import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import d4rl
import random
import argparse
from distutils.util import strtobool
import os
import time
from diffusion_model import UNetDiffusion
from data_utils import AntMazeDiffusionDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, models will be saved")
    parser.add_argument("--wandb-project-name", type=str, default="advantage-diffusion",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default='swish',
        help="the entity (team) of wandb's project")
    
    parser.add_argument("--env-id", type=str, default="antmaze-medium-diverse-v1",
        help="the id of the environment")
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=float, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--schedule", type=str, default='linear')
    parser.add_argument("--n-epochs", type=int, default=100000)
    parser.add_argument("--predict", type=str, default='epsilon')
    parser.add_argument("--num-eval", type=int, default=5)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    filename = args.env_id+"_"+args.exp_name
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    env = gym.make(args.env_id)
    dataset = env.get_dataset()
    data = AntMazeDiffusionDataset(dataset, H=args.H)
    
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    
    model = UNetDiffusion(s_dim=env.observation_space.shape[0], a_dim=env.action_space.shape[0], H=args.H, diffusion_steps=args.diffusion_steps, predict=args.predict).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(args.n_epochs):
        diffusion_loss_epoch = 0.0
        idm_loss_epoch = 0.0
        for states, s0, sg, actions in dataloader:
            states = states.float().to(device)
            actions = actions.float().to(device)
            s0 = s0.float().to(device)
            sg = sg.float().to(device)
            optimizer.zero_grad()
            diffusion_loss = model.compute_diffusion_loss(states)
            idm_loss = model.compute_idm_loss(s0, sg, actions)
            loss = diffusion_loss + idm_loss
            loss.backward()
            optimizer.step()
            diffusion_loss_epoch += diffusion_loss.item()
            idm_loss_epoch += idm_loss.item()
        if args.save_model and epoch % 1000 == 0:
            torch.save(model.state_dict(), "models/"+filename+".pth")
            writer.add_scalar("losses/diffusion_loss", diffusion_loss_epoch, epoch)
            writer.add_scalar("losses/idm_loss", idm_loss_epoch, epoch)