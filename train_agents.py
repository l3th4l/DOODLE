import os
import math
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from gymnasium import spaces
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from env import DifferentiableHeliostatEnv 
# -------------------------
# Global Hyperparameters
# -------------------------
SEED = 123
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPISODES = 1500
MAX_STEPS = 10
LR = 0.000002
GAMMA = 0.99
TAU = 0.005


# -------------------------
# Simple Policy Gradient (PG) Agent
# -------------------------
class PGPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(PGPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

def train_pg_agent(env, policy_net, optimizer, num_episodes=200):
    device = env.device
    rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        obs = obs.to(device)
        cumulative_reward = 0.0
        optimizer.zero_grad()
        for t in range(env.max_steps):
            obs_tensor = obs.unsqueeze(0)
            action = policy_net(obs_tensor).squeeze(0)
            obs, reward, done, _, _ = env.step(action)
            cumulative_reward += reward
            obs = obs.to(device)
            if done:
                break
        loss = -cumulative_reward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        optimizer.step()
        cumulative_reward_scaled = cumulative_reward.detach().cpu()/(env.max_steps * env.num_heliostats)
        rewards.append(cumulative_reward_scaled)
        print(f"PG Agent - Episode {episode}: Reward = {cumulative_reward_scaled}")
    return rewards

# -------------------------
# Simple SAC Agent
# -------------------------
class SimpleSACPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(SimpleSACPolicy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 128)
        )
        self.mean_layer = nn.Linear(128, action_dim)
        self.log_std_layer = nn.Linear(128, action_dim)
    def forward(self, x):
        h = self.fc(x)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std
    def sample(self, x):
        unsqueezed = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            unsqueezed = True
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t  # scale=1, bias=0 for simplicity
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        if unsqueezed:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            mean = mean.squeeze(0)
        return action, log_prob, mean

class SimpleQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(SimpleQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

class SACAgent(object):
    def __init__(self, env, lr=0.0003, gamma=0.99, tau=0.005, num_episodes=NUM_EPISODES):
        self.env = env
        self.device = torch.device(DEVICE)
        self.gamma = gamma
        self.tau = tau

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        self.policy = SimpleSACPolicy(obs_dim, act_dim).to(self.device)
        self.q1 = SimpleQNetwork(obs_dim, act_dim).to(self.device)
        self.q2 = SimpleQNetwork(obs_dim, act_dim).to(self.device)
        self.q_target1 = SimpleQNetwork(obs_dim, act_dim).to(self.device)
        self.q_target2 = SimpleQNetwork(obs_dim, act_dim).to(self.device)
        self.q_target1.load_state_dict(self.q1.state_dict())
        self.q_target2.load_state_dict(self.q2.state_dict())

        # Automatic entropy tuning.
        self.automatic_entropy_tuning = True
        if self.automatic_entropy_tuning:
            self.target_entropy = -float(act_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = ALPHA

        self.policy_opt = optim.Adam(self.policy.parameters(), lr=lr, weight_decay=0.1)
        self.q_opt = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr, weight_decay=0.1)

        # Create linear decay LR schedulers (linearly decay from 1.0 to 0.0 over num_episodes).
        self.policy_scheduler = optim.lr_scheduler.LambdaLR(self.policy_opt, lr_lambda=lambda ep: 1 - ep/num_episodes)
        self.q_scheduler = optim.lr_scheduler.LambdaLR(self.q_opt, lr_lambda=lambda ep: 1 - ep/num_episodes)
        if self.automatic_entropy_tuning:
            self.alpha_scheduler = optim.lr_scheduler.LambdaLR(self.alpha_opt, lr_lambda=lambda ep: 1 - ep/num_episodes)

    def select_action(self, state, evaluate=False):
        state_scale = 0.1
        state = state * state_scale
        if state.dim() == 1:
            state = state.unsqueeze(0)
        action, _, _ = self.policy.sample(state)
        return action.squeeze(0)

    def train_episode(self):
        obs = self.env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            state = obs.to(self.device).detach()
            action = self.select_action(state).detach()  # Detach to prevent gradients through env.
            next_obs, reward, done, _, _ = self.env.step(action)
            episode_reward += reward.item() if isinstance(reward, torch.Tensor) else reward
            next_state = next_obs.to(self.device).detach()

            with torch.no_grad():
                next_action, next_log_prob, _ = self.policy.sample(next_state.unsqueeze(0))
                q1_next = self.q_target1(next_state.unsqueeze(0), next_action)
                q2_next = self.q_target2(next_state.unsqueeze(0), next_action)
                q_next = torch.min(q1_next, q2_next) - next_log_prob
                target_q = reward + self.gamma * q_next.item() if not done else reward

            state_input = state.unsqueeze(0)
            action_input = action.unsqueeze(0)
            q1_val = self.q1(state_input, action_input)
            q2_val = self.q2(state_input, action_input)
            target_tensor = torch.tensor([[target_q]], device=self.device, dtype=torch.float32)
            q_loss = F.mse_loss(q1_val, target_tensor) + F.mse_loss(q2_val, target_tensor)

            self.q_opt.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 1.0)
            self.q_opt.step()

            new_action, log_prob, _ = self.policy.sample(state_input)
            q1_pi = self.q1(state_input, new_action)
            q2_pi = self.q2(state_input, new_action)
            q_pi = torch.min(q1_pi, q2_pi)
            if self.automatic_entropy_tuning:
                policy_loss = (self.alpha * log_prob - q_pi).mean()
                # Update entropy coefficient.
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_opt.zero_grad()
                alpha_loss.backward()
                torch.nn.utils.clip_grad_norm_([self.log_alpha], 1.0)
                self.alpha_opt.step()
                self.alpha = self.log_alpha.exp()
            else:
                policy_loss = (log_prob - q_pi).mean()

            self.policy_opt.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.policy_opt.step()

            # Polyak averaging for target networks.
            for target, net in zip([self.q_target1, self.q_target2], [self.q1, self.q2]):
                for target_param, param in zip(target.parameters(), net.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

            obs = next_obs.detach()
        return episode_reward

    def step_schedulers(self, current_episode):
        self.policy_scheduler.step(current_episode)
        self.q_scheduler.step(current_episode)
        if self.automatic_entropy_tuning:
            self.alpha_scheduler.step(current_episode)

def train_sac_agent(env, sac_agent, num_episodes=200):
    rewards = []
    for episode in range(num_episodes):
        ep_reward = sac_agent.train_episode()
        ep_reward_scaled = ep_reward / (env.max_steps * env.num_heliostats)
        rewards.append(ep_reward_scaled)
        sac_agent.step_schedulers(episode)
        print(f"SAC Agent - Episode {episode}: Reward = {ep_reward_scaled}")
    return rewards


# -------------------------
# Main Training Loop and Comparison
# -------------------------
def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    TRAIN_SAC = False

    #torch.manual_seed(SEED)
    #np.random.seed(SEED)
    #random.seed(SEED)

    # Create the environment.
    '''
    env = GaussianBlobEnv(image_size=IMG_SIZE, num_blobs=NUM_BLOBS, sigma=SIGMA,
                          amplitude=AMPLITUDE, max_steps=MAX_STEPS, observation_type="true_positions", DEVICE=DEVICE)
    '''
    env = DifferentiableHeliostatEnv(control_method='m_pos', num_heliostats=50, device=DEVICE, error_magnitude=1.145916)
    env.reset()
    env.render()

    # Compute dimensions.
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    # Initialize PG Agent.
    pg_policy = PGPolicy(obs_dim, act_dim).to(env.device)
    pg_optimizer = optim.AdamW(pg_policy.parameters(), lr=0.001, weight_decay=0.05)
    print("Training PG Agent...")
    pg_rewards = train_pg_agent(env, pg_policy, pg_optimizer, num_episodes=NUM_EPISODES)

    env.render(mode="rgb_array")

    if TRAIN_SAC:
        # Reset environment.
        env.reset()
        # Initialize SAC Agent.
        sac_agent = SACAgent(env, lr=LR, gamma=GAMMA, tau=TAU)
        print("Training SAC Agent...")
        sac_rewards = train_sac_agent(env, sac_agent, num_episodes=NUM_EPISODES)

    def moving_average(data, window=10):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    def running_std(data, window=10):
        stds = []
        for i in range(len(data) - window + 1):
            stds.append(np.std(data[i:i+window]))
        return np.array(stds)

    window = 20
    
    episodes_smoothed = np.arange(window - 1, NUM_EPISODES)
    
    pg_avg = moving_average(pg_rewards, window)
    pg_std = running_std(pg_rewards, window)

    if TRAIN_SAC:
        sac_avg = moving_average(sac_rewards, window)
        sac_std = running_std(sac_rewards, window)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes_smoothed, pg_avg, label="APG Agent (smoothed)")
    plt.fill_between(episodes_smoothed, pg_avg - pg_std, pg_avg + pg_std, alpha=0.3)

    if TRAIN_SAC:
        plt.plot(episodes_smoothed, sac_avg, label="SAC Agent (smoothed)")
        plt.fill_between(episodes_smoothed, sac_avg - sac_std, sac_avg + sac_std, alpha=0.3)
            
        plt.title("Reward Comparison: APG vs SAC (Smoothed) [Linear Scale]")
    else:
            
        plt.title("Reward: APG (Smoothed) [Linear Scale]")

    plt.xlabel("Episode")
    plt.ylabel("Average Reward per heliostat (meters)")
    
    plt.legend()
    plt.show()


    plt.figure(figsize=(10, 6))
    plt.plot(episodes_smoothed, pg_avg , label="APG Agent (smoothed)")

    if TRAIN_SAC:
        plt.plot(episodes_smoothed, sac_avg, label="SAC Agent (smoothed)")

        plt.title("Reward Comparison: APG vs SAC (Smoothed) [Log Scale]")
    else:
            
        plt.title("Reward: APG (Smoothed) [Log Scale]")

    plt.xlabel("Episode")
    plt.ylabel("Average Reward (meters)")
    plt.legend()
    plt.yscale('log')
    plt.show()

    env.close()

if __name__ == '__main__':
    main()