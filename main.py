import torch
import torch.optim as optim
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import safety_gym
import mujoco_py
from collections import deque
import copy
import gym
import random

#device, CPU or the GPU of your choice
GPU = 0
DEVICE = device = torch.device('cuda')

#environment names
RAM_ENV_NAME = 'Safexp-PointGoal2-v0'
VISUAL_ENV_NAME = None

#Agent parameters
BATCH_SIZE = 256
LR1 = 0.0001
LR2 = 0.001
SPEED1 = 1
SPEED2 = 1
STEP = 1
TAU = 0.001
LEARNING_TIME = 1
OUN=True,
BN=False,
CLIP=False,
INIT=True,
HIDDEN=[400, 300]

#Training parameters
L = 1000
N = 1


class ReplayBuffer:

    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)

    def add(self, state, action, reward, log,next_state, done):
        self.memory.append((state, action, reward,log, next_state, done))

    def sample(self, batchsize):
        experiences = random.sample(self.memory, k=batchsize)

        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(device)
        costs = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[4] for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[5] for e in experiences]).astype(np.uint8)).float().to(device)
        return states, actions, rewards,costs, next_states, dones


class OUNoise:

    def __init__(self, action_size, mu=0, theta=0.15, sigma=0.05):
        self.action_size = action_size
        self.mu = mu * np.ones(action_size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.uniform(size=self.action_size)
        self.state = x + dx
        return self.state

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1 / math.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):

    def __init__(self, state_size, action_size, batchnorm, initialize, hidden=[256, 256]):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], action_size)
        self.bn1 = nn.BatchNorm1d(hidden[0])
        self.bn2 = nn.BatchNorm1d(hidden[1])
        self.batchnorm = batchnorm
        if initialize:
            self.initialize()

    def forward(self, x):
        if self.batchnorm:
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

    def initialize(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.bias.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.bias.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)


class Critic(nn.Module):

    def __init__(self, state_size, action_size,out_size, batchnorm, initialize, hidden=[256, 256]):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0] + action_size, hidden[1])
        self.fc3 = nn.Linear(hidden[1], out_size)
        self.bn1 = nn.BatchNorm1d(hidden[0])
        self.batchnorm = batchnorm
        if initialize:
            self.initialize()

    def forward(self, state, action):
        if self.batchnorm:
            x = F.relu(self.bn1(self.fc1(state)))
        else:
            x = F.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def initialize(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.bias.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.bias.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-4, 3e-4)
        self.fc3.bias.data.uniform_(-3e-4, 3e-4)


class DDPG_Agent:

    def __init__(self, env,num_cost, lr1=0.0001, lr2=0.001, tau=0.001, speed1=1, speed2=1, step=1, learning_time=1,
                 batch_size=64, OUN_noise=True, batchnorm=True, clip=True, initialize=True, hidden=[256, 256]):

        # Initialize environment
        state_size, action_size = env.observation_space.shape[0], env.action_space.shape[0]
        self.env = env

        # Initialize some hyper parameters of agent
        self.lr1 = lr1
        self.lr2 = lr2
        self.tau = tau
        self.speed1 = speed1
        self.speed2 = speed2
        self.learning_time = learning_time
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = 0.99
        self.step = step
        self.OUN_noise = OUN_noise
        self.batchnorm = batchnorm
        self.clip = clip
        self.initialize = initialize
        self.hidden = hidden
        self.num_cost=num_cost-1

        # Initialize agent (networks, replyabuffer and noise)
        self.actor_local = Actor(self.state_size, self.action_size, self.batchnorm, initialize, hidden).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, self.batchnorm, initialize, hidden).to(device)
        self.critic_local = Critic(self.state_size, self.action_size,1, self.batchnorm, initialize, hidden).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, 1,self.batchnorm, initialize, hidden).to(device)
        self.cost_local = Critic(self.state_size, self.action_size, self.num_cost , self.batchnorm, initialize,
                                 hidden).to(device)
        self.cost_target = Critic(self.state_size, self.action_size, self.num_cost , self.batchnorm, initialize,
                                  hidden).to(device)
        # self.actor_local.eval()
        # self.actor_target.eval()
        # self.critic_local.eval()
        # self.critic_target.eval()

        self.soft_update(self.actor_local, self.actor_target, 1)
        self.soft_update(self.critic_local, self.critic_target, 1)
        self.soft_update(self.cost_local, self.cost_target, 1)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr1)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr2)
        self.cost_optimizer = optim.Adam(self.cost_local.parameters(), lr=self.lr2)
        self.memory = ReplayBuffer(int(1e5))
        self.noise = OUNoise(action_size=self.action_size)

    def act(self, state, i):
        state = torch.tensor(state, dtype=torch.float).to(device).view(1, -1)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).detach().view(-1).cpu().numpy()
        self.actor_local.train()
        if self.OUN_noise:
            noise = self.noise.sample()
        else:
            noise = self.noise.sigma * np.random.standard_normal(self.action_size)
        action += noise / math.sqrt(i)
        action = np.clip(action, -1, 1)
        return action

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for local_layer, target_layer in zip(local_model.modules(), target_model.modules()):
            for local_parameter, target_parameter in zip(local_layer.parameters(), target_layer.parameters()):
                target_parameter.data.copy_(tau * local_parameter.data + (1 - tau) * target_parameter.data)
            try:
                target_layer.running_mean = tau * local_layer.running_mean + (1 - tau) * target_layer.running_mean
                target_layer.running_var = tau * local_layer.running_var + (1 - tau) * target_layer.running_var
            except:
                None

    def learn(self):
        # Calculate the total times of actor update on reward or cost direction
        t1=0
        t2=0

        #sample from replay buffer
        experiences = self.memory.sample(256)
        states, actions, rewards, costs,next_states, dones = experiences
        rewards=rewards*torch.abs(costs.mean()/rewards.mean())

        # Reward_critic update
        with torch.no_grad():
            expected_rewards = rewards + (1 - dones) * self.gamma * self.critic_target(next_states,
                                                                                       self.actor_target(next_states))

        for _ in range(self.speed1):
            observed_rewards = self.critic_local(states, actions)
            L = (expected_rewards - observed_rewards).pow(2).mean()
            self.critic_optimizer.zero_grad()
            L.backward()
            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
            self.critic_optimizer.step()

        # Cost critic update
        dones = torch.Tensor(self.batch_size, self.num_cost).copy_(dones).to(device)
        with torch.no_grad():
            expected_costs = costs+ (1 - dones) * self.gamma * self.cost_target(next_states,self.actor_target( next_states))
        for _ in range(self.speed1):
            observed_costs = self.cost_local(states, actions)
            L = (expected_costs - observed_costs).pow(2).mean()
            self.cost_optimizer.zero_grad()
            L.backward()
            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
            self.cost_optimizer.step()

        # Determine the direction of policy update
        J_function=observed_costs.mean(dim=0).detach()
        penalty_objects = []
        for i in range(len(J_function.shape)):
            if J_function[i] >= 0.3:
                penalty_objects.append(i)

        # Preform reward direction update
        if len(penalty_objects)==0:
            for _ in range(self.speed2):
                t1+=1
                L = - self.critic_local(states, self.actor_local(states)).mean()
                self.actor_optimizer.zero_grad()
                L.backward()
                if self.clip:
                    torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
                self.actor_optimizer.step()

        #Preform cost direction update
        else:
            for _ in range(self.speed2):
                t2+=1
                update_index = random.choice(penalty_objects)
                L = self.cost_local(states, self.actor_local(states)).mean(dim=0)[update_index]
                self.actor_optimizer.zero_grad()
                L.backward()
                if self.clip:
                    torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
                self.actor_optimizer.step()

        # Update target network
        self.soft_update(self.actor_local, self.actor_target, self.tau)
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.cost_local, self.cost_target, self.tau)
        return t1,t2


def train(n_episodes, env, agent):
    rewards = []
    average_rewards = []
    costs = []
    average_cost = []
    for i in range(1, n_episodes + 1):
        episodic_reward = 0
        episodic_cost = 0
        reward_update = 0
        cost_update = 0
        state = env.reset()
        agent.noise.reset()
        action = agent.act(state, i)
        t = 0

        while True:
            next_state, reward, done, info = env.step(action)
            episodic_reward += reward
            episodic_cost += info['cost']
            log = []
            for key in info.keys():
                if (key != 'cost' and key != 'goal_met'):
                    log.append(info[key])
            agent.memory.add(state, action, reward,log, next_state, done)
            t += 1

            if len(agent.memory.memory) > agent.batch_size:
                if t % agent.step == 0:
                    for _ in range(agent.learning_time):
                        t1,t2=agent.learn()
                        reward_update+=t1
                        cost_update+=t2
            if done:
                break

            state = next_state.copy()
            action = agent.act(state, i)

        rewards.append(episodic_reward)
        costs.append(episodic_cost)
        average_rewards.append(np.mean(rewards[-100:]))
        average_cost.append(np.mean(costs[-100:]))

        print(
            '\rEpisode {}, Reward {:.2f}, Average Reward {:.2f}, Cost {:.2f}, Average Cost {:.2f}, '
            'reward update times:{}, cost update times:{}'.format(
                i, episodic_reward, average_rewards[-1], episodic_cost, average_cost[-1], reward_update, cost_update),
            end='')
        if i % 100 == 0:
            print('')

    return rewards, average_rewards


if __name__ == '__main__':
    env = gym.make(RAM_ENV_NAME)

    rewards = np.zeros((N, L))
    average_rewards = np.zeros((N, L))
    state = env.reset()
    action = np.random.randn(env.action_space.shape[0])
    _, _, _, info = env.step(action)
    num_cost = len(info)
    for i in range(N):
        print('{}/{}'.format(i + 1, N))
        agent = DDPG_Agent(env=env,
                           num_cost=num_cost,
                           lr1=LR1,
                           lr2=LR2,
                           tau=TAU,
                           speed1=SPEED1,
                           speed2=SPEED2,
                           step=STEP,
                           learning_time=LEARNING_TIME,
                           batch_size=BATCH_SIZE,
                           OUN_noise=OUN,
                           batchnorm=BN,
                           clip=CLIP,
                           initialize=INIT,
                           hidden=HIDDEN)
        rewards[i, :], average_rewards[i, :] = train(L, env, agent)

    np.save('./rewards_{}_{}_{}_{}_{}_{}.npy'.format(agent.OUN_noise, agent.batchnorm, agent.clip, agent.initialize,
                                                     agent.hidden[0], agent.hidden[1]), rewards)
    np.save(
        './average_rewards_{}_{}_{}_{}_{}_{}.npy'.format(agent.OUN_noise, agent.batchnorm, agent.clip, agent.initialize,
                                                         agent.hidden[0], agent.hidden[1]), average_rewards)