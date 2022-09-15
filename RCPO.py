import torch
import torch.optim as optim
import numpy as np
import math
import gym
import safety_gym
import mujoco_py
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
#device, CPU or the GPU of your choice
GPU = 0
DEVICE = torch.device('cuda')

#environment names
RAM_CONTINUOUS_ENV_NAME = 'Safexp-PointGoal2-v0'
CONSTANT = 90

#Agent parameters
LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC=1.5e-4
LEARNING_RATE_LAM=1e-6
LAM_DECAY=1-1e-9
GAMMA = 0.99
BETA = 0
EPS = 0.1
TAU = 0.99
MODE = 'TD'
SHARE = False
CRITIC = True
NORMALIZE = False
HIDDEN_CONTINUOUS = [64,64]
ALPHA=0.05

#Training parameters
RAM_NUM_EPISODE = 1000
VISUAL_NUM_EPISODE = 5000
SCALE = 1
MAX_T = 10000
NUM_FRAME = 2
N_UPDATE = 5
UPDATE_FREQUENCY = 1


class Actor_continuous(nn.Module):

    def __init__(self, state_size, action_size, hidden):
        super(Actor_continuous, self).__init__()
        hidden = [state_size] + hidden
        self.feature = nn.ModuleList(nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(hidden[:-1], hidden[1:]))
        self.mu = nn.Linear(hidden[-1], action_size)
        self.logstd = nn.Linear(hidden[-1], action_size)

    def forward(self, state):
        x = state
        for layer in self.feature:
            x = F.relu(layer(x))
        mu = self.mu(x)
        logstd = self.logstd(x)
        return mu, logstd


class Critic(nn.Module):

    def __init__(self, state_size, hidden):
        super(Critic, self).__init__()
        hidden = [state_size] + hidden
        self.feature = nn.ModuleList(nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(hidden[:-1], hidden[1:]))
        self.output = nn.Linear(hidden[-1], 1)

    def forward(self, state):
        x = state
        for layer in self.feature:
            x = F.relu(layer(x))
        values = self.output(x)
        return values


class Actor_Critic_continuous(nn.Module):

    def __init__(self, state_size, action_size, hidden):
        super(Actor_Critic_continuous, self).__init__()
        hidden = [state_size] + hidden
        self.feature = nn.ModuleList(nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(hidden[:-1], hidden[1:]))
        self.mu = nn.Linear(hidden[-1], action_size)
        self.logstd = nn.Linear(hidden[-1], action_size)
        self.output = nn.Linear(hidden[-1], 1)

    def forward(self, state):
        x = state
        for layer in self.feature:
            x = F.relu(layer(x))
        mu = self.mu(x)
        logstd = self.logstd(x)
        values = self.output(x)
        return mu, logstd, values


class Agent_continuous:

    def __init__(self, state_size, action_size, lr_actor,lr_critic, beta, eps, tau, gamma, device, hidden, share=False,
                 mode='MC', use_critic=False, normalize=False):
        self.state_size = state_size
        self.action_size = action_size
        self.lr_actor = lr_actor
        self.lr_critic=lr_critic
        self.beta = beta
        self.eps = eps
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.share = share
        self.mode = mode
        self.use_critic = use_critic
        self.normalize = normalize

        if self.share:
            self.Actor_Critic = Actor_Critic_continuous(self.state_size, self.action_size, hidden).to(self.device)
            self.optimizer = optim.Adam(self.Actor_Critic.parameters(), lr_actor)
        else:
            self.Actor = Actor_continuous(state_size, action_size, hidden).to(self.device)
            self.Critic = Critic(state_size, hidden).to(self.device)
            self.actor_optimizer = optim.Adam(self.Actor.parameters(), lr_actor)
            self.critic_optimizer = optim.Adam(self.Critic.parameters(), lr_critic)

    def act(self, states):
        with torch.no_grad():
            states = torch.tensor(states).float().view(-1, self.state_size).to(self.device)
            if self.share:
                mu, logstd, _ = self.Actor_Critic(states)
            else:
                mu, logstd = self.Actor(states)
            actions = torch.distributions.Normal(mu, logstd.exp()).sample()
            actions = actions.cpu().numpy().reshape(-1)
        return actions

    def process_data(self, states, actions, rewards,costs, dones, batch_size):

        actions.append(np.zeros(self.action_size))
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device).view(-1, self.action_size)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device).view(-1, 1)

        # calculate log probabilities and state values
        N = states.size(0)  # N-1 is the length of actions, rewards and dones
        log_probs = torch.zeros((N, self.action_size)).to(self.device)
        old_mu = torch.zeros((N, self.action_size)).to(self.device)
        old_logstd = torch.zeros((N, self.action_size)).to(self.device)
        step = math.ceil(N / batch_size)

        for ind in range(step):
            if self.share:
                mu, logstd, _ = self.Actor_Critic(states[ind * batch_size:(ind + 1) * batch_size, :])
            else:
                mu, logstd = self.Actor(states[ind * batch_size:(ind + 1) * batch_size, :])
            distribution = torch.distributions.normal.Normal(mu, logstd.exp())
            log_probs[ind * batch_size:(ind + 1) * batch_size, :] = distribution.log_prob(
                actions[ind * batch_size:(ind + 1) * batch_size, :])
            old_mu[ind * batch_size:(ind + 1) * batch_size, :] = mu
            old_logstd[ind * batch_size:(ind + 1) * batch_size, :] = logstd

        log_probs = log_probs[:-1, :]  # remove the last one, which corresponds to no actions
        old_mu = old_mu[:-1, :]
        old_logstd = old_logstd[:-1, :]
        actions = actions[:-1, :]
        log_probs = log_probs.sum(dim=1, keepdim=True)

        rewards = np.array(rewards)
        costs = np.array(costs)
        # r_t

        return states, actions, old_mu.detach(), old_logstd.detach(), log_probs.detach(), rewards,costs, dones

    def learn(self, states, actions, old_mu, old_logstd, log_probs, rewards, costs,lam,dones):
        if self.share:
            new_mu, new_logstd, state_values = self.Actor_Critic(states)
            new_mu = new_mu[:-1, :]
            new_logstd = new_logstd[:-1, :]
        else:
            new_mu, new_logstd = self.Actor(states)
            new_mu = new_mu[:-1, :]
            new_logstd = new_logstd[:-1, :]
            state_values = self.Critic(states)
        new_distribution = torch.distributions.normal.Normal(new_mu, new_logstd.exp())
        new_log_probs = new_distribution.log_prob(actions).sum(dim=1, keepdim=True)

        KL = new_logstd - old_logstd + (old_logstd.exp().pow(2) + (new_mu - old_mu).pow(2)) / (
                    2 * new_logstd.exp().pow(2) + 1e-6) - 0.5
        KL = KL.sum(dim=1, keepdim=True)

        L = rewards.shape[0]
        with torch.no_grad():
            G = []
            return_value = 0
            if self.mode == 'MC':
                for i in range(L - 1, -1, -1):
                    return_value = rewards[i] + self.gamma * return_value * (1 - dones[i])
                    G.append(return_value)
                G = G[::-1]
                G = torch.tensor(G, dtype=torch.float).view(-1, 1).to(self.device)
            else:
                rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
                costs = torch.tensor(costs, dtype=torch.float).view(-1, 1).to(self.device)
                lam=torch.tensor(lam,dtype=torch.float).to(self.device)
                G = rewards-lam*costs + (1 - dones) * self.gamma * state_values[1:, :]

        Critic_Loss = 0.5 * (state_values[:-1, :] - G).pow(2).mean()

        with torch.no_grad():
            if self.use_critic:
                G = G - state_values[:-1, :]  # advantage
            for i in range(L - 2, -1, -1):
                G[i] += G[i + 1] * self.gamma * (1 - dones[i]) * self.tau  # cumulated advantage
            if self.normalize:
                G = (G - G.mean()) / (G.std() + 0.00001)

        ratio = (new_log_probs - log_probs).exp()
        Actor_Loss1 = ratio * G
        Actor_Loss2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * G
        Actor_Loss = -torch.min(Actor_Loss1, Actor_Loss2)
        Actor_Loss += self.beta * KL
        Actor_Loss = Actor_Loss.mean()

        if self.share:
            Loss = Actor_Loss + Critic_Loss
            self.optimizer.zero_grad()
            Loss.backward()
            torch.nn.utils.clip_grad_norm_(self.Actor_Critic.parameters(), 1)
            self.optimizer.step()
        else:
            self.critic_optimizer.zero_grad()
            Critic_Loss.backward()
            torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), 1)
            self.critic_optimizer.step()
            self.actor_optimizer.zero_grad()
            Actor_Loss.backward()
            torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), 1)
            self.actor_optimizer.step()


def train(agent, env, n_episode, n_update=8, update_frequency=1, max_t=1500, scale=1):
    rewards_log = []
    average_log = []
    cost_log=[]
    average_cost=[]
    state_history = []
    action_history = []
    done_history = []
    reward_history = []
    cost_history=[]
    lam=0.0
    lr_lam=LEARNING_RATE_LAM
    log1=[]
    log2=[]

    for i in range(1, n_episode + 1):
        state = env.reset()
        done = False
        t = 0
        if len(state_history) == 0:
            state_history.append(list(state))
        else:
            state_history[-1] = list(state)
        episodic_reward = 0
        episodic_cost=0

        while not done and t < max_t:
            t+=1
            action = agent.act(state)
            next_state, reward, done, cost = env.step(action)
            cost=cost['cost']
            episodic_reward += reward
            episodic_cost+=cost
            log1.append(reward)
            log2.append(cost)
            action_history.append(action)
            done_history.append(done)
            reward_history.append(reward * scale)
            cost_history.append(cost*scale)
            state = next_state
            state_history.append(list(state))
            lr_lam *= LAM_DECAY

        if i % update_frequency == 0:
            states, actions, old_mu, old_logstd, log_probs, rewards, costs,dones = agent.process_data(state_history,
                                                                                                action_history,
                                                                                                reward_history,
                                                                                                cost_history,
                                                                                                done_history,
                                                                                                      648)
            for _ in range(n_update):
                agent.learn(states, actions, old_mu, old_logstd, log_probs, rewards,costs, lam,dones)
            state_history = []
            action_history = []
            done_history = []
            reward_history = []
            cost_history=[]

        dlam = episodic_cost - ALPHA*t
        lam_temp = lam + dlam*lr_lam
        if (lam_temp > 0):
            lam = lam_temp
        else:
            lam = 0
        rewards_log.append(episodic_reward)
        average_log.append(np.mean(rewards_log[-100:]))
        cost_log.append(episodic_cost)
        average_cost.append(np.mean(cost_log[-100:]))

        print('\rEpisode {}, Total step {} Reward {:.2f}, Average Reward {:.2f}, Cost {:.2f}, Average Cost {:.2f},dlam:{:.2f},lam:{:.7f}'.format(i, t,episodic_reward, average_log[-1],episodic_cost,average_cost[-1],dlam,lam), end='')
        if not done:
            print('\nEpisode {} did not end'.format(i))
        if i % 200 == 0:
            print()

    return rewards_log, average_log,cost_log,average_cost


env = gym.make(RAM_CONTINUOUS_ENV_NAME)
agent = Agent_continuous(state_size=env.observation_space.shape[0],
                             action_size=env.action_space.shape[0],
                             lr_actor=LEARNING_RATE_ACTOR,
                             lr_critic=LEARNING_RATE_CRITIC,
                             beta=BETA,
                             eps=EPS,
                             tau=TAU,
                             gamma=GAMMA,
                             device=DEVICE,
                             hidden=HIDDEN_CONTINUOUS,
                             share=SHARE,
                             mode=MODE,
                             use_critic=CRITIC,
                             normalize=NORMALIZE)
rewards_log, average_reward,cost_log,average_cost= train(agent=agent,
                           env=env,
                           n_episode=RAM_NUM_EPISODE,
                           n_update=N_UPDATE,
                           update_frequency=UPDATE_FREQUENCY,
                           max_t=MAX_T,
                           scale=SCALE)
x=np.arange(len(rewards_log))
plt.plot(x,rewards_log)
plt.show()
plt.plot(x,cost_log)
plt.show()
np.save('{}_rewards.npy'.format(RAM_CONTINUOUS_ENV_NAME), rewards_log)
np.save('{}_costs.npy'.format(RAM_CONTINUOUS_ENV_NAME), cost_log)