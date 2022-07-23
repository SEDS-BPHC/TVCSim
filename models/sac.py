import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

epsilon = 1e-6


def weights_init_(m):
	if isinstance(m, nn.Linear):
		torch.nn.init.xavier_normal_(m.weight, gain=1)
		torch.nn.init.constant(m.bias, 0)


class Critic(nn.Module):
	def __init__(self, num_inputs, num_actions, hidden_layers):
		super(Critic, self).__init__()

		self.linear1 = nn.Linear(num_inputs + num_actions, hidden_layers)
		self.linear2 = nn.Linear(hidden_layers, hidden_layers)
		self.linear3 = nn.Linear(hidden_layers, 1)

		self.linear4 = nn.Linear(num_inputs + num_actions, hidden_layers)
		self.linear5 = nn.Linear(hidden_layers, hidden_layers)
		self.linear6 = nn.Linear(hidden_layers, 1)

		self.apply(weights_init_)

	def forward(self, state, action):
		xu = torch.cat([state, action], 1)

		x1 = F.relu(self.linear1(xu))
		x1 = F.relu(self.linear2(x1))
		x1 = self.linear3(x1)

		x2 = F.relu(self.linear4(xu))
		x2 = F.relu(self.linear5(x2))
		x2 = self.linear6(x2)

		return x1, x2


def hard_update(target, source):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(param.data)


class SAC(object):
	def __init__(self, num_inputs, action_space, args):
		self.gamma = args.gamma
		self.tau = args.tau

		self.policy_type = args.policy
		self.target_update_interval = args.target_update_interval
		self.automatic_entropy_tuning = args.automatic_entropy_tuning

		self.device = torch.device('cuda' if args.cuda else "cpu")

		self.critic = Critic(num_inputs, action_space, 32).to(self.device)

		self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

		self.critic_target = Critic(num_inputs, action_space, 32).to(device=self.device)
		hard_update(self.critic_target, self.critic)

		if self.automatic_entropy_tuning:
			self.target_entropy = -torch.prod(torch.Tensor(action_space.shape)).to(self.device).itemn()
			self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
			self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
		self.actor = Actor(num_inputs, action_space.shape, 32, action_space)


class Actor(nn.Module):
	def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
		super(Actor, self).__init__()
		self.linear1 = nn.Linear(num_inputs, hidden_dim)
		self.linear2 = nn.Linear(hidden_dim, hidden_dim)

		self.mean = nn.Linear(hidden_dim, num_actions)
		self.noise = torch.Tensor(num_actions)

		self.apply(weights_init_)

		# action rescaling
		if action_space is None:
			self.action_scale = 1.
			self.action_bias = 0.
		else:
			self.action_scale = torch.FloatTensor(
				(action_space.high - action_space.low) / 2.)
			self.action_bias = torch.FloatTensor(
				(action_space.high + action_space.low) / 2.)

	def forward(self, state):
		x = F.relu(self.linear1(state))
		x = F.relu(self.linear2(x))
		mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
		return mean

	def sample(self, state):
		mean = self.forward(state)
		noise = self.noise.normal_(0., std=0.1)
		noise = noise.clamp(-0.25, 0.25)
		action = mean + noise
		return action, torch.tensor(0.), mean

	def to(self, device):
		self.action_scale = self.action_scale.to(device)
		self.action_bias = self.action_bias.to(device)
		self.noise = self.noise.to(device)
		return super(Actor, self).to(device)
