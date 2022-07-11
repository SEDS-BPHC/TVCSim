from torch import nn
from torch import distributions
from torch import optim
import numpy as np
from TVCSim.env import env


class Actor(nn.Module):
	def __init__(self, hidden_layers, input_size, output_size):
		super(Actor, self).__init__()
		self.hidden = [nn.Linear(input_size, hidden_layers[0])]
		for i in range(1, len(hidden_layers)):
			self.hidden.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax(dim=output_size)

	def forward(self, x):
		for layer in self.hidden:
			x = layer(x)
			x = self.relu(x)
		x = self.softmax(x)
		return x

	def predict_action(self, x):
		probs = self.forward(x)
		dist = distributions.Categorical(probs)
		action = m.sample()
		log_prob = dist.log_prob(action)
		return action, log_prob


class Critic(nn.Module):
	def __init__(self, hidden_layers, input_size):
		super(Critic, self).__init__()
		self.hidden = []
		self.hidden = [nn.Linear(input_size, hidden_layers[0])]
		for i in range(1, len(hidden_layers)):
			self.hidden.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
		self.relu = nn.ReLU()
		self.output = nn.Linear(hidden_layers[-1], 1)

	def forward(self, x):
		for layer in self.hidden:
			x = layer(x)
			x = self.relu(x)
		x = self.output(x)
		return x

	def q_value(self, x):
		return self.forward(x)


class Agent:
	def __init__(self, state_dims, action_space, actor_hidden_layer=[64, 32], critic_hidden_layer=[32], gamma=0.9,
				 l_r=0.01):
		self.actor = Actor(actor_hidden_layer, state_dims, action_space)
		self.critic = Critic(critic_hidden_layer, state_dims)
		self.optimizer = optim.Adam(np.array([self.actor.parameters(), self.critic.parameters()]), lr=l_r)
		self.action_buffer = []
		self.log_probs_buffer = []
		self.q_value_buffer = []
		self.gamma = gamma

	def predict_action(self, x):
		action, log_probs = self.actor.predict_action(x)
		q_value = self.critic.forward(x)
		self.action_buffer.append(action)
		self.log_probs_buffer.append(log_probs)
		return action

	def reset_buffer(self):
		self.action_buffer = []
		self.log_probs_buffer = []

	def calculate_loss(self, rewards):
		cum_reward = self.compute_discounted_R(rewards)
		advantage = cum_reward - np.array(self.q_value_buffer)
		critic_loss = 0.5 * advantage ** 2
		critic_loss = critic_loss.mean()
		log_probs = np.array(self.log_probs_buffer)
		actor_loss = -log_probs * advantage
		actor_loss = actor_loss.mean()
		return actor_loss, critic_loss

	def compute_discounted_R(self, rewards):
		discounted_r = np.zeros_like(rewards, dtype=np.float32)
		running_add = 0
		for t in reversed(range(len(rewards))):
			running_add = running_add * self.gamma + rewards[t]
			discounted_r[t] = running_add
		return discounted_r

	def train(self, rewards):
		actor_loss, critic_loss = self.calculate_loss(rewards)
		print("Actor Loss : {} Critic Loss : {}".format(actor_loss, critic_loss))
		self.optimizer.zero_grad()
		loss = actor_loss + critic_loss
		loss.backward()
		self.optimizer.step()


class A2CEnv:
	def __init__(self, num_episodes, max_len):
		self.env = env.Env()  # the environment takes values of direction of TVC but the agent gives actions which is to
		# rotate the tvc

		self.agent = Agent()
