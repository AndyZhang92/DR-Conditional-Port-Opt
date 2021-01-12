import numpy as np

class SingleStockPI(object):

	def __init__(self, alphas, log_returns):
		
		self.num_train = 0 # Define the placeholders
		self.num_group = 0 # Define the placeholders
		self.group_id = [] # Define the placeholders
		self.group_mean_log_ret = []  # Define the placeholders
		self.trans_prob = [[]]  # Define the placeholders
		self.log_returns = log_returns
		self.discretize(alphas)
		
		self.nS = 2 * self.num_group
		self.nA = 2
		self.gamma = 0.99
		self.friction = 0.001

	def discretize(self, alphas):

		self.splits = [-np.inf] + [np.quantile(self.log_returns,alpha) for alpha in alphas] + [np.inf]
		self.num_train = self.log_returns.shape[0]
		self.num_group = len(self.splits) - 1
		self.group_id = np.zeros(self.num_train)
		for i in range(self.num_group):
			bool_id = (self.splits[i] <= self.log_returns) & (self.log_returns < self.splits[i+1])
			self.group_id[bool_id] = i
			self.trans_prob = np.zeros((self.num_group, self.num_group))
			a = np.column_stack((self.group_id[1:],self.group_id[:-1])).astype(np.int)
		for i in range(self.num_group):
			for j in range(self.num_group):
				self.trans_prob[i,j] = np.sum((a[:,0] == i) & (a[:,1] == j)) / np.sum(a[:,0] == i)

		self.group_mean_log_ret = np.zeros(self.num_group)
		for i in range(self.num_group):
			self.group_mean_log_ret[i] = np.mean(self.log_returns[self.group_id == i])

	def policy_evaluation(self, policy, tol=1e-10):
		"""Evaluate the value function from a given policy.
		"""

		P = np.zeros((self.nS,self.nS))
		mean_reward = np.zeros(self.nS)

		for i in range(self.nS):
			last_act = i//self.num_group
			act = policy[i]
			P[i, self.num_group*act:self.num_group*(act+1)] = self.trans_prob[i % self.num_group, :]
			if act == 0:
				mean_reward[i] = -self.friction*(last_act!=act)
			else:
				mean_reward[i] = self.trans_prob[i % self.num_group, :] @ self.group_mean_log_ret - self.friction*(last_act!=act)

		value_function, new_value_function = np.zeros(self.nS), np.zeros(self.nS)

		while True:
			new_value_function = mean_reward + self.gamma * (P @ value_function)
			if np.max(np.abs(new_value_function - value_function)) < tol:
				break
			value_function = new_value_function

		return value_function

	def policy_improvement(self, value_function):
		"""Given the value function from policy improve the policy.
		"""
		new_policy = np.zeros(self.nS, dtype=np.int)
		for i in range(self.nS):
			maxval = -np.inf
			maxid = 0
			last_act = i//self.num_group

			for act in range(self.nA):
				probs = np.zeros(self.nS)
				probs[self.num_group*act:self.num_group*(act+1)] = self.trans_prob[i % self.num_group, :]

				if act == 0:
					reward = -self.friction*(last_act!=act)
				else:
					reward = self.trans_prob[i % self.num_group, :] @ self.group_mean_log_ret - self.friction*(last_act!=act)
				Q = reward + self.gamma * (probs @ value_function)

				if Q > maxval:
					maxid = act
					maxval = Q

			new_policy[i] = maxid

		return new_policy

	def policy_iteration(self, tol=1e-4):
		"""Runs policy iteration.
		"""

		value_function = np.zeros(self.nS)
		policy = np.zeros(self.nS, dtype=np.int)
		ITER = 0
		while True:
			value_function = self.policy_evaluation(policy, tol = tol)
			new_policy = self.policy_improvement(value_function)
			if list(new_policy) == list(policy):
				break
			ITER = ITER + 1
			policy = new_policy
		return value_function, policy

	def policy_test(self, policy, log_returns):
		num_test = log_returns.shape[0]

		group_id = np.zeros(num_test)
		for i in range(self.num_group):
			bool_id = (self.splits[i] <= log_returns) & (log_returns < self.splits[i+1])
			group_id[bool_id] = i

		actions = np.zeros(num_test)

		for i in range(1,num_test):
			state = int(actions[i-1] * self.num_group + group_id[i-1])
			actions[i] = policy[state]

		policy_log_return = log_returns * actions
		policy_log_return[1:] -= self.friction * (actions[1:]!=actions[:-1])

		return policy_log_return