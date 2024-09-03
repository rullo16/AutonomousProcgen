import torch
import numpy as np
from collections import deque
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class DistilledTrajectory:

    def __init__(self, obs_shape, num_steps, num_envs):

        self.observation_shape = obs_shape
        self.collection_steps = num_steps
        self.num_environments = num_envs
        self.observations = torch.zeros(self.collection_steps + 1, self.num_environments, *self.observation_shape)
        self.actions = torch.zeros(self.collection_steps, self.num_environments)
        self.ext_int_rewards = torch.zeros(self.collection_steps,2,self.num_environments)
        self.combined_ext_int_rewards = torch.zeros(self.collection_steps, self.num_environments)
        self.dones = torch.zeros(self.collection_steps, self.num_environments)
        self.action_log_probabilities = torch.zeros(self.collection_steps, self.num_environments)
        self.extrinsic_values = torch.zeros(self.collection_steps+1, self.num_environments)
        self.intrinsic_values = torch.zeros(self.collection_steps+1, self.num_environments)
        self.returns = torch.zeros(self.collection_steps, self.num_environments)
        self.advantages = torch.zeros(self.collection_steps, self.num_environments)
        self.intrinsic_ref_values = torch.zeros(self.collection_steps, self.num_environments)
        self.extrinsic_ref_values = torch.zeros(self.collection_steps, self.num_environments)

        self.steps=0

    def store(self, obs, action, reward, done, log_prob, value_ext, value_int):
        reward[0] = np.clip(reward[0], -10, 10)
        reward[1] = np.clip(reward[1], -10, 10)
        self.observations[self.steps] = torch.tensor(obs.copy())
        self.actions[self.steps]= torch.tensor(action.copy())
        self.ext_int_rewards[self.steps] = torch.tensor(reward.copy())
        self.combined_ext_int_rewards[self.steps] = torch.tensor(reward.sum().copy())
        self.dones[self.steps]= torch.tensor(done.copy())
        self.action_log_probabilities[self.steps]= torch.tensor(log_prob.copy())
        self.extrinsic_values[self.steps]= torch.tensor(value_ext.copy())
        self.intrinsic_values[self.steps]= torch.tensor(value_int.copy())

        self.steps = (self.steps + 1) % self.collection_steps

    def store_last_state(self, obs, value_ext, value_int):
        self.observations[-1] = torch.tensor(obs.copy())
        self.extrinsic_values[-1]= torch.tensor(value_ext.copy())
        self.intrinsic_values[-1]= torch.tensor(value_int.copy())


    def compute_combined_rewards_advantage(self, gamma=0.999, gae_lambda=0.95, advantages_norm=True):
        reward_batch = self.combined_ext_int_rewards
        for i in reversed(range(self.collection_steps - 1)):
            reward = reward_batch[i]
            done = self.dones[i]
            value = self.intrinsic_values[i] + self.extrinsic_values[i]
            next_value = self.intrinsic_values[i + 1] + self.extrinsic_values[i + 1]

            delta = reward + gamma * next_value * (1 - done) - value
            self.advantages[i] = delta + gamma * gae_lambda * (1 - done) * self.advantages[i + 1]

        self.returns = self.advantages + (self.extrinsic_values[:-1] + self.intrinsic_values[:-1])
        if advantages_norm:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def compute_intrinsic_and_extrinsic_reference_values(self,gamma=0.999, gae_lambda=0.95):
        reward_ext_batch = self.ext_int_rewards[:,0]
        reward_int_batch = self.ext_int_rewards[:,1]
        ext = torch.zeros(self.collection_steps, self.num_environments)
        int = torch.zeros(self.collection_steps, self.num_environments)
        for i in reversed(range(self.collection_steps-1)):
            extrinsics_reward = reward_ext_batch[i]
            intrinsic_rewards = reward_int_batch[i]
            done = self.dones[i]

            extrinsic_value = self.extrinsic_values[i]
            extrinsic_next_value = self.extrinsic_values[i + 1]
            intrinsic_value = self.intrinsic_values[i]
            intrinsic_next_value = self.intrinsic_values[i + 1]

            delta_ext = extrinsics_reward + gamma * extrinsic_next_value * (1 - done) - extrinsic_value
            delta_int = intrinsic_rewards + gamma * intrinsic_next_value * (1 - done) - intrinsic_value

            ext[i] = delta_ext + gamma * gae_lambda * (1 - done) * ext[i + 1]
            int[i] = delta_int + gamma * gae_lambda * (1 - done) * int[i + 1]

        self.extrinsic_ref_values = ext + self.extrinsic_values[:-1]
        self.intrinsic_ref_values = int + self.intrinsic_values[:-1]

    def gather_experience(self, batch_size=None, minimum_batch=None):
        '''
        Use the BatchSampler to sample a batch of experiences from the trajectory.
        SubsetRandomSampler is used to shuffle the indices before sampling.
        '''
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), minimum_batch, drop_last=True)
        for idxs in sampler:
            obs = torch.Tensor(self.observations[:-1]).reshape(-1, *self.observation_shape)[idxs]
            actions = torch.Tensor(self.actions).reshape(-1)[idxs]
            returns = torch.Tensor(self.returns).reshape(-1)[idxs]
            advantages = torch.Tensor(self.advantages).reshape(-1)[idxs]
            log_probs = torch.Tensor(self.action_log_probabilities).reshape(-1)[idxs]
            ext_vals = torch.Tensor(self.extrinsic_ref_values).reshape(-1)[idxs]
            int_vals = torch.Tensor(self.intrinsic_ref_values).reshape(-1)[idxs]
            yield obs, actions, ext_vals,int_vals, returns, advantages, log_probs