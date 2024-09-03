import torch
import numpy as np
from torch.utils.data import BatchSampler, SubsetRandomSampler

class ExperienceBuffer:

    def __init__(self, observation_dimensions, max_steps, num_simulations):
        """
        Initializes the experience buffer for storing simulation data over multiple steps.
        
        Args:
            observation_dimensions (tuple): Dimensions of each observation per environment.
            max_steps (int): Maximum number of steps per trajectory.
            num_simulations (int): Number of parallel simulation environments.
        """
        self.obs_dims = observation_dimensions
        self.total_steps = max_steps
        self.sim_count = num_simulations
        
        # Buffers to hold simulation data
        self.obs_memory = torch.zeros(self.total_steps + 1, self.sim_count, *self.obs_dims)
        self.action_memory = torch.zeros(self.total_steps, self.sim_count)
        self.reward_split = torch.zeros(self.total_steps, 2, self.sim_count)  # Splitting internal and external rewards
        self.accumulated_rewards = torch.zeros(self.total_steps, self.sim_count)
        self.termination_flags = torch.zeros(self.total_steps, self.sim_count)
        self.action_log_probs = torch.zeros(self.total_steps, self.sim_count)
        self.ext_value_estimations = torch.zeros(self.total_steps + 1, self.sim_count)
        self.int_value_estimations = torch.zeros(self.total_steps + 1, self.sim_count)
        self.future_returns = torch.zeros(self.total_steps, self.sim_count)
        self.adv_estimations = torch.zeros(self.total_steps, self.sim_count)
        self.int_ref_vals = torch.zeros(self.total_steps, self.sim_count)
        self.ext_ref_vals = torch.zeros(self.total_steps, self.sim_count)

        self.curr_step = 0  # Track the current step

    def log_transition(self, observation, action_taken, rewards, done_flag, log_probability, ext_value, int_value):
        """
        Logs a single simulation transition into the buffer.

        Args:
            observation (ndarray): The observation at the current step.
            action_taken (ndarray): The action taken during this step.
            rewards (ndarray): The intrinsic and extrinsic rewards.
            done_flag (ndarray): Boolean flag indicating whether the episode is complete.
            log_probability (ndarray): Log probability of the action.
            ext_value (ndarray): Predicted extrinsic value.
            int_value (ndarray): Predicted intrinsic value.
        """
        # Clipping rewards to ensure they stay within a reasonable range
        rewards[0] = np.clip(rewards[0], -10, 10)
        rewards[1] = np.clip(rewards[1], -10, 10)
        
        # Store data
        self.obs_memory[self.curr_step] = torch.tensor(observation.copy())
        self.action_memory[self.curr_step] = torch.tensor(action_taken.copy())
        self.reward_split[self.curr_step] = torch.tensor(rewards.copy())
        self.accumulated_rewards[self.curr_step] = torch.tensor(rewards.sum().copy())
        self.termination_flags[self.curr_step] = torch.tensor(done_flag.copy())
        self.action_log_probs[self.curr_step] = torch.tensor(log_probability.copy())
        self.ext_value_estimations[self.curr_step] = torch.tensor(ext_value.copy())
        self.int_value_estimations[self.curr_step] = torch.tensor(int_value.copy())

        # Move to next step
        self.curr_step = (self.curr_step + 1) % self.total_steps

    def save_final_state(self, observation, ext_value, int_value):
        """
        Logs the final state of the trajectory.
        
        Args:
            observation (ndarray): The last observation in the trajectory.
            ext_value (ndarray): The final extrinsic value estimate.
            int_value (ndarray): The final intrinsic value estimate.
        """
        self.obs_memory[-1] = torch.tensor(observation.copy())
        self.ext_value_estimations[-1] = torch.tensor(ext_value.copy())
        self.int_value_estimations[-1] = torch.tensor(int_value.copy())

    def calculate_advantages_and_returns(self, discount_factor=0.999, lambda_factor=0.95, normalize_advantages=True):
        """
        Compute the generalized advantages and returns from combined rewards (intrinsic + extrinsic).

        Args:
            discount_factor (float): Discount rate for future rewards.
            lambda_factor (float): Smoothing parameter for GAE.
            normalize_advantages (bool): Whether to normalize the calculated advantages.
        """
        rewards = self.accumulated_rewards
        # Compute advantages in reverse using Generalized Advantage Estimation
        for step in reversed(range(self.total_steps - 1)):
            combined_value = self.int_value_estimations[step] + self.ext_value_estimations[step]
            next_combined_value = self.int_value_estimations[step + 1] + self.ext_value_estimations[step + 1]
            delta = rewards[step] + discount_factor * next_combined_value * (1 - self.termination_flags[step]) - combined_value
            self.adv_estimations[step] = delta + discount_factor * lambda_factor * (1 - self.termination_flags[step]) * self.adv_estimations[step + 1]

        # Calculate future returns
        self.future_returns = self.adv_estimations + (self.ext_value_estimations[:-1] + self.int_value_estimations[:-1])

        # Normalize advantages if specified
        if normalize_advantages:
            self.adv_estimations = (self.adv_estimations - self.adv_estimations.mean()) / (self.adv_estimations.std() + 1e-8)

    def compute_reference_values(self, discount_factor=0.999, lambda_factor=0.95):
        """
        Compute reference values separately for intrinsic and extrinsic rewards.

        Args:
            discount_factor (float): Discount rate for future rewards.
            lambda_factor (float): Smoothing parameter for GAE.
        """
        ext_rewards = self.reward_split[:, 0]
        int_rewards = self.reward_split[:, 1]
        extrinsic_buffer = torch.zeros(self.total_steps, self.sim_count)
        intrinsic_buffer = torch.zeros(self.total_steps, self.sim_count)
        
        # Process extrinsic and intrinsic advantages separately
        for step in reversed(range(self.total_steps - 1)):
            delta_ext = ext_rewards[step] + discount_factor * self.ext_value_estimations[step + 1] * (1 - self.termination_flags[step]) - self.ext_value_estimations[step]
            delta_int = int_rewards[step] + discount_factor * self.int_value_estimations[step + 1] * (1 - self.termination_flags[step]) - self.int_value_estimations[step]

            extrinsic_buffer[step] = delta_ext + discount_factor * lambda_factor * (1 - self.termination_flags[step]) * extrinsic_buffer[step + 1]
            intrinsic_buffer[step] = delta_int + discount_factor * lambda_factor * (1 - self.termination_flags[step]) * intrinsic_buffer[step + 1]

        self.ext_ref_vals = extrinsic_buffer + self.ext_value_estimations[:-1]
        self.int_ref_vals = intrinsic_buffer + self.int_value_estimations[:-1]

    def sample_experiences(self, batch_total=None, mini_batch_size=None):
        """
        Samples experience batches from the buffer for training.

        Args:
            batch_total (int): Total number of samples to extract from the buffer.
            mini_batch_size (int): Size of the individual mini-batches returned.
        """
        sample_indices = BatchSampler(SubsetRandomSampler(range(batch_total)), mini_batch_size, drop_last=True)
        for indices in sample_indices:
            # Fetch sampled data by reshaping and indexing
            obs_batch = torch.Tensor(self.obs_memory[:-1]).reshape(-1, *self.obs_dims)[indices]
            actions_batch = torch.Tensor(self.action_memory).reshape(-1)[indices]
            return_batch = torch.Tensor(self.future_returns).reshape(-1)[indices]
            advantage_batch = torch.Tensor(self.adv_estimations).reshape(-1)[indices]
            log_prob_batch = torch.Tensor(self.action_log_probs).reshape(-1)[indices]
            ext_value_batch = torch.Tensor(self.ext_ref_vals).reshape(-1)[indices]
            int_value_batch = torch.Tensor(self.int_ref_vals).reshape(-1)[indices]
            
            # Return the sampled data
            yield obs_batch, actions_batch, ext_value_batch, int_value_batch, return_batch, advantage_batch, log_prob_batch
