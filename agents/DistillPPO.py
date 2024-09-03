import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb

from common.env.env import make_env
from common.trajectories import DistilledTrajectory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Initialize weights of the network using xavier and orthogonal initialization
This should be done for the distilled network as well as the network to be distilled
and should yield better results than random initialization
'''

def xavier_inititalization(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module

def orthogonal_initialization(module, gain=nn.init.calculate_gain('relu')):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module

'''
ImpalaBlock with Residual Connections, from the IMPALA Paper
Helps in better extraction for the features from the images
used instead of the standard Convolutional Layers
that were previously used following A2C architecture
'''

class ResidualBlock(nn.Module):
    def __init__(self, input_shape):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape, input_shape, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(input_shape, input_shape, kernel_size=3, stride=1, padding=1),
        )
    
    def forward(self, x):
        return x + self.conv(x)
    
class ImpalaBlock(nn.Module):
    def __init__(self, input_shape, out_shape):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape, out_shape, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(out_shape),
            nn.ReLU(),
            ResidualBlock(out_shape)
        )
    
    def forward(self, x):
        return self.conv(x)

## Network to be distilled

'''
Distilled Network Architecture
With Two ImpalaBlocks and 
Three Linear Layers.
Same architecture for both the teacher and the student
one more layer than the intrinsic critic in the PPO network
'''
class DistilledArchitecture(nn.Module):
    def __init__(self, input_shape):
        super(DistilledArchitecture, self).__init__()

        self.conv = nn.Sequential(
            ImpalaBlock(input_shape[0], 16),
            ImpalaBlock(16, 32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.5)
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.ff = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        fx = x.float()/256
        conv_out = self.conv(fx).view(fx.size()[0],-1)
        return self.ff(conv_out)
    
## Distilled PPO Network
'''
Main PPO Architecture, using two critics,
one for environment rewards (extrinsic)
and one from the teacher/student rewards (intrinsic)
'''
class PPONetwork(nn.Module):
    
    def __init__(self, obs_size, n_actions):
        super(PPONetwork, self).__init__()

        self.conv = nn.Sequential(
            ImpalaBlock(obs_size[0], 16),
            ImpalaBlock(16, 32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.5)
        )

        self.conv.apply(xavier_inititalization)

        conv_out_size = self._get_conv_out(obs_size)


        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )


        self.critic_ext = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.critic_int = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
        self.actor.apply(orthogonal_initialization)
        self.critic_ext.apply(orthogonal_initialization)
        self.critic_int.apply(orthogonal_initialization)
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        fx = x.float()/256
        conv_out = self.conv(fx).view(fx.size()[0],-1)
        return self.actor(conv_out), self.critic_ext(conv_out), self.critic_int(conv_out)
    
# Distilled PPO Agent
class DistillPPOAgent:
    def __init__(self, obs_size, action_size, params):
        self.params = params
        self.net = PPONetwork(obs_size, action_size).to(device)
        self.student = DistilledArchitecture(obs_size).to(device)
        self.student.train(False)
        self.teacher = DistilledArchitecture(obs_size).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr = self.params.lr)
        self.distillation_optimizer = optim.Adam(self.teacher.parameters(), lr = self.params.lr_distill)
        self.trajectories = DistilledTrajectory(obs_size, self.params.n_steps, self.params.n_envs)
        self.minimum_batch_size = params.min_batch_size
        self.training_batch_size = params.batch_per_epoch
    
    
    def select_action(self, state, train=False):
        if train:
            if not isinstance(state, torch.Tensor):
                state = np.array(state)
                state = torch.tensor(state, dtype=torch.float32).to(device)

            #During testing, the state is not passed through the wrappers, only the distillation wrapper, so I need to permute dimensions
            state = state.permute(2,0,1).unsqueeze(0) if state.dim() == 3 else state
            policy, extrinsics_val, intrinsic_val = self.net(state)
            intrinsic_val = intrinsic_val.squeeze(-1)
            extrinsics_val = extrinsics_val.squeeze(-1)
            #Take the softmax of the policy and sample an action from the Categorical distribution
            #Decided to clip the action probabilities to avoid big negative values
            #First tried to use Sigmoid but it was not working well
            #Still not sure if this is the best approach
            action_probs = F.log_softmax(policy, dim=1)
            # action_probs = torch.clip(action_probs, -1)
            action_probs = torch.distributions.Categorical(logits=(action_probs))
            action = action_probs.sample()
            return action.cpu().numpy(), action_probs, extrinsics_val, intrinsic_val
        
        #During sample collection, we don't need the gradients
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = np.array(state)
                state = torch.tensor(state, dtype=torch.float32).to(device)
            state = state.permute(2,0,1).unsqueeze(0) if state.dim() == 3 else state
            policy, extrinsics_val, intrinsic_val = self.net(state)
            intrinsic_val = intrinsic_val.squeeze(-1)
            extrinsics_val = extrinsics_val.squeeze(-1)
            action_probs = F.log_softmax(policy, dim=1)
            # action_probs = torch.clip(action_probs, -1)
            action_probs = torch.distributions.Categorical(logits=(action_probs))
            action = action_probs.sample()
        return action.cpu().numpy(), action_probs.log_prob(action).cpu().detach().numpy(), extrinsics_val.cpu().detach().numpy(), intrinsic_val.cpu().detach().numpy()
    
    '''
    Function used for testing both agents in the environment
    and during training, to evaluate the performance of the agent
    flag used to decide wheter to visualize the environmetns when loading a saved model
    '''
    @torch.no_grad()
    def testing(self, game, test=True, viz=False, count=500):
        env = make_env(game,params=self.params, test=test, viz=viz, teacher_model=self.teacher, student_model=self.student, sum_rewards=False)
        test_rewards = 0.0
        test_steps = 0
        for _ in range(count):
            obs = env.reset()
            while True:
                action, _, _,_ = self.select_action(obs)
                if test:
                    action = action[0]
                obs, reward, done, _ = env.step(action)
                test_rewards += reward[0]
                test_steps += 1
                if done:
                    break
        return test_rewards / count, test_steps//count
    
    '''
    Function used to improve the learning rate of the optimizer
    during training, to avoid the learning rate to be too high
    '''
    def improv_lr(self, optimizer, init_lr, timesteps, num_timesteps):
        lr = init_lr * (1-(timesteps/num_timesteps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer
    
    '''
    Training function
    '''
    def train(self, steps):
        batch = self.params.n_envs * self.params.n_steps // self.minimum_batch_size
        if batch < self.minimum_batch_size:
            self.minimum_batch_size = batch
        #Accumulate gradients for the batch
        gradient_accumulation_steps = batch // self.minimum_batch_size
        gradient_steps = 1

        self.net.train()
        for _ in range(self.params.ppo_epochs):
            samples = self.trajectories.gather_experience(batch_size=batch, minimum_batch=self.minimum_batch_size)

            for sample in samples:
                observations, actions, old_extrinsic_values, old_intrinsic_values, returns, advantages, old_action_log_probabilities = sample
                
                observations = observations.to(device)
                actions = actions.to(device)
                old_extrinsic_values = old_extrinsic_values.to(device)
                old_intrinsic_values = old_intrinsic_values.to(device)
                returns = returns.to(device)
                advantages = advantages.to(device)
                old_action_log_probabilities = old_action_log_probabilities.to(device)

                _ , action_distribution, extrinsic_values, intrinsic_values = self.select_action(observations, train=True)

                #Calculate the ratio of the new and old action probabilities
                action_log_probabilities = action_distribution.log_prob(actions)
                ratio = torch.exp(action_log_probabilities - old_action_log_probabilities)
                unclipped_advantages = ratio * advantages
                clipped_advantages = torch.clamp(ratio, 1.0 - self.params.ppo_eps, 1.0 + self.params.ppo_eps) * advantages
                action_loss = -torch.min(unclipped_advantages, clipped_advantages).mean()

                #Clipped Bellman-Error distilled
                clipped_extrinsic_values = old_extrinsic_values + (extrinsic_values - old_extrinsic_values).clamp(-self.params.ppo_eps, self.params.ppo_eps)
                clipped_intrinsic_values = old_intrinsic_values + (intrinsic_values - old_intrinsic_values).clamp(-self.params.ppo_eps, self.params.ppo_eps)
                unclipped_values = (extrinsic_values - returns).pow(2) + (intrinsic_values - returns).pow(2)
                clipped_values = (clipped_extrinsic_values - returns).pow(2) + (clipped_intrinsic_values - returns).pow(2)
                value_loss = self.params.val_loss_coef * torch.max(unclipped_values, clipped_values).mean()

                # distillation training
                self.distillation_optimizer.zero_grad()
                teacher_out = self.teacher(observations)
                student_out = self.student(observations)
                loss_distill = F.mse_loss(student_out, teacher_out)
                loss_distill.backward()

                #Policy entropy
                entropy_beta = max(0.01, self.params.entropy_beta * (1-steps/self.params.n_total_steps))
                entropy_loss = action_distribution.entropy().mean()
                loss= (action_loss + (self.params.val_loss_coef * value_loss)) - (entropy_beta * entropy_loss)
                loss.backward()

                if gradient_steps % gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                    nn.utils.clip_grad_norm_(self.teacher.parameters(), 0.5)
                    self.distillation_optimizer.step()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                gradient_steps += 1
                
                wandb.log({"Action Loss": action_loss.item(), "Value Loss": value_loss.item(), "Entropy Loss": entropy_loss.item(), "Distill Loss": loss_distill.item()})