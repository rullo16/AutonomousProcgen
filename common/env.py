import gym.wrappers
import gym.wrappers
import gym.wrappers.normalize
import gym.wrappers.record_video
import gym.wrappers.transform_observation
from procgen import ProcgenEnv
from common.procgen_wrappers import *
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def distill_reward(obs,student_model = None, teacher_model = None):
    with torch.no_grad():
        if not isinstance(obs, torch.Tensor):
            obs = np.array(obs)
            obs = torch.tensor(obs,dtype=torch.float32).to(device)
        if obs.dim() == 3:
            obs = obs.permute(2,0,1)
            obs = obs.unsqueeze(0)
        res = (student_model(obs) - teacher_model(obs))
        res = np.array([r.abs()[0].item() for r in res]) # Done to avoid negative rewards extra rewards
        return res

# Distilled Wrapper for Procgen
class DistillationRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_callable, rewards_scale=1.0, sum_rewards=True, student_model = None, teacher_model = None):
        super(DistillationRewardWrapper, self).__init__(env)
        self.reward_callable = reward_callable
        self.rewards_scale = rewards_scale
        self.sum_rewards = sum_rewards
        self.student_model = student_model
        self.teacher_model = teacher_model

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        extra_reward = self.reward_callable(obs, student_model = self.student_model, teacher_model = self.teacher_model)
        extra_reward = extra_reward[0] if reward.ndim == 0 else extra_reward
        if self.sum_rewards:
            res_rewards = reward + self.rewards_scale * extra_reward
        else:
            res_rewards  = np.array([reward, extra_reward * self.rewards_scale])
        return obs, res_rewards, done, info



def make_env(game, n_envs=1,params=None, test=False, viz = False, student_model = None, teacher_model = None, sum_rewards=False):
    if test:
        env = gym.make(f'procgen:procgen-{game}-v0', num_levels=200, start_level=1, distribution_mode='easy', render_mode='human' if viz else 'rgb_array')
    else:
        env = ProcgenEnv(num_envs=n_envs, env_name=game, num_levels=200, start_level=1, distribution_mode='easy')
        env = VecExtractDictObs(env, "rgb")
        env = VecNormalize(env, ob=False)
        env = TransposeFrame(env)
        env = ScaledFloatFrame(env)
    if params and params.name == 'distill':
        env = DistillationRewardWrapper(env,distill_reward,rewards_scale=params.distillation_scale, student_model = student_model, teacher_model = teacher_model, sum_rewards=sum_rewards)
    return env