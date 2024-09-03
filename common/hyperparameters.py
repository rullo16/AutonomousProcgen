from types import SimpleNamespace

# Hyperparameters

HYPERPARAMS = {
    'distill': SimpleNamespace(**{
        'run_name' : 'distill',
        'stop_test_reward': 10,
        'lr': 5e-4,
        'gamma': 0.999,
        'ppo_epochs' : 2,
        'ppo_eps':0.2,
        'val_loss_coef': 0.5,
        'gae_lambda': 0.95,
        'entropy_beta': 0.01,
        'n_envs': 24,
        'n_steps': 256,
        'n_total_steps': 2000000,
        'lr_distill': 2e-4,
        'distill_scale': 10.0,
        'distill_loss_weight': 0.5,
        'batch_per_epoch': 8,
        'min_batch_size': 2048,
        'test_freq': 10,
    }),
}