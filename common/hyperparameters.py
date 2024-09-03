from types import SimpleNamespace

# Hyperparameters

HYPERPARAMS = {
    'distill': SimpleNamespace(**{
        'name' : 'distill',
        'stop_reward': 10,
        'lr': 5e-4,
        'gamma': 0.999,
        'train_epochs' : 2,
        'epsilon':0.2,
        'val_coef': 0.5,
        'lambda': 0.95,
        'entropy_beta': 0.01,
        'num_envs': 24,
        'traj_steps': 256,
        'total_epochs': 2000000,
        'lr_distillation': 2e-4,
        'distillation_scale': 10.0,
        'epoch_batches': 8,
        'minimum_batch_size': 2048,
        'tests_to_do': 10,
    }),
}