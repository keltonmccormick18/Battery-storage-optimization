"""Frozen PPO configuration, selected by docs/experiments/01_ppo_config_freeze.md.
Do not edit after tag rl-config-frozen."""

PPO = dict(
    gamma=1.0,
    gae_lambda=0.95,
    n_steps=512,
    batch_size=512,
    n_epochs=10,
    learning_rate=3e-4,
    ent_coef=0.01,
    policy_kwargs=dict(net_arch=[128, 128]),
    device="cpu",
)
N_ENVS = 16
TOTAL_TIMESTEPS = 2_000_000
EVAL_FREQ = 100_000
