import jax
import jax.numpy as jnp
from typing import Tuple
from env.robotaxi import RobotaxiEnv
from algorithms.ppo import PPOAgent, PPOMemory, train_step
import chex


def collect_trajectory(env, env_params, init_state, agent: PPOAgent, 
                      trajectory_length: int = 256, eval_mode: bool = False) -> Tuple[PPOMemory, float]:
    """
    Collect a trajectory by rolling out the policy in the environment.
    """
    key = jax.random.PRNGKey(0)
    
    observations = []
    actions = []
    rewards_list = []
    values = []
    log_probs = []
    dones = []
    
    state = init_state
    episode_return = 0.0
    
    for step in range(trajectory_length):
        key, subkey = jax.random.split(key)
        
        # Get observation
        obs = env.get_observation(state, env_params)
        obs_vector = obs.as_vector()
        observations.append(obs_vector)
        
        # Get action and log_prob
        mean, log_std = agent.policy(obs_vector)
        
        if eval_mode:
            action = mean
            log_prob = jnp.zeros(mean.shape[0])
        else:
            std = jnp.exp(log_std)
            noise = jax.random.normal(subkey, mean.shape)
            action = mean + std * noise
            log_prob = -0.5 * ((action - mean) / std) ** 2 - 0.5 * jnp.log(2 * jnp.pi) - log_std
            log_prob = log_prob.sum(axis=-1)
        
        value = agent.value(obs_vector)
        
        actions.append(action)
        values.append(value)
        log_probs.append(log_prob)
        
        # Step environment
        key, subkey = jax.random.split(key)
        _, state, reward, done, info = env.step(subkey, state, action, env_params)
        
        rewards_list.append(reward)
        dones.append(float(done))
        episode_return += reward
        
        if bool(done):
            break
    
    memory = PPOMemory(
        observations=jnp.stack(observations),
        actions=jnp.stack(actions),
        rewards=jnp.array(rewards_list),
        values=jnp.stack(values),
        log_probs=jnp.array(log_probs),
        dones=jnp.array(dones),
    )
    
    return memory, episode_return


def train_ppo(num_episodes: int = 100, trajectory_length: int = 256, 
              num_updates_per_episode: int = 4, map_id: int = 1):
    """
    Train PPO agent on robotaxi environment.
    """
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    
    env_params, init_state = RobotaxiEnv.init_params(
        key=subkey,
        map_id=map_id,
        max_steps=1000,
        discretization_scale=1,
        path_length=100,
        fps=20,
        perception_radius=5.0,
        num_ray_sensors=32,
    )
    
    # Get observation dimension
    _, test_state = RobotaxiEnv.reset(subkey, env_params, init_state)
    test_obs = RobotaxiEnv.get_observation(test_state, env_params)
    obs_dim = test_obs.as_vector().shape[0]
    action_dim = 2  # accel and steering_rate
    
    print(f"Environment initialized:")
    print(f"  Observation dimension: {obs_dim}")
    print(f"  Action dimension: {action_dim}")
    
    # Initialize agent
    key, subkey = jax.random.split(key)
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        key=subkey,
        learning_rate=1e-4,
        hidden_dim=256,
    )
    
    
    print(f"\nStarting training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        key, subkey = jax.random.split(key)
        
        # Collect trajectory
        memory, episode_return = collect_trajectory(
            RobotaxiEnv, env_params, init_state, agent,
            trajectory_length=trajectory_length,
            eval_mode=False
        )
        
        # Perform multiple PPO updates
        for update in range(num_updates_per_episode):
            key, subkey = jax.random.split(key)
            agent, (policy_loss, value_loss) = train_step(agent, memory, subkey)
        
        if (episode + 1) % max(1, num_episodes // 10) == 0 or (episode + 1) == num_episodes:
            print(f"Episode {episode + 1}/{num_episodes} | Return: {episode_return:.2f} | "
                  f"Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f}")
    
    print("\nTraining completed!")
    return agent, env_params, init_state


if __name__ == "__main__":
    agent, env_params, init_state = train_ppo(
        num_episodes=30,
        trajectory_length=512,
        num_updates_per_episode=8,
        map_id=1,
    )
