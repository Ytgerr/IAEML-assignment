"""
Complete training and visualization script for PPO on robotaxi path planning.
Run this script to train an agent and then visualize its behavior.
"""
import jax
import jax.numpy as jnp
from env.robotaxi import RobotaxiEnv
from algorithms.train_ppo import train_ppo
from utils.renderer import PygameFrontend
from utils.autoreset import AutoResetWrapper


class TrainedAgentWrapper:
    """Wrapper to use trained agent in environment."""
    
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.last_structured_obs = None
        
    def reset(self, key, env_params, init_state):
        _, state = self.env.reset(key, env_params, init_state)
        self.last_structured_obs = self.env.get_observation(state, env_params)
        return self.last_structured_obs, state
        
    def step(self, key, state, action, env_params):
        _, new_state, reward, done, info = self.env.step(key, state, action, env_params)
        self.last_structured_obs = self.env.get_observation(new_state, env_params)
        info["structured_obs"] = self.last_structured_obs
        return self.last_structured_obs, new_state, reward, done, info


def get_agent_action(agent, obs, key, deterministic: bool = True):
    """
    Get action from trained agent.
    
    Args:
        agent: Trained PPO agent
        obs: Observation vector
        key: JAX random key
        deterministic: If True, use mean action. If False, sample from distribution.
    
    Returns:
        action: Agent's action
    """
    mean, log_std = agent.policy(obs)
    
    if deterministic:
        return mean
    else:
        std = jnp.exp(log_std)
        noise = jax.random.normal(key, mean.shape)
        return mean + std * noise


def visualize_agent(agent, env_params, init_state, num_steps: int = 500, 
                   deterministic: bool = True):
    """
    Run trained agent in environment with visualization.
    
    Args:
        agent: Trained PPO agent
        env_params: Environment parameters
        init_state: Initial environment state
        num_steps: Number of steps to run
        deterministic: If True, use deterministic actions
    """
    key = jax.random.PRNGKey(123)
    
    # Create environment wrapper
    env_wrapper = TrainedAgentWrapper(RobotaxiEnv, RobotaxiEnv)
    
    # Reset environment
    key, subkey = jax.random.split(key)
    obs, state = env_wrapper.reset(subkey, env_params, init_state)
    
    # Run episode with agent
    total_reward = 0.0
    
    for step in range(num_steps):
        key, subkey = jax.random.split(key)
        
        # Get observation vector
        obs_vector = obs.as_vector() if hasattr(obs, 'as_vector') else obs
        
        # Get action from agent
        action = get_agent_action(agent, obs_vector, subkey, deterministic=deterministic)
        
        # Step environment
        key, subkey = jax.random.split(key)
        obs, state, reward, done, info = env_wrapper.step(
            subkey, state, action, env_params
        )
        
        total_reward += reward
        
        if bool(done):
            print(f"Episode ended at step {step} with total reward: {total_reward:.2f}")
            break
    
    print(f"Visualization complete. Total reward: {total_reward:.2f}")
    return total_reward


def main():
    """Main function: Train and visualize."""
    
    print("=" * 60)
    print("PPO Training & Visualization for Robotaxi Path Planning")
    print("=" * 60)
    
    config = {
        'num_episodes': 100, 
        'trajectory_length': 2048,   
        'num_updates_per_episode': 16,
        'map_id': 1,
        'eval_steps': 1000,          
    }
    
    print("\nTraining Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Step 1: Train agent
    print("\n" + "=" * 60)
    print("Step 1: Training PPO Agent")
    print("=" * 60)
    
    agent, env_params, init_state = train_ppo(
        num_episodes=config['num_episodes'],
        trajectory_length=config['trajectory_length'],
        num_updates_per_episode=config['num_updates_per_episode'],
        map_id=config['map_id'],
    )
    
    print("\n✓ Training completed!")
    
    # Step 2: Evaluate agent (deterministic actions)
    print("\n" + "=" * 60)
    print("Step 2: Evaluating Trained Agent")
    print("=" * 60)
    
    eval_reward = visualize_agent(
        agent, env_params, init_state,
        num_steps=config['eval_steps'],
        deterministic=True
    )
    print(eval_reward)
    
    print("\n✓ Evaluation completed!")
    
    # Step 3: Interactive visualization (optional)
    print("\n" + "=" * 60)
    print("Step 3: Live Visualization (Pygame)")
    print("=" * 60)
    print("Starting Pygame visualization...")
    print("Close the window to exit.\n")
    
    try:
        # Create wrapper for visualization
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        _, test_state = RobotaxiEnv.reset(subkey, env_params, init_state)
        
        class VisualizationEnv:
            def __init__(self, agent, env_params):
                self.agent = agent
                self.env_params = env_params
                self.last_structured_obs = None
                
            def reset(self, key, env_params, init_state):
                _, state = RobotaxiEnv.reset(key, env_params, init_state)
                self.last_structured_obs = RobotaxiEnv.get_observation(state, env_params)
                return self.last_structured_obs, state
                
            def step(self, key, state, action, env_params):
                _, new_state, reward, done, info = RobotaxiEnv.step(
                    key, state, action, env_params
                )
                self.last_structured_obs = RobotaxiEnv.get_observation(new_state, env_params)
                info["structured_obs"] = self.last_structured_obs
                return self.last_structured_obs, new_state, reward, done, info
        
        vis_env = VisualizationEnv(agent, env_params)
        
        class AgentControlledEnv:
            def __init__(self, agent, vis_env):
                self.agent = agent
                self.vis_env = vis_env
                self.key = jax.random.PRNGKey(42)
                
            def reset(self, key, env_params, init_state):
                self.key = key
                return self.vis_env.reset(key, env_params, init_state)
                
            def step(self, key, state, _action, env_params):
                # Ignore human input, use agent's action instead
                obs = self.vis_env.last_structured_obs
                obs_vector = obs.as_vector()
                
                # Get agent action
                mean, _ = self.agent.policy(obs_vector)
                agent_action = mean
                
                self.key, subkey = jax.random.split(self.key)
                return self.vis_env.step(subkey, state, agent_action, env_params)
        
        agent_env = AgentControlledEnv(agent, vis_env)
        frontend = PygameFrontend(agent_env, env_params, init_state, video_name="ppo_result")
        frontend.run()
        
    except Exception as e:
        print(f"Visualization skipped (pygame may not be available): {e}")
    
    print("\n" + "=" * 60)
    print("All steps completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
