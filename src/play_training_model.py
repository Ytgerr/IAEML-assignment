import jax
import jax.numpy as jnp
from env.robotaxi import RobotaxiEnv
from algorithms.ppo import PPOAgent
from utils.renderer import PygameFrontend

# --------------------------------------------------
# Init
# --------------------------------------------------
key = jax.random.PRNGKey(0)

env_params, init_state = RobotaxiEnv.init_params(
    key=key,
    map_id=1,
    max_steps=1000,
    discretization_scale=1,
    path_length=100,
    fps=20,
    perception_radius=5.0,
    num_ray_sensors=32,
)

# Get obs/action dims
_, state = RobotaxiEnv.reset(key, env_params, init_state)
obs = RobotaxiEnv.get_observation(state, env_params)

obs_dim = obs.as_vector().shape[0]
action_dim = 2

# --------------------------------------------------
# Load agent
# --------------------------------------------------
agent = PPOAgent(
    obs_dim=obs_dim,
    action_dim=action_dim,
    key=key,
    hidden_dim=256,
)

agent = PPOAgent.load("src/model/ppo_robotaxi.eqx", agent)

# --------------------------------------------------
# PURE PYTHON ENV (NO JIT, NO AUTORESET)
# --------------------------------------------------
class AgentEnv:
    def __init__(self, agent):
        self.agent = agent

    def reset(self, key, env_params, init_state):
        _, state = RobotaxiEnv.reset(key, env_params, init_state)
        obs = RobotaxiEnv.get_observation(state, env_params)
        return obs, state

    def step(self, key, state, _, env_params):
        # observation
        obs = RobotaxiEnv.get_observation(state, env_params)
        obs_vec = obs.as_vector()

        # agent action
        mean, _ = self.agent.policy(obs_vec)
        action = mean  # deterministic play

        # env step
        _, new_state, reward, done, info = RobotaxiEnv.step(
            key, state, action, env_params
        )
       
        new_obs = RobotaxiEnv.get_observation(new_state, env_params)
        info["structured_obs"] = new_obs

        return new_obs, new_state, reward, done, info


env = AgentEnv(agent)

# --------------------------------------------------
# Frontend
# --------------------------------------------------
frontend = PygameFrontend(
    env,
    env_params,
    init_state,
    video_name="ready_PPO",
    stop_on_done=True,
)

frontend.run()
