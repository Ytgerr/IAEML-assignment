import jax
from env.robotaxi import RobotaxiEnv
from env.base import BaseEnv
from utils.renderer import PygameFrontend
from utils.autoreset import AutoResetWrapper
import imageio

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

class RendererCompatibleEnv:
    def __init__(self, env, env_params):
            self.env = env
            self.env_params = env_params
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
    
compatible_env = RendererCompatibleEnv(RobotaxiEnv, env_params)
env = AutoResetWrapper(compatible_env, env_params, init_state)
frames = []

frontend = PygameFrontend(env, env_params, init_state, stop_on_done=False, video_name="user_test")
frontend.run()
