import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Dict, Any
from functools import partial

from env.base import BaseEnv, BaseEnvParams, BaseEnvState, BaseEnvObservation


class VehicleDynamics:
    @staticmethod
    @partial(jax.jit, static_argnames=("dt", "friction_coeff"))
    def step_dynamics(state, action, dt, friction_coeff=0.2):

        x, y, theta, v, delta = state
        accel, steering_rate = action

        accel = jnp.clip(accel, -5.0, 5.0)
        steering_rate = jnp.clip(steering_rate, -1.5, 1.5)
        delta = jnp.clip(delta, -0.5, 0.5)

        new_delta = delta + steering_rate * dt
        new_delta = jnp.clip(new_delta, -0.5, 0.5)

        friction_force = friction_coeff * v
        new_v = v + (accel - friction_force) * dt
        new_v = jnp.maximum(new_v, 0.0)

        L = 2.5

        def straight_motion(_):
            new_x = x + new_v * jnp.cos(theta) * dt
            new_y = y + new_v * jnp.sin(theta) * dt
            new_theta = theta
            return jnp.array([new_x, new_y, new_theta, new_v, new_delta])

        def turning_motion(_):
            turning_radius = L / jnp.tan(new_delta)
            omega = new_v / turning_radius

            new_x = x + turning_radius * \
                (jnp.sin(theta + omega * dt) - jnp.sin(theta))
            new_y = y - turning_radius * \
                (jnp.cos(theta + omega * dt) - jnp.cos(theta))
            new_theta = theta + omega * dt
            return jnp.array([new_x, new_y, new_theta, new_v, new_delta])
        is_straight = jnp.abs(new_delta) < 1e-6

        new_state = jax.lax.cond(
            is_straight,
            straight_motion,
            turning_motion,
            None
        )

        new_x, new_y, new_theta, new_v, new_delta = new_state
        new_theta = jnp.mod(new_theta, 2 * jnp.pi)

        return jnp.array([new_x, new_y, new_theta, new_v, new_delta])


class RobotaxiState(BaseEnvState):
    vehicle_state: jnp.ndarray
    last_action: jnp.ndarray


class RobotaxiObservation(BaseEnvObservation):
    vehicle_state: jnp.ndarray
    relative_goal: jnp.ndarray
    goal_distance: jnp.array
    goal_angle: jnp.array

    def as_vector(self) -> jnp.ndarray:
        return jnp.concatenate([
            self.vehicle_state,
            self.relative_goal,
            jnp.array([self.goal_distance]),
            jnp.array([self.goal_angle]),
            self.collision_rays.flatten()
        ])


class RobotaxiEnv(BaseEnv):
    @staticmethod
    def init_params(
        key: jnp.ndarray,
        map_id: int = 1,
        max_steps: int = 1000,
        path_length: int = 100,
        discretization_scale: int = 1,
        perception_radius: float = 10.0,
        num_ray_sensors: int = 16,
        fov: float = jnp.pi,
        fps: int = 60,
    ) -> Tuple[BaseEnvParams, RobotaxiState]:

        env_params, base_state = BaseEnv.init_params(
            key=key,
            map_id=map_id,
            max_steps=max_steps,
            path_length=path_length,
            discretization_scale=discretization_scale,
            perception_radius=perception_radius,
            num_ray_sensors=num_ray_sensors,
            fov=fov,
            fps=fps,
        )

        initial_vehicle_state = jnp.array([
            base_state.agent_pos[0],
            base_state.agent_pos[1],
            jnp.arctan2(
                base_state.agent_forward_dir[1], base_state.agent_forward_dir[0]),  # theta
            0.0,
            0.0
        ])

        robotaxi_state = RobotaxiState(
            time=base_state.time,
            goal_pos=base_state.goal_pos,
            agent_pos=base_state.agent_pos,
            agent_forward_dir=base_state.agent_forward_dir,
            static_obstacles=base_state.static_obstacles,
            kinematic_obstacles=base_state.kinematic_obstacles,
            kinematic_obst_velocities=base_state.kinematic_obst_velocities,
            path_array=base_state.path_array,
            vehicle_state=initial_vehicle_state,
            last_action=jnp.zeros(2)
        )

        return env_params, robotaxi_state

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def reset(
        key: jnp.ndarray,
        env_params: BaseEnvParams,
        init_state: RobotaxiState | None = None
    ) -> Tuple[jnp.ndarray, RobotaxiState]:

        state = init_state

        obs = RobotaxiEnv.get_observation(state, env_params)
        return obs.as_vector(), state

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def step(
        key: jnp.ndarray,
        env_state: RobotaxiState,
        action: jnp.ndarray,
        env_params: BaseEnvParams,
    ) -> Tuple[jnp.ndarray, RobotaxiState, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:

        action = jnp.clip(action, jnp.array(
            [-1.0, -1.0]), jnp.array([1.0, 1.0]))

        scaled_action = jnp.array([
            action[0] * 5.0,
            action[1] * 1.5
        ])

        dt = env_params.step_size
        new_vehicle_state = VehicleDynamics.step_dynamics(
            env_state.vehicle_state, scaled_action, dt
        )
        new_agent_pos = new_vehicle_state[:2]
        new_agent_forward_dir = jnp.array([
            jnp.cos(new_vehicle_state[2]),
            jnp.sin(new_vehicle_state[2])
        ])

        new_kinematic_obstacles = RobotaxiEnv._move_kinematic_obstacles(
            env_state, env_params
        )

        obstacles = jnp.concatenate([
            env_state.static_obstacles, new_kinematic_obstacles
        ], axis=0)

        goal_done = RobotaxiEnv._check_goal(new_agent_pos, env_state.goal_pos)
        collision_done = RobotaxiEnv._check_collisions(
            new_agent_pos, obstacles)
        time_done = env_state.time >= env_params.max_steps_in_episode

        done = jnp.logical_or(
            goal_done, jnp.logical_or(collision_done, time_done))

        reward = RobotaxiEnv._compute_reward(
            env_state, new_agent_pos, new_vehicle_state, goal_done, collision_done, obstacles, env_params
        )

        pred = jnp.mod(env_state.time + 1, env_params.fps) == 0
        new_path_array = jax.lax.cond(
            pred,
            lambda _: RobotaxiEnv._find_path(
                new_agent_pos, env_state.goal_pos, obstacles, env_params),
            lambda _: env_state.path_array,
            None
        )

        new_state = RobotaxiState(
            time=env_state.time + 1,
            goal_pos=env_state.goal_pos,
            agent_pos=new_agent_pos,
            agent_forward_dir=new_agent_forward_dir,
            static_obstacles=env_state.static_obstacles,
            kinematic_obstacles=new_kinematic_obstacles,
            kinematic_obst_velocities=env_state.kinematic_obst_velocities,
            path_array=new_path_array,
            vehicle_state=new_vehicle_state,
            last_action=scaled_action
        )

        obs = RobotaxiEnv.get_observation(new_state, env_params)
        info = {
            "time": new_state.time,
            "vehicle_state": new_vehicle_state,
            "goal_reached": goal_done,
            "collision": collision_done,
            "distance_to_goal": jnp.linalg.norm(new_agent_pos - env_state.goal_pos)
        }

        return obs.as_vector(), new_state, reward, done, info

    @staticmethod
    def get_observation(env_state: RobotaxiState, env_params: BaseEnvParams) -> RobotaxiObservation:
        base_obs = BaseEnv.get_observation(env_state, env_params)

        vehicle_state = env_state.vehicle_state
        goal_pos = env_state.goal_pos

        global_goal_vec = goal_pos - vehicle_state[:2]
        theta = vehicle_state[2]

        rot_matrix = jnp.array([
            [jnp.cos(theta), jnp.sin(theta)],
            [-jnp.sin(theta), jnp.cos(theta)]
        ])
        relative_goal = rot_matrix @ global_goal_vec

        goal_distance = jnp.linalg.norm(global_goal_vec)
        goal_angle = jnp.arctan2(relative_goal[1], relative_goal[0])

        return RobotaxiObservation(
            distance_to_path=base_obs.distance_to_path,
            direction_of_path=base_obs.direction_of_path,
            collision_rays=base_obs.collision_rays,
            vehicle_state=vehicle_state,
            relative_goal=relative_goal,
            goal_distance=goal_distance,
            goal_angle=goal_angle
        )

    @staticmethod
    def _compute_reward(
        old_state: RobotaxiState,
        new_agent_pos: jnp.ndarray,
        new_vehicle_state: jnp.ndarray,
        goal_done: jnp.ndarray,
        collision_done: jnp.ndarray,
        obstacles: jnp.ndarray,
        env_params: BaseEnvParams
    ) -> jnp.ndarray:

        base_obs = BaseEnv.get_observation(old_state, env_params)

        goal_reward = jax.lax.cond(
            goal_done,
            lambda _: 100.0,
            lambda _: 0.0,
            None
        )

        collision_penalty = jax.lax.cond(
            collision_done,
            lambda _: -50.0,
            lambda _: 0.0,
            None
        )

        old_goal_dist = jnp.linalg.norm(
            old_state.agent_pos - old_state.goal_pos)
        new_goal_dist = jnp.linalg.norm(new_agent_pos - old_state.goal_pos)
        progress_reward = (old_goal_dist - new_goal_dist) * 10.0

        map_height, map_width = env_params.map_height_width
        x, y = new_agent_pos
        out_of_bounds = (x < 0) | (x > map_width) | (y < 0) | (y > map_height)
        out_of_bounds_penalty = jax.lax.cond(
            out_of_bounds,
            lambda _: -10.0,
            lambda _: 0.0,
            None
        )

        action_penalty = -0.01 * jnp.sum(jnp.square(old_state.last_action))

        speed = new_vehicle_state[3]
        min_ray_distance = jnp.min(base_obs.collision_rays)
        speed_penalty = jax.lax.cond(
            (min_ray_distance < 2.0) & (speed > 2.0),
            lambda _: -5.0,
            lambda _: 0.0,
            None
        )
        time_penalty = -0.1

        total_reward = (
            goal_reward +
            collision_penalty +
            progress_reward +
            out_of_bounds_penalty +
            action_penalty +
            speed_penalty +
            time_penalty
        )

        return total_reward
