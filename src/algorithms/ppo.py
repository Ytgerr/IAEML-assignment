"""
Proximal Policy Optimization (PPO) algorithm for robotaxi path planning.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Tuple, NamedTuple
import chex


class PPOMemory(NamedTuple):
    """Stores trajectories for PPO training."""
    observations: chex.Array
    actions: chex.Array
    rewards: chex.Array
    values: chex.Array
    log_probs: chex.Array
    dones: chex.Array


class PolicyNetwork(eqx.Module):
    layers: list

    def __init__(self, obs_dim: int, action_dim: int, key: chex.PRNGKey, hidden_dim: int = 256):
        keys = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Linear(obs_dim, hidden_dim, key=keys[0]),
            eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[1]),
            eqx.nn.Linear(hidden_dim, action_dim * 2, key=keys[2]),
        ]

    def _forward_single(self, obs: chex.Array) -> Tuple[chex.Array, chex.Array]:
        x = obs
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        output = self.layers[-1](x)
        action_dim = output.shape[-1] // 2
        mean = output[..., :action_dim]
        log_std = jnp.clip(output[..., action_dim:], -20, 2)
        return mean, log_std

    def __call__(self, obs: chex.Array) -> Tuple[chex.Array, chex.Array]:
        if obs.ndim == 1:
            return self._forward_single(obs)
        else:
            return jax.vmap(self._forward_single)(obs)


class ValueNetwork(eqx.Module):
    layers: list

    def __init__(self, obs_dim: int, key: chex.PRNGKey, hidden_dim: int = 256):
        keys = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(obs_dim, hidden_dim, key=keys[0]),
            eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[1]),
            eqx.nn.Linear(hidden_dim, 1, key=keys[2]),
        ]

    def _forward_single(self, obs: chex.Array) -> chex.Array:
        x = obs
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x).squeeze(-1)

    def __call__(self, obs: chex.Array) -> chex.Array:
        if obs.ndim == 1:
            return self._forward_single(obs)
        else:
            return jax.vmap(self._forward_single)(obs)


class PPOAgent(eqx.Module):
    policy: PolicyNetwork
    value: ValueNetwork

    policy_optimizer: optax.GradientTransformation = eqx.static_field()
    value_optimizer: optax.GradientTransformation = eqx.static_field()
    policy_opt_state: optax.OptState
    value_opt_state: optax.OptState

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        key: chex.PRNGKey,
        learning_rate: float = 3e-4,
        hidden_dim: int = 256,
    ):
        k1, k2 = jax.random.split(key, 2)

        self.policy = PolicyNetwork(obs_dim, action_dim, k1, hidden_dim)
        self.value = ValueNetwork(obs_dim, k2, hidden_dim)

        self.policy_optimizer = optax.adam(learning_rate)
        self.value_optimizer = optax.adam(learning_rate)

        self.policy_opt_state = self.policy_optimizer.init(
            eqx.filter(self.policy, eqx.is_array)
        )
        self.value_opt_state = self.value_optimizer.init(
            eqx.filter(self.value, eqx.is_array)
        )

    # ================= SAVE =================

    def save(self, path: str):
        to_save = {
            "policy": self.policy,
            "value": self.value,
            "policy_opt_state": self.policy_opt_state,
            "value_opt_state": self.value_opt_state,
        }
        with open(path, "wb") as f:
            eqx.tree_serialise_leaves(f, to_save)
        print(f"Model saved to {path}")

    # ================= LOAD =================

    @staticmethod
    def load(path: str, agent_template: "PPOAgent") -> "PPOAgent":
        with open(path, "rb") as f:
            loaded = eqx.tree_deserialise_leaves(
                f,
                {
                    "policy": agent_template.policy,
                    "value": agent_template.value,
                    "policy_opt_state": agent_template.policy_opt_state,
                    "value_opt_state": agent_template.value_opt_state,
                },
            )

        return eqx.tree_at(
            lambda a: (
                a.policy,
                a.value,
                a.policy_opt_state,
                a.value_opt_state,
            ),
            agent_template,
            (
                loaded["policy"],
                loaded["value"],
                loaded["policy_opt_state"],
                loaded["value_opt_state"],
            ),
        )

def compute_advantages(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    trajectory_length = len(rewards)
    values_extended = jnp.concatenate([values, jnp.array([0.0])])
    advantages = []
    gae = 0.0
    for t in range(trajectory_length - 1, -1, -1):
        delta = rewards[t] + gamma * values_extended[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, float(gae))
    advantages = jnp.array(advantages)
    returns = advantages + values
    return advantages, returns


def ppo_loss(policy, value, batch: PPOMemory, clip_ratio=0.2, value_coeff=0.5, entropy_coeff=0.02):
    advantages, returns = compute_advantages(batch.rewards, batch.values, batch.dones)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    mean, log_std = policy(batch.observations)
    std = jnp.exp(log_std)
    new_log_probs = -0.5 * ((batch.actions - mean) / std) ** 2 - 0.5 * jnp.log(2 * jnp.pi) - log_std
    new_log_probs = new_log_probs.sum(axis=-1)
    new_values = value(batch.observations)
    ratio = jnp.exp(new_log_probs - batch.log_probs)
    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
    policy_loss = -jnp.minimum(surr1, surr2).mean()
    value_loss = 0.5 * ((new_values - returns) ** 2).mean()
    entropy = -new_log_probs.mean()
    return policy_loss - entropy_coeff * entropy, value_coeff * value_loss


def train_step(agent: PPOAgent, batch: PPOMemory, key: chex.PRNGKey):
    # Policy update
    def policy_loss_fn(policy):
        loss, _ = ppo_loss(policy, agent.value, batch)
        return loss

    policy_loss, policy_grads = eqx.filter_value_and_grad(policy_loss_fn)(agent.policy)
    policy_updates, new_policy_opt_state = agent.policy_optimizer.update(policy_grads, agent.policy_opt_state, agent.policy)
    new_policy = eqx.apply_updates(agent.policy, policy_updates)

    # Value update
    def value_loss_fn(value):
        _, loss = ppo_loss(agent.policy, value, batch)
        return loss

    value_loss, value_grads = eqx.filter_value_and_grad(value_loss_fn)(agent.value)
    value_updates, new_value_opt_state = agent.value_optimizer.update(value_grads, agent.value_opt_state, agent.value)
    new_value = eqx.apply_updates(agent.value, value_updates)

    new_agent = eqx.tree_at(
        lambda a: (a.policy, a.value, a.policy_opt_state, a.value_opt_state),
        agent,
        (new_policy, new_value, new_policy_opt_state, new_value_opt_state)
    )

    return new_agent, (policy_loss, value_loss)
