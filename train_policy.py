"""Train a more stable actor-critic policy for the protein folding environment.

This trainer stays within the project constraints:
- standard Python
- numpy only
- no external RL libraries

It improves on the earlier REINFORCE baseline by adding:
- a learned value baseline
- clipped reward normalization
- action pruning to avoid the worst moves
- best-checkpoint saving by evaluation score
- CSV training logs for later inspection

Examples:
    python train_policy.py --task task_1 --episodes 400
    python train_policy.py --task task_2 --episodes 600 --eval-every 20
    python train_policy.py --task task_1 --mode eval --model-file rl/xero/models/task1_best_policy.npz
"""

from __future__ import annotations

import argparse
import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

warnings.filterwarnings(
    "ignore",
    message="websockets.legacy is deprecated",
    category=DeprecationWarning,
)

try:
    from models import ProteinAction, ProteinObservation
    from server.xero_environment import ProteinFoldingEnvironment
    from test import build_action_candidates, format_action
except ImportError:
    from xero.models import ProteinAction, ProteinObservation
    from xero.server.xero_environment import ProteinFoldingEnvironment
    from xero.test import build_action_candidates, format_action


@dataclass
class EpisodeStep:
    """Transition data used for policy and value updates."""

    features: np.ndarray
    action_index: int
    reward: float
    value_estimate: float


class RunningStat:
    """Track running mean and variance for reward normalization."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 1.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.variance, 1e-6)))

    def normalize(self, value: float, clip_value: float) -> float:
        normalized = (value - self.mean) / self.std
        return float(np.clip(normalized, -clip_value, clip_value))


class ActorCriticPolicy:
    """Linear actor-critic with softmax policy and scalar value head."""

    def __init__(self, num_features: int, num_actions: int, rng: np.random.Generator):
        self.num_features = num_features
        self.num_actions = num_actions
        self.rng = rng
        self.policy_weights = rng.normal(0.0, 0.03, size=(num_features, num_actions))
        self.policy_bias = np.zeros(num_actions, dtype=float)
        self.value_weights = rng.normal(0.0, 0.03, size=num_features)
        self.value_bias = 0.0

    def policy_logits(self, features: np.ndarray) -> np.ndarray:
        """Compute action logits."""
        return features @ self.policy_weights + self.policy_bias

    def action_probabilities(
        self,
        features: np.ndarray,
        candidate_indices: np.ndarray,
    ) -> np.ndarray:
        """Softmax over the pruned action subset."""
        logits = self.policy_logits(features)[candidate_indices]
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    def value(self, features: np.ndarray) -> float:
        """Predict state value."""
        return float(features @ self.value_weights + self.value_bias)

    def sample_action(
        self,
        features: np.ndarray,
        candidate_indices: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        """Sample from the current policy over the candidate set."""
        probs = self.action_probabilities(features, candidate_indices)
        local_index = int(self.rng.choice(len(candidate_indices), p=probs))
        return int(candidate_indices[local_index]), probs

    def greedy_action(
        self,
        features: np.ndarray,
        candidate_indices: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        """Pick the highest-probability action over the candidate set."""
        probs = self.action_probabilities(features, candidate_indices)
        local_index = int(np.argmax(probs))
        return int(candidate_indices[local_index]), probs

    def update(
        self,
        trajectory: list[EpisodeStep],
        gamma: float,
        actor_lr: float,
        critic_lr: float,
        entropy_coef: float,
        reward_stats: RunningStat,
    ) -> tuple[float, float]:
        """Apply actor and critic updates using advantage estimates."""
        normalized_rewards = []
        for step in trajectory:
            reward_stats.update(step.reward)
            normalized_rewards.append(reward_stats.normalize(step.reward, clip_value=3.0))

        returns = discounted_returns(normalized_rewards, gamma)
        advantages = returns - np.array([step.value_estimate for step in trajectory], dtype=float)
        advantages = normalize_vector(advantages)

        actor_loss = 0.0
        critic_loss = 0.0
        grad_policy_w = np.zeros_like(self.policy_weights)
        grad_policy_b = np.zeros_like(self.policy_bias)
        grad_value_w = np.zeros_like(self.value_weights)
        grad_value_b = 0.0

        for step, ret, advantage in zip(trajectory, returns, advantages):
            logits = self.policy_logits(step.features)
            logits = logits - np.max(logits)
            probs = np.exp(logits)
            probs /= np.sum(probs)

            one_hot = np.zeros(self.num_actions, dtype=float)
            one_hot[step.action_index] = 1.0

            policy_delta = (one_hot - probs) * advantage
            grad_policy_w += np.outer(step.features, policy_delta)
            grad_policy_b += policy_delta

            entropy = -float(np.sum(probs * np.log(probs + 1e-8)))
            grad_policy_b += entropy_coef * (-np.log(probs + 1e-8) - 1.0)
            entropy_grad = entropy_coef * probs * (entropy - np.log(probs + 1e-8))
            grad_policy_w += np.outer(step.features, entropy_grad)
            actor_loss += -float(np.log(probs[step.action_index] + 1e-8)) * advantage

            value_error = ret - step.value_estimate
            grad_value_w += step.features * value_error
            grad_value_b += value_error
            critic_loss += 0.5 * (value_error**2)

        scale = 1.0 / max(len(trajectory), 1)
        self.policy_weights += actor_lr * grad_policy_w * scale
        self.policy_bias += actor_lr * grad_policy_b * scale
        self.value_weights += critic_lr * grad_value_w * scale
        self.value_bias += critic_lr * grad_value_b * scale
        return float(actor_loss * scale), float(critic_loss * scale)

    def save(self, model_file: str, action_count: int) -> None:
        """Persist policy and value parameters."""
        path = Path(model_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            policy_weights=self.policy_weights,
            policy_bias=self.policy_bias,
            value_weights=self.value_weights,
            value_bias=self.value_bias,
            num_features=self.num_features,
            action_count=action_count,
        )

    @classmethod
    def load(cls, model_file: str, rng: np.random.Generator) -> "ActorCriticPolicy":
        """Restore an actor-critic model from disk."""
        payload = np.load(model_file)
        policy = cls(
            num_features=int(payload["num_features"]),
            num_actions=int(payload["action_count"]),
            rng=rng,
        )
        policy.policy_weights = payload["policy_weights"]
        policy.policy_bias = payload["policy_bias"]
        policy.value_weights = payload["value_weights"]
        policy.value_bias = float(payload["value_bias"])
        return policy


def discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    """Compute discounted returns."""
    returns = np.zeros(len(rewards), dtype=float)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + gamma * running
        returns[index] = running
    return returns


def normalize_vector(values: np.ndarray) -> np.ndarray:
    """Normalize a vector for stable optimization."""
    if values.size == 0:
        return values
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-8:
        return values - mean
    return (values - mean) / (std + 1e-8)


def extract_features(observation: ProteinObservation, max_steps: int) -> np.ndarray:
    """Create a compact feature vector from the current observation."""
    torsions = np.asarray(observation.torsion_angles, dtype=float)
    coords = np.asarray(observation.coordinates, dtype=float)
    score_components = observation.metadata.get("score_components", {})

    if torsions.size == 0:
        mean_phi = mean_psi = std_phi = std_psi = torsion_abs_mean = 0.0
    else:
        mean_phi = float(np.mean(torsions[:, 0]) / 180.0)
        mean_psi = float(np.mean(torsions[:, 1]) / 180.0)
        std_phi = float(np.std(torsions[:, 0]) / 180.0)
        std_psi = float(np.std(torsions[:, 1]) / 180.0)
        torsion_abs_mean = float(np.mean(np.abs(torsions)) / 180.0)

    if coords.size == 0:
        radius_mean = radius_max = compactness = z_spread = 0.0
    else:
        radii = np.linalg.norm(coords, axis=1)
        radius_mean = float(np.mean(radii) / 10.0)
        radius_max = float(np.max(radii) / 10.0)
        compactness = float(1.0 / (1.0 + np.var(radii)))
        z_spread = float(np.std(coords[:, 2]) / 5.0)

    feature_values = np.array(
        [
            1.0,
            observation.energy / 50.0,
            observation.step_count / max(max_steps, 1),
            observation.hydrophobic_contacts / 10.0,
            observation.collisions / 10.0,
            float(observation.metadata.get("score", 0.0)),
            float(score_components.get("energy_reduction_ratio", 0.0)),
            float(score_components.get("hydrophobic_contact_ratio", 0.0)),
            float(score_components.get("stability_score", 0.0)),
            mean_phi,
            mean_psi,
            std_phi,
            std_psi,
            torsion_abs_mean,
            radius_mean,
            radius_max,
            compactness,
            z_spread,
        ],
        dtype=float,
    )
    return feature_values


def candidate_subset(
    policy: ActorCriticPolicy,
    features: np.ndarray,
    actions: list[ProteinAction],
    action_prune_k: int,
) -> np.ndarray:
    """Keep only the most promising policy actions for exploration/evaluation."""
    logits = policy.policy_logits(features)
    ranked = np.argsort(logits)[::-1]
    keep = max(1, min(action_prune_k, len(actions)))
    return ranked[:keep]


def run_episode(
    env: ProteinFoldingEnvironment,
    task_id: str,
    seed: int | None,
    policy: ActorCriticPolicy,
    actions: list[ProteinAction],
    greedy: bool,
    action_prune_k: int,
) -> tuple[list[EpisodeStep], ProteinObservation, float]:
    """Roll out one episode with the current actor-critic policy."""
    observation = env.reset(seed=seed, task_id=task_id)
    trajectory: list[EpisodeStep] = []
    total_reward = 0.0
    max_steps = env.TASKS[task_id].max_steps

    while not observation.done and env.state.step_count < max_steps:
        features = extract_features(observation, max_steps)
        value_estimate = policy.value(features)
        candidates = candidate_subset(policy, features, actions, action_prune_k)

        if greedy:
            action_index, _ = policy.greedy_action(features, candidates)
        else:
            action_index, _ = policy.sample_action(features, candidates)

        action = actions[action_index]
        next_observation = env.step(action)
        reward = float(next_observation.reward or 0.0)
        trajectory.append(
            EpisodeStep(
                features=features,
                action_index=action_index,
                reward=reward,
                value_estimate=value_estimate,
            )
        )
        total_reward += reward
        observation = next_observation

    return trajectory, observation, total_reward


def evaluate_policy(
    task_id: str,
    policy: ActorCriticPolicy,
    actions: list[ProteinAction],
    episodes: int,
    seed: int,
    action_prune_k: int,
) -> tuple[float, float, float]:
    """Evaluate the policy greedily on fresh seeds."""
    rewards: list[float] = []
    scores: list[float] = []
    energies: list[float] = []

    for episode in range(episodes):
        env = ProteinFoldingEnvironment()
        _, observation, total_reward = run_episode(
            env=env,
            task_id=task_id,
            seed=seed + episode,
            policy=policy,
            actions=actions,
            greedy=True,
            action_prune_k=action_prune_k,
        )
        rewards.append(total_reward)
        scores.append(float(observation.metadata.get("score", 0.0)))
        energies.append(float(observation.energy))

    return float(np.mean(rewards)), float(np.mean(scores)), float(np.mean(energies))


def write_csv_row(csv_file: str, row: dict[str, float | int]) -> None:
    """Append a row to the training log CSV."""
    path = Path(csv_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    fieldnames = list(row.keys())
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train(args: argparse.Namespace) -> None:
    """Train the actor-critic policy and save the best checkpoint."""
    rng = np.random.default_rng(args.seed)
    env = ProteinFoldingEnvironment()
    initial_observation = env.reset(seed=args.seed, task_id=args.task)
    actions = build_action_candidates(len(initial_observation.coordinates))
    num_features = len(extract_features(initial_observation, env.TASKS[args.task].max_steps))

    policy = ActorCriticPolicy(num_features=num_features, num_actions=len(actions), rng=rng)
    reward_stats = RunningStat()
    best_eval_score = -float("inf")

    print(f"Training on {args.task} with {len(actions)} discrete actions and {num_features} features")
    print(
        f"Episodes={args.episodes}, gamma={args.gamma}, "
        f"actor_lr={args.actor_lr}, critic_lr={args.critic_lr}, prune_k={args.action_prune_k}"
    )

    for episode in range(1, args.episodes + 1):
        episode_seed = args.seed + episode
        env = ProteinFoldingEnvironment()
        trajectory, observation, total_reward = run_episode(
            env=env,
            task_id=args.task,
            seed=episode_seed,
            policy=policy,
            actions=actions,
            greedy=False,
            action_prune_k=args.action_prune_k,
        )
        actor_loss, critic_loss = policy.update(
            trajectory=trajectory,
            gamma=args.gamma,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            entropy_coef=args.entropy_coef,
            reward_stats=reward_stats,
        )

        if episode == 1 or episode % args.eval_every == 0:
            eval_reward, eval_score, eval_energy = evaluate_policy(
                task_id=args.task,
                policy=policy,
                actions=actions,
                episodes=args.eval_episodes,
                seed=args.seed + 10_000 + episode,
                action_prune_k=args.action_prune_k,
            )

            row = {
                "episode": episode,
                "train_reward": round(total_reward, 6),
                "train_score": round(float(observation.metadata.get("score", 0.0)), 6),
                "train_energy": round(float(observation.energy), 6),
                "actor_loss": round(actor_loss, 6),
                "critic_loss": round(critic_loss, 6),
                "eval_reward": round(eval_reward, 6),
                "eval_score": round(eval_score, 6),
                "eval_energy": round(eval_energy, 6),
            }
            write_csv_row(args.metrics_file, row)

            print(
                f"episode={episode:04d} "
                f"train_reward={total_reward:8.3f} "
                f"train_score={float(observation.metadata.get('score', 0.0)):.3f} "
                f"actor_loss={actor_loss:8.3f} "
                f"critic_loss={critic_loss:8.3f} "
                f"eval_reward={eval_reward:8.3f} "
                f"eval_score={eval_score:.3f} "
                f"eval_energy={eval_energy:8.3f}"
            )

            if eval_score > best_eval_score:
                best_eval_score = eval_score
                policy.save(args.best_model_file, action_count=len(actions))
                print(
                    f"  best checkpoint updated -> {args.best_model_file} "
                    f"(eval_score={eval_score:.3f})"
                )

    policy.save(args.model_file, action_count=len(actions))
    print(f"Saved final policy to {args.model_file}")
    print(f"Best policy saved to {args.best_model_file}")
    print(f"Metrics CSV written to {args.metrics_file}")


def evaluate(args: argparse.Namespace) -> None:
    """Load a saved policy and run a greedy evaluation episode."""
    rng = np.random.default_rng(args.seed)
    env = ProteinFoldingEnvironment()
    initial_observation = env.reset(seed=args.seed, task_id=args.task)
    actions = build_action_candidates(len(initial_observation.coordinates))
    policy = ActorCriticPolicy.load(args.model_file, rng=rng)

    trajectory, observation, total_reward = run_episode(
        env=env,
        task_id=args.task,
        seed=args.seed,
        policy=policy,
        actions=actions,
        greedy=True,
        action_prune_k=args.action_prune_k,
    )

    print(f"Loaded policy from {args.model_file}")
    print(f"Task={args.task}")
    print(f"Steps={len(trajectory)}")
    print(f"Total reward={total_reward:.3f}")
    print(f"Final energy={observation.energy:.3f}")
    print(f"Final score={float(observation.metadata.get('score', 0.0)):.3f}")
    print(f"Hydrophobic contacts={observation.hydrophobic_contacts}")
    print(f"Collisions={observation.collisions}")

    if trajectory:
        print("Greedy action trace:")
        env = ProteinFoldingEnvironment()
        obs = env.reset(seed=args.seed, task_id=args.task)
        for step_index, step in enumerate(trajectory, start=1):
            action = actions[step.action_index]
            obs = env.step(action)
            print(
                f"  {step_index:02d}. {format_action(action)} "
                f"=> reward={float(obs.reward or 0.0):.3f}, "
                f"energy={obs.energy:.3f}, "
                f"score={float(obs.metadata.get('score', 0.0)):.3f}"
            )


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description="Train an actor-critic protein folding agent")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument(
        "--task",
        choices=tuple(ProteinFoldingEnvironment.TASKS.keys()),
        default="task_1",
    )
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--actor-lr", type=float, default=0.01)
    parser.add_argument("--critic-lr", type=float, default=0.02)
    parser.add_argument("--entropy-coef", type=float, default=0.0005)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--action-prune-k", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--model-file",
        default="rl/xero/models/protein_policy_final.npz",
        help="Path used to save or load the final policy parameters",
    )
    parser.add_argument(
        "--best-model-file",
        default="rl/xero/models/protein_policy_best.npz",
        help="Path used to save the best evaluation checkpoint",
    )
    parser.add_argument(
        "--metrics-file",
        default="rl/xero/logs/training_metrics.csv",
        help="CSV file for evaluation snapshots during training",
    )
    return parser


def main() -> None:
    """Entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
