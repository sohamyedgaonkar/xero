"""Greedy search harness for the protein folding environment.

This script resets the environment for a selected task and then searches for
high-value actions using short-horizon rollout planning. It does not guarantee
the global optimum, but it consistently picks strong actions from the current
state instead of using random exploration.

Example:
    python test.py --task task_1 --seed 7
    python test.py --task task_3 --seed 42 --depth 3 --beam-width 8
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    from models import ProteinAction, ProteinObservation
    from server.xero_environment import ProteinFoldingEnvironment
except ImportError:
    from xero.models import ProteinAction, ProteinObservation
    from xero.server.xero_environment import ProteinFoldingEnvironment


@dataclass
class SearchResult:
    """Container for a simulated action sequence."""

    action_path: list[ProteinAction]
    final_observation: ProteinObservation
    objective: float
    env: ProteinFoldingEnvironment


class HumanLogger:
    """Write human-readable logs to stdout and an optional file."""

    def __init__(self, log_path: str | None):
        self._path = Path(log_path) if log_path else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text("", encoding="utf-8")

    def log(self, message: str = "") -> None:
        print(message)
        if self._path is not None:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")

    @property
    def path(self) -> Path | None:
        return self._path


def build_action_candidates(length: int) -> list[ProteinAction]:
    """Create a diverse but bounded set of valid actions for the current chain."""
    candidates: list[ProteinAction] = []
    angle_choices = (-60.0, -30.0, 30.0, 60.0)

    for residue_index in range(1, max(length - 1, 1)):
        for angle_delta in angle_choices:
            candidates.append(
                ProteinAction(
                    action_type="rotate_phi",
                    residue_index=residue_index,
                    angle_delta=angle_delta,
                )
            )
            candidates.append(
                ProteinAction(
                    action_type="rotate_psi",
                    residue_index=residue_index,
                    angle_delta=angle_delta,
                )
            )

    for residue_index in range(0, max(length - 2, 1)):
        for angle_delta in angle_choices:
            candidates.append(
                ProteinAction(
                    action_type="pivot_rotation",
                    residue_index=residue_index,
                    angle_delta=angle_delta,
                )
            )

    window_sizes = (3, 4, 5)
    for window_size in window_sizes:
        if window_size > length:
            continue
        step = 1 if length <= 12 else 2
        for start in range(0, length - window_size + 1, step):
            end = start + window_size - 1
            candidates.append(
                ProteinAction(
                    action_type="segment_flip",
                    segment_start=start,
                    segment_end=end,
                )
            )
            for angle_delta in (-45.0, 45.0):
                candidates.append(
                    ProteinAction(
                        action_type="crankshaft_move",
                        segment_start=start,
                        segment_end=end,
                        angle_delta=angle_delta,
                    )
                )

    for angle_delta in (-45.0, -20.0, 20.0, 45.0):
        candidates.append(
            ProteinAction(
                action_type="end_move_forward",
                angle_delta=angle_delta,
            )
        )
        candidates.append(
            ProteinAction(
                action_type="end_move_backward",
                angle_delta=angle_delta,
            )
        )

    return candidates


def observation_objective(observation: ProteinObservation) -> float:
    """Score an observation for search ranking.

    The environment already exposes a normalized task score, so we use that as
    the main signal and add smaller tie-breakers for lower energy and fewer
    collisions.
    """
    score = float(observation.metadata.get("score", 0.0))
    reward = float(observation.reward or 0.0)
    return (
        score * 100.0
        + reward
        - observation.energy * 0.5
        - observation.collisions * 4.0
        + observation.hydrophobic_contacts * 2.0
        - observation.step_count * 0.05
    )


def format_action(action: ProteinAction) -> str:
    """Format an action in a compact human-readable way."""
    return (
        f"{action.action_type} "
        f"(residue={action.residue_index}, "
        f"segment={action.segment_start}:{action.segment_end}, "
        f"delta={action.angle_delta})"
    )


def describe_observation(observation: ProteinObservation) -> str:
    """Summarize the current conformation in plain language."""
    score_components = observation.metadata.get("score_components", {})
    return (
        f"energy={observation.energy:.3f}, "
        f"reward={float(observation.reward or 0.0):.3f}, "
        f"score={float(observation.metadata.get('score', 0.0)):.3f}, "
        f"contacts={observation.hydrophobic_contacts}, "
        f"collisions={observation.collisions}, "
        f"energy_reduction_ratio={float(score_components.get('energy_reduction_ratio', 0.0)):.3f}, "
        f"hydrophobic_contact_ratio={float(score_components.get('hydrophobic_contact_ratio', 0.0)):.3f}, "
        f"stability_score={float(score_components.get('stability_score', 0.0)):.3f}"
    )


def explain_reward(observation: ProteinObservation) -> str:
    """Turn reward metadata into a readable explanation."""
    breakdown = observation.metadata.get("reward_breakdown", {})
    return (
        f"reward_breakdown: "
        f"energy_term={float(breakdown.get('energy_term', 0.0)):.3f}, "
        f"progress_term={float(breakdown.get('progress_term', 0.0)):.3f}, "
        f"stability_term={float(breakdown.get('stability_term', 0.0)):.3f}, "
        f"collision_penalty={float(breakdown.get('collision_penalty', 0.0)):.3f}, "
        f"invalid_action={float(breakdown.get('invalid_action', 0.0)):.3f}"
    )


def simulate_action(
    env: ProteinFoldingEnvironment,
    action: ProteinAction,
) -> tuple[ProteinFoldingEnvironment, ProteinObservation]:
    """Simulate one action on a copied environment."""
    env_copy = copy.deepcopy(env)
    observation = env_copy.step(action)
    return env_copy, observation


def choose_best_action(
    env: ProteinFoldingEnvironment,
    candidates: list[ProteinAction],
    depth: int,
    beam_width: int,
) -> tuple[SearchResult, list[SearchResult]]:
    """Run a small beam search and return the best first action found."""
    frontier: list[SearchResult] = []

    for action in candidates:
        next_env, observation = simulate_action(env, action)
        frontier.append(
            SearchResult(
                action_path=[action],
                final_observation=observation,
                objective=observation_objective(observation),
                env=next_env,
            )
        )

    frontier.sort(key=lambda item: item.objective, reverse=True)
    frontier = frontier[:beam_width]
    best_result = frontier[0]

    for _ in range(1, depth):
        expanded: list[SearchResult] = []
        for result in frontier:
            if result.final_observation.done:
                expanded.append(result)
                continue

            for action in candidates:
                next_env, observation = simulate_action(result.env, action)
                objective = observation_objective(observation)
                expanded.append(
                    SearchResult(
                        action_path=result.action_path + [action],
                        final_observation=observation,
                        objective=objective,
                        env=next_env,
                    )
                )

        expanded.sort(key=lambda item: item.objective, reverse=True)
        frontier = expanded[:beam_width]
        if frontier and frontier[0].objective > best_result.objective:
            best_result = frontier[0]

    return best_result, frontier


def log_top_candidates(
    logger: HumanLogger,
    frontier: list[SearchResult],
    step_number: int,
    top_k: int,
) -> None:
    """Log the best candidate actions considered at a decision point."""
    logger.log(f"Top candidate actions considered for step {step_number}:")
    for rank, result in enumerate(frontier[:top_k], start=1):
        candidate_action = result.action_path[0]
        logger.log(
            f"  {rank}. {format_action(candidate_action)} "
            f"=> objective={result.objective:.3f}, "
            f"{describe_observation(result.final_observation)}"
        )
    logger.log()


def run_episode(
    task_id: str,
    seed: int,
    depth: int,
    beam_width: int,
    log_path: str | None,
    top_k: int,
) -> None:
    """Reset the environment and execute the strongest action found each step."""
    logger = HumanLogger(log_path)
    env = ProteinFoldingEnvironment()
    observation = env.reset(seed=seed, task_id=task_id)
    candidates = build_action_candidates(len(observation.coordinates))

    logger.log("Protein Folding Search Log")
    logger.log(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    logger.log(f"Task: {task_id}")
    logger.log(f"Episode: {env.state.episode_id}")
    logger.log(
        f"Search settings: depth={depth}, beam_width={beam_width}, candidates={len(candidates)}"
    )
    logger.log(
        "Reset -> "
        f"{describe_observation(observation)}"
    )
    logger.log()

    history: list[ProteinAction] = []
    while not observation.done and env.state.step_count < env.TASKS[task_id].max_steps:
        best_result, frontier = choose_best_action(
            env,
            candidates,
            depth=depth,
            beam_width=beam_width,
        )
        best_action = best_result.action_path[0]
        log_top_candidates(
            logger=logger,
            frontier=frontier,
            step_number=env.state.step_count + 1,
            top_k=top_k,
        )

        previous_energy = observation.energy
        previous_contacts = observation.hydrophobic_contacts
        previous_collisions = observation.collisions
        observation = env.step(best_action)
        history.append(best_action)

        logger.log(
            f"Step {observation.step_count:03d}: "
            f"chosen_action={format_action(best_action)}"
        )
        logger.log(f"  before: energy={previous_energy:.3f}, contacts={previous_contacts}, collisions={previous_collisions}")
        logger.log(f"  after:  {describe_observation(observation)}")
        logger.log(f"  {explain_reward(observation)}")
        logger.log(
            "  reasoning: selected because it produced the strongest search objective "
            "among the candidate rollouts while favoring lower energy, fewer collisions, "
            "and stronger hydrophobic packing."
        )
        logger.log()

        if observation.done:
            break

    logger.log("Final summary")
    logger.log(f"  completed={observation.done}")
    logger.log(f"  steps={observation.step_count}")
    logger.log(f"  energy={observation.energy:.3f}")
    logger.log(f"  score={float(observation.metadata.get('score', 0.0)):.3f}")
    logger.log(f"  hydrophobic_contacts={observation.hydrophobic_contacts}")
    logger.log(f"  collisions={observation.collisions}")
    logger.log("  chosen_actions=")
    for index, action in enumerate(history, start=1):
        logger.log(
            f"    {index:02d}. {action.action_type} "
            f"residue={action.residue_index} "
            f"segment={action.segment_start}:{action.segment_end} "
            f"delta={action.angle_delta}"
        )
    if logger.path is not None:
        logger.log()
        logger.log(f"Log saved to: {logger.path}")


def main() -> None:
    """Parse CLI arguments and run a search-driven episode."""
    parser = argparse.ArgumentParser(description="Protein folding search harness")
    parser.add_argument(
        "--task",
        choices=tuple(ProteinFoldingEnvironment.TASKS.keys()),
        default="task_1",
        help="Task identifier from openenv.yaml",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used when resetting the environment",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Lookahead depth for beam search",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=6,
        help="Number of rollout branches kept at each search depth",
    )
    parser.add_argument(
        "--log-file",
        default="rl/xero/logs/protein_folding_run.log",
        help="Optional path for a human-readable log file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many top candidate actions to log per decision step",
    )
    args = parser.parse_args()
    run_episode(
        args.task,
        args.seed,
        args.depth,
        args.beam_width,
        args.log_file,
        args.top_k,
    )


if __name__ == "__main__":
    main()
