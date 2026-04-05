# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Protein folding environment implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import numpy as np
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import ProteinAction, ProteinObservation
except ImportError:
    # This handles the case when running via uvicorn server.app
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import ProteinAction, ProteinObservation


@dataclass(frozen=True)
class TaskConfig:
    """Per-task configuration for the simplified folding problem."""

    task_id: str
    protein_length: int
    goal: str
    folding_ratio: float
    max_steps: int = 200


class ProteinFoldingEnvironment(Environment):
    """Simplified protein folding environment with task-specific chain lengths."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    TASKS: dict[str, TaskConfig] = {
        "task_1": TaskConfig(
            task_id="task_1",
            protein_length=15,
            goal="reduce energy by 30%",
            folding_ratio=0.85,
        ),
        "task_2": TaskConfig(
            task_id="task_2",
            protein_length=15,
            goal="form hydrophobic core",
            folding_ratio=0.85,
        ),
        "task_3": TaskConfig(
            task_id="task_3",
            protein_length=20,
            goal="reach minimum energy",
            folding_ratio=0.55,
        ),
    }

    BOND_LENGTH = 1.5
    CONTACT_THRESHOLD = 2.6
    COLLISION_THRESHOLD = 1.05
    ALLOWED_TORSION_RANGE = 180.0

    def __init__(self):
        """Initialize the protein folding environment."""
        super().__init__()
        self._rng = np.random.default_rng()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task = self.TASKS["task_1"]
        self._coordinates = np.zeros((0, 3), dtype=float)
        self._torsion_angles = np.zeros((0, 2), dtype=float)
        self._contact_map = np.zeros((0, 0), dtype=int)
        self._hydrophobic_mask = np.zeros(0, dtype=bool)
        self._energy = 0.0
        self._hydrophobic_contacts = 0
        self._collisions = 0
        self._initial_energy = 0.0
        self._max_hydrophobic_contacts = 1
        self._folding_threshold = 0.0

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> ProteinObservation:
        """Create a new unfolded conformation for the selected task."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        task_id = str(kwargs.get("task_id", self._task.task_id))
        self._task = self.TASKS.get(task_id, self.TASKS["task_1"])
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        length = self._task.protein_length

        # Start from a straight chain, then randomize torsions to create an unfolded pose.
        self._coordinates = self._generate_straight_chain(length)
        self._torsion_angles = self._rng.uniform(-180.0, 180.0, size=(length, 2))
        self._torsion_angles[0] = np.array([0.0, 0.0])
        self._coordinates = self._build_coordinates_from_torsions(self._torsion_angles)

        self._hydrophobic_mask = self._build_hydrophobic_mask(length)
        self._update_metrics()
        self._initial_energy = self._energy
        self._folding_threshold = self._initial_energy * self._task.folding_ratio
        self._max_hydrophobic_contacts = max(
            1,
            self._estimate_max_hydrophobic_contacts(length),
        )
        return self._make_observation(reward=0.0, done=False, invalid_action=False)

    def step(
        self,
        action: ProteinAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> ProteinObservation:
        """Apply a structural move, recompute energy, and emit shaped reward."""
        del timeout_s
        if not self._is_initialized():
            self.reset(
                seed=kwargs.get("seed"),
                task_id=kwargs.get("task_id", self._task.task_id),
                episode_id=kwargs.get("episode_id"),
            )

        previous_energy = self._energy
        previous_contacts = self._hydrophobic_contacts
        invalid_action = False

        next_torsions = np.array(self._torsion_angles, copy=True)
        next_coordinates = np.array(self._coordinates, copy=True)

        action_name = action.action_type.strip().lower()
        try:
            next_torsions, next_coordinates = self._apply_action(
                action_name,
                action,
                next_torsions,
                next_coordinates,
            )
        except ValueError:
            invalid_action = True

        self._state.step_count += 1

        if invalid_action:
            reward = 0.0
            done = self._is_done()
            observation = self._make_observation(
                reward=reward,
                done=done,
                invalid_action=True,
            )
            observation.metadata["reward_breakdown"]["invalid_action"] = 0.0
            return observation

        self._torsion_angles = self._normalize_angles(next_torsions)
        self._coordinates = next_coordinates
        self._update_metrics()

        done = self._is_done()

        # Compute normalized reward components in [0, 1]
        initial_energy_magnitude = max(abs(self._initial_energy), 1e-6)
        energy_improvement = (previous_energy - self._energy) / initial_energy_magnitude
        energy_term = np.clip(energy_improvement, 0.0, 1.0)
        
        progress_term = np.clip(
            float(self._hydrophobic_contacts) / max(self._max_hydrophobic_contacts, 1),
            0.0,
            1.0
        )
        
        collision_cost = np.clip(float(self._collisions) / 10.0, 0.0, 1.0)
        stability_term = 1.0 - collision_cost
        
        # Weighted average to keep reward in [0, 1]
        reward = float(0.4 * energy_term + 0.35 * progress_term + 0.25 * stability_term)
        reward = float(np.round(reward, 4))

        observation = self._make_observation(
            reward=reward,
            done=done,
            invalid_action=False,
        )
        observation.metadata["reward_breakdown"] = {
            "energy_term": float(np.round(energy_term, 4)),
            "progress_term": float(np.round(progress_term, 4)),
            "collision_cost": float(np.round(collision_cost, 4)),
            "total_reward": reward,
        }
        return observation

    @property
    def state(self) -> State:
        """Return the environment state for OpenEnv introspection."""
        return self._state

    def _apply_action(
        self,
        action_name: str,
        action: ProteinAction,
        torsions: np.ndarray,
        coordinates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Dispatch and apply a supported structural move."""
        delta = float(action.angle_delta if action.angle_delta is not None else 15.0)

        if action_name == "rotate_phi":
            index = self._require_index(action.residue_index, torsions.shape[0])
            torsions[index, 0] += delta
            return torsions, self._build_coordinates_from_torsions(torsions)

        if action_name == "rotate_psi":
            index = self._require_index(action.residue_index, torsions.shape[0])
            torsions[index, 1] += delta
            return torsions, self._build_coordinates_from_torsions(torsions)

        if action_name == "pivot_rotation":
            index = self._require_index(action.residue_index, torsions.shape[0] - 1)
            torsions[index + 1 :, 0] += delta
            torsions[index + 1 :, 1] -= delta * 0.5
            return torsions, self._build_coordinates_from_torsions(torsions)

        if action_name == "segment_flip":
            start, end = self._require_segment(
                action.segment_start,
                action.segment_end,
                torsions.shape[0],
            )
            torsions[start : end + 1] = torsions[start : end + 1][::-1]
            return torsions, self._build_coordinates_from_torsions(torsions)

        if action_name == "crankshaft_move":
            start, end = self._require_segment(
                action.segment_start,
                action.segment_end,
                torsions.shape[0],
            )
            if end - start < 2:
                raise ValueError("crankshaft_move requires a segment of at least length 3")
            torsions[start + 1 : end, 0] += delta
            torsions[start + 1 : end, 1] -= delta
            return torsions, self._build_coordinates_from_torsions(torsions)

        if action_name == "end_move_forward":
            torsions[-1, 0] += delta
            torsions[-1, 1] += delta * 0.5
            return torsions, self._build_coordinates_from_torsions(torsions)

        if action_name == "end_move_backward":
            torsions[0, 0] -= delta * 0.5
            torsions[0, 1] -= delta
            return torsions, self._build_coordinates_from_torsions(torsions)

        raise ValueError(f"Unsupported action type: {action.action_type}")

    def _generate_straight_chain(self, length: int) -> np.ndarray:
        """Generate a straight chain used as the base unfolded conformation."""
        coordinates = np.zeros((length, 3), dtype=float)
        coordinates[:, 0] = np.arange(length, dtype=float) * self.BOND_LENGTH
        return coordinates

    def _build_coordinates_from_torsions(self, torsions: np.ndarray) -> np.ndarray:
        """Convert torsion angles into a coarse 3D chain geometry."""
        length = torsions.shape[0]
        coordinates = np.zeros((length, 3), dtype=float)
        if length <= 1:
            return coordinates

        for index in range(1, length):
            phi_rad = np.deg2rad(torsions[index - 1, 0])
            psi_rad = np.deg2rad(torsions[index - 1, 1])
            direction = np.array(
                [
                    np.cos(phi_rad) * np.cos(psi_rad),
                    np.sin(phi_rad) * np.cos(psi_rad),
                    np.sin(psi_rad),
                ],
                dtype=float,
            )
            norm = np.linalg.norm(direction)
            if norm < 1e-8:
                direction = np.array([1.0, 0.0, 0.0], dtype=float)
            else:
                direction /= norm
            coordinates[index] = coordinates[index - 1] + direction * self.BOND_LENGTH

        # Center the chain around the origin to keep transforms numerically stable.
        coordinates -= np.mean(coordinates, axis=0, keepdims=True)
        return coordinates

    def _build_hydrophobic_mask(self, length: int) -> np.ndarray:
        """Mark every other residue as hydrophobic to create clustering pressure."""
        return np.array([(index % 2) == 0 for index in range(length)], dtype=bool)

    def _update_metrics(self) -> None:
        """Recompute geometry-derived metrics after each move."""
        self._contact_map = self._compute_contact_map(self._coordinates)
        self._collisions = self._count_collisions(self._coordinates)
        self._hydrophobic_contacts = self._count_hydrophobic_contacts(self._contact_map)
        self._energy = self._compute_energy(
            self._coordinates,
            self._torsion_angles,
            self._hydrophobic_contacts,
            self._collisions,
        )

    def _compute_contact_map(self, coordinates: np.ndarray) -> np.ndarray:
        """Build a non-local binary contact map from pairwise distances."""
        length = coordinates.shape[0]
        contact_map = np.zeros((length, length), dtype=int)
        for i in range(length):
            for j in range(i + 2, length):
                distance = float(np.linalg.norm(coordinates[i] - coordinates[j]))
                if distance <= self.CONTACT_THRESHOLD:
                    contact_map[i, j] = 1
                    contact_map[j, i] = 1
        return contact_map

    def _count_hydrophobic_contacts(self, contact_map: np.ndarray) -> int:
        """Count contacts where both residues are hydrophobic."""
        hydrophobic_pairs = np.outer(self._hydrophobic_mask, self._hydrophobic_mask)
        masked = np.triu(contact_map * hydrophobic_pairs.astype(int), k=2)
        return int(np.sum(masked))

    def _count_collisions(self, coordinates: np.ndarray) -> int:
        """Count steric clashes between non-bonded residues."""
        collisions = 0
        length = coordinates.shape[0]
        for i in range(length):
            for j in range(i + 2, length):
                distance = float(np.linalg.norm(coordinates[i] - coordinates[j]))
                if distance < self.COLLISION_THRESHOLD:
                    collisions += 1
        return collisions

    def _compute_energy(
        self,
        coordinates: np.ndarray,
        torsions: np.ndarray,
        hydrophobic_contacts: int,
        collisions: int,
    ) -> float:
        """Combine the simplified energy terms into a scalar objective."""
        bond_lengths = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
        bond_energy = float(np.sum((bond_lengths - self.BOND_LENGTH) ** 2))
        angle_excess = np.maximum(np.abs(torsions) - self.ALLOWED_TORSION_RANGE, 0.0)
        angle_energy = float(np.sum(angle_excess**2) / 180.0)
        hydrophobic_energy = -1.0 * hydrophobic_contacts
        steric_energy = 5.0 * collisions
        total_energy = hydrophobic_energy + steric_energy + bond_energy + angle_energy
        return float(total_energy)

    def _compute_score(self) -> dict[str, float]:
        """Compute task score components and normalize them to [0, 1]."""
        denominator = max(abs(self._initial_energy), 1e-6)
        energy_reduction_ratio = np.clip(
            (self._initial_energy - self._energy) / denominator,
            0.0,
            1.0,
        )
        hydrophobic_contact_ratio = np.clip(
            self._hydrophobic_contacts / max(self._max_hydrophobic_contacts, 1),
            0.0,
            1.0,
        )
        stability_score = 1.0 if self._collisions == 0 else float(1.0 / (1 + self._collisions))
        total_score = float(
            np.clip(
                (energy_reduction_ratio + hydrophobic_contact_ratio + stability_score) / 3.0,
                0.0,
                1.0,
            )
        )
        return {
            "energy_reduction_ratio": float(energy_reduction_ratio),
            "hydrophobic_contact_ratio": float(hydrophobic_contact_ratio),
            "stability_score": float(stability_score),
            "score": total_score,
        }

    def _make_observation(
        self,
        reward: float,
        done: bool,
        invalid_action: bool,
    ) -> ProteinObservation:
        """Create an observation enriched with task metrics and score details."""
        score = self._compute_score()
        metadata = {
            "task_id": self._task.task_id,
            "goal": self._task.goal,
            "folding_threshold": self._folding_threshold,
            "invalid_action": invalid_action,
            "score": score["score"],
            "score_components": score,
            "reward_breakdown": {
                "energy_term": 0.0,
                "progress_term": 0.0,
                "stability_term": 0.0,
                "collision_penalty": 0.0,
                "invalid_action": 0.0,
            },
        }
        return ProteinObservation(
            coordinates=self._coordinates.round(4).tolist(),
            torsion_angles=self._normalize_angles(self._torsion_angles).round(4).tolist(),
            contact_map=self._contact_map.astype(int).tolist(),
            energy=float(round(self._energy, 6)),
            step_count=self._state.step_count,
            hydrophobic_contacts=int(self._hydrophobic_contacts),
            collisions=int(self._collisions),
            done=done,
            reward=float(round(reward, 6)),
            metadata=metadata,
        )

    def _is_done(self) -> bool:
        """Check episode termination conditions based on task-specific goals."""
        # 1. Global termination: Max steps reached
        if self._state.step_count >= self._task.max_steps:
            return True

        # 2. Task-specific goal evaluation
        if self._task.task_id == "task_1":
            # Goal: Reduce energy by 30% (folding_ratio=0.70)
            # Logic: Has the energy dropped below 70% of the starting energy?
            # We use a threshold calculated during reset: self._folding_threshold
            return self._energy <= self._folding_threshold

        elif self._task.task_id == "task_2":
            # Goal: Form hydrophobic core (folding_ratio=0.62)
            # Logic: Has the ratio of hydrophobic contacts reached the target?
            contact_ratio = self._hydrophobic_contacts / max(self._max_hydrophobic_contacts, 1)
            return contact_ratio >= self._task.folding_ratio

       # TASK 3: No early termination. 
        # The agent should keep optimizing until max_steps is reached to find the absolute minimum.
        if self._task.task_id == "task_3":
            return False

        return False

    def _require_index(self, index: int | None, upper_bound: int) -> int:
        """Validate a residue index."""
        if index is None:
            raise ValueError("residue_index is required")
        if index < 0 or index >= upper_bound:
            raise ValueError("residue_index out of range")
        return index

    def _require_segment(
        self,
        start: int | None,
        end: int | None,
        length: int,
    ) -> tuple[int, int]:
        """Validate a segment range."""
        if start is None or end is None:
            raise ValueError("segment_start and segment_end are required")
        if start < 0 or end >= length or start >= end:
            raise ValueError("invalid segment range")
        return start, end

    def _normalize_angles(self, torsions: np.ndarray) -> np.ndarray:
        """Wrap torsion angles into [-180, 180]."""
        return ((torsions + 180.0) % 360.0) - 180.0

    def _is_initialized(self) -> bool:
        """Check whether the environment has a live episode state."""
        return bool(self._coordinates.shape[0] and self._torsion_angles.shape[0])

    def _estimate_max_hydrophobic_contacts(self, length: int) -> int:
        """Estimate a reasonable upper bound for normalization."""
        hydrophobic_count = int(np.sum(self._build_hydrophobic_mask(length)))
        return max(1, hydrophobic_count * (hydrophobic_count - 1) // 2)


# Backward-compatible alias for the scaffolded environment class name.
MyEnvironment = ProteinFoldingEnvironment
