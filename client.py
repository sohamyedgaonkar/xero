# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client for the protein folding OpenEnv environment."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ProteinAction, ProteinObservation


class ProteinFoldingEnv(EnvClient[ProteinAction, ProteinObservation, State]):
    """Client wrapper for the protein folding environment."""

    def _step_payload(self, action: ProteinAction) -> Dict:
        """Serialize a protein action into the request payload."""
        return {
            "action_type": action.action_type,
            "residue_index": action.residue_index,
            "segment_start": action.segment_start,
            "segment_end": action.segment_end,
            "angle_delta": action.angle_delta,
            "metadata": action.metadata,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ProteinObservation]:
        """Parse the server step/reset payload into a typed observation."""
        obs_data = payload.get("observation", {})
        observation = ProteinObservation(
            coordinates=obs_data.get("coordinates", []),
            torsion_angles=obs_data.get("torsion_angles", []),
            contact_map=obs_data.get("contact_map", []),
            energy=obs_data.get("energy", 0.0),
            step_count=obs_data.get("step_count", 0),
            hydrophobic_contacts=obs_data.get("hydrophobic_contacts", 0),
            collisions=obs_data.get("collisions", 0),
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse the state endpoint payload."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )


# Backward-compatible alias for the scaffolded client name.
MyEnv = ProteinFoldingEnv
