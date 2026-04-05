# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the protein folding OpenEnv environment."""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ProteinAction(Action):
    """Structural edit applied to the simplified protein chain."""
    # ADD THIS
    model_config = {
        "extra": "ignore"
    }

    action_type: str = Field(..., description="Type of structural move to apply")
    residue_index: int | None = Field(
        default=None,
        description="Residue index used by local moves such as rotate_phi/rotate_psi",
    )
    segment_start: int | None = Field(
        default=None,
        description="Inclusive segment start index for segment-based moves",
    )
    segment_end: int | None = Field(
        default=None,
        description="Inclusive segment end index for segment-based moves",
    )
    angle_delta: float | None = Field(
        default=None,
        description="Rotation amount in degrees for angle-based moves",
    )


class ProteinObservation(Observation):
    """Observable state of the protein folding environment."""
    # ADD THIS
    model_config = {
        "extra": "ignore"
    }

    coordinates: list[list[float]] = Field(
        default_factory=list,
        description="Residue coordinates as 3D vectors",
    )
    torsion_angles: list[list[float]] = Field(
        default_factory=list,
        description="Per-residue torsion angles [phi, psi] in degrees",
    )
    contact_map: list[list[int]] = Field(
        default_factory=list,
        description="Binary non-local contact map between residues",
    )
    energy: float = Field(default=0.0, description="Current total energy")
    step_count: int = Field(default=0, description="Current episode step count")
    hydrophobic_contacts: int = Field(
        default=0,
        description="Number of hydrophobic residue contacts",
    )
    collisions: int = Field(default=0, description="Number of steric collisions")
