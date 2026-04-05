---
title: Protein Folding RL Environment
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
tags:
- openenv
- reinforcement-learning
- bioinformatics
---

# Protein Folding Optimization with OpenEnv

An agentic RL environment for learning how structural moves can reduce protein energy, designed for the Meta OpenEnv Hackathon.

## Project Overview

This project models protein folding as a sequential decision-making problem. An agent (LLM or RL policy) learns to reshape a protein chain into lower-energy, more stable conformations by applying structural transformations.

### The Problem
Proteins are linear chains that must fold into specific 3D shapes to function. This environment turns this into a tractable task:
- **State**: 3D coordinates, torsion angles, and contact maps.
- **Actions**: Structural edits (rotations, flips, pivots).
- **Goal**: Minimize energy while maximizing hydrophobic packing and avoiding steric collisions.

---

## Environment Design (OpenEnv Spec)

This environment is fully compliant with the OpenEnv specification, featuring typed Pydantic models and a multi-task curriculum.

### Tasks
Three difficulty levels are defined in `openenv.yaml`:
- **Task 1 (Easy)**: Length 8. Goal: 30% energy reduction.
- **Task 2 (Medium)**: Length 15. Goal: Form a dense hydrophobic core (>85% contact ratio).
- **Task 3 (Hard)**: Length 20. Goal: Deep optimization (Minimum energy + 0 collisions).

### Observation Space
- `coordinates`: 3D vectors for each residue.
- `torsion_angles`: `[phi, psi]` angles in degrees.
- `energy`: Scalar total energy (Hydrophobic + Steric + Bond + Angle).
- `hydrophobic_contacts`: Count of favorable non-local interactions.
- `collisions`: Count of steric clashes.

### Action Space
- `rotate_phi` / `rotate_psi`: Local residue rotations.
- `pivot_rotation`: Rotating the entire chain tail from a point.
- `segment_flip` / `crankshaft_move`: Coordinated multi-residue moves.
- `end_move_forward` / `end_move_backward`: Terminal residue adjustments.

---

## The Reward System (Normalized)

To ensure stable learning, the reward is strictly bounded between **0.0 and 1.0** using a weighted normalization:

$$Reward = (0.4 \times EnergyTerm) + (0.35 \times ProgressTerm) + (0.25 \times StabilityTerm)$$

- **Energy Term**: Normalized improvement in total system energy.
- **Progress Term**: Ratio of achieved hydrophobic contacts vs. theoretical maximum.
- **Stability Term**: Penalty-based score that drops if steric collisions are detected.

---

## Technical Architecture

- **Server**: FastAPI/Uvicorn-based OpenEnv server.
- **Inference**: LLM-driven baseline using Llama-3.1-70B via Hugging Face/OpenAI routers.
- **Heuristics**: A technical shortlisting layer that pre-ranks candidate moves before presenting them to the LLM.

## Setup and Usage

### Local Development
1. Install dependencies: `pip install -r requirements.txt`
2. Start the server: `uvicorn server.app:app --host 0.0.0.0 --port 8000`
3. Run local validation: `python inference.py`

### Deployment
This environment is designed to be deployed as a containerized Hugging Face Space.
```bash
openenv push