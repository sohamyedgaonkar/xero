"""
Protein Folding Inference Script
=================================
Calls the protein folding environment at https://arya2004-xero.hf.space

Environment variables:
    API_BASE_URL     LLM API endpoint        (default: HF router)
    MODEL_NAME       Model identifier        (default: Llama-3.1-70B)
    API_KEY          LLM API key             (required)
    HF_TOKEN         Alternative to API_KEY
    ENV_URL          Override env base URL   (default: https://arya2004-xero.hf.space)
    TASK_ID          Task to run             (default: task_2)
    EPISODE_SEED     Episode seed            (default: 7)
    MAX_STEPS        Steps per episode       (default: 10)

Usage:
    API_KEY=xxx python inference.py
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
import textwrap
from typing import Any
# from dotenv import load_dotenv
# load_dotenv() 
import numpy as np
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Local module imports
# ---------------------------------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models import ProteinAction, ProteinObservation
from server.xero_environment import ProteinFoldingEnvironment

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-70B-Instruct")
ENV_URL      = os.environ.get("ENV_URL", "https://arya2004-xero.hf.space").rstrip("/")

TASK_ID       = os.environ.get("TASK_ID", "task_2")
EPISODE_SEED  = int(os.environ.get("EPISODE_SEED", "7"))
MAX_STEPS     = int(os.environ.get("MAX_STEPS", "10"))
TEMPERATURE   = float(os.environ.get("TEMPERATURE", "0.2"))
MAX_TOKENS    = int(os.environ.get("MAX_TOKENS", "250"))
SHORTLIST_SIZE = int(os.environ.get("SHORTLIST_SIZE", "8"))

BENCHMARK     = "protein_folding"
ACTION_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)

# ---------------------------------------------------------------------------
# Stdout logging  (mandatory OpenEnv format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# HTTP client  — mirrors EnvClient in the SQLab reference exactly
# ---------------------------------------------------------------------------

class EnvClient:
    """Thin synchronous HTTP client for the protein folding FastAPI server."""

    def __init__(self, base_url: str, timeout: int = 60):
        self.base    = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def health(self) -> dict:
        r = self.session.get(f"{self.base}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def reset(self, task_id: str, seed: int = 7) -> dict:
        r = self.session.post(
            f"{self.base}/reset",
            json={"task_id": task_id, "seed": seed},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def step(self, action: ProteinAction) -> dict:
        r = self.session.post(
            f"{self.base}/step",
            json={"action": action.model_dump(exclude_none=True)},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self.session.close()

# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------

def parse_observation(resp: dict) -> ProteinObservation:
    obs_data = resp.get("observation", resp)
    if not isinstance(obs_data, dict):
        obs_data = {}
    # Mirror top-level reward/done into obs_data when missing
    for key in ("reward", "done"):
        if key not in obs_data and key in resp:
            obs_data[key] = resp[key]
    return ProteinObservation(**obs_data)


def parse_reward(resp: dict, obs: ProteinObservation) -> float:
    top = resp.get("reward")
    return float(top if top is not None else getattr(obs, "reward", 0.0) or 0.0)


def parse_done(resp: dict, obs: ProteinObservation) -> bool:
    top = resp.get("done")
    return bool(top if top is not None else getattr(obs, "done", False))

# ---------------------------------------------------------------------------
# Action candidates
# ---------------------------------------------------------------------------

def build_action_candidates(length: int) -> list[ProteinAction]:
    candidates: list[ProteinAction] = []
    angle_choices = (-60.0, -30.0, 30.0, 60.0)

    for i in range(1, max(length - 1, 1)):
        for delta in angle_choices:
            candidates.append(ProteinAction(action_type="rotate_phi", residue_index=i, angle_delta=delta))
            candidates.append(ProteinAction(action_type="rotate_psi", residue_index=i, angle_delta=delta))

    for i in range(0, max(length - 2, 1)):
        for delta in angle_choices:
            candidates.append(ProteinAction(action_type="pivot_rotation", residue_index=i, angle_delta=delta))

    for window in (3, 4, 5):
        if window > length:
            continue
        step = 1 if length <= 12 else 2
        for start in range(0, length - window + 1, step):
            end = start + window - 1
            candidates.append(ProteinAction(action_type="segment_flip", segment_start=start, segment_end=end))
            for delta in (-45.0, 45.0):
                candidates.append(ProteinAction(action_type="crankshaft_move", segment_start=start, segment_end=end, angle_delta=delta))

    for delta in (-45.0, -20.0, 20.0, 45.0):
        candidates.append(ProteinAction(action_type="end_move_forward",  angle_delta=delta))
        candidates.append(ProteinAction(action_type="end_move_backward", angle_delta=delta))

    return candidates


def format_action(action: ProteinAction) -> str:
    return (
        f"{action.action_type}(residue={action.residue_index},"
        f"segment={action.segment_start}:{action.segment_end},"
        f"delta={action.angle_delta})"
    )

# ---------------------------------------------------------------------------
# Heuristic scoring
# ---------------------------------------------------------------------------

def estimate_action_quality(obs: ProteinObservation, task_id: str) -> float:
    score  = float(obs.metadata.get("score", 0.0))
    reward = float(obs.reward or 0.0)
    if task_id == "task_1":
        return (reward * 10.0) - (obs.energy * 2.0) - (obs.collisions * 20.0)
    elif task_id == "task_2":
        return (obs.hydrophobic_contacts * 50.0) + (reward * 5.0) - (obs.collisions * 30.0)
    else:
        return (score * 100.0) - (obs.energy * 1.0) - (obs.collisions * 100.0)


def estimate_score_from_observation(obs: ProteinObservation, initial_energy: float) -> float:
    length = max(1, len(obs.coordinates))
    hc     = (length + 1) // 2
    max_hc = max(1, hc * (hc - 1) // 2)
    denom  = max(abs(initial_energy), 1e-6)

    energy_ratio     = float(np.clip((initial_energy - obs.energy) / denom, 0.0, 1.0))
    hydrophobic_ratio = float(np.clip(obs.hydrophobic_contacts / max_hc, 0.0, 1.0))
    stability         = 1.0 if obs.collisions == 0 else 1.0 / (1 + obs.collisions)

    return float(np.clip((energy_ratio + hydrophobic_ratio + stability) / 3.0, 0.0, 1.0))


def shortlist_candidates(
    current_obs: ProteinObservation,
    candidates: list[ProteinAction],
    shortlist_size: int,
    task_id: str,
) -> list[tuple[ProteinAction, ProteinObservation, float]]:
    sim = ProteinFoldingEnvironment()
    sim.reset(task_id=task_id)
    sim._torsion_angles = np.array(current_obs.torsion_angles)
    sim._coordinates    = np.array(current_obs.coordinates)
    sim._update_metrics()

    ranked: list[tuple[ProteinAction, ProteinObservation, float]] = []
    for action in candidates:
        try:
            obs = copy.deepcopy(sim).step(action)
            ranked.append((action, obs, estimate_action_quality(obs, task_id)))
        except Exception:
            continue

    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked[: max(1, shortlist_size)]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are controlling a simplified protein folding environment.
    Your job is to choose exactly one structural action that improves the protein conformation.

    Return exactly one JSON object with this schema:
    {
      "action_type": "rotate_phi | rotate_psi | pivot_rotation | segment_flip | crankshaft_move | end_move_forward | end_move_backward",
      "residue_index": int or null,
      "segment_start": int or null,
      "segment_end": int or null,
      "angle_delta": number or null
    }

    Rules:
    - Use only one of the candidate actions listed in the user message.
    - Return valid JSON only. No markdown fences. No explanations.
""").strip()

TASK_GOALS = {
    "task_1": "Reach a 30% reduction in energy as quickly as possible.",
    "task_2": "Maximize hydrophobic contacts. Move hydrophobic residues together to form a core.",
    "task_3": "Long-horizon optimization. Keep 0 collisions and drive energy to the absolute minimum.",
}


def summarize_observation(obs: ProteinObservation) -> str:
    sc = obs.metadata.get("score_components", {})
    return textwrap.dedent(f"""
        energy:                   {obs.energy:.3f}
        step_count:               {obs.step_count}
        hydrophobic_contacts:     {obs.hydrophobic_contacts}
        collisions:               {obs.collisions}
        normalized_score:         {float(obs.metadata.get('score', 0.0)):.3f}
        energy_reduction_ratio:   {float(sc.get('energy_reduction_ratio', 0.0)):.3f}
        hydrophobic_contact_ratio:{float(sc.get('hydrophobic_contact_ratio', 0.0)):.3f}
        stability_score:          {float(sc.get('stability_score', 0.0)):.3f}
        first_5_torsions:         {obs.torsion_angles[:5]}
    """).strip()


def build_user_prompt(
    obs: ProteinObservation,
    shortlisted: list[tuple[ProteinAction, ProteinObservation, float]],
    history: list[str],
    task_id: str,
) -> str:
    goal = TASK_GOALS.get(task_id, "Lower energy and improve packing.")
    history_text = "\n".join(history[-5:]) if history else "None"

    candidate_lines = []
    for i, (action, next_obs, quality) in enumerate(shortlisted, 1):
        candidate_lines.append(textwrap.dedent(f"""
            {i}. {format_action(action)}
               estimated_next_energy:  {next_obs.energy:.3f}
               estimated_reward:       {float(next_obs.reward or 0.0):.3f}
               estimated_score:        {float(next_obs.metadata.get('score', 0.0)):.3f}
               estimated_collisions:   {next_obs.collisions}
               estimated_contacts:     {next_obs.hydrophobic_contacts}
               heuristic_quality:      {quality:.3f}
        """).strip())

    collision_warning = (
        f"\nCRITICAL: You currently have {obs.collisions} collisions. Fix these immediately!"
        if obs.collisions > 0 else ""
    )

    return textwrap.dedent(f"""
        Task: {task_id}
        Mission: {goal}
        Objective: lower energy, reduce collisions, improve hydrophobic packing.

        Current environment summary:
        {summarize_observation(obs)}
        {collision_warning}

        Recent action history:
        {history_text}

        Candidate actions:
        {chr(10).join(candidate_lines)}

        Choose exactly one candidate and return only its JSON object.
    """).strip()

# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def action_to_payload(action: ProteinAction) -> dict:
    return {
        "action_type":    action.action_type,
        "residue_index":  action.residue_index,
        "segment_start":  action.segment_start,
        "segment_end":    action.segment_end,
        "angle_delta":    action.angle_delta,
    }


def parse_action_response(text: str, candidates: list[ProteinAction]) -> ProteinAction:
    if not text:
        return candidates[0]
    match = ACTION_JSON_RE.search(text)
    if not match:
        return candidates[0]
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return candidates[0]
    for action in candidates:
        if action_to_payload(action) == payload:
            return action
    return candidates[0]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        raise SystemExit("Set API_KEY or HF_TOKEN before running.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env    = EnvClient(ENV_URL)

    rewards: list[float] = []
    steps_taken = 0
    score   = 0.0
    success = False
    observation: ProteinObservation | None = None

    log_start(task=TASK_ID, env=BENCHMARK, model=MODEL_NAME)

    try:
        # ---- reset --------------------------------------------------------
        resp        = env.reset(task_id=TASK_ID, seed=EPISODE_SEED)
        observation = parse_observation(resp)
        done        = parse_done(resp, observation)
        initial_energy = float(observation.energy)
        history: list[str] = []

        loop_limit = MAX_STEPS if TASK_ID != "task_3" else max(MAX_STEPS, 15)

        # ---- episode loop -------------------------------------------------
        for step in range(1, loop_limit + 1):
            if done:
                break

            # 1. build & shortlist candidates locally
            candidates  = build_action_candidates(len(observation.coordinates))
            shortlisted = shortlist_candidates(observation, candidates, SHORTLIST_SIZE, TASK_ID)
            cand_actions = [item[0] for item in shortlisted]

            # 2. ask the LLM
            user_prompt = build_user_prompt(observation, shortlisted, history, TASK_ID)
            step_error: str | None = None
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc:
                step_error    = str(exc)
                response_text = ""

            # 3. parse → step the remote env
            chosen = parse_action_response(response_text, cand_actions)
            try:
                resp        = env.step(chosen)
                observation = parse_observation(resp)
                done        = parse_done(resp, observation)
                reward      = parse_reward(resp, observation)
            except Exception as exc:
                step_error = str(exc)
                reward = 0.0
                done   = False

            rewards.append(reward)
            steps_taken = step
            action_desc = format_action(chosen)

            log_step(step=step, action=action_desc, reward=reward, done=done, error=step_error)
            history.append(f"Step {step}: {action_desc} -> reward {reward:.2f}")

        # ---- score --------------------------------------------------------
        if observation is not None:
            meta_score = float((observation.metadata or {}).get("score", 0.0) or 0.0)
            score = meta_score if meta_score > 0.0 else estimate_score_from_observation(observation, initial_energy)
            score = max(0.0, min(1.0, score))
            success = score >= 0.7

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        try:
            env.close()
        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()