"""Baseline inference script for the protein folding OpenEnv environment.

MANDATORY
- Before running, define the following environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- This script is named `inference.py` and is placed in the project root.
- All LLM calls are made through the OpenAI client.
"""

from __future__ import annotations

import copy
import json
import os
import re
import textwrap
from typing import Any
import sys
from dotenv import load_dotenv
load_dotenv() # Load your OpenAI key from .env

# ADD THIS: Ensures the script can find the 'xero' folder
sys.path.append(os.path.dirname(__file__))

from openai import OpenAI
import asyncio  # Required for Docker lifecycle
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
import numpy as np

try:
    from models import ProteinAction, ProteinObservation
    from server.xero_environment import ProteinFoldingEnvironment
    from test import build_action_candidates, format_action
except ImportError:
    from xero.models import ProteinAction, ProteinObservation
    from xero.server.xero_environment import ProteinFoldingEnvironment
    from xero.test import build_action_candidates, format_action


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
#API_BASE_URL = os.getenv("API_BASE_URL")  
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
#API_KEY = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")
TASK_ID = os.getenv("TASK_ID", "task_2")
EPISODE_SEED = int(os.getenv("EPISODE_SEED", "7"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "3"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "250"))
SHORTLIST_SIZE = int(os.getenv("SHORTLIST_SIZE", "8"))
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "protein-env:latest")

ACTION_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
class ProteinFoldingEnvClient(EnvClient[ProteinAction, ProteinObservation, Any]):
    """Bridge that communicates with the Docker container via HTTP."""
    
    def _step_payload(self, action: ProteinAction) -> dict:
        # EnvClient automatically wraps this in {"action": ...}
        # exclude_none=True prevents sending null fields that might trigger validation errors
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict) -> StepResult:
        # 1. Extract the actual observation data from the nested key
        # If 'observation' isn't a key, it falls back to the payload itself
        obs_data = payload.get("observation", payload)
        if not isinstance(obs_data, dict):
            obs_data = {}

        top_level_reward = payload.get("reward")
        top_level_done = payload.get("done")

        # Some OpenEnv servers return reward/done top-level only. Mirror them into
        # observation fields when missing so downstream logic is consistent.
        if "reward" not in obs_data and top_level_reward is not None:
            obs_data["reward"] = top_level_reward
        if "done" not in obs_data and top_level_done is not None:
            obs_data["done"] = top_level_done
        
        # 2. Reconstruct the Observation model
        # We use .get() for reward and done as they are part of the StepResult, 
        # not necessarily the Observation model itself.
        observation = ProteinObservation(**obs_data)
        
        parsed_reward = (
            top_level_reward
            if top_level_reward is not None
            else getattr(observation, "reward", 0.0)
        )
        parsed_done = (
            top_level_done
            if top_level_done is not None
            else getattr(observation, "done", False)
        )

        return StepResult(
            observation=observation,
            reward=float(parsed_reward or 0.0),
            done=bool(parsed_done),
        )

    def _parse_state(self, payload: dict) -> Any:
        return payload
SYSTEM_PROMPT = textwrap.dedent(
    """
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
    - Use only one of the candidate actions provided in the user message.
    - Return valid JSON only.
    - Do not add markdown fences.
    - Do not add explanations.
    """
).strip()


def summarize_observation(observation: ProteinObservation) -> str:
    """Create a readable environment summary for the language model."""
    score_components = observation.metadata.get("score_components", {})
    return textwrap.dedent(
        f"""
        energy: {observation.energy:.3f}
        step_count: {observation.step_count}
        hydrophobic_contacts: {observation.hydrophobic_contacts}
        collisions: {observation.collisions}
        normalized_score: {float(observation.metadata.get('score', 0.0)):.3f}
        energy_reduction_ratio: {float(score_components.get('energy_reduction_ratio', 0.0)):.3f}
        hydrophobic_contact_ratio: {float(score_components.get('hydrophobic_contact_ratio', 0.0)):.3f}
        stability_score: {float(score_components.get('stability_score', 0.0)):.3f}
        first_5_torsions: {observation.torsion_angles[:5]}
        """
    ).strip()


def estimate_action_quality(observation: ProteinObservation, task_id: str) -> float:
    """Rank candidate actions based on the specific goal of the current task."""
    score = float(observation.metadata.get("score", 0.0))
    reward = float(observation.reward or 0.0)
    
    # Task 1: Focus purely on Energy reduction
    if task_id == "task_1":
        return (reward * 10.0) - (observation.energy * 2.0) - (observation.collisions * 20.0)
    
    # Task 2: Focus on Hydrophobic Contacts
    elif task_id == "task_2":
        return (observation.hydrophobic_contacts * 50.0) + (reward * 5.0) - (observation.collisions * 30.0)
    
    # Task 3: Focus on deep optimization (Stability is key to avoid getting stuck)
    else: 
        # Collision penalty is highest here because one bad move ruins a long trajectory
        return (score * 100.0) - (observation.energy * 1.0) - (observation.collisions * 100.0)


def estimate_score_from_observation(
    observation: ProteinObservation,
    initial_energy: float,
) -> float:
    """Reconstruct normalized score when metadata is missing from API payloads."""
    length = max(1, len(observation.coordinates))
    hydrophobic_count = (length + 1) // 2
    max_hydrophobic_contacts = max(1, hydrophobic_count * (hydrophobic_count - 1) // 2)

    denominator = max(abs(initial_energy), 1e-6)
    energy_reduction_ratio = float(
        np.clip((initial_energy - observation.energy) / denominator, 0.0, 1.0)
    )
    hydrophobic_contact_ratio = float(
        np.clip(observation.hydrophobic_contacts / max_hydrophobic_contacts, 0.0, 1.0)
    )
    stability_score = 1.0 if observation.collisions == 0 else float(1.0 / (1 + observation.collisions))

    return float(
        np.clip(
            (energy_reduction_ratio + hydrophobic_contact_ratio + stability_score) / 3.0,
            0.0,
            1.0,
        )
    )


def shortlist_candidates(
    current_obs: ProteinObservation,
    candidates: list[ProteinAction],
    shortlist_size: int,
    task_id: str, # Add this
) -> list[tuple[ProteinAction, ProteinObservation, float]]:
    """Evaluate all legal actions once and keep the strongest immediate moves."""
     # 1. Create a local simulator (physics engine)
    simulator = ProteinFoldingEnvironment()
    simulator.reset(task_id=task_id) # Set the right length
    
    # 2. Sync the local simulator to the Docker environment's current state
    simulator._torsion_angles = np.array(current_obs.torsion_angles)
    simulator._coordinates = np.array(current_obs.coordinates)
    simulator._update_metrics()
    ranked: list[tuple[ProteinAction, ProteinObservation, float]] = []
    for action in candidates:
        env_copy = copy.deepcopy(simulator)
        try:
            obs = env_copy.step(action)
            ranked.append((action, obs, estimate_action_quality(obs, task_id)))
        except:
            continue

    ranked.sort(key=lambda item: item[2], reverse=True)
    return ranked[: max(1, shortlist_size)]





def build_user_prompt(
    observation: ProteinObservation,
    candidates: list[tuple[ProteinAction, ProteinObservation, float]],
    history: list[str],
    task_id: str,
) -> str:
    """Build the user prompt sent to the model."""
    TASK_GOALS = {
    "task_1": "Your goal is to reach a 30 percent reduction in energy as quickly as possible.",
    "task_2": "Your goal is to maximize hydrophobic contacts to form a core. Focus on moving hydrophobic residues together.",
    "task_3": "This is a long-horizon optimization. Maintain stability (0 collisions) and reduce energy to the absolute minimum over the full episode."
}
    if history:
        history_text = "\n".join(history[-5:])
    else:
        history_text = "None"
    goal_statement = TASK_GOALS.get(task_id, "Lower energy and improve packing.")
    candidate_lines = []
    for index, (action, next_obs, quality) in enumerate(candidates, start=1):
        candidate_lines.append(
            textwrap.dedent(
                f"""
                {index}. {format_action(action)}
                   estimated_next_energy: {next_obs.energy:.3f}
                   estimated_reward: {float(next_obs.reward or 0.0):.3f}
                   estimated_score: {float(next_obs.metadata.get('score', 0.0)):.3f}
                   estimated_collisions: {next_obs.collisions}
                   estimated_contacts: {next_obs.hydrophobic_contacts}
                   heuristic_quality: {quality:.3f}
                """
            ).strip()
        )

    return textwrap.dedent(
        f"""
        Task: {task_id}
        Mission: {goal_statement}
        Objective: lower energy, reduce collisions, and improve hydrophobic packing.

        Current environment summary:
        {summarize_observation(observation)}

        {f"CRITICAL: You currently have {observation.collisions} collisions. Fix these immediately!" if observation.collisions > 0 else ""}

        Recent action history:
        {history_text}

        Candidate actions:
        {chr(10).join(candidate_lines)}

        Choose exactly one candidate and return only the JSON object for that action.
        """
    ).strip()


def action_to_payload(action: ProteinAction) -> dict[str, Any]:
    """Convert an action model to a plain JSON dictionary."""
    return {
        "action_type": action.action_type,
        "residue_index": action.residue_index,
        "segment_start": action.segment_start,
        "segment_end": action.segment_end,
        "angle_delta": action.angle_delta,
    }


def parse_action_response(
    response_text: str,
    candidate_actions: list[ProteinAction],
) -> ProteinAction:
    """Parse the model response and fall back safely if needed."""
    if not response_text:
        return candidate_actions[0]

    json_match = ACTION_JSON_RE.search(response_text)
    if not json_match:
        return candidate_actions[0]

    try:
        payload = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return candidate_actions[0]

    for action in candidate_actions:
        if action_to_payload(action) == payload:
            return action

    return candidate_actions[0]


def ensure_required_env() -> None:
    """Check the required model configuration."""
    missing = [
        name
        for name, value in (
            ("API_BASE_URL", API_BASE_URL),
            ("MODEL_NAME", MODEL_NAME),
            ("HF_TOKEN", API_KEY),
        )
        if not value
    ]
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {missing_text}")


async def main() -> None:
    """Run one inference episode with LLM-selected structural actions."""
    ensure_required_env()
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await ProteinFoldingEnvClient.from_docker_image(LOCAL_IMAGE_NAME)
    tasks_to_evaluate = ["task_1", "task_2", "task_3"]
    try:
        for current_task in tasks_to_evaluate:
            print(f"\n\n{'='*60}")
            print(f"EVALUATING: {current_task.upper()}")
            print(f"{'='*60}")

            # 1. Initialize Environment for the specific task
            
            result = await env.reset(seed=EPISODE_SEED, task_id=current_task)
            observation = result.observation
            episode_done = bool(result.done if result.done is not None else getattr(observation, "done", False))
            initial_energy = float(observation.energy)
            print(f"[START] task={current_task} env=protein_folding model={MODEL_NAME}", flush=True)

            

            # History tracks the trajectory to help the LLM reason
            history: list[str] = []
            rewards: list[float] = []
            
            # Determine how many steps to allow based on task complexity
            # Task 3 runs for the full duration (50+ steps) as requested
            loop_limit = MAX_STEPS if current_task != "task_3" else max(MAX_STEPS, 15)
        
        

            for step in range(1, loop_limit + 1):
                if episode_done:
                    print("Environment signalled done. Stopping early.")
                    break
                candidates = build_action_candidates(len(observation.coordinates))
                shortlisted = shortlist_candidates(observation, candidates, SHORTLIST_SIZE, current_task)
                candidate_actions = [item[0] for item in shortlisted]
                # 3. Build Prompt (including history so LLM learns from steps)
                user_prompt = build_user_prompt(observation, shortlisted, history,current_task)

                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=TEMPERATURE,
                        max_completion_tokens=MAX_TOKENS,
            
                    )
                    response_text = completion.choices[0].message.content or ""
                except Exception as exc:  # noqa: BLE001
                    print(f"Model request failed ({exc}). Falling back to strongest heuristic action.")
                    response_text = ""

                chosen_action = parse_action_response(response_text, candidate_actions)
                result = await env.step(chosen_action)
                observation = result.observation
                episode_done = bool(result.done if result.done is not None else getattr(observation, "done", False))

                reward = float(
                    result.reward
                    if result.reward is not None
                    else getattr(observation, "reward", 0.0)
                )
                rewards.append(reward)
                action_desc = format_action(chosen_action).replace(" ", "_")
                done_val = str(episode_done).lower()
                energy_val = float(getattr(observation, "energy", 0.0))
                print(
                    f"[STEP] step={step} action={action_desc} reward={reward:.4f} energy={energy_val:.2f} done={done_val} error=null",
                    flush=True,
                )
                history.append(f"Step {step}: {format_action(chosen_action)} -> reward {reward:.2f}")

                

            
        

            # Final Score calculation
            metadata_score = float(getattr(observation, "metadata", {}).get("score", 0.0) or 0.0)
            final_score = (
                metadata_score
                if metadata_score > 0.0
                else estimate_score_from_observation(observation, initial_energy)
            )
            success = str(final_score >= 0.7).lower() # Example threshold
            
            # [END] line is MANDATORY
            # Format: [END] success=<bool> steps=<n> score=<0.000> rewards=<r1,r2,rn>
            rewards_str = ",".join([f"{r:.4f}" for r in rewards])
            print(f"[END] success={success} steps={len(rewards)} score={final_score:.3f} rewards={rewards_str}", flush=True)
    finally:

        await env.close()


if __name__ == "__main__":
    asyncio.run(main())
