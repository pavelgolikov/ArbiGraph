"""Train Qwen3.5-9B with TRL and ArbiGraph's existing terminal reward."""

import argparse
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "rlvr"), str(REPO_ROOT / "source"), str(REPO_ROOT / "eval")]

from arbigraph_rlvr import DatasetLoader, terminal_reward
from mcp_server import calculator as existing_calculator
# Qualified so this module imports the same way as a script and as full_train.train.
from full_train.prepare import TRAIN_TOPOLOGIES


# Exact system prompt from the phase-3 pilot.
SYSTEM_PROMPT = (
    "You are a helpful reasoning assistant with access to a calculator tool. Think step by step. "
    "You MUST use the calculator tool EVERY TIME you need to perform arithmetic operations. "
    "When several arithmetic expressions are useful, submit them together in one calculator call. "
)


def calculator(expressions: list[str]) -> str:
    """Evaluate arithmetic expressions in order using + - * / // % **, sqrt, abs, round, and pow.

    Use pow(base, exponent, modulus) for modular exponentiation, e.g. pow(3, 998244351, 998244353).
    Use pow(base, -1, modulus) for a modular inverse when one exists.
    Integer powers followed by % are also evaluated efficiently. Batch independent expressions in one call.

    Args:
        expressions: Mathematical expressions to evaluate in order, e.g. ['17 * 43 + 5', '100 // 7', 'round(sqrt(2), 3)'].

    Returns:
        JSON containing the results in the same order.
    """
    return existing_calculator(expressions)


def load_training_rows(data_dir, condition):
    rows, verifiers = [], {}
    for topology in TRAIN_TOPOLOGIES:
        loader = DatasetLoader(data_dir / "train" / topology / f"{condition}.json", topology, condition)
        for episode in loader.iter_episodes():
            observation = episode.observation
            rows.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": observation.prompt},
                ],
                "episode_id": observation.episode_id,
            })
            verifiers[observation.episode_id] = episode.verifier_state
    if len(rows) != 384 or len(verifiers) != len(rows):
        raise ValueError("Expected 384 distinct training prompts")
    return rows, verifiers


def make_reward(verifiers):
    def target_reward(completions, episode_id, **kwargs):
        rewards = []
        for messages, key in zip(completions, episode_id, strict=True):
            # Transformers parses reasoning separately from the final assistant content.
            text = "\n".join(
                message[field]
                for message in messages if message["role"] == "assistant"
                for field in ("reasoning_content", "content") if message.get(field)
            )
            rewards.append(terminal_reward(text, verifiers[key]))
        return rewards
    return target_reward


def check_training_logs(logs):
    for name in ("loss", "grad_norm"):
        if name in logs and not math.isfinite(logs[name]):
            raise FloatingPointError(f"Non-finite {name}: {logs[name]}")
    # TRL reports tool execution exceptions in this metric instead of re-raising them.
    if logs.get("tools/failure_frequency", 0) > 0:
        raise RuntimeError("TRL reported a failed tool call")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "generate/generated/rlvr_full_train")
    parser.add_argument("--condition", choices=("full", "edge_cut"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume_from_checkpoint:
        parser.error("Output directory is not empty; use a new directory or an explicit resume checkpoint")

    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    class FailOnTrainingError(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            check_training_logs(logs or {})

    config = GRPOConfig(
        output_dir=str(args.output_dir),
        seed=args.seed,
        data_seed=args.seed,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=3,
        num_generations=4,
        num_iterations=1,
        loss_type="dapo",
        scale_rewards="group",
        beta=0.0,
        learning_rate=1e-6,
        warmup_steps=10,
        lr_scheduler_type="constant_with_warmup",
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        model_init_kwargs={"dtype": "bfloat16", "attn_implementation": "sdpa"},
        use_vllm=False,
        generation_kwargs={"use_cache": True},
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        max_completion_length=40960,
        max_tool_calling_iterations=32,
        mask_truncated_completions=True,
        chat_template_kwargs={"reasoning_effort": "xhigh", "preserve_thinking": True},
        remove_unused_columns=False,
        logging_steps=1,
        logging_nan_inf_filter=False,
        report_to="none",
        save_steps=32,
        save_total_limit=1,
        deepspeed={
            "bf16": {"enabled": True},
            "train_micro_batch_size_per_gpu": "auto",
            "train_batch_size": "auto",
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": "auto",
            "zero_optimization": {
                # Keep weights local for independent multi-turn generation on each rank.
                "stage": 2,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
            },
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": "auto", "betas": "auto", "eps": "auto", "weight_decay": "auto"},
            },
        },
    )
    if config.world_size != 4:
        raise ValueError("This configuration requires exactly four training processes")
    rows, verifiers = load_training_rows(args.data_dir, args.condition)
    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=Dataset.from_list(rows),
        reward_funcs=make_reward(verifiers),
        tools=[calculator],
        callbacks=[FailOnTrainingError()],
    )
    trainer.model.config.get_text_config().use_cache = False
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.save_model(str(args.output_dir))
    trainer.save_state()


if __name__ == "__main__":
    main()
