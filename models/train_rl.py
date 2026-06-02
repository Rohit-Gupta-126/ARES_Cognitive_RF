"""
models/train_rl.py
────────────────────────────────────────────────────────────
A.R.E.S. Stage 5 — DQN Training Arena.

Runs the DQN agent through NUM_EPISODES episodes of simulated
drone RF evasion, printing live progress and saving the best
checkpoint to models/dqn_brain.pth.

Usage (from project root)
--------------------------
    python -m models.train_rl
    python -m models.train_rl --episodes 3000 --max-steps 500
    python -m models.train_rl --hop-tax -2.0 --top-k 5
    python -m models.train_rl --resume models/dqn_brain.pth

Output
------
    models/dqn_brain.pth         — best checkpoint (highest mean reward)
    data/rl_training_log.csv     — per-episode metrics
    models/dqn_training.png      — reward + PDR training curves
"""

import sys
import os
import argparse
import csv
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless backend for server/subprocess use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Allow running as `python -m models.train_rl` from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.rl_environment import RLEnvironment
from models.dqn_agent import DQNAgent


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def _plot_training(log_path: str, out_path: str) -> None:
    """Generate a dark-themed dual-axis training curve and save to out_path."""
    episodes, rewards, pdrs, epsilons = [], [], [], []

    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["mean_reward"]))
            pdrs.append(float(row["pdr_pct"]))
            epsilons.append(float(row["epsilon"]))

    if not episodes:
        return

    # Moving average (window = 50)
    window = 50
    def ma(arr):
        if len(arr) < window:
            return arr
        return np.convolve(arr, np.ones(window) / window, mode="valid")

    ma_ep  = episodes[window - 1:]
    ma_rew = ma(rewards)
    ma_pdr = ma(pdrs)

    dark = "#0a0f1e"
    cyan = "#22d3ee"
    violet = "#a78bfa"
    gold = "#fbbf24"
    grey = "#475569"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), facecolor=dark)
    fig.suptitle(
        "A.R.E.S. Stage 5 — DQN Training Convergence",
        color="#f1f5f9", fontsize=15, fontweight="bold", y=0.98
    )

    for ax in (ax1, ax2):
        ax.set_facecolor(dark)
        ax.tick_params(colors=grey, labelsize=8)
        ax.spines["bottom"].set_color(grey)
        ax.spines["left"].set_color(grey)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, color=grey, alpha=0.15, linestyle="--")

    # ── Panel 1: Episode reward ─────────────────────────────────────────
    ax1.plot(episodes, rewards, color=cyan, alpha=0.2, linewidth=0.6, label="Raw reward")
    if len(ma_ep) == len(ma_rew):
        ax1.plot(ma_ep, ma_rew, color=cyan, linewidth=2.0, label=f"MA-{window}")
    ax1.set_ylabel("Episode Reward", color=cyan, fontsize=9)
    ax1.tick_params(axis="y", colors=cyan)
    ax1.set_xlabel("Episode", color=grey, fontsize=8)
    ax1.legend(loc="lower right", fontsize=7, facecolor=dark, labelcolor="#f1f5f9")

    # ε on twin axis
    ax1b = ax1.twinx()
    ax1b.set_facecolor(dark)
    ax1b.plot(episodes, epsilons, color=gold, linewidth=1.0, linestyle=":", alpha=0.7, label="ε (exploration)")
    ax1b.set_ylabel("Epsilon (ε)", color=gold, fontsize=9)
    ax1b.tick_params(axis="y", colors=gold, labelsize=8)
    ax1b.spines["right"].set_color(gold)
    ax1b.spines["top"].set_visible(False)
    ax1b.legend(loc="upper right", fontsize=7, facecolor=dark, labelcolor="#f1f5f9")

    # ── Panel 2: PDR ───────────────────────────────────────────────────
    ax2.plot(episodes, pdrs, color=violet, alpha=0.2, linewidth=0.6, label="Raw PDR")
    if len(ma_ep) == len(ma_pdr):
        ax2.plot(ma_ep, ma_pdr, color=violet, linewidth=2.0, label=f"MA-{window}")
    ax2.axhline(90, color="#34d399", linewidth=1.0, linestyle="--", alpha=0.8, label="90% target")
    ax2.set_ylabel("PDR (%)", color=violet, fontsize=9)
    ax2.tick_params(axis="y", colors=violet)
    ax2.set_xlabel("Episode", color=grey, fontsize=8)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.set_ylim(0, 105)
    ax2.legend(loc="lower right", fontsize=7, facecolor=dark, labelcolor="#f1f5f9")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=dark)
    plt.close()
    print(f"[TRAIN] Training curve saved -> {out_path}")


# ─── Main training arena ──────────────────────────────────────────────────────

def train(args) -> None:
    # ── Device banner ────────────────────────────────────────────────────
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "None (CPU mode)"
    )
    print("\n" + "=" * 60)
    print("  A.R.E.S. Stage 5 — DQN Training Arena")
    print("=" * 60)
    print(f"  Device        : {device_str.upper()} — {gpu_name}")
    print(f"  Episodes      : {args.episodes}")
    print(f"  Max steps/ep  : {args.max_steps}")
    print(f"  Reward shaping: success=+{args.reward_success}, "
          f"collision={args.reward_collision}, hop-tax={args.hop_tax}")
    print(f"  Top-K (ZEK)   : {args.top_k}")
    print("=" * 60 + "\n")

    # ── Environment + Agent ───────────────────────────────────────────────
    env = RLEnvironment(
        max_steps=args.max_steps,
        cycle_interval=args.cycle_interval,
        reward_success=args.reward_success,
        reward_collision=args.reward_collision,
        reward_hop_tax=args.hop_tax,
    )

    agent = DQNAgent(
        state_dim=env.state_dim,
        num_actions=env.action_dim,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        target_update_freq=args.target_update_freq,
    )

    if args.resume and os.path.exists(args.resume):
        print(f"[TRAIN] Resuming from checkpoint: {args.resume}")
        agent.load(args.resume, inference_only=False)

    # ── Output paths ──────────────────────────────────────────────────────
    os.makedirs("data",   exist_ok=True)
    os.makedirs("models", exist_ok=True)
    checkpoint_path = args.output
    log_path        = args.log
    plot_path       = args.plot

    # Prepare CSV log
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward", "mean_reward", "pdr_pct",
                         "successes", "collisions", "epsilon", "loss"])

    # ── Training loop ─────────────────────────────────────────────────────
    best_mean_reward = -float("inf")
    recent_rewards   = []  # rolling window for best-model tracking
    start_time       = time.time()

    print(f"{'Episode':>8}  {'Reward':>10}  {'PDR':>7}  {'Eps':>7}  {'Loss':>10}  {'Best':>10}")
    print("-" * 62)

    for episode in range(1, args.episodes + 1):
        state    = env.reset()
        ep_reward   = 0.0
        ep_successes = 0
        ep_losses   = []

        for _ in range(args.max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                ep_losses.append(loss)

            state      = next_state
            ep_reward += reward
            if info["success"]:
                ep_successes += 1

            if done:
                break

        ep_collisions = args.max_steps - ep_successes
        pdr = (ep_successes / args.max_steps) * 100.0
        mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

        # Rolling best-model tracking (window = 100 episodes)
        recent_rewards.append(ep_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
        mean_recent = float(np.mean(recent_rewards))

        if mean_recent > best_mean_reward and episode >= 50:
            best_mean_reward = mean_recent
            agent.save(checkpoint_path)
            best_tag = f"* {mean_recent:.1f}"
        else:
            best_tag = ""

        # CSV log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, f"{ep_reward:.2f}", f"{mean_recent:.2f}",
                f"{pdr:.2f}", ep_successes, ep_collisions,
                f"{agent.epsilon:.4f}", f"{mean_loss:.5f}",
            ])

        # Console print every 50 episodes
        if episode % 50 == 0 or episode == 1:
            elapsed = _format_duration(time.time() - start_time)
            print(
                f"{episode:>8}  {ep_reward:>10.1f}  {pdr:>6.1f}%  "
                f"{agent.epsilon:>7.4f}  {mean_loss:>10.5f}  {best_tag:<10}  [{elapsed}]"
            )

    # ── Post-training ─────────────────────────────────────────────────────
    total_time = _format_duration(time.time() - start_time)
    print("\n" + "=" * 62)
    print(f"  Training complete in {total_time}")
    print(f"  Best mean reward (window-100): {best_mean_reward:.2f}")
    print(f"  Checkpoint saved  -> {checkpoint_path}")
    print("=" * 62)

    # Generate training curve plot
    _plot_training(log_path, plot_path)

    print(f"\n[TRAIN] To run the HIL loop with this DQN brain:")
    print(f"  python run.py --dqn-model {checkpoint_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A.R.E.S. Stage 5 — DQN Training Arena",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Episode config
    parser.add_argument("--episodes",       type=int,   default=2000,
                        help="Total training episodes")
    parser.add_argument("--max-steps",      type=int,   default=500,
                        help="Steps per episode (MDP horizon)")
    parser.add_argument("--cycle-interval", type=int,   default=50,
                        help="Jammer rotation interval within an episode")

    # Reward shaping
    parser.add_argument("--reward-success",  type=float, default=10.0,
                        help="Reward for a successful packet delivery (+R)")
    parser.add_argument("--reward-collision", type=float, default=-100.0,
                        help="Penalty for a jamming collision (-R)")
    parser.add_argument("--hop-tax",         type=float, default=-1.0,
                        help="Penalty for an unnecessary channel hop")

    # Agent hyperparameters
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--gamma",           type=float, default=0.99)
    parser.add_argument("--epsilon-start",   type=float, default=1.0)
    parser.add_argument("--epsilon-end",     type=float, default=0.05)
    parser.add_argument("--epsilon-decay",   type=int,   default=10_000,
                        help="Steps over which ε decays from start to end")
    parser.add_argument("--batch-size",      type=int,   default=64)
    parser.add_argument("--buffer-capacity", type=int,   default=50_000)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--top-k",           type=int,   default=5,
                        help="Safe-Zone size (top-K Q-value channels fed to ZEK)")

    # I/O paths
    parser.add_argument("--output", type=str, default="models/dqn_brain.pth",
                        help="Checkpoint save path")
    parser.add_argument("--log",    type=str, default="data/rl_training_log.csv",
                        help="Per-episode CSV log path")
    parser.add_argument("--plot",   type=str, default="models/dqn_training.png",
                        help="Training curve output path")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from an existing checkpoint")

    args = parser.parse_args()
    train(args)
