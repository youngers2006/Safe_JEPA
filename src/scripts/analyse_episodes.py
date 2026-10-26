import argparse
import csv
import re
import sys
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import orbax.checkpoint as ocp
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

from safe_jepa.world_model.world_model import WorldModel
from train_world_model import resolve_dims, episode_ids

FEAS_TOL = 1e-3 # planned risk more than this above the threshold counts as over it
LATE_WINDOW = 30 # steps before the end of an episode that count as "late"
EARLY_OVER = 0.3 # a share of over-threshold plans before the late window above this counts as "throughout"
VERDICT_COLOURS = {"no violation": "tab:green", "critic saw no risk": "tab:purple", "warned late": "tab:orange",
                   "over the threshold throughout": "tab:red"}
FAR = 50 # dataset states at least this many steps before a violation count as far from one
K_ACTIONS = 64 # sampled actions for the safest-action risk
N_DATA = 200_000 # dataset rows scored in check 2
N_BANK = 100_000 # dataset rows searched for the nearest neighbour in check 3

def load_episodes(results, controller):
    files = sorted(results.glob(f"{controller}_seed_*.npz"), key=lambda p: int(re.search(r"seed_(\d+)", p.name).group(1)))
    return [(int(re.search(r"seed_(\d+)", p.name).group(1)), dict(np.load(p))) for p in files]

def return_to_go(rewards, gamma):
    # Discounted return actually earned from each step to the end of the episode
    g, acc = np.zeros(len(rewards)), 0.0
    for t in range(len(rewards) - 1, -1, -1):
        acc = rewards[t] + gamma * acc
        g[t] = acc
    return g

def check_threshold(ep, tau):
    # Check 1: when did the plans stop meeting the threshold, relative to the end of the episode?
    steps, violated = len(ep["rewards"]), bool(ep["violations"][-1])
    over = ep["info_pred_risk"].max(axis=1) > tau + FEAS_TOL
    final_run = 0
    for o in over[::-1]:
        if not o:
            break
        final_run += 1
    early = over[:max(steps - LATE_WINDOW, 0)]
    early_frac = float(early.mean()) if early.size else float("nan")
    if not violated:
        verdict = "no violation"
    elif final_run == 0:
        verdict = "critic saw no risk"
    elif early.size and early_frac > EARLY_OVER:
        verdict = "over the threshold throughout"
    else:
        verdict = "warned late"
    return {
        "steps": steps,
        "violated": violated,
        "return": float(np.sum(ep["rewards"])),
        "verdict": verdict,
        "frac_over": float(over.mean()),
        "early_frac_over": early_frac,
        "final_over_run": final_run,
        "first_over": int(np.argmax(over)) if over.any() else -1
    }, over

def load_world_model(cfg):
    # Rebuild the world model from the experiment's merged config and restore its checkpoint (as sweep_runner does)
    with h5py.File(ROOT / cfg["dataset_filename"], 'r') as f:
        data = {k: f[k][:] for k in f.keys()}
    cfg = resolve_dims(cfg, data["observations"].shape[-1], data["actions"].shape[-1])
    cfg.setdefault("rho", 0.5) # configs saved before multi-step training
    obs_dim = data["observations"].shape[-1]
    world_model = WorldModel(cfg, jnp.zeros(obs_dim), jnp.ones(obs_dim), rngs=nnx.Rngs(cfg["seed"]))
    _, target = nnx.split(world_model)
    ckpt_path = ROOT / "src/save_data" / cfg["data_dir"] / "Model/state"
    nnx.update(world_model, ocp.StandardCheckpointer().restore(ckpt_path, target))
    return world_model, data

class Critic:
    # The world model's encoder and safety critic, evaluated in chunks
    def __init__(self, world_model, action_bounds, key=jax.random.key(0)):
        self.wm, self.bounds, self.key = world_model, action_bounds, key

    def encode(self, obs, size=65536):
        return np.concatenate([np.asarray(self.wm.encoder(jnp.asarray(obs[i:i + size]))) for i in range(0, len(obs), size)])

    def risk(self, z, u, size=65536):
        return np.concatenate([np.asarray(self.wm.safety_critic(jnp.asarray(z[i:i + size]), jnp.asarray(u[i:i + size])).squeeze(-1))
                               for i in range(0, len(z), size)])

    def safest_risk(self, z, d_u, size=4096):
        # Lowest risk over K uniformly sampled actions: the floor the critic allows at each state
        out = []
        for i in range(0, len(z), size):
            zc = jnp.asarray(z[i:i + size])
            self.key, sub = jax.random.split(self.key)
            u = jax.random.uniform(sub, (zc.shape[0] * K_ACTIONS, d_u), minval=self.bounds[0], maxval=self.bounds[1])
            q = self.wm.safety_critic(jnp.repeat(zc, K_ACTIONS, axis=0), u).reshape(zc.shape[0], K_ACTIONS)
            out.append(np.asarray(q.min(axis=-1)))
        return np.concatenate(out)

def nearest(bank_z, z, size=256):
    # Index of the nearest bank latent (Euclidean) for each row of z
    bank = jnp.asarray(bank_z)
    bank_sq = jnp.sum(bank ** 2, axis=-1)
    out = []
    for i in range(0, len(z), size):
        q = jnp.asarray(z[i:i + size])
        d = jnp.sum(q ** 2, axis=-1)[:, None] + bank_sq[None] - 2.0 * q @ bank.T
        out.append(np.asarray(jnp.argmin(d, axis=-1)))
    return np.concatenate(out)

def check_dataset(critic, data, tau, d_u):
    # Check 2: the critic's risk on the dataset, at the data's actions and at the safest sampled action
    n = len(data["rewards"])
    idx = np.random.default_rng(0).choice(n, min(N_DATA, n), replace=False)
    episode = episode_ids(data)
    last_row = np.r_[np.flatnonzero(np.diff(episode)), n - 1] # last row of each trajectory
    steps_to_end = last_row[episode] - np.arange(n)
    ends_in_violation = (data["costs"][last_row] > 0)[episode]
    far = (~ends_in_violation | (steps_to_end >= FAR))[idx]

    z = critic.encode(data["observations"][idx])
    q_data = critic.risk(z, data["actions"][idx])
    q_min = critic.safest_risk(z, d_u)
    print(f"\nCheck 2 - the critic on {len(idx)} dataset rows ({far.mean():.0%} at least {FAR} steps from a violation):")
    for name, q in (("risk at the dataset action", q_data), ("risk at the safest sampled action", q_min)):
        for label, sel in (("all rows", slice(None)), (f">= {FAR} steps from a violation", far)):
            print(f"  {name:34s} {label:28s} median {np.median(q[sel]):.4f}  90% {np.percentile(q[sel], 90):.4f}  "
                  f"over tau {np.mean(q[sel] > tau):.1%}")
    return q_data, q_min, far

def check_visited(critic, ep, bank_z, bank_u, d_u):
    # Check 3: at each visited state, risk at the executed action, at the safest sampled action and at the action of the
    # nearest dataset state
    z = critic.encode(ep["obs"])
    nn = nearest(bank_z, z)
    return {
        "exec": critic.risk(z, ep["actions"]),
        "safest": critic.safest_risk(z, d_u),
        "nearest": critic.risk(z, bank_u[nn]),
        "nn_dist": np.linalg.norm(z - bank_z[nn], axis=-1)
    }

def plot_episode(path, seed, ep, tau, gamma, over, result, visited=None):
    steps = np.arange(len(ep["rewards"]))
    is_scp = "info_iterations" in ep
    fig, axes = plt.subplots(4 if is_scp else 3, 1, figsize=(12, 11 if is_scp else 8.5), sharex=True)

    # Planned risk against the threshold, with the critic's floor and the nearest dataset action's risk
    ax = axes[0]
    ax.plot(steps, ep["info_pred_risk"][:, 0], lw=1, label="planned risk, executed step")
    ax.plot(steps, ep["info_pred_risk"].max(axis=1), lw=1, label="planned risk, plan maximum")
    if visited is not None:
        ax.plot(steps, visited["safest"], lw=1, label="safest sampled action")
        ax.plot(steps, visited["nearest"], lw=1, alpha=0.7, label="nearest dataset action")
    ax.axhline(tau, color="k", ls="--", lw=1, label="threshold")
    for t in np.flatnonzero(over):
        ax.axvspan(t - 0.5, t + 0.5, color="red", alpha=0.08, lw=0)
    if result["violated"]:
        ax.axvline(steps[-1], color="red", lw=1.5, label="violation")
    ax.set_ylabel("risk")
    ax.set_title(f"seed {seed}: {result['verdict']} | {result['steps']} steps, return {result['return']:.2f}, "
                 f"plans over the threshold {result['frac_over']:.0%} (shaded)")
    ax.legend(fontsize=7, loc="upper left", ncol=3)

    # Reward the planner predicted for the executed step against the real one
    ax = axes[1]
    ax.plot(steps, ep["rewards"], lw=1, label="real reward")
    ax.plot(steps, ep["info_pred_rewards"][:, 0], lw=1, label="predicted reward, executed step")
    ax.set_ylabel("reward")
    ax.legend(fontsize=7, loc="upper left")

    # Planned return against the discounted return actually earned from each step
    ax = axes[2]
    ax.plot(steps, ep["info_pred_return"], lw=1, label="planned return (rewards over the horizon + terminal value)")
    ax.plot(steps, return_to_go(ep["rewards"], gamma), lw=1, label="discounted return earned from this step")
    ax.set_ylabel("return")
    ax.legend(fontsize=7, loc="upper left")

    # SCP iterations, coloured by stop reason, and plan time
    if is_scp:
        ax = axes[3]
        reasons = ep["info_stop_reason"].astype(str)
        for reason, colour in (("converged", "tab:green"), ("converged_infeasible", "tab:orange"), ("max_iters", "tab:red")):
            sel = reasons == reason
            ax.scatter(steps[sel], ep["info_iterations"][sel], s=6, color=colour, label=reason)
        ax.set_ylabel("SCP iterations")
        ax.legend(fontsize=7, loc="upper left", ncol=3)
        twin = ax.twinx()
        twin.plot(steps, ep["plan_times"], color="grey", lw=0.8, alpha=0.7)
        twin.set_ylabel("plan time (s)", color="grey")
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

def plot_summary(path, results, episodes, tau, gamma, dataset=None, visited=None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Check 1: share of plans over the threshold before the late window (bars, coloured by verdict) and overall (dots)
    ax = axes[0, 0]
    x = np.arange(len(results))
    ax.bar(x, [r["early_frac_over"] for r in results], 0.7, color=[VERDICT_COLOURS[r["verdict"]] for r in results])
    ax.scatter(x, [r["frac_over"] for r in results], color="k", s=12, zorder=3, label="whole episode")
    for verdict, colour in VERDICT_COLOURS.items():
        ax.bar([0], [0], color=colour, label=verdict)
    ax.set_xticks(x, [str(r["seed"]) for r in results], fontsize=7)
    ax.set_xlabel("seed")
    ax.set_ylabel(f"plans over the threshold before the last {LATE_WINDOW} steps")
    ax.set_title("Check 1: when the plans stopped meeting the threshold")
    ax.legend(fontsize=7)

    # Planned return against the return actually earned
    ax = axes[0, 1]
    pred = np.concatenate([ep["info_pred_return"] for _, ep in episodes])
    real = np.concatenate([return_to_go(ep["rewards"], gamma) for _, ep in episodes])
    ax.scatter(real, pred, s=2, alpha=0.3)
    lim = [min(real.min(), pred.min()), max(real.max(), pred.max())]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("discounted return earned from the step")
    ax.set_ylabel("planned return")
    ax.set_title("Planned against earned return (dashed: equal)")

    # Check 2: the critic's risk on the dataset
    ax = axes[1, 0]
    if dataset is not None:
        q_data, q_min, far = dataset
        # The critic's output is not clipped to [0, 1], so the bins cover whatever it predicts
        bins = np.linspace(min(0.0, q_data.min(), q_min.min()), max(1.0, q_data.max(), q_min.max()), 51)
        ax.hist(q_data, bins=bins, histtype="step", density=True, label="dataset action")
        ax.hist(q_min, bins=bins, histtype="step", density=True, label="safest sampled action")
        ax.hist(q_min[far], bins=bins, histtype="step", density=True, label=f"safest, >= {FAR} steps from a violation")
        ax.axvline(tau, color="k", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("critic risk")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "needs the world model\n(run without --no_model)", ha="center", va="center")
        ax.axis("off")
    ax.set_title("Check 2: the critic's risk on the dataset (dashed: threshold)")

    # Check 3: risk at the executed action against the nearest dataset action, coloured by steps to the end
    ax = axes[1, 1]
    if visited is not None:
        exec_q = np.concatenate([v["exec"] for v in visited])
        near_q = np.concatenate([v["nearest"] for v in visited])
        to_end = np.concatenate([np.arange(len(v["exec"]))[::-1] for v in visited])
        sc = ax.scatter(near_q, exec_q, c=np.minimum(to_end, 100), s=3, cmap="viridis_r", alpha=0.6)
        fig.colorbar(sc, ax=ax, label="steps to the end of the episode (capped at 100)")
        lim = [0.0, max(exec_q.max(), near_q.max())]
        ax.plot(lim, lim, "k--", lw=1)
        ax.axhline(tau, color="red", ls=":", lw=1)
        ax.axvline(tau, color="red", ls=":", lw=1)
        ax.set_xlabel("risk at the nearest dataset state's action")
        ax.set_ylabel("risk at the executed action")
    else:
        ax.text(0.5, 0.5, "needs the world model\n(run without --no_model)", ha="center", va="center")
        ax.axis("off")
    ax.set_title("Check 3: executed against dataset-like action (dotted: threshold)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="folder with {controller}_seed_*.npz and {controller}_config.yaml")
    parser.add_argument("--controller", default="SCP")
    parser.add_argument("--no_model", action="store_true", help="skip checks 2 and 3, which load the world model")
    args = parser.parse_args()

    results_dir = Path(args.results)
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    cfg = yaml.safe_load(open(results_dir / f"{args.controller}_config.yaml"))
    tau, gamma, d_u = cfg["safety_threshold"], cfg["discount"], cfg["d_u"]
    episodes = load_episodes(results_dir, args.controller)
    out_dir = results_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    print(f"{results_dir}: {len(episodes)} {args.controller} episodes, threshold {tau}, world model {cfg['data_dir']}")

    # Checks 2 and 3 need the world model
    dataset, critic = None, None
    if not args.no_model:
        world_model, data = load_world_model(cfg)
        critic = Critic(world_model, cfg["action_bounds"])
        dataset = check_dataset(critic, data, tau, d_u)
        bank = np.random.default_rng(1).choice(len(data["rewards"]), min(N_BANK, len(data["rewards"])), replace=False)
        bank_z, bank_u = critic.encode(data["observations"][bank]), data["actions"][bank]

    # Check 1 for every episode, check 3 at its visited states, and its figure
    rows, visited_all = [], []
    print(f"\nCheck 1 - plans over the threshold (planned risk > {tau} + {FEAS_TOL}){' and check 3' if critic else ''}:")
    for seed, ep in episodes:
        result, over = check_threshold(ep, tau)
        result = {"seed": seed, **result}
        visited = None
        if critic is not None:
            visited = check_visited(critic, ep, bank_z, bank_u, d_u)
            visited_all.append(visited)
            result.update({
                "steps_no_safe_action": float(np.mean(visited["safest"] > tau)),
                "median_exec_minus_nearest": float(np.median(visited["exec"] - visited["nearest"])),
                "steps_exec_over_nearest_under": float(np.mean((visited["exec"] > tau) & (visited["nearest"] <= tau))),
                "max_abs_exec_vs_planned": float(np.max(np.abs(visited["exec"] - ep["info_pred_risk"][:, 0])))
            })
        rows.append(result)
        plot_episode(out_dir / f"{args.controller}_seed_{seed}.png", seed, ep, tau, gamma, over, result, visited)

        line = (f"  seed {seed:3d}: {result['verdict']:30s} {result['steps']:5d} steps | over the threshold "
                f"{result['frac_over']:.0%} overall, {result['early_frac_over']:.0%} before the last {LATE_WINDOW}, "
                f"final run {result['final_over_run']} steps")
        if critic is not None:
            line += (f" | no safe sampled action on {result['steps_no_safe_action']:.0%} of steps, executed minus "
                     f"nearest-dataset risk median {result['median_exec_minus_nearest']:+.4f}")
        print(line)

    verdicts = [r["verdict"] for r in rows]
    print("\nVerdicts: " + ", ".join(f"{v}: {verdicts.count(v)}" for v in dict.fromkeys(verdicts)))
    if critic is not None:
        # The executed-action risk recomputed here should match what the planner logged
        print(f"Largest difference between the recomputed and the logged executed-step risk: "
              f"{max(r['max_abs_exec_vs_planned'] for r in rows):.2e}")

    with open(out_dir / f"{args.controller}_checks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot_summary(out_dir / f"{args.controller}_summary.png", rows, episodes, tau, gamma, dataset, visited_all or None)
    print(f"\nFigures and {args.controller}_checks.csv written to {out_dir}")

if __name__ == "__main__":
    main()
