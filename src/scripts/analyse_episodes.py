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
from safe_jepa.nn.mlp import MarginBarrier
from train_world_model import resolve_dims, episode_ids, safety_margin

RESULTS_PATH = ROOT / "src/save_data/experiments"

FEAS_TOL = 1e-3 # planned risk more than this above the threshold counts as over it
LATE_WINDOW = 30 # steps before the end of an episode that count as "late"
EARLY_OVER = 0.3 # a share of over-threshold plans before the late window above this counts as "throughout"
TAU = 0.0 # the margin barrier is met at risk <= 0
MARGIN_BANDS = ((-np.inf, 0.0), (0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, np.inf)) # real next margins in check 2
K_ACTIONS = 64 # sampled actions for the safest-action risk
N_DATA = 200_000 # dataset rows scored in check 2
N_BANK = 100_000 # dataset rows searched for the nearest neighbour in check 3
DATA_CHUNK = 8192 # rows per network call on the dataset
EPISODE_CHUNK = 256 # rows per network call on an episode

CONTROLLER_COLOURS = {"SCP": "tab:blue", "MPPI": "tab:orange", "CEM": "tab:green"}
VERDICT_COLOURS = {"no violation": "tab:green", "no warning": "tab:purple", "warned late": "tab:orange",
                   "over the threshold throughout": "tab:red"}
SWEEP_LABELS = {"eps": "margin buffer eps", "J": "sampled sequences per iteration"}

# Loading saved episodes
def load_episodes(results_dir, controller):
    files = sorted(results_dir.glob(f"{controller}_seed_*.npz"), key=lambda p: int(re.search(r"seed_(\d+)", p.name).group(1)))
    return [(int(re.search(r"seed_(\d+)", p.name).group(1)), dict(np.load(p))) for p in files]

def return_to_go(rewards, gamma):
    # Discounted return actually earned from each step to the end of the episode
    g, acc = np.zeros(len(rewards)), 0.0
    for t in range(len(rewards) - 1, -1, -1):
        acc = rewards[t] + gamma * acc
        g[t] = acc
    return g

def sweep_value(tag):
    # "e1/p_0.1" -> ("p", 0.1)
    prefix, value = tag.rsplit("/", 1)[-1].rsplit("_", 1)
    return prefix, float(value)

# Check 1: when did the plans stop meeting the threshold, relative to the end of the episode?

def check_threshold(ep, tau):
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

# Checks 2 and 3 use the world model

def load_world_model(cfg):
    # Rebuild the world model from an experiment's merged config and restore its checkpoint (as sweep_runner does)
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

# Jitted network calls. They only ever see chunks of DATA_CHUNK or EPISODE_CHUNK rows (the last chunk is padded), so
# each compiles once per chunk size rather than once per episode length

@nnx.jit
def _encode(encoder, obs):
    return encoder(obs)

@nnx.jit
def _risk(barrier, z, u):
    return barrier(z, u).squeeze(-1)

@nnx.jit
def _margin_next(dynamics, margin_fn, z, u):
    return margin_fn(dynamics(z, u)).squeeze(-1)

@nnx.jit
def _safest(barrier, z, u):
    # Lowest risk over the K sampled actions of each state, u: (B * K, d_u)
    q = barrier(jnp.repeat(z, K_ACTIONS, axis=0), u).reshape(z.shape[0], K_ACTIONS)
    return q.min(axis=-1)

@jax.jit
def _nearest(bank_z, bank_sq, z):
    # Index of the nearest bank latent (Euclidean) for each row of z
    d = jnp.sum(z ** 2, axis=-1)[:, None] + bank_sq[None] - 2.0 * z @ bank_z.T
    return jnp.argmin(d, axis=-1)

def in_chunks(fn, arrays, size):
    # Apply fn to fixed-size chunks of the arrays, padding the last chunk by repeating its last row
    n = len(arrays[0])
    out = []
    for i in range(0, n, size):
        part = [a[i:i + size] for a in arrays]
        k = len(part[0])
        if k < size:
            part = [np.concatenate([p, np.repeat(p[-1:], size - k, axis=0)]) for p in part]
        out.append(np.asarray(fn(*[jnp.asarray(p) for p in part]))[:k])
    return np.concatenate(out)

class Safety:
    def __init__(self, world_model, action_bounds, d_u):
        self.encoder, self.dynamics, self.margin_fn = world_model.encoder, world_model.dynamics, world_model.margin_fn
        self.bounds, self.d_u = action_bounds, d_u
        self.rng = np.random.default_rng(0)
        self.barrier = None

    def encode(self, obs, size):
            return in_chunks(lambda o: _encode(self.encoder, o), [obs], size)
    
    def risk(self, z, u, size):
        return in_chunks(lambda zz, uu: _risk(self.safety_critic, zz, uu), [z, u], size)

    def safest_risk(self, z, size):
        # Lowest risk over K uniformly sampled actions: the floor the critic allows at each state
        def fn(zz):
            u = self.rng.uniform(self.bounds[0], self.bounds[1], (zz.shape[0] * K_ACTIONS, self.d_u))
            return _safest(self.safety_critic, zz, jnp.asarray(u))
        return in_chunks(fn, [z], size)

    def set_bank(self, obs, actions):
        self.bank_z = jnp.asarray(self.encode(obs, DATA_CHUNK))
        self.bank_sq = jnp.sum(self.bank_z ** 2, axis=-1)
        self.bank_u = actions

    def nearest_action(self, z, size):
        nn = in_chunks(lambda zz: _nearest(self.bank_z, self.bank_sq, zz), [z], size)
        return self.bank_u[nn]

    def set_barrier(self, eps, alpha):
        # The barrier of one experiment config, as the planners used it
        self.barrier = MarginBarrier(self.dynamics, self.margin_fn, eps, alpha)

    def margin_next(self, z, u, size):
        # The margin head's prediction for the next state, as the barrier uses it
        return in_chunks(lambda zz, uu: _margin_next(self.dynamics, self.margin_fn, zz, uu), [z, u], size)

def check_dataset(safety, data, sources):
    # Check 2: the critic's risk on the dataset, at the data's actions and at the safest sampled action
    n = len(data["rewards"])
    idx = np.random.default_rng(0).choice(n, min(N_DATA, n), replace=False)
    episode = episode_ids(data)
    last_row = np.r_[np.flatnonzero(np.diff(episode)), n - 1] # last row of each trajectory
    steps_to_end = last_row[episode] - np.arange(n)
    ends_in_violation = (data["costs"][last_row] > 0)[episode]
    far = (~ends_in_violation | (steps_to_end >= FAR))[idx]
    pred = safety.margin_next(z, data["actions"][idx], DATA_CHUNK)
    real = safety_margin(data["next_observations"][idx], sources)

    z = critic.encode(data["observations"][idx], DATA_CHUNK)
    q_data = critic.risk(z, data["actions"][idx], DATA_CHUNK)
    q_min = critic.safest_risk(z, DATA_CHUNK)
    print(f"Check 2 - the critic on {len(idx)} dataset rows ({far.mean():.0%} at least {FAR} steps from a violation): "
          f"share with risk over each threshold")
    print(f"  {'threshold':>9s}  {'dataset action':>14s}  {'... far from violations':>23s}  {'safest action':>13s}  "
          f"{'... far from violations':>23s}")
    for tau in taus:
        print(f"  {tau:9g}  {np.mean(q_data > tau):14.1%}  {np.mean(q_data[far] > tau):23.1%}  "
              f"{np.mean(q_min > tau):13.1%}  {np.mean(q_min[far] > tau):23.1%}")
    return q_data, q_min, far

def check_visited(critic, ep):
    # Check 3: at each visited state, risk at the executed action, at the safest sampled action and at the action of the
    # nearest dataset state
    z = critic.encode(ep["obs"], EPISODE_CHUNK)
    return {
        "exec": critic.risk(z, ep["actions"], EPISODE_CHUNK),
        "safest": critic.safest_risk(z, EPISODE_CHUNK),
        "nearest": critic.risk(z, critic.nearest_action(z, EPISODE_CHUNK), EPISODE_CHUNK)
    }

# Figures

def plot_episode(path, controller, seed, ep, tau, gamma, over, result, visited=None):
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
    ax.fill_between(steps, 0, 1, where=over, step="mid", color="red", alpha=0.08, lw=0, transform=ax.get_xaxis_transform())
    if result["violated"]:
        ax.axvline(steps[-1], color="red", lw=1.5, label="violation")
    ax.set_ylabel("risk")
    ax.set_title(f"{controller} seed {seed}: {result['verdict']} | {result['steps']} steps, return {result['return']:.2f}, "
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

def plot_dataset(path, dataset, taus):
    # Check 2: the critic's risk on the dataset, with every threshold of the experiment
    q_data, q_min, far = dataset
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(min(0.0, q_data.min(), q_min.min()), max(1.0, q_data.max(), q_min.max()), 51) # output not clipped to [0, 1]
    ax.hist(q_data, bins=bins, histtype="step", density=True, label="dataset action")
    ax.hist(q_min, bins=bins, histtype="step", density=True, label="safest sampled action")
    ax.hist(q_min[far], bins=bins, histtype="step", density=True, label=f"safest, >= {FAR} steps from a violation")
    for tau in taus:
        ax.axvline(tau, color="k", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("critic risk")
    ax.set_title("Check 2: the critic's risk on the dataset (dashed: thresholds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

def boxes(ax, data, controllers, ylabel, log=False):
    parts = ax.boxplot([data[c] for c in controllers], showfliers=False, patch_artist=True)
    for patch, c in zip(parts["boxes"], controllers):
        patch.set_facecolor(CONTROLLER_COLOURS.get(c, "grey"))
        patch.set_alpha(0.5)
    ax.set_xticks(range(1, len(controllers) + 1), controllers)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")

def needs_model(ax):
    ax.text(0.5, 0.5, "needs the world model\n(set MODEL_CHECKS = True)", ha="center", va="center")
    ax.axis("off")

def plot_comparison(path, title, point):
    # The controllers at one sweep point, side by side
    controllers = list(point)
    tau = point[controllers[0]]["tau"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    # Episodes without a violation, against step
    ax = axes[0, 0]
    for c in controllers:
        steps = np.array([r["steps"] for r in point[c]["rows"]])
        violated = np.array([r["violated"] for r in point[c]["rows"]])
        t = np.arange(steps.max() + 1)
        survival = np.mean(~violated[None, :] | (steps[None, :] > t[:, None]), axis=1)
        ax.step(t, survival, where="post", color=CONTROLLER_COLOURS.get(c), label=f"{c} ({violated.mean():.0%} violated)")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("step")
    ax.set_ylabel("episodes without a violation")
    ax.legend(fontsize=8)
    ax.set_title("Survival")

    # Episode return
    ax = axes[0, 1]
    returns = {c: [r["return"] for r in point[c]["rows"]] for c in controllers}
    boxes(ax, returns, controllers, "episode return")
    for i, c in enumerate(controllers):
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(returns[c]))
        ax.scatter(i + 1 + jitter, returns[c], s=8, color="k", alpha=0.5)
    ax.set_title("Return")

    # Plan time per step
    ax = axes[0, 2]
    boxes(ax, {c: np.concatenate([ep["plan_times"] for ep in point[c]["episodes"]]) for c in controllers}, controllers,
          "plan time per step (s)", log=True)
    ax.set_title("Plan time")

    # Check 1 verdicts
    ax = axes[0, 3]
    bottom = np.zeros(len(controllers))
    for verdict, colour in VERDICT_COLOURS.items():
        counts = np.array([sum(r["verdict"] == verdict for r in point[c]["rows"]) for c in controllers])
        ax.bar(controllers, counts, bottom=bottom, color=colour, label=verdict)
        bottom += counts
    ax.set_ylabel("episodes")
    ax.legend(fontsize=7)
    ax.set_title("Check 1: how the episodes ended")

    # How far the plans went over the threshold
    ax = axes[1, 0]
    boxes(ax, {c: np.concatenate([ep["info_pred_risk"].max(axis=1) for ep in point[c]["episodes"]]) - tau for c in controllers},
          controllers, "planned risk (plan maximum) minus threshold")
    ax.axhline(0.0, color="k", ls="--", lw=1)
    ax.set_title("Planned risk against the threshold")

    # Planned against earned return
    ax = axes[1, 1]
    boxes(ax, {c: np.concatenate([ep["info_pred_return"] - return_to_go(ep["rewards"], point[c]["gamma"])
                                  for ep in point[c]["episodes"]]) for c in controllers},
          controllers, "planned minus earned return")
    ax.axhline(0.0, color="k", ls="--", lw=1)
    ax.set_title("Model optimism")

    # Check 3: was any safe action available, and how risky were the executed actions against dataset-like ones
    if point[controllers[0]]["visited"] is not None:
        ax = axes[1, 2]
        share = [np.mean(np.concatenate([v["safest"] for v in point[c]["visited"]]) > tau) for c in controllers]
        ax.bar(controllers, share, color=[CONTROLLER_COLOURS.get(c, "grey") for c in controllers], alpha=0.7)
        ax.set_ylabel("visited steps with no safe sampled action")
        ax.set_title("Check 3: was a safe action available?")

        ax = axes[1, 3]
        boxes(ax, {c: np.concatenate([v["exec"] - v["nearest"] for v in point[c]["visited"]]) for c in controllers},
              controllers, "risk at executed minus nearest dataset action")
        ax.axhline(0.0, color="k", ls="--", lw=1)
        ax.set_title("Check 3: executed against dataset-like actions")
    else:
        needs_model(axes[1, 2])
        needs_model(axes[1, 3])

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

def point_summary(entry):
    # One row of the sweep summary: one controller at one sweep point
    rows, episodes, tau, gamma = entry["rows"], entry["episodes"], entry["tau"], entry["gamma"]
    returns = np.array([r["return"] for r in rows])
    over = np.concatenate([ep["info_pred_risk"].max(axis=1) > tau + FEAS_TOL for ep in episodes])
    summary = {
        "episodes": len(rows),
        "violation_rate": float(np.mean([r["violated"] for r in rows])),
        "mean_return": float(returns.mean()),
        "sem_return": float(returns.std() / np.sqrt(len(returns))),
        "median_plan_time": float(np.median(np.concatenate([ep["plan_times"] for ep in episodes]))),
        "plans_over_threshold": float(over.mean()),
        "critic_saw_no_risk": float(np.mean([r["verdict"] == "critic saw no risk" for r in rows])),
        "warned_late": float(np.mean([r["verdict"] == "warned late" for r in rows])),
        "over_throughout": float(np.mean([r["verdict"] == "over the threshold throughout" for r in rows])),
        "planned_minus_earned": float(np.median(np.concatenate(
            [ep["info_pred_return"] - return_to_go(ep["rewards"], gamma) for ep in episodes])))
    }
    if entry["visited"] is not None:
        summary["no_safe_action"] = float(np.mean(np.concatenate([v["safest"] for v in entry["visited"]]) > tau))
        summary["exec_minus_nearest"] = float(np.median(np.concatenate([v["exec"] - v["nearest"] for v in entry["visited"]])))
    return summary

def plot_sweep(path, title, sweep):
    # Each controller across the sweep points (e.g. the thresholds of E1)
    prefix = sweep_value(sweep[0][0])[0]
    panels = [("violation_rate", "episodes with a violation"), ("mean_return", "mean episode return (± s.e.)"),
              ("median_plan_time", "median plan time (s)"), ("plans_over_threshold", "plans over the threshold"),
              ("critic_saw_no_risk", "violated, last plan within the threshold"),
              ("over_throughout", "violated, plans over the threshold throughout"),
              ("planned_minus_earned", "median planned minus earned return"),
              ("no_safe_action", "visited steps with no safe sampled action"),
              ("exec_minus_nearest", "median executed minus dataset-like risk")]
    controllers = list(dict.fromkeys(c for _, point in sweep for c in point))
    summaries = {(tag, c): point_summary(point[c]) for tag, point in sweep for c in point}
    values = [sweep_value(tag)[1] for tag, _ in sweep]
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    for ax, (key, label) in zip(axes.flat, panels):
        plotted = False
        for c in controllers:
            pts = [(sweep_value(tag)[1], summaries[(tag, c)]) for tag, point in sweep if c in point]
            pts = [(v, s) for v, s in pts if key in s]
            if not pts:
                continue
            x = [v for v, _ in pts]
            y = [s[key] for _, s in pts]
            ax.plot(x, y, "o-", color=CONTROLLER_COLOURS.get(c), label=c)
            if key == "mean_return":
                err = [s["sem_return"] for _, s in pts]
                ax.fill_between(x, np.subtract(y, err), np.add(y, err), color=CONTROLLER_COLOURS.get(c), alpha=0.15)
            plotted = True
        if not plotted:
            needs_model(ax)
            continue
        if max(values) / max(min(values), 1e-12) > 10:
            ax.set_xscale("log")
        ax.set_xlabel(SWEEP_LABELS.get(prefix, prefix))
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

# Running the analysis

def analyse_point(results_dir, title, controllers, critic=None, episode_figures=True):
    # Check 1 (and check 3) for every episode of every controller at one sweep point, their figures, and the comparison
    out_dir = results_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    point = {}
    for controller in controllers:
        cfg_path = results_dir / f"{controller}_config.yaml"
        if not cfg_path.exists():
            print(f"  {controller}: no results")
            continue
        cfg = yaml.safe_load(open(cfg_path))
        tau, gamma = TAU, cfg["discount"]
        episodes = load_episodes(results_dir, controller)
        if not episodes or "info_pred_risk" not in episodes[0][1]:
            print(f"  {controller}: no planner logs (only runs made with sweep_runner have them)")
            continue

        rows, visited_all = [], []
        for seed, ep in episodes:
            result, over = check_threshold(ep, tau)
            result = {"seed": seed, **result}
            visited = None
            if safety is not None:
                visited = check_visited(critic, ep)
                result.update({
                    "steps_no_safe_action": float(np.mean(visited["safest"] > tau)),
                    "median_exec_minus_nearest": float(np.median(visited["exec"] - visited["nearest"])),
                    "max_abs_exec_vs_planned": float(np.max(np.abs(visited["exec"] - ep["info_pred_risk"][:, 0])))
                })
                visited_all.append(visited)
            rows.append(result)
            if episode_figures:
                plot_episode(out_dir / f"{controller}_seed_{seed}.png", controller, seed, ep, tau, gamma, over, result, visited)

        with open(out_dir / f"{controller}_checks.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

        point[controller] = {"tau": tau, "gamma": gamma, "rows": rows, "episodes": [ep for _, ep in episodes],
                             "visited": visited_all if critic is not None else None}
        verdicts = [r["verdict"] for r in rows]
        line = (f"  {controller:5s} {len(rows)} episodes, {np.mean([r['violated'] for r in rows]):.0%} violated, "
                f"mean return {np.mean([r['return'] for r in rows]):.2f} | "
                + ", ".join(f"{v}: {verdicts.count(v)}" for v in VERDICT_COLOURS if v in verdicts))
        if critic is not None:
            # The executed-step risk recomputed here should match what the planner logged
            line += f" | recomputed vs logged risk {max(r['max_abs_exec_vs_planned'] for r in rows):.1e}"
        print(line)

    if point:
        tau = point[next(iter(point))]["tau"]
        plot_comparison(out_dir / "comparison.png", f"{title} (threshold {tau})", point)
    return point

def analyse_experiment(experiment, tags, controllers, model_checks=True, episode_figures=True):
    exp_dir = RESULTS_PATH / experiment
    configs = [exp_dir / tag / f"{c}_config.yaml" for tag in tags for c in controllers]
    configs = [p for p in configs if p.exists()]
    if not configs:
        print(f"{experiment}: no results under {exp_dir}")
        return
    out_dir = exp_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    print(f"\n{experiment}")

    # Checks 2 and 3 need the world model, which every config of the experiment shares
    critic = None
    if model_checks:
        cfg = yaml.safe_load(open(configs[0]))
        world_model, data = load_world_model(cfg)
        critic = Critic(world_model, cfg["action_bounds"], cfg["d_u"])
        taus = sorted({yaml.safe_load(open(p))["safety_threshold"] for p in configs})
        dataset = check_dataset(critic, data, taus)
        plot_dataset(out_dir / "check2_critic_on_dataset.png", dataset, taus)
        rows = np.random.default_rng(1).choice(len(data["rewards"]), min(N_BANK, len(data["rewards"])), replace=False)
        critic.set_bank(data["observations"][rows], data["actions"][rows])

    # Every sweep point, then each sweep (e.g. e1, e2) across its points
    sweeps = {}
    for tag in tags:
        if not (exp_dir / tag).exists():
            print(f"{tag}: not found")
            continue
        print(f"{tag}:")
        point = analyse_point(exp_dir / tag, f"{Path(experiment).name}: {tag}", controllers, critic, episode_figures)
        if point:
            sweeps.setdefault(tag.split("/")[0], []).append((tag, point))

    summary_rows = []
    for name, sweep in sweeps.items():
        sweep.sort(key=lambda tp: sweep_value(tp[0])[1])
        plot_sweep(out_dir / f"sweep_{name}.png", f"{Path(experiment).name}: {name}", sweep)
        for tag, point in sweep:
            for c, entry in point.items():
                summary_rows.append({"tag": tag, "controller": c, **point_summary(entry)})
    if summary_rows:
        with open(out_dir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in summary_rows for k in r)))
            w.writeheader()
            w.writerows(summary_rows)
    print(f"Written to {out_dir}, and to analysis/ in each sweep point")

if __name__ == "__main__":
    # Experiment folders under src/save_data/experiments, and the sweep points saved in each (as run_e1.py saves them)
    EXPERIMENTS = {
        "T5_pointgoal1_c_1_5_seed_0": [f"e1/p_{p}" for p in [0.5, 0.3, 0.2, 0.1, 0.05, 0.02]],
    }
    CONTROLLERS = ["SCP", "MPPI", "CEM"]
    MODEL_CHECKS = True # checks 2 and 3 load the world model
    EPISODE_FIGURES = True # one figure per episode

    for experiment, tags in EXPERIMENTS.items():
        analyse_experiment(experiment, tags, CONTROLLERS, MODEL_CHECKS, EPISODE_FIGURES)