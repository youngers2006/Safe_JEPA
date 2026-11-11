import csv
import glob
import re
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[2]

# Reads every saved episode under RESULTS (no world model or GPU needed) and writes summary tables, paired tests and
# figures to OUT: python analyse_results.py
RESULTS = ROOT / "src/save_data/experiments" # the merged experiments folder
OUT = RESULTS / "analysis_results"
PREFIX = "M2_" # only world model folders starting with this ("" for all)
SEED_CAP = 30 # only seeds 0 to SEED_CAP - 1 are read, so every point uses the same seeds; one with fewer keeps its own
# Runs at another budget, saved in their own folder under the plain planner names, join the point they extend so they
# pair with its seeds, named with their budget from their config: a sampler's sequences ("MPPI@65536"), SCP's iteration
# cap ("SCP+MPPI@100it"). Each rule maps such a folder (its path below the world model's folder) to that point's
BUDGET_DIRS = {r"e1/eps_t_matched_(.+)": r"e1/eps_\1", # time-matched samplers beside E1's margin_eps sweep
               r"e4_extended_budget[^/]*/(.+)": r"e4/\1"} # larger budgets beside the horizon tests, e.g.
                                                          # e4_extended_budget_130k/H_30 beside e4_extended_budget/H_30

MAIN = "SCP+MPPI" # the planner compared against every other one
PLANNER_ORDER = ["SCP+MPPI", "SCP", "MPPI", "MPPI+CBF", "CEM", "CEM+CBF"]
COLOURS = {"SCP+MPPI": "tab:purple", "SCP": "tab:blue", "MPPI": "tab:orange", "MPPI+CBF": "tab:brown",
           "CEM": "tab:green", "CEM+CBF": "tab:olive"}
BOUND_COLOURS = {1.2: "tab:red", 1.5: "tab:blue", np.inf: "tab:gray"}
TAU, FEAS_TOL = 0.0, 1e-3 # a plan meets the barrier in the model when its worst step's risk is at most TAU + FEAS_TOL
ACTIVE_TOL = 0.01 # a plan that meets the barrier has it active when its worst step's risk is within this of TAU
# SCP's feas_tol (scp.yaml): SCP counts as converged when the barrier is met to this, so certified_tol certifies a
# converged step whose rollout meets the barrier to it (the logged "certified" checks the rollout with no tolerance)
SCP_FEAS_TOL = 1e-4
GOAL_THRESHOLD = 0.5 # a step whose reward exceeds this reached the goal
DISCOUNT = 0.99 # the world models' discount (wmTraining configs), in the plans' predicted returns
OPTIMISM_TAIL = 0.05 # model optimism uses the steps whose unseen return after the episode's end weighs at most this
# The agent builds a new controller at every reset, so JAX recompiles on each episode's first two steps (about 10x a
# normal step): JIT compilation, not planning, so left out of every plan-time statistic
COMPILE_STEPS = 2
POOL_MIN_EPS = 0.1 # pooled tests use every margin_eps from this up (the points with a buffer)
N_BOOT = 10_000
RNG = np.random.default_rng(0)

# Statistics

def wilson(k, n, z=1.96):
    # 95% interval for a proportion, sensible at 0 or n
    if n == 0:
        return np.nan, np.nan
    p, d = k / n, 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return max(0.0, centre - half), min(1.0, centre + half)

def boot_ci(x):
    # 95% bootstrap interval for a mean
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    means = RNG.choice(x, (N_BOOT, len(x))).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))

def binom_two_sided(k, n):
    if n == 0:
        return 1.0
    try:
        return float(stats.binomtest(k, n, 0.5).pvalue)
    except AttributeError: # scipy < 1.7
        return float(stats.binom_test(k, n, 0.5))

def cmh(tables):
    # Cochran-Mantel-Haenszel test over strata of 2x2 tables [[a, b], [c, d]] (rows: planners, columns: violated or
    # not), with continuity correction; also the Mantel-Haenszel common odds ratio
    num = var = r = s = 0.0
    for (a, b), (c, d) in tables:
        n = a + b + c + d
        if n < 2:
            continue
        num += a - (a + b) * (a + c) / n
        var += (a + b) * (c + d) * (a + c) * (b + d) / (n ** 2 * (n - 1))
        r, s = r + a * d / n, s + b * c / n
    if var == 0:
        return 1.0, np.nan
    chi = max(abs(num) - 0.5, 0.0) ** 2 / var
    return float(stats.chi2.sf(chi, 1)), (r / s if s > 0 else np.inf)

def logrank(strata):
    # Log-rank test of time to violation, summed over strata of (steps A, violated A, steps B, violated B)
    o_minus_e = var = 0.0
    for t1, e1, t2, e2 in strata:
        t, e = np.concatenate([t1, t2]), np.concatenate([e1, e2]).astype(bool)
        first = np.concatenate([np.ones(len(t1), bool), np.zeros(len(t2), bool)])
        for time in np.unique(t[e]):
            at_risk = t >= time
            n, n1 = at_risk.sum(), (at_risk & first).sum()
            d, d1 = (e & (t == time)).sum(), (e & (t == time) & first).sum()
            o_minus_e += d1 - d * n1 / n
            if n > 1:
                var += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    return float(stats.chi2.sf(o_minus_e ** 2 / var, 1)) if var > 0 else 1.0

def nan_stat(fn, x, *args):
    x = np.asarray(x, float)
    return float(fn(x[~np.isnan(x)], *args)) if np.any(~np.isnan(x)) else np.nan

def holm(ps):
    ps = np.asarray(ps, float)
    adj, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(np.argsort(ps)):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj

# Loading

def episode_summary(path):
    d = np.load(path)
    rewards, violations = d["rewards"], d["violations"].astype(bool)
    T = len(rewards)
    ep = {"steps": T, "violated": bool(violations[-1]), "return": float(rewards.sum()),
          "goals": int(np.sum(rewards > GOAL_THRESHOLD)),
          "plan_times": d["plan_times"][COMPILE_STEPS:] if "plan_times" in d.files else np.full(T, np.nan)}
    if "info_pred_return" in d.files: # model optimism: the executed plan's predicted return (its discounted predicted
        # rewards plus the value at its end, which carries the goal bonus) minus the discounted return received from
        # that step, goal bonuses included. Episodes end at a violation or the step limit while the value assumes they
        # go on, so only steps at least K before the end count, where the unseen remainder weighs at most OPTIMISM_TAIL
        received = lfilter([1.0], [1.0, -DISCOUNT], rewards[::-1])[::-1] # sum over k of DISCOUNT^k r[t + k]
        K = int(np.ceil(np.log(OPTIMISM_TAIL) / np.log(DISCOUNT)))
        gap = d["info_pred_return"].reshape(T)[:T - K + 1] - received[:T - K + 1] if T >= K else np.array([])
        ep["optimism"] = float(gap.mean()) if len(gap) else np.nan
    if "info_pred_risk" in d.files: # per step: the executed plan's worst step risk, and does it meet the barrier?
        ep["max_risk"] = d["info_pred_risk"].reshape(T, -1).max(axis=1)
        ep["latent_safe"] = ep["max_risk"] <= TAU + FEAS_TOL
    for k in ("stop_reason", "start"):
        if f"info_{k}" in d.files:
            ep[k] = d[f"info_{k}"].astype(str)
    for k in ("certified", "iterations"):
        if f"info_{k}" in d.files:
            ep[k] = d[f"info_{k}"].astype(float)
    if "stop_reason" in ep and "max_risk" in ep:
        ep["certified_tol"] = ((ep["stop_reason"] == "converged") & (ep["max_risk"] <= TAU + SCP_FEAS_TOL)).astype(float)
    if "info_filter_active" in d.files: # the one-step safety filter: did it change the action, and did that work?
        ep["filter_active"] = d["info_filter_active"].astype(bool)
        ep["filter_met"] = d["info_filter_q_after"] <= TAU + FEAS_TOL
    return ep

def tag_parts(tag):
    # The last folder names the swept value, the rest the sweep: "e1/eps_0.2" -> ("e1", "eps", 0.2),
    # "e2/eps_0.5/J_512" -> ("e2/eps_0.5", "J", 512); "timing/mppi_J_512" -> ("timing", "mppi_J_512", nan)
    sweep, _, rest = tag.rpartition("/")
    m = re.fullmatch(r"([A-Za-z]+)_(-?[\d.]+(?:e-?\d+)?)", rest)
    return (sweep, m.group(1), float(m.group(2))) if m else (sweep, rest, np.nan)

def model_info(data_dir, cfg):
    # Task, Lipschitz bound and planning horizon, from the saved config where it has them, else from the folder name
    task = re.search(r"PointGoal(\d)", str(cfg.get("dataset_filename", ""))) or re.search(r"pointgoal(\d)", data_dir)
    bound = cfg.get("DynamicsParams", {}).get("lipschitz_bound")
    if bound is None:
        m = re.search(r"_c_(inf|\d+_\d+)", data_dir)
        bound = np.inf if m is None or m.group(1) == "inf" else float(m.group(1).replace("_", "."))
    horizon = cfg.get("horizon")
    if horizon is None:
        m = re.search(r"_h(\d+)$", data_dir)
        horizon = int(m.group(1)) if m else 10
    task = int(task.group(1)) if task else 0
    bound, horizon = float(bound), int(horizon)
    return {"task": task, "bound": bound, "horizon": horizon,
            "model": f"PG{task} c{'inf' if np.isinf(bound) else f'{bound:g}'} H{horizon}"}

def check_point(point_dir, sources, configs, warnings):
    # Overlapping seed ranges for a planner (e.g. a test run's CSV beside the final one) and settings that differ
    # between the planners of one point. sources: each planner's folder and the name its files were saved under
    for folder, name in sources.values():
        ranges = []
        for f in folder.glob(f"{glob.escape(name)}_seeds_*.csv"):
            m = re.fullmatch(rf"{re.escape(name)}_seeds_(\d+)-(\d+)\.csv", f.name)
            if m:
                ranges.append((int(m.group(1)), int(m.group(2))))
        ranges.sort()
        if any(b0 <= a1 for (a0, a1), (b0, b1) in zip(ranges, ranges[1:])):
            warnings.append(f"{folder}: {name} has overlapping seed ranges {ranges}; check which run wrote the "
                            f"seed files")
    for key in ("rho_u", "barrier_alpha", "horizon", "data_dir", "margin_eps"):
        values = {p: str(c.get(key)) for p, c in configs.items() if c}
        if len(set(values.values())) > 1:
            warnings.append(f"{point_dir}: {key} differs between planners: {values}")

def point_name(folder, saved, rel):
    # The point (tag) and planner name a seed file counts under: its own folder and saved name, except in a
    # BUDGET_DIRS folder, which joins the point it extends under its name and budget
    tag = "/".join(rel[1:])
    for pattern, point in BUDGET_DIRS.items():
        m = re.fullmatch(pattern, tag)
        if m:
            cfg_file = folder / f"{saved}_config.yaml"
            cfg = yaml.safe_load(open(cfg_file)) if cfg_file.exists() else {}
            budget = cfg.get("num_seqs") if cfg.get("controller") in ("MPPI", "CEM") else \
                (f"{cfg['scp_iters']}it" if "scp_iters" in cfg else None)
            return m.expand(point), f"{saved}@{budget or 'budget'}"
    return tag, saved

def load_results(results, prefix):
    raw, sources = {}, {}
    for f in sorted(results.rglob("*_seed_*.npz")):
        m = re.fullmatch(r"(.+)_seed_(\d+)\.npz", f.name)
        rel = f.parent.relative_to(results).parts
        if m is None or len(rel) < 2 or not rel[0].startswith(prefix) or int(m.group(2)) >= SEED_CAP:
            continue
        tag, planner = point_name(f.parent, m.group(1), rel)
        sources.setdefault((rel[0], tag), {})[planner] = (f.parent, m.group(1))
        raw.setdefault((rel[0], tag), {}).setdefault(planner, {})[int(m.group(2))] = episode_summary(f)
    points, warnings = [], []
    for (data_dir, tag), planners in sorted(raw.items()):
        point_dir, src = results / data_dir / tag, sources[(data_dir, tag)]
        configs = {p: (yaml.safe_load(open(folder / f"{name}_config.yaml"))
                       if (folder / f"{name}_config.yaml").exists() else {}) for p, (folder, name) in src.items()}
        check_point(point_dir, src, configs, warnings)
        cfg = next((c for c in configs.values() if c), {})
        sweep, name, value = tag_parts(tag)
        eps = cfg.get("margin_eps", value if name == "eps" else np.nan)
        points.append({"data_dir": data_dir, "tag": tag, "sweep": sweep, "name": name, "value": value,
                       "eps": float(np.atleast_1d(eps)[0]), "planners": planners, "configs": configs,
                       **model_info(data_dir, cfg)})
    return points, warnings

# Per-point summaries

def planner_row(episodes, horizon):
    eps = list(episodes.values())
    n, k = len(eps), sum(e["violated"] for e in eps)
    ret, goals = np.array([e["return"] for e in eps]), np.array([e["goals"] for e in eps])
    pt = np.concatenate([e["plan_times"] for e in eps])
    row = {"n": n, "violations": k, "rate": k / n, "rate_lo": wilson(k, n)[0], "rate_hi": wilson(k, n)[1],
           "return": ret.mean(), "return_lo": boot_ci(ret)[0], "return_hi": boot_ci(ret)[1],
           "goals": goals.mean(), "goals_lo": boot_ci(goals)[0], "goals_hi": boot_ci(goals)[1],
           "steps": np.mean([e["steps"] for e in eps]),
           "plan_time_med": nan_stat(np.median, pt), "plan_time_mean": nan_stat(np.mean, pt),
           "plan_time_p90": nan_stat(np.percentile, pt, 90), "plan_time_p99": nan_stat(np.percentile, pt, 99)}
    # Latent safety: the share of executed plans that met the barrier in the model, and how often a violation followed
    # within one horizon of a plan that did or did not
    if all("latent_safe" in e for e in eps):
        safe = np.concatenate([e["latent_safe"] for e in eps])
        soon = np.concatenate([np.arange(e["steps"]) >= e["steps"] - horizon if e["violated"]
                               else np.zeros(e["steps"], bool) for e in eps])
        warned = [not e["latent_safe"][-horizon:].all() for e in eps if e["violated"]]
        row.update({"latent_safe": safe.mean(),
                    "viol_soon_after_safe": soon[safe].mean() if safe.any() else np.nan,
                    "viol_soon_after_unsafe": soon[~safe].mean() if (~safe).any() else np.nan,
                    "violations_warned": np.mean(warned) if warned else np.nan,
                    "steps_safe": int(safe.sum()), "steps_safe_viol_soon": int(soon[safe].sum()),
                    "steps_unsafe": int((~safe).sum()), "steps_unsafe_viol_soon": int(soon[~safe].sum())})
        ct = crash_types(eps, horizon)
        row.update({k: ct[k] for k in ("crashes_unwarned", "barrier_active", "barrier_slack_med")})
    # SCP's certificates: converged with the plan feasible (KKT point), converged with slack in use (no safe plan
    # nearby, to first order), or stopped at the iteration cap (no certificate)
    if all("stop_reason" in e for e in eps):
        stops = np.concatenate([e["stop_reason"] for e in eps])
        row.update({"converged": np.mean(stops == "converged"),
                    "converged_infeasible": np.mean(stops == "converged_infeasible"),
                    "max_iters": np.mean(stops == "max_iters")})
        if all("certified" in e for e in eps):
            row["certified"] = np.concatenate([e["certified"] for e in eps]).mean()
        if all("certified_tol" in e for e in eps):
            row["certified_tol"] = np.concatenate([e["certified_tol"] for e in eps]).mean()
        if all("iterations" in e for e in eps):
            row["iterations"] = np.concatenate([e["iterations"] for e in eps]).mean()
    if all("start" in e for e in eps):
        row["sampled_start"] = np.mean(np.concatenate([e["start"] for e in eps]) == "sampled")
    if any("optimism" in e for e in eps):
        row["optimism"] = nan_stat(np.mean, [e.get("optimism", np.nan) for e in eps])
    if all("filter_active" in e for e in eps):
        active = np.concatenate([e["filter_active"] for e in eps])
        met = np.concatenate([e["filter_met"] for e in eps])
        row.update({"filter_active": active.mean(), "filter_met": met[active].mean() if active.any() else np.nan})
    return row

def episode_crash(e, horizon):
    # One episode: did it crash with no warning (every executed plan in its last horizon met the barrier in the model,
    # so the model was wrong), and on what share of the steps whose plan met the barrier was it active
    if "max_risk" not in e:
        return np.nan, np.nan
    slack = TAU - e["max_risk"][e["latent_safe"]]
    return (bool(e["violated"]) and bool(e["latent_safe"][-horizon:].all()),
            float(np.mean(slack <= ACTIVE_TOL)) if len(slack) else np.nan)

def crash_types(eps, horizon):
    # Pooled over episodes: crashes the model warned of (an executed plan in the last horizon failed the barrier) and
    # ones it did not, and how close the plans that met the barrier sat to it: slack is TAU minus the worst step risk,
    # and the barrier is active where the slack is at most ACTIVE_TOL
    eps = [e for e in eps if "max_risk" in e]
    if not eps:
        return {}
    slack = np.concatenate([TAU - e["max_risk"][e["latent_safe"]] for e in eps])
    crashes, unwarned = sum(bool(e["violated"]) for e in eps), sum(episode_crash(e, horizon)[0] for e in eps)
    return {"episodes": len(eps), "crashes_warned": crashes - unwarned, "crashes_unwarned": unwarned,
            "barrier_active": float(np.mean(slack <= ACTIVE_TOL)) if len(slack) else np.nan,
            "barrier_slack_med": float(np.median(slack)) if len(slack) else np.nan}

def prob_positive(seeds, x, rng):
    # Probability that the mean of x over pairs is above 0, given the data: the Bayesian bootstrap (Rubin, 1981) over
    # whole seeds, i.e. the share of draws of flat Dirichlet weights on the seeds whose weighted mean is positive
    seeds, x = np.asarray(seeds), np.asarray(x, float)
    keep = ~np.isnan(x)
    keys, inv = np.unique(seeds[keep], return_inverse=True)
    if len(keys) == 0:
        return np.nan
    sums, counts = np.bincount(inv, x[keep]), np.bincount(inv)
    w = rng.dirichlet(np.ones(len(keys)), size=N_BOOT)
    return float(np.mean(w @ sums / (w @ counts) > 0))

def seed_boot(seeds, x, rng):
    # 95% interval for the mean of x over pairs, resampling whole seeds: seed k is the same layout in every model and
    # margin_eps, so its pairs are not independent
    seeds, x = np.asarray(seeds), np.asarray(x, float)
    keep = ~np.isnan(x)
    keys, inv = np.unique(seeds[keep], return_inverse=True)
    if len(keys) < 2:
        return np.nan, np.nan
    sums, counts = np.bincount(inv, weights=x[keep]), np.bincount(inv)
    pick = rng.integers(len(keys), size=(N_BOOT, len(keys)))
    return tuple(np.percentile(sums[pick].sum(axis=1) / counts[pick].sum(axis=1), [2.5, 97.5]))

def pooled_planner(pts, pl):
    # One planner's episodes pooled over the points with margin_eps >= POOL_MIN_EPS
    eps = [e for p in pts if pl in p["planners"] and p["eps"] >= POOL_MIN_EPS for e in p["planners"][pl].values()]
    if not eps:
        return None
    n, k = len(eps), sum(e["violated"] for e in eps)
    ret, goals = np.array([e["return"] for e in eps]), np.array([e["goals"] for e in eps])
    (lo, hi), (rlo, rhi) = wilson(k, n), boot_ci(ret)
    return {"episodes": n, "violations": k, "rate": k / n, "rate_lo": lo, "rate_hi": hi, "return": ret.mean(),
            "return_lo": rlo, "return_hi": rhi, "goals": goals.mean()}

def compare(a, b):
    # MAIN (a) against another planner (b) on the seeds both ran: paired violations (McNemar exact), unpaired counts
    # (Fisher exact), time to violation (log-rank) and paired return difference (bootstrap interval, Wilcoxon)
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    va, vb = (np.array([x[s]["violated"] for s in seeds]) for x in (a, b))
    ta, tb = (np.array([x[s]["steps"] for s in seeds]) for x in (a, b))
    dr = np.array([a[s]["return"] - b[s]["return"] for s in seeds])
    n, only_a, only_b = len(seeds), int(np.sum(va & ~vb)), int(np.sum(~va & vb))
    wil = stats.wilcoxon(dr).pvalue if np.any(dr != 0) else 1.0
    return {"n_paired": n, "viol_main": int(va.sum()), "viol_other": int(vb.sum()), "only_main": only_a,
            "only_other": only_b, "p_mcnemar": binom_two_sided(min(only_a, only_b), only_a + only_b),
            "p_fisher": float(stats.fisher_exact([[va.sum(), n - va.sum()], [vb.sum(), n - vb.sum()]])[1]),
            "p_logrank": logrank([(ta, va, tb, vb)]), "return_diff": dr.mean(), "return_diff_lo": boot_ci(dr)[0],
            "return_diff_hi": boot_ci(dr)[1], "p_wilcoxon_return": float(wil),
            "_strata": ((ta, va, tb, vb), [[int(va.sum()), int(n - va.sum())], [int(vb.sum()), int(n - vb.sum())]])}

def pooled(rows):
    # The same comparison pooled over sweep points: summed discordant pairs (exact), CMH and stratified log-rank
    only_a, only_b = sum(r["only_main"] for r in rows), sum(r["only_other"] for r in rows)
    p_cmh, odds = cmh([r["_strata"][1] for r in rows])
    return {"points": len(rows), "n_paired": sum(r["n_paired"] for r in rows),
            "viol_main": sum(r["viol_main"] for r in rows), "viol_other": sum(r["viol_other"] for r in rows),
            "only_main": only_a, "only_other": only_b,
            "p_mcnemar": binom_two_sided(min(only_a, only_b), only_a + only_b), "p_cmh": p_cmh,
            "odds_ratio": odds, "p_logrank": logrank([r["_strata"][0] for r in rows])}

# Figures

def ordered(planners):
    # PLANNER_ORDER first, each sampler followed by its other budgets (e.g. MPPI, then MPPI@32768), then the rest
    base = lambda p: p.split("@")[0]
    budget = lambda p: int(re.match(r"\d*", p.split("@")[1]).group() or 0) if "@" in p else 0 # "100it" -> 100
    rank = lambda p: PLANNER_ORDER.index(base(p)) if base(p) in PLANNER_ORDER else len(PLANNER_ORDER)
    return sorted(planners, key=lambda p: (rank(p), base(p), budget(p)))

def style(planner):
    # A sampler at another budget (e.g. MPPI@32768) takes its sampler's colour, dashed
    return {"color": COLOURS.get(planner.split("@")[0], "grey"), "ls": "--" if "@" in planner else "-"}

def errorbar(ax, x, rows, key, planner, offset=0.0, **kw):
    y = np.array([r[key] for r in rows])
    lo, hi = np.array([r[f"{key}_lo"] for r in rows]), np.array([r[f"{key}_hi"] for r in rows])
    ax.errorbar(np.asarray(x) + offset, y, yerr=[y - lo, hi - y], marker="o", capsize=3, label=planner,
                **{**style(planner), **kw})

def plot_e1(path, title, pts, summary):
    values, name = [p["value"] for p in pts], pts[0]["name"]
    planners = ordered({pl for p in pts for pl in p["planners"]})
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    width = 0.01 * (max(values) - min(values) or 1)
    for i, pl in enumerate(planners):
        xs = [p["value"] for p in pts if pl in p["planners"]]
        rows = [summary[(p["data_dir"], p["tag"], pl)] for p in pts if pl in p["planners"]]
        off = (i - (len(planners) - 1) / 2) * width
        errorbar(axes[0], xs, rows, "rate", pl, off)
        errorbar(axes[1], xs, rows, "return", pl, off)
        errorbar(axes[2], xs, rows, "goals", pl, off)
        axes[3].plot([r["rate"] for r in rows], [r["return"] for r in rows], marker="o", label=pl, **style(pl))
        for x, r in zip(xs, rows):
            axes[3].annotate(f"{x:g}", (r["rate"], r["return"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    for ax, label in zip(axes, ["violation rate (95% Wilson interval)", "episode return (95% bootstrap interval)",
                                "goals reached per episode", "episode return"]):
        ax.set_ylabel(label)
        ax.set_xlabel(name if ax is not axes[3] else f"violation rate (labels: {name})")
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_grid(path, models, key, ylabel, summary):
    # One panel per task (rows) and Lipschitz bound (columns), every planner against margin_eps
    tasks, bounds = sorted({t for t, _ in models}), sorted({b for _, b in models})
    fig, axes = plt.subplots(len(tasks), len(bounds), figsize=(5 * len(bounds), 4 * len(tasks)), squeeze=False,
                             sharey=True)
    for i, task in enumerate(tasks):
        for j, bound in enumerate(bounds):
            ax = axes[i][j]
            if (task, bound) not in models:
                ax.axis("off")
                continue
            data_dir, pts = models[(task, bound)]
            planners = ordered({pl for p in pts for pl in p["planners"]})
            width = 0.01 * (max(p["eps"] for p in pts) - min(p["eps"] for p in pts) or 1)
            for k, pl in enumerate(planners):
                sel = [p for p in pts if pl in p["planners"]]
                errorbar(ax, [p["eps"] for p in sel], [summary[(data_dir, p["tag"], pl)] for p in sel], key, pl,
                         (k - (len(planners) - 1) / 2) * width)
            ax.set_title(f"PointGoal{task}, bound {'inf' if np.isinf(bound) else f'{bound:g}'}", fontsize=10)
            ax.set_xlabel("margin_eps")
            if j == 0:
                ax.set_ylabel(ylabel)
    # One legend for every panel's planners, as some appear only in some panels (e.g. a time-matched sampler)
    labels = {}
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            labels.setdefault(l, h)
    order = ordered(labels)
    axes[0][0].legend([labels[l] for l in order], order, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_survival(path, title, pts, tests):
    fig, axes = plt.subplots(1, len(pts), figsize=(4.5 * len(pts), 4), squeeze=False)
    for ax, p in zip(axes[0], pts):
        for pl in ordered(p["planners"]):
            eps = list(p["planners"][pl].values())
            # Episodes only end early at a violation (no other censoring), so survival is the share not yet violated
            steps, viol = np.array([e["steps"] for e in eps]), np.array([e["violated"] for e in eps])
            t = np.arange(steps.max() + 1)
            ax.step(t, 1 - np.mean(viol[None] & (steps[None] <= t[:, None]), axis=1), where="post",
                    label=f"{pl} ({viol.sum()}/{len(viol)})", **style(pl))
        t = tests.get((p["data_dir"], p["tag"], "MPPI"))
        ax.set_title(f"{p['name']} {p['value']:g}" + (f", vs MPPI log-rank p={t['p_logrank']:.2g}" if t else ""),
                     fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("step")
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("episodes without a violation")
    fig.suptitle(f"{title}: time to first violation")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_e2(path, model, e2_pts, e1_pt, summary, timing):
    # Violation rate and return against plan time per step: the samplers over their budgets, SCP+MPPI and SCP as
    # single points. Filled markers use idle-machine timing (check_timing.py), hollow ones the times logged in the runs
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    def time_of(pt, pl, J):
        t = timing.get((pt["data_dir"], pl, J) if J is not None else (pt["data_dir"], pl, None, pt["eps"]))
        return ((t.mean(), True) if t is not None
                else (summary[(pt["data_dir"], pt["tag"], pl)]["plan_time_mean"], False))
    for pl in ordered({pl for p in e2_pts for pl in p["planners"]}):
        pts = sorted((p for p in e2_pts if pl in p["planners"]), key=lambda p: p["value"])
        xs, idle = zip(*[time_of(p, pl, int(p["value"])) for p in pts])
        rows = [summary[(p["data_dir"], p["tag"], pl)] for p in pts]
        for ax, key in zip(axes, ("rate", "return")):
            y = np.array([r[key] for r in rows])
            err = [y - [r[f"{key}_lo"] for r in rows], [r[f"{key}_hi"] for r in rows] - y]
            ax.errorbar(xs, y, yerr=err, color=COLOURS.get(pl), capsize=3, label=f"{pl} (sampled sequences)")
            for x, yy, p, filled in zip(xs, y, pts, idle):
                ax.plot(x, yy, "o", color=COLOURS.get(pl), mfc=COLOURS.get(pl) if filled else "white")
                ax.annotate(f"{int(p['value'])}", (x, yy), fontsize=7, xytext=(3, 3), textcoords="offset points")
    if e1_pt is not None:
        for pl in ("SCP+MPPI", "SCP"):
            if pl not in e1_pt["planners"]:
                continue
            x, filled = time_of(e1_pt, pl, None)
            r = summary[(e1_pt["data_dir"], e1_pt["tag"], pl)]
            for ax, key in zip(axes, ("rate", "return")):
                ax.errorbar([x], [r[key]], yerr=[[r[key] - r[f"{key}_lo"]], [r[f"{key}_hi"] - r[key]]], marker="*",
                            ms=14, color=COLOURS.get(pl), mfc=COLOURS.get(pl) if filled else "white", capsize=3,
                            label=f"{pl} (E1, eps {e1_pt['eps']:g})")
    for ax, label in zip(axes, ("violation rate (95% Wilson interval)", "episode return (95% bootstrap interval)")):
        ax.set_xscale("log")
        ax.set_xlabel("mean plan time per step (s); filled: idle timing, hollow: logged during the runs")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
    fig.suptitle(f"{model}: E2 at eps {e2_pts[0]['eps']:g}, plan time against safety and return")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_lipschitz(path, task, by_bound, summary):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for bound, pts in sorted(by_bound.items()):
        for pl, style in ((MAIN, "-"), ("MPPI", "--")):
            pts_pl = [p for p in pts if pl in p["planners"]]
            if not pts_pl:
                continue
            rows = [summary[(p["data_dir"], p["tag"], pl)] for p in pts_pl]
            label = f"{pl}, bound {'inf' if np.isinf(bound) else f'{bound:g}'}"
            for ax, key in zip(axes, ("rate", "return")):
                y = np.array([r[key] for r in rows])
                err = [y - [r[f"{key}_lo"] for r in rows], [r[f"{key}_hi"] for r in rows] - y]
                ax.errorbar([p["eps"] for p in pts_pl], y, yerr=err, ls=style, marker="o", capsize=3,
                            color=BOUND_COLOURS.get(bound, "k"), label=label)
    for ax, label in zip(axes, ("violation rate (95% Wilson interval)", "episode return (95% bootstrap interval)")):
        ax.set_xlabel("margin_eps")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
    fig.suptitle(f"PointGoal{task}: Lipschitz bound")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_horizon(path, task, by_eps, summary):
    fig, axes = plt.subplots(2, len(by_eps), figsize=(5 * len(by_eps), 8), squeeze=False)
    for col, (eps, pts) in enumerate(sorted(by_eps.items())):
        pts = sorted(pts, key=lambda p: p["horizon"])
        for pl in ordered({pl for p in pts for pl in p["planners"]}):
            pts_pl = [p for p in pts if pl in p["planners"]]
            rows = [summary[(p["data_dir"], p["tag"], pl)] for p in pts_pl]
            for row, key in enumerate(("rate", "return")):
                errorbar(axes[row][col], [p["horizon"] for p in pts_pl], rows, key, pl)
        axes[0][col].set_title(f"eps {eps:g}")
        for row, label in enumerate(("violation rate", "episode return")):
            axes[row][col].set_xlabel("planning horizon (steps)")
            axes[row][col].set_ylabel(label)
            axes[row][col].legend(fontsize=8)
    fig.suptitle(f"PointGoal{task}, bound 1.5: planning horizon")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_certificates(path, title, pts, summary):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    scp = [pl for pl in ("SCP+MPPI", "SCP") if any(pl in p["planners"] for p in pts)]
    labels, bottom = [], None
    for reason, colour in (("converged", "tab:green"), ("converged_infeasible", "tab:orange"),
                           ("max_iters", "tab:red")):
        heights = [summary[(p["data_dir"], p["tag"], pl)].get(reason, np.nan) for pl in scp for p in pts
                   if pl in p["planners"]]
        labels = [f"{pl}\n{p['name']} {p['value']:g}" for pl in scp for p in pts if pl in p["planners"]]
        bottom = np.zeros(len(heights)) if bottom is None else bottom
        axes[0].bar(range(len(heights)), heights, bottom=bottom, color=colour, label=reason)
        bottom = bottom + np.nan_to_num(heights)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_ylabel("share of steps")
    axes[0].set_ylim(0, 1.15)
    axes[0].set_title("How SCP's steps ended", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper center", ncol=3)
    planners = ordered({pl for p in pts for pl in p["planners"]})
    for i, pl in enumerate(planners):
        rows = [summary[(p["data_dir"], p["tag"], pl)] for p in pts if pl in p["planners"] and p["eps"] >= POOL_MIN_EPS]
        for j, key in enumerate(("viol_soon_after_safe", "viol_soon_after_unsafe")):
            axes[1].bar(i + 0.4 * j - 0.2, nan_stat(np.mean, [r.get(key, np.nan) for r in rows]), width=0.4,
                        color=style(pl)["color"], alpha=1.0 if j == 0 else 0.4, hatch=None if j == 0 else "//")
    axes[1].set_xticks(range(len(planners)))
    axes[1].set_xticklabels(planners)
    axes[1].set_ylabel(f"share of steps followed by a violation within one horizon (eps >= {POOL_MIN_EPS})")
    axes[1].set_title("Solid: the plan met the barrier in the model; hatched: it did not", fontsize=10)
    fig.suptitle(f"{title}: certificates and latent safety")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

# Report

def fmt_time(t):
    return f"{'-':>9s}" if np.isnan(t) else f"{t:8.3f}s"

def fmt_p(p):
    return "nan" if p is None or np.isnan(p) else (f"{p:.3f}" if p >= 1e-3 else f"{p:.1e}")

def is_e2(p):
    return p["sweep"] == "e2" or p["sweep"].startswith("e2/")

def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r if not k.startswith("_")))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    points, warnings = load_results(RESULTS, PREFIX)
    lines = [f"Results in {RESULTS} (world model folders starting '{PREFIX}'): {len(points)} sweep points, "
             f"seeds 0-{SEED_CAP - 1} only (SEED_CAP)"]
    if not points:
        print(lines[0])
        return

    summary, tests, summary_rows, episode_rows = {}, {}, [], []
    for p in points:
        for pl, episodes in p["planners"].items():
            row = planner_row(episodes, p["horizon"])
            summary[(p["data_dir"], p["tag"], pl)] = row
            summary_rows.append({"model": p["model"], "data_dir": p["data_dir"], "tag": p["tag"], "eps": p["eps"],
                                 "horizon": p["horizon"], "bound": p["bound"], "task": p["task"], "planner": pl,
                                 "seeds": f"{min(episodes)}-{max(episodes)}", "seed_cap": SEED_CAP, **row})
            for s, e in sorted(episodes.items()):
                episode_rows.append({"model": p["model"], "data_dir": p["data_dir"], "tag": p["tag"], "eps": p["eps"],
                                     "planner": pl, "seed": s, "steps": e["steps"], "violated": e["violated"],
                                     "return": e["return"], "goals": e["goals"],
                                     "plan_time_med": nan_stat(np.median, e["plan_times"]),
                                     **dict(zip(("crash_unwarned", "barrier_active"),
                                                episode_crash(e, p["horizon"]))),
                                     "optimism": e.get("optimism", np.nan)})
        if MAIN in p["planners"]:
            for pl in p["planners"]:
                if pl != MAIN:
                    t = compare(p["planners"][MAIN], p["planners"][pl])
                    if t is not None:
                        tests[(p["data_dir"], p["tag"], pl)] = t

    # Idle-machine plan times from check_timing.py, every timed step: (data_dir, sampler, sampled sequences) for the
    # samplers, whose time barely depends on margin_eps, and (data_dir, planner, None, margin_eps) for SCP
    timing = {}
    for p in points:
        if p["sweep"] == "timing":
            m = re.search(r"_J_(\d+)", p["name"])
            e = re.search(r"_eps(-?[\d.]+)$", p["name"])
            for pl, episodes in p["planners"].items():
                t = np.concatenate([e_["plan_times"] for e_ in episodes.values()])
                if np.any(~np.isnan(t)):
                    key = ((p["data_dir"], pl, int(m.group(1))) if m else
                           (p["data_dir"], pl, None, float(e.group(1)) if e else 0.2))
                    timing[key] = t[~np.isnan(t)]
    for data_dir in sorted({k[0] for k in timing}):
        # A sampler's time is nearly constant while SCP+MPPI's varies with its iterations and with margin_eps: the
        # matched budget is the smallest whose mean is at least SCP+MPPI's at its slowest eps (at least as much compute
        # per episode at every eps)
        lines += ["", f"=== Plan time per step (s), idle timing runs ({data_dir}) ===",
                  f"{'planner':11s} {'sequences':>9s} {'eps':>5s} {'steps':>6s} {'median':>7s} {'mean':>7s} "
                  f"{'p90':>7s} {'p99':>7s}"]
        keys = sorted((k for k in timing if k[0] == data_dir), key=lambda k: (
            PLANNER_ORDER.index(k[1]) if k[1] in PLANNER_ORDER else len(PLANNER_ORDER), k[2] or 0,
            k[3] if len(k) > 3 else 0))
        for k in keys:
            t, J = timing[k], k[2]
            lines.append(f"{k[1]:11s} {'-' if J is None else J:>9} {'-' if len(k) < 4 else f'{k[3]:g}':>5} "
                         f"{len(t):6d} {np.median(t):7.3f} {t.mean():7.3f} {np.percentile(t, 90):7.3f} "
                         f"{np.percentile(t, 99):7.3f}")
        refs = [timing[k] for k in keys if k[1] == MAIN and k[2] is None]
        ref = max(refs, key=np.mean) if refs else None
        for s in ("MPPI", "CEM"):
            budgets = sorted(k[2] for k in timing if k[0] == data_dir and k[1] == s and k[2] is not None)
            matched = [J for J in budgets if ref is not None and timing[(data_dir, s, J)].mean() >= ref.mean()]
            if ref is not None and matched:
                t = timing[(data_dir, s, matched[0])]
                lines.append(f"  {s}: matched budget {matched[0]} (mean {t.mean():.3f} s, at least {MAIN}'s "
                             f"{ref.mean():.3f} s at its slowest eps); {MAIN} planned for longer than that on "
                             f"{np.mean(ref > t.mean()):.1%} of its steps there")
            elif ref is not None and budgets:
                lines.append(f"  {s}: no budget reaches {MAIN}'s mean ({ref.mean():.3f} s)")

    # E1-type points: every sweep but E2 and the timing runs (E1's margin_eps, the horizon tests, any later sweep),
    # grouped by world model and sweep
    e1 = {}
    for p in points:
        if not is_e2(p) and p["sweep"] != "timing" and not np.isnan(p["eps"]):
            e1.setdefault((p["task"], p["bound"], p["horizon"], p["data_dir"], p["sweep"]), []).append(p)
    same_model = {}
    for task, bound, horizon, data_dir, sweep in e1:
        if sweep == "e1":
            same_model.setdefault((task, bound, horizon), []).append(data_dir)
    warnings += [f"{dirs} share task, bound and horizon; the Lipschitz and horizon figures use the last"
                 for dirs in same_model.values() if len(dirs) > 1]
    test_rows, pooled_rows = [], []
    for (task, bound, horizon, data_dir, sweep), pts in sorted(e1.items()):
        pts.sort(key=lambda p: p["value"])
        model, name = pts[0]["model"], pts[0]["name"]
        lines += ["", f"=== {model} ({data_dir}), {sweep} ===",
                  f"{name:>5s} {'planner':11s} {'n':>3s} {'violations':>10s} {'rate [95% CI]':>18s} "
                  f"{'return [95% CI]':>22s} {'goals':>6s} {'plan time':>9s} {'latent-safe':>11s} {'certified':>9s} "
                  f"{'sampled start':>13s}"]
        for p in pts:
            for pl in ordered(p["planners"]):
                r = summary[(data_dir, p["tag"], pl)]
                lines.append(f"{p['value']:5g} {pl:11s} {r['n']:3d} {r['violations']:>4d}/{r['n']:<5d} "
                             f"{r['rate']:5.2f} [{r['rate_lo']:.2f}, {r['rate_hi']:.2f}] "
                             f"{r['return']:7.2f} [{r['return_lo']:6.2f}, {r['return_hi']:6.2f}] {r['goals']:6.2f} "
                             f"{fmt_time(r['plan_time_med'])} {r.get('latent_safe', np.nan):11.2f} "
                             f"{r.get('certified', np.nan):9.2f} {r.get('sampled_start', np.nan):13.2f}")
        others = ordered({pl for p in pts for pl in p["planners"]} - {MAIN})
        if any(MAIN in p["planners"] for p in pts):
            lines.append(f"  {MAIN} against each planner on paired seeds (only-main:only-other = episodes where just "
                         f"one of the two violated); Holm adjusts the McNemar p over this sweep's points")
        for pl in others:
            rows = [(p, tests[(data_dir, p["tag"], pl)]) for p in pts if (data_dir, p["tag"], pl) in tests]
            if not rows:
                continue
            adj = holm([t["p_mcnemar"] for _, t in rows])
            for (p, t), a in zip(rows, adj):
                t["p_mcnemar_holm"] = a
                test_rows.append({"model": model, "data_dir": data_dir, "tag": p["tag"], "eps": p["eps"],
                                  "main": MAIN, "other": pl, **t})
                lines.append(f"  vs {pl:5s} {name} {p['value']:4g}: violations {t['viol_main']}/{t['n_paired']} vs "
                             f"{t['viol_other']}/{t['n_paired']} ({t['only_main']}:{t['only_other']}), McNemar p "
                             f"{fmt_p(t['p_mcnemar'])} (Holm {fmt_p(a)}), log-rank p {fmt_p(t['p_logrank'])}, return "
                             f"difference {t['return_diff']:+.2f} [{t['return_diff_lo']:+.2f}, "
                             f"{t['return_diff_hi']:+.2f}] (Wilcoxon p {fmt_p(t['p_wilcoxon_return'])})")
            pool = [t for p, t in rows if p["eps"] >= POOL_MIN_EPS]
            if pool:
                q = pooled(pool)
                pooled_rows.append({"model": model, "data_dir": data_dir, "sweep": sweep, "main": MAIN,
                                    "other": pl, "eps": f">= {POOL_MIN_EPS}", **q})
                lines.append(f"  vs {pl:5s} pooled over eps >= {POOL_MIN_EPS} ({q['points']} points): violations "
                             f"{q['viol_main']}/{q['n_paired']} vs {q['viol_other']}/{q['n_paired']} "
                             f"({q['only_main']}:{q['only_other']}), McNemar p {fmt_p(q['p_mcnemar'])}, CMH p "
                             f"{fmt_p(q['p_cmh'])} (odds ratio {q['odds_ratio']:.2f}), stratified log-rank p "
                             f"{fmt_p(q['p_logrank'])}")
        lines.append(f"  latent safety, pooled over eps >= {POOL_MIN_EPS}: share of steps followed by a violation "
                     f"within one horizon, after a plan that met the barrier in the model / one that did not; share of "
                     f"violations preceded by such a warning; crashes with no warning (the model was wrong); share of "
                     f"the plans meeting the barrier that had it active (slack <= {ACTIVE_TOL:g}), and their median slack")
        for pl in ordered({pl for p in pts for pl in p["planners"]}):
            rows = [summary[(data_dir, p["tag"], pl)] for p in pts if pl in p["planners"] and p["eps"] >= POOL_MIN_EPS]
            vals = {k: nan_stat(np.mean, [r.get(k, np.nan) for r in rows])
                    for k in ("viol_soon_after_safe", "viol_soon_after_unsafe", "violations_warned", "filter_active",
                              "filter_met")}
            ct = crash_types([e for p in pts if pl in p["planners"] and p["eps"] >= POOL_MIN_EPS
                              for e in p["planners"][pl].values()], pts[0]["horizon"])
            lines.append(f"    {pl:11s} {vals['viol_soon_after_safe']:.4f} / {vals['viol_soon_after_unsafe']:.4f}, "
                         f"warned {vals['violations_warned']:.2f}"
                         + (f"; unwarned crashes {ct['crashes_unwarned']} in {ct['episodes']} episodes; barrier active "
                            f"on {ct['barrier_active']:.1%} (median slack {ct['barrier_slack_med']:.3f})" if ct else "")
                         + (f"; safety filter changed the action on {vals['filter_active']:.1%} of steps and met the "
                            f"barrier after {vals['filter_met']:.0%} of those"
                            if not np.isnan(vals["filter_active"]) else ""))
        title, stem = f"{model}: {sweep}", f"{sweep}_{data_dir}"
        plot_e1(OUT / f"{stem}.png", title, pts, summary)
        plot_survival(OUT / f"survival_{stem}.png", title, pts, tests)
        if any(pl in p["planners"] for p in pts for pl in ("SCP+MPPI", "SCP")):
            plot_certificates(OUT / f"certificates_{stem}.png", title, pts, summary)

    # Headline: each world model (task and bound, horizon 10, E1), each planner pooled over margin_eps >= POOL_MIN_EPS,
    # with SCP+MPPI's pooled paired tests against it, Holm-adjusted over the world models for each comparison
    main_models = {(t, b): (d, pts) for (t, b, h, d, sw), pts in sorted(e1.items()) if sw == "e1" and h == 10}
    pooled_by = {(r["data_dir"], r["other"]): r for r in pooled_rows if r["sweep"] == "e1"}
    holm_by = {}
    for pl in {r["other"] for r in pooled_rows}:
        keys = [(d, pl) for d, _ in main_models.values() if (d, pl) in pooled_by]
        holm_by.update(zip(keys, holm([pooled_by[k]["p_mcnemar"] for k in keys])))
    top, headline_rows, lip_rows, lip_pooled, all_rows, bound_rows = [], [], [], [], [], []
    bound_str = lambda b: "inf" if np.isinf(b) else f"{b:g}"
    if main_models:
        top += ["", f"=== Headline: each planner pooled over eps >= {POOL_MIN_EPS}; {MAIN} against it on paired seeds: "
                    f"pooled McNemar p (Holm over the {len(main_models)} world models) and CMH p ===",
                f"{'task':4s} {'bound':>5s} {'planner':15s} {'episodes':>8s} {'violations':>10s} "
                f"{'rate [95% CI]':>18s} {'return [95% CI]':>22s} {'goals':>6s}  vs {MAIN}"]
        for (task, bound), (data_dir, pts) in sorted(main_models.items()):
            for pl in ordered({pl for p in pts for pl in p["planners"]}):
                r = pooled_planner(pts, pl)
                if r is None:
                    continue
                q = pooled_by.get((data_dir, pl))
                top.append(f"PG{task:<2d} {bound_str(bound):>5s} {pl:15s} {r['episodes']:8d} {r['violations']:10d} "
                           f"{r['rate']:5.2f} [{r['rate_lo']:.2f}, {r['rate_hi']:.2f}] {r['return']:7.2f} "
                           f"[{r['return_lo']:6.2f}, {r['return_hi']:6.2f}] {r['goals']:6.2f}  "
                           + (f"McNemar p {fmt_p(q['p_mcnemar'])} (Holm {fmt_p(holm_by[(data_dir, pl)])}), CMH p "
                              f"{fmt_p(q['p_cmh'])}" if q else ""))
                headline_rows.append({"task": task, "bound": bound, "data_dir": data_dir, "planner": pl, **r,
                                      **({"p_mcnemar_vs_main": q["p_mcnemar"],
                                          "p_mcnemar_vs_main_holm": holm_by[(data_dir, pl)],
                                          "p_cmh_vs_main": q["p_cmh"]} if q else {})})

        # All world models: MAIN against each planner on the same model, margin_eps (>= POOL_MIN_EPS) and seed, pooled
        # over the horizon-10 models, with intervals that resample whole seeds. The crash types and barrier slack test
        # whether the stronger optimisers add crashes the model did not predict by planning closer to the barrier
        rng = np.random.default_rng(1) # its own stream, so every other interval stays as it was
        pairs, by_bound = {}, {}
        for (_, bnd), (data_dir, pts) in main_models.items():
            for p in pts:
                if p["eps"] < POOL_MIN_EPS or MAIN not in p["planners"]:
                    continue
                a = p["planners"][MAIN]
                for pl, b in p["planners"].items():
                    for s in (sorted(set(a) & set(b)) if pl != MAIN else []):
                        (ua, xa), (ub, xb) = (episode_crash(x[s], p["horizon"]) for x in (a, b))
                        dv, dr = float(a[s]["violated"]) - float(b[s]["violated"]), a[s]["return"] - b[s]["return"]
                        pairs.setdefault(pl, []).append((s, dv, dr, float(ua) - float(ub), xa - xb))
                        by_bound.setdefault((bnd, pl), []).append((s, dv, dr))
        if pairs:
            top += ["", f"=== All {len(main_models)} horizon-10 world models, eps >= {POOL_MIN_EPS}: {MAIN} against each "
                        f"planner on the same model, eps and seed (95% intervals resample seeds) ===",
                    f"  {'vs':13s} {'pairs':>5s}  only {MAIN}:only other violated, violation difference (points), "
                    f"McNemar p, return difference"]
            for pl in ordered(pairs):
                s, dv, dr, du, dx = (np.array(c, float) for c in zip(*pairs[pl]))
                b, c = int(np.sum(dv > 0)), int(np.sum(dv < 0))
                (vlo, vhi), (rlo, rhi) = seed_boot(s, dv, rng), seed_boot(s, dr, rng)
                ub, uc = int(np.sum(du > 0)), int(np.sum(du < 0))
                (xlo, xhi), keep = seed_boot(s, dx, rng), ~np.isnan(dx)
                all_rows.append({"other": pl, "pairs": len(s), "only_main": b, "only_other": c,
                                 "viol_diff": dv.mean(), "viol_diff_lo": vlo, "viol_diff_hi": vhi,
                                 "p_mcnemar": binom_two_sided(min(b, c), b + c), "return_diff": dr.mean(),
                                 "return_diff_lo": rlo, "return_diff_hi": rhi, "unwarned_only_main": ub,
                                 "unwarned_only_other": uc, "p_mcnemar_unwarned": binom_two_sided(min(ub, uc), ub + uc),
                                 "active_diff": dx[keep].mean() if keep.any() else np.nan, "active_diff_lo": xlo,
                                 "active_diff_hi": xhi})
                r = all_rows[-1]
                top.append(f"  {pl:13s} {len(s):5d}  {b:4d}:{c:<4d} {100 * r['viol_diff']:+6.1f} [{100 * vlo:+.1f}, "
                           f"{100 * vhi:+.1f}], p {fmt_p(r['p_mcnemar'])}, {r['return_diff']:+.2f} [{rlo:+.2f}, "
                           f"{rhi:+.2f}]")
            top += [f"  crash types and barrier slack, each planner over the same models and eps: crashes per 100 "
                    f"episodes the model warned of / did not; share of the plans meeting the barrier that had it active "
                    f"(slack <= {ACTIVE_TOL:g}) and their median slack;",
                    f"  {MAIN} against it on paired episodes: unwarned crashes (only {MAIN}:only other, McNemar p) and "
                    f"the difference in the active share (points)"]
            cts = {pl: crash_types([e for _, pts in main_models.values() for p in pts if p["eps"] >= POOL_MIN_EPS
                                    for e in p["planners"].get(pl, {}).values()], 10) # main_models are horizon 10
                   for pl in set(pairs) | {MAIN}}
            for r in all_rows:
                r.update({**{f"main_{k}": v for k, v in cts[MAIN].items()},
                          **{f"other_{k}": v for k, v in cts[r["other"]].items()}})
            for pl in ordered(cts):
                ct, r = cts[pl], next((r for r in all_rows if r["other"] == pl), None)
                if not ct:
                    continue
                top.append(f"  {pl:13s} {ct['episodes']:5d} episodes: {100 * ct['crashes_warned'] / ct['episodes']:5.1f} / "
                           f"{100 * ct['crashes_unwarned'] / ct['episodes']:4.1f}, active {ct['barrier_active']:6.1%}, "
                           f"median slack {ct['barrier_slack_med']:.3f}"
                           + (f"; vs {MAIN}: unwarned {r['unwarned_only_main']}:{r['unwarned_only_other']}, p "
                              f"{fmt_p(r['p_mcnemar_unwarned'])}, active {100 * r['active_diff']:+.1f} "
                              f"[{100 * r['active_diff_lo']:+.1f}, {100 * r['active_diff_hi']:+.1f}]" if r else ""))

            # The same comparison for each Lipschitz bound, pooled over the tasks: the paper's tables, since pooling the
            # bounds averages the main models with the ones the paper argues against
            rng_bound = np.random.default_rng(2) # its own stream, so the intervals above stay as they were
            rng_prob = np.random.default_rng(3) # and the probabilities theirs, so these intervals stay as they were
            top += [f"  per Lipschitz bound, both tasks: {MAIN} against each planner as above"]
            for bnd in sorted({bnd for bnd, _ in by_bound}):
                for pl in ordered({pl for b_, pl in by_bound if b_ == bnd}):
                    s, dv, dr = (np.array(c, float) for c in zip(*by_bound[(bnd, pl)]))
                    b, c = int(np.sum(dv > 0)), int(np.sum(dv < 0))
                    (vlo, vhi), (rlo, rhi) = seed_boot(s, dv, rng_bound), seed_boot(s, dr, rng_bound)
                    bound_rows.append({"bound": bnd, "other": pl, "pairs": len(s), "only_main": b, "only_other": c,
                                       "viol_diff": dv.mean(), "viol_diff_lo": vlo, "viol_diff_hi": vhi,
                                       "p_mcnemar": binom_two_sided(min(b, c), b + c), "return_diff": dr.mean(),
                                       "return_diff_lo": rlo, "return_diff_hi": rhi,
                                       "p_safer": prob_positive(s, -dv, rng_prob),
                                       "p_higher_return": prob_positive(s, dr, rng_prob)})
                    top.append(f"  c {bound_str(bnd):3s} {pl:13s} {len(s):5d}  {b:4d}:{c:<4d} {100 * dv.mean():+6.1f} "
                               f"[{100 * vlo:+.1f}, {100 * vhi:+.1f}], p {fmt_p(bound_rows[-1]['p_mcnemar'])}, "
                               f"{dr.mean():+.2f} [{rlo:+.2f}, {rhi:+.2f}]; P(safer) "
                               f"{bound_rows[-1]['p_safer']:.3f}, P(higher return) {bound_rows[-1]['p_higher_return']:.3f}")

        # Lipschitz bound: each planner on the bound-1.5 model against the same planner on each other bound, on paired
        # seeds (seed k is the same layout for every model), per margin_eps and pooled over margin_eps >= POOL_MIN_EPS
        for task in sorted({t for t, _ in main_models}):
            others = sorted(b for t, b in main_models if t == task and not np.isclose(b, 1.5))
            if (task, 1.5) not in main_models or not others:
                continue
            top += ["", f"=== Lipschitz bound, PointGoal{task}: each planner with bound 1.5 against bound "
                        + ", ".join(map(bound_str, others)) + f", paired seeds, pooled over eps >= {POOL_MIN_EPS} "
                        f"(only-1.5:only-other = episodes where just one of the two violated) ==="]
            ref_pts = main_models[(task, 1.5)][1]
            for pl in ordered({pl for p in ref_pts for pl in p["planners"]}):
                for b in others:
                    other_by_eps = {round(p["eps"], 6): p for p in main_models[(task, b)][1]}
                    rows, diffs = [], []
                    for p in ref_pts:
                        q = other_by_eps.get(round(p["eps"], 6))
                        if q is None or pl not in p["planners"] or pl not in q["planners"]:
                            continue
                        t = compare(p["planners"][pl], q["planners"][pl])
                        if t is None:
                            continue
                        lip_rows.append({"task": task, "planner": pl, "eps": p["eps"], "ref_bound": 1.5,
                                         "other_bound": b, **{k.replace("main", "ref"): v for k, v in t.items()}})
                        if p["eps"] >= POOL_MIN_EPS:
                            rows.append(t)
                            a, c = p["planners"][pl], q["planners"][pl]
                            diffs += [a[s]["return"] - c[s]["return"] for s in sorted(set(a) & set(c))]
                    if not rows:
                        continue
                    qd, (lo, hi) = pooled(rows), boot_ci(diffs)
                    lip_pooled.append({"task": task, "planner": pl, "ref_bound": 1.5, "other_bound": b,
                                       **{k.replace("main", "ref"): v for k, v in qd.items()},
                                       "return_diff": np.mean(diffs), "return_diff_lo": lo, "return_diff_hi": hi})
                    top.append(f"  {pl:15s} 1.5 vs {bound_str(b):>3s}: violations {qd['viol_main']}/"
                               f"{qd['n_paired']} vs {qd['viol_other']}/{qd['n_paired']} "
                               f"({qd['only_main']}:{qd['only_other']}), "
                               f"McNemar p {fmt_p(qd['p_mcnemar'])}, CMH p {fmt_p(qd['p_cmh'])}, return difference "
                               f"{np.mean(diffs):+.2f} [{lo:+.2f}, {hi:+.2f}]")
        plot_grid(OUT / "e1_grid_violations.png", main_models, "rate", "violation rate (95% Wilson interval)", summary)
        plot_grid(OUT / "e1_grid_return.png", main_models, "return", "episode return (95% bootstrap interval)",
                  summary)
    lines[1:1] = top

    # E2: samplers over their budgets at each margin_eps run, with the same model's E1 point at that margin_eps
    for data_dir, sweep in sorted({(p["data_dir"], p["sweep"]) for p in points if is_e2(p)}):
        e2_pts = sorted((p for p in points if p["sweep"] == sweep and p["data_dir"] == data_dir),
                        key=lambda p: p["value"])
        e1_pt = next((p for p in points if p["data_dir"] == data_dir and p["sweep"] == "e1"
                      and np.isclose(p["eps"], e2_pts[0]["eps"])), None)
        model = e2_pts[0]["model"]
        lines += ["", f"=== {model} ({data_dir}): E2 at eps {e2_pts[0]['eps']:g} ===",
                  f"{'planner':11s} {'sequences':>9s} {'n':>3s} {'rate [95% CI]':>18s} {'return':>7s} "
                  f"{'mean plan time logged / idle':>29s}"]
        for p in e2_pts:
            for pl in ordered(p["planners"]):
                r = summary[(data_dir, p["tag"], pl)]
                idle = timing.get((data_dir, pl, int(p["value"])))
                lines.append(f"{pl:11s} {int(p['value']):9d} {r['n']:3d} {r['rate']:5.2f} [{r['rate_lo']:.2f}, "
                             f"{r['rate_hi']:.2f}] {r['return']:7.2f} {r['plan_time_mean']:15.3f}s / "
                             f"{'-' if idle is None else f'{idle.mean():.3f}s'}")
        if e1_pt is not None:
            for pl in ("SCP+MPPI", "SCP"):
                if pl in e1_pt["planners"]:
                    r = summary[(data_dir, e1_pt["tag"], pl)]
                    idle = timing.get((data_dir, pl, None, e1_pt["eps"]))
                    lines.append(f"{pl:11s} {'(E1)':>9s} {r['n']:3d} {r['rate']:5.2f} [{r['rate_lo']:.2f}, "
                                 f"{r['rate_hi']:.2f}] {r['return']:7.2f} {r['plan_time_mean']:15.3f}s / "
                                 f"{'-' if idle is None else f'{idle.mean():.3f}s'}")
        plot_e2(OUT / f"e2_eps{e2_pts[0]['eps']:g}_{data_dir}.png", model, e2_pts, e1_pt, summary, timing)

    # Lipschitz bound: the horizon-10 models of each task, by bound, on E1
    for task in sorted({k[0] for k in e1}):
        by_bound = {k[1]: v for k, v in e1.items() if k[0] == task and k[2] == 10 and k[4] == "e1"}
        if len(by_bound) > 1:
            plot_lipschitz(OUT / f"lipschitz_pg{task}.png", task, by_bound, summary)
    # Planning horizon: the bound-1.5 models of each task, at every margin_eps run at two or more horizons (E1's points
    # and the horizon tests')
    for task in sorted({k[0] for k in e1}):
        by_eps = {}
        for k, pts in e1.items():
            if k[0] == task and np.isclose(k[1], 1.5) and k[4] in ("e1", "e4"):
                for p in pts:
                    by_eps.setdefault(round(p["eps"], 6), []).append(p)
        by_eps = {e: v for e, v in by_eps.items() if len({p["horizon"] for p in v}) > 1}
        if by_eps:
            plot_horizon(OUT / f"horizon_pg{task}.png", task, by_eps, summary)

    if warnings:
        lines += ["", "Warnings:"] + [f"  {w}" for w in warnings]
    write_csv(OUT / "summary.csv", summary_rows)
    write_csv(OUT / "tests.csv", test_rows)
    write_csv(OUT / "pooled_tests.csv", pooled_rows)
    write_csv(OUT / "episodes.csv", episode_rows)
    write_csv(OUT / "headline.csv", headline_rows)
    write_csv(OUT / "lipschitz_tests.csv", lip_rows)
    write_csv(OUT / "lipschitz_pooled.csv", lip_pooled)
    write_csv(OUT / "all_models.csv", all_rows)
    write_csv(OUT / "by_bound.csv", bound_rows)
    report = "\n".join(lines)
    (OUT / "report.txt").write_text(report + "\n")
    print(report)
    print(f"\nWritten to {OUT}: report.txt, headline.csv, summary.csv, tests.csv, pooled_tests.csv, "
          f"lipschitz_tests.csv, lipschitz_pooled.csv, all_models.csv, by_bound.csv, episodes.csv and the figures")

if __name__ == "__main__":
    main()
