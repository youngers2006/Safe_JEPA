import argparse
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

ROOT = Path(__file__).resolve().parents[2]

# Reads every saved episode under the experiments folder (no world model or GPU needed) and writes summary tables,
# paired tests and figures, e.g. python analyse_results.py --results path/to/experiments --prefix M2_
ARGS = argparse.ArgumentParser()
ARGS.add_argument("--results", type=Path, default=ROOT / "src/save_data/experiments", help="experiments folder")
ARGS.add_argument("--out", type=Path, default=None, help="output folder (default <results>/analysis_results)")
ARGS.add_argument("--prefix", default="M2_", help="only world model folders starting with this ('' for all)")

MAIN = "SCP+MPPI" # the planner compared against every other one
PLANNER_ORDER = ["SCP+MPPI", "SCP", "MPPI", "CEM"]
COLOURS = {"SCP+MPPI": "tab:purple", "SCP": "tab:blue", "MPPI": "tab:orange", "CEM": "tab:green"}
BOUND_COLOURS = {1.2: "tab:red", 1.5: "tab:blue", np.inf: "tab:gray"}
TAU, FEAS_TOL = 0.0, 1e-3 # a plan meets the barrier in the model when its worst step's risk is at most TAU + FEAS_TOL
GOAL_THRESHOLD = 0.5 # a step whose reward exceeds this reached the goal
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
          "plan_times": d["plan_times"] if "plan_times" in d.files else np.full(T, np.nan)}
    if "info_pred_risk" in d.files: # per step: does the executed plan meet the barrier in the model?
        ep["latent_safe"] = d["info_pred_risk"].reshape(T, -1).max(axis=1) <= TAU + FEAS_TOL
    for k in ("stop_reason", "start"):
        if f"info_{k}" in d.files:
            ep[k] = d[f"info_{k}"].astype(str)
    for k in ("certified", "iterations"):
        if f"info_{k}" in d.files:
            ep[k] = d[f"info_{k}"].astype(float)
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

def check_point(point_dir, planners, configs, warnings):
    # Overlapping seed ranges for a planner (e.g. a test run's CSV beside the final one) and settings that differ
    # between the planners of one point
    for planner in planners:
        ranges = []
        for f in point_dir.glob(f"{glob.escape(planner)}_seeds_*.csv"):
            m = re.fullmatch(rf"{re.escape(planner)}_seeds_(\d+)-(\d+)\.csv", f.name)
            if m:
                ranges.append((int(m.group(1)), int(m.group(2))))
        ranges.sort()
        if any(b0 <= a1 for (a0, a1), (b0, b1) in zip(ranges, ranges[1:])):
            warnings.append(f"{point_dir}: {planner} has overlapping seed ranges {ranges}; check which run wrote the "
                            f"seed files")
    for key in ("rho_u", "barrier_alpha", "horizon", "data_dir", "margin_eps"):
        values = {p: str(c.get(key)) for p, c in configs.items() if c}
        if len(set(values.values())) > 1:
            warnings.append(f"{point_dir}: {key} differs between planners: {values}")

def load_results(results, prefix):
    raw = {}
    for f in sorted(results.rglob("*_seed_*.npz")):
        m = re.fullmatch(r"(.+)_seed_(\d+)\.npz", f.name)
        rel = f.parent.relative_to(results).parts
        if m is None or len(rel) < 2 or not rel[0].startswith(prefix):
            continue
        raw.setdefault((rel[0], "/".join(rel[1:])), {}).setdefault(m.group(1), {})[int(m.group(2))] = \
            episode_summary(f)
    points, warnings = [], []
    for (data_dir, tag), planners in sorted(raw.items()):
        point_dir = results / data_dir / tag
        configs = {p: (yaml.safe_load(open(point_dir / f"{p}_config.yaml"))
                       if (point_dir / f"{p}_config.yaml").exists() else {}) for p in planners}
        check_point(point_dir, planners, configs, warnings)
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
           "plan_time_med": nan_stat(np.median, pt), "plan_time_p90": nan_stat(np.percentile, pt, 90)}
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
                    "violations_warned": np.mean(warned) if warned else np.nan})
    # SCP's certificates: converged with the plan feasible (KKT point), converged with slack in use (no safe plan
    # nearby, to first order), or stopped at the iteration cap (no certificate)
    if all("stop_reason" in e for e in eps):
        stops = np.concatenate([e["stop_reason"] for e in eps])
        row.update({"converged": np.mean(stops == "converged"),
                    "converged_infeasible": np.mean(stops == "converged_infeasible"),
                    "max_iters": np.mean(stops == "max_iters")})
        if all("certified" in e for e in eps):
            row["certified"] = np.concatenate([e["certified"] for e in eps]).mean()
        if all("iterations" in e for e in eps):
            row["iterations"] = np.concatenate([e["iterations"] for e in eps]).mean()
    if all("start" in e for e in eps):
        row["sampled_start"] = np.mean(np.concatenate([e["start"] for e in eps]) == "sampled")
    return row

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
    return [p for p in PLANNER_ORDER if p in planners] + sorted(p for p in planners if p not in PLANNER_ORDER)

def errorbar(ax, x, rows, key, planner, offset=0.0, **kw):
    y = np.array([r[key] for r in rows])
    lo, hi = np.array([r[f"{key}_lo"] for r in rows]), np.array([r[f"{key}_hi"] for r in rows])
    ax.errorbar(np.asarray(x) + offset, y, yerr=[y - lo, hi - y], marker="o", capsize=3,
                color=COLOURS.get(planner), label=planner, **kw)

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
        axes[3].plot([r["rate"] for r in rows], [r["return"] for r in rows], "o-", color=COLOURS.get(pl), label=pl)
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

def plot_survival(path, title, pts, tests):
    fig, axes = plt.subplots(1, len(pts), figsize=(4.5 * len(pts), 4), squeeze=False)
    for ax, p in zip(axes[0], pts):
        for pl in ordered(p["planners"]):
            eps = list(p["planners"][pl].values())
            # Episodes only end early at a violation (no other censoring), so survival is the share not yet violated
            steps, viol = np.array([e["steps"] for e in eps]), np.array([e["violated"] for e in eps])
            t = np.arange(steps.max() + 1)
            ax.step(t, 1 - np.mean(viol[None] & (steps[None] <= t[:, None]), axis=1), where="post",
                    color=COLOURS.get(pl), label=f"{pl} ({viol.sum()}/{len(viol)})")
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
    # single points. Filled markers use idle-machine timing (run_timing.py), hollow ones the times logged in the runs
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    def time_of(pt, pl, J):
        t = timing.get((pt["data_dir"], pl, J))
        return (t, True) if t is not None else (summary[(pt["data_dir"], pt["tag"], pl)]["plan_time_med"], False)
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
        ax.set_xlabel("plan time per step (s); filled: idle timing, hollow: logged during the runs")
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
                        color=COLOURS.get(pl), alpha=1.0 if j == 0 else 0.4, hatch=None if j == 0 else "//")
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
    args = ARGS.parse_args()
    results, out = args.results, args.out or args.results / "analysis_results"
    out.mkdir(parents=True, exist_ok=True)
    points, warnings = load_results(results, args.prefix)
    lines = [f"Results in {results} (world model folders starting '{args.prefix}'): {len(points)} sweep points"]
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
                                 "seeds": f"{min(episodes)}-{max(episodes)}", **row})
            for s, e in sorted(episodes.items()):
                episode_rows.append({"model": p["model"], "data_dir": p["data_dir"], "tag": p["tag"], "eps": p["eps"],
                                     "planner": pl, "seed": s, "steps": e["steps"], "violated": e["violated"],
                                     "return": e["return"], "goals": e["goals"],
                                     "plan_time_med": nan_stat(np.median, e["plan_times"])})
        if MAIN in p["planners"]:
            for pl in p["planners"]:
                if pl != MAIN:
                    t = compare(p["planners"][MAIN], p["planners"][pl])
                    if t is not None:
                        tests[(p["data_dir"], p["tag"], pl)] = t

    # Idle-machine plan times from run_timing.py: (data_dir, planner, sampled sequences or None) -> median per step
    timing = {}
    for p in points:
        if p["sweep"] == "timing":
            m = re.search(r"_J_(\d+)", p["name"])
            for pl, episodes in p["planners"].items():
                timing[(p["data_dir"], pl, int(m.group(1)) if m else None)] = nan_stat(np.median, np.concatenate(
                    [e["plan_times"] for e in episodes.values()]))

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
                  f"{name:>5s} {'planner':9s} {'n':>3s} {'violations':>10s} {'rate [95% CI]':>18s} "
                  f"{'return [95% CI]':>22s} {'goals':>6s} {'plan time':>9s} {'latent-safe':>11s} {'certified':>9s} "
                  f"{'sampled start':>13s}"]
        for p in pts:
            for pl in ordered(p["planners"]):
                r = summary[(data_dir, p["tag"], pl)]
                lines.append(f"{p['value']:5g} {pl:9s} {r['n']:3d} {r['violations']:>4d}/{r['n']:<5d} "
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
                     f"violations preceded by such a warning")
        for pl in ordered({pl for p in pts for pl in p["planners"]}):
            rows = [summary[(data_dir, p["tag"], pl)] for p in pts if pl in p["planners"] and p["eps"] >= POOL_MIN_EPS]
            vals = {k: nan_stat(np.mean, [r.get(k, np.nan) for r in rows])
                    for k in ("viol_soon_after_safe", "viol_soon_after_unsafe", "violations_warned")}
            lines.append(f"    {pl:9s} {vals['viol_soon_after_safe']:.4f} / {vals['viol_soon_after_unsafe']:.4f}, "
                         f"warned {vals['violations_warned']:.2f}")
        title, stem = f"{model}: {sweep}", f"{sweep}_{data_dir}"
        plot_e1(out / f"{stem}.png", title, pts, summary)
        plot_survival(out / f"survival_{stem}.png", title, pts, tests)
        if any(pl in p["planners"] for p in pts for pl in ("SCP+MPPI", "SCP")):
            plot_certificates(out / f"certificates_{stem}.png", title, pts, summary)

    # E2: samplers over their budgets at each margin_eps run, with the same model's E1 point at that margin_eps
    for data_dir, sweep in sorted({(p["data_dir"], p["sweep"]) for p in points if is_e2(p)}):
        e2_pts = sorted((p for p in points if p["sweep"] == sweep and p["data_dir"] == data_dir),
                        key=lambda p: p["value"])
        e1_pt = next((p for p in points if p["data_dir"] == data_dir and p["sweep"] == "e1"
                      and np.isclose(p["eps"], e2_pts[0]["eps"])), None)
        model = e2_pts[0]["model"]
        lines += ["", f"=== {model} ({data_dir}): E2 at eps {e2_pts[0]['eps']:g} ===",
                  f"{'planner':9s} {'sequences':>9s} {'n':>3s} {'rate [95% CI]':>18s} {'return':>7s} "
                  f"{'plan time logged / idle':>24s}"]
        for p in e2_pts:
            for pl in ordered(p["planners"]):
                r = summary[(data_dir, p["tag"], pl)]
                idle = timing.get((data_dir, pl, int(p["value"])))
                lines.append(f"{pl:9s} {int(p['value']):9d} {r['n']:3d} {r['rate']:5.2f} [{r['rate_lo']:.2f}, "
                             f"{r['rate_hi']:.2f}] {r['return']:7.2f} {r['plan_time_med']:10.3f}s / "
                             f"{'-' if idle is None else f'{idle:.3f}s'}")
        if e1_pt is not None:
            for pl in ("SCP+MPPI", "SCP"):
                if pl in e1_pt["planners"]:
                    r = summary[(data_dir, e1_pt["tag"], pl)]
                    idle = timing.get((data_dir, pl, None))
                    lines.append(f"{pl:9s} {'(E1)':>9s} {r['n']:3d} {r['rate']:5.2f} [{r['rate_lo']:.2f}, "
                                 f"{r['rate_hi']:.2f}] {r['return']:7.2f} {r['plan_time_med']:10.3f}s / "
                                 f"{'-' if idle is None else f'{idle:.3f}s'}")
        plot_e2(out / f"e2_eps{e2_pts[0]['eps']:g}_{data_dir}.png", model, e2_pts, e1_pt, summary, timing)

    # Lipschitz bound: the horizon-10 models of each task, by bound, on E1
    for task in sorted({k[0] for k in e1}):
        by_bound = {k[1]: v for k, v in e1.items() if k[0] == task and k[2] == 10 and k[4] == "e1"}
        if len(by_bound) > 1:
            plot_lipschitz(out / f"lipschitz_pg{task}.png", task, by_bound, summary)
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
            plot_horizon(out / f"horizon_pg{task}.png", task, by_eps, summary)

    if warnings:
        lines += ["", "Warnings:"] + [f"  {w}" for w in warnings]
    write_csv(out / "summary.csv", summary_rows)
    write_csv(out / "tests.csv", test_rows)
    write_csv(out / "pooled_tests.csv", pooled_rows)
    write_csv(out / "episodes.csv", episode_rows)
    report = "\n".join(lines)
    (out / "report.txt").write_text(report + "\n")
    print(report)
    print(f"\nWritten to {out}: report.txt, summary.csv, tests.csv, pooled_tests.csv, episodes.csv and the figures")

if __name__ == "__main__":
    main()
