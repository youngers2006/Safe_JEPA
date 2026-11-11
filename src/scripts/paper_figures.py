import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

ROOT = Path(__file__).resolve().parents[2]

# The paper's figures, drawn from analyse_results.py's outputs (run that first): python paper_figures.py
# Saved as PDF (for the paper) and PNG (to preview) in OUT, one folder per place in the paper
ANALYSIS = ROOT / "src/save_data/experiments/analysis_results"
OUT = ROOT / "paper_graphs"
# The seeds are capped in analyse_results.py, which reads the episodes; this must match its SEED_CAP so the figures say
# which seeds they show. The analysis outputs record their cap, and the figures stop if the two differ
SEED_CAP = 30
TEXT_WIDTH = 6.0 # inches, the paper's text width: every figure spans it
MAIN_BOUND = 1.5 # the main world models' Lipschitz bound
SURVIVAL_TASK = 1 # the main model whose survival curves go in the appendix
EPS_TICKS = [0.0, 0.1, 0.2, 0.3, 0.5]
POOL_MIN_EPS = 0.1 # pooled figures use the margin_eps sweep from here up, as the report does
HORIZON_TASK, HORIZON_EPS = 1, 0.2 # the horizon tests: PointGoal1's bound-MAIN_BOUND models at margin_eps 0.2
TIME_MATCHED = {10: 65536, 20: 95040, 30: 120000} # each horizon's time-matched sampler budget
SAMPLE_BUDGET = 512 # the samplers' budget in the runs named plain MPPI and CEM
# The baselines in the per-bound comparison tables, in order; add "MPPI" and "CEM" for the 512-sequence samplers
POOLED_ROWS = ["SCP", "MPPI@65536", "CEM@65536"]
# The baselines in the main-text horizon table ("@tm": each horizon's TIME_MATCHED budget); the appendix version lists
# every planner run at each horizon
HORIZON_TABLE_ROWS = ["SCP", "MPPI@tm", "CEM@tm"]
# Return against compute uses the plan times logged during the runs, comparable within these horizons' runs (H10's
# time-matched runs shared GPUs differently; its compute matching is the idle timing in the report)
COMPUTE_HORIZONS = (20, 30)

# Each planner as it appears in the paper: label, colour, line style, marker, filled marker. The time-matched samplers
# are the main-text baselines (solid); the 512-sequence samplers appear only in the ablation and the appendix (dashed)
STYLE = {"SCP+MPPI": ("SCP+MPPI", "tab:purple", "-", "o", True),
         "SCP": ("SCP", "tab:blue", "-", "s", True),
         "MPPI@65536": ("MPPI (time-matched)", "tab:orange", "-", "^", True),
         "CEM@65536": ("CEM (time-matched)", "tab:green", "-", "v", True),
         "MPPI": ("MPPI (512 samples)", "tab:orange", "--", "^", False),
         "CEM": ("CEM (512 samples)", "tab:green", "--", "v", False)}
BOUND_COLOURS = {1.2: "tab:red", 1.5: "tab:blue", np.inf: "0.4"}
MAIN_TEXT = ["SCP+MPPI", "SCP", "MPPI@65536", "CEM@65536"]
ABLATION = ["SCP+MPPI", "MPPI", "CEM", "MPPI@65536", "CEM@65536"]
EVERY = ["SCP+MPPI", "SCP", "MPPI", "MPPI@65536", "CEM", "CEM@65536"]
TRADE_OFF = ["SCP+MPPI", "MPPI", "MPPI@65536", "CEM", "CEM@65536"]
SURVIVAL = MAIN_TEXT
# The horizon figures' series: "@tm" is the sampler at each horizon's TIME_MATCHED budget
HORIZON_MAIN = ["SCP+MPPI", "SCP", "MPPI@tm", "CEM@tm"]
HORIZON_ALL = ["SCP+MPPI", "SCP", "MPPI@tm", "CEM@tm", "MPPI", "CEM"]
# How SCP+MPPI's steps ended: (summary column, label, colour). Certified is certified_tol where the summary has it
# (the rollout meets the barrier to SCP's convergence tolerance, 1e-4), else the logged certified (no tolerance);
# "converged_uncertified" is converged minus certified. Categories that never occur are left out
STOPS = [("certified", "Certified: converged, rollout meets the barrier", "tab:green"),
         ("converged_uncertified", "Converged, rollout misses the barrier", "yellowgreen"),
         ("converged_infeasible", "Converged with slack: no safe plan nearby", "tab:orange"),
         ("max_iters", "Iteration cap", "tab:red")]

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9,
                     "axes.titlesize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "lines.linewidth": 1.2, "lines.markersize": 4, "errorbar.capsize": 2,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

# Helpers

def load(name):
    with open(ANALYSIS / name) as f:
        return list(csv.DictReader(f))

def style(pl):
    # A run's style; a sampler at any budget but 512 takes its time-matched style
    base, _, budget = pl.partition("@")
    return STYLE[pl] if pl in STYLE else STYLE[f"{base}@65536"] if budget else STYLE[pl]

def budget_of(pl):
    return int(pl.partition("@")[2] or SAMPLE_BUDGET)

def budget_label(pl):
    J = budget_of(pl)
    return f"{J // 1000}k" if J >= 1000 else f"{J}"

def run_name(series, H):
    # A horizon series' run at horizon H: the time-matched samplers' budget differs by horizon
    return f"{series[:-3]}@{TIME_MATCHED[H]}" if series.endswith("@tm") else series

def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z ** 2 / n
    centre, half = (p + z ** 2 / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return max(0.0, centre - half), min(1.0, centre + half)

def boot_ci(x, rng):
    x = np.asarray(x, float)
    return tuple(np.percentile(rng.choice(x, (10_000, len(x))).mean(axis=1), [2.5, 97.5])) if len(x) > 1 \
        else (np.nan, np.nan)

def sweep_of(rows, task, bound):
    # One horizon-10 world model's margin_eps sweep: {planner: [(eps, summary row)] in eps order}
    out = {}
    for r in rows:
        if r["tag"].startswith("e1/") and int(r["task"]) == task and int(r["horizon"]) == 10 \
                and np.isclose(float(r["bound"]), bound):
            out.setdefault(r["planner"], []).append((float(r["eps"]), r))
    return {pl: sorted(v, key=lambda x: x[0]) for pl, v in out.items()}

def horizon_points(rows):
    # The horizon tests: {H: {planner: summary row}}; H10 is that model's margin_eps sweep point
    out = {}
    for r in rows:
        if int(r["task"]) != HORIZON_TASK or not np.isclose(float(r["bound"]), MAIN_BOUND) \
                or not np.isclose(float(r["eps"]), HORIZON_EPS):
            continue
        H = int(r["horizon"])
        if r["tag"] == (f"e1/eps_{HORIZON_EPS}" if H == 10 else f"e4/H_{H}"):
            out.setdefault(H, {})[r["planner"]] = r
    return out

def pooled_episodes(episodes):
    # Every horizon-10 world model's margin_eps sweep from POOL_MIN_EPS up
    return [r for r in episodes if r["tag"].startswith("e1/") and float(r["eps"]) >= POOL_MIN_EPS]

def bound_label(b):
    return r"$c=\infty$" if np.isinf(b) else f"$c={b:g}$"

def bound_colour(b):
    return next(c for k, c in BOUND_COLOURS.items() if np.isclose(k, b) or (np.isinf(k) and np.isinf(b)))

def plot_sweep(ax, sweep, planners, key):
    # One quantity against margin_eps with its 95% interval; planners offset a little so the intervals do not overlap
    shown = [pl for pl in planners if pl in sweep]
    for i, pl in enumerate(shown):
        label, colour, ls, marker, filled = STYLE[pl]
        eps = np.array([e for e, _ in sweep[pl]])
        y, lo, hi = (np.array([float(r[k]) for _, r in sweep[pl]]) for k in (key, f"{key}_lo", f"{key}_hi"))
        ax.errorbar(eps + (i - (len(shown) - 1) / 2) * 0.008, y, yerr=[y - lo, hi - y], color=colour, ls=ls,
                    marker=marker, mfc=colour if filled else "white", label=label)
    ax.set_xticks(EPS_TICKS)

def top_legend(fig, axes, ncol=None):
    # One legend above the figure for every planner drawn in any panel
    seen = {}
    for ax in np.ravel(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
               ncol=ncol or len(seen))

def save(fig, folder, name):
    (OUT / folder).mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / folder / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print(f"  {folder}/{name}.pdf")

def skip(name, why):
    print(f"  (skipped {name}: {why})")

# Figures

def fig_sweep(rows, planners, folder, name):
    # Violation rate and return against margin_eps, each task's main world model in a column
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 3.9), sharex=True, sharey="row")
    for col, task in enumerate((1, 2)):
        sweep = sweep_of(rows, task, MAIN_BOUND)
        plot_sweep(axes[0][col], sweep, planners, "rate")
        plot_sweep(axes[1][col], sweep, planners, "return")
        axes[0][col].set_title(f"PointGoal{task}")
        axes[1][col].set_xlabel(r"margin $\epsilon$")
    axes[0][0].set_ylabel("violation rate")
    axes[1][0].set_ylabel("episode return")
    fig.tight_layout()
    top_legend(fig, axes)
    save(fig, folder, name)

def fig_stop_reasons(rows):
    # How SCP+MPPI's steps ended at each margin_eps, on each task's main world model
    cert = "certified_tol" if "certified_tol" in rows[0] else "certified"
    if cert == "certified":
        print("  (summary.csv has no certified_tol: re-run analyse_results.py to certify at SCP's 1e-4 tolerance)")
    sweeps = [sweep_of(rows, task, MAIN_BOUND).get("SCP+MPPI", []) for task in (1, 2)]
    heights = {}
    for key, _, _ in STOPS:
        if key == "converged_uncertified":
            heights[key] = [np.array([max(0.0, float(r["converged"]) - float(r[cert])) for _, r in s]) for s in sweeps]
        else:
            heights[key] = [np.array([float(r[cert if key == "certified" else key]) for _, r in s]) for s in sweeps]
    shown = [(k, l, c) for k, l, c in STOPS if any(np.any(h > 1e-9) for h in heights[k])]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.4), sharey=True)
    for i, (ax, task, sweep) in enumerate(zip(axes, (1, 2), sweeps)):
        x, bottom = np.arange(len(sweep)), np.zeros(len(sweep))
        for key, label, colour in shown:
            ax.bar(x, heights[key][i], bottom=bottom, color=colour, width=0.7, label=label)
            bottom += heights[key][i]
        ax.set_xticks(x, [f"{e:g}" for e, _ in sweep])
        ax.set_xlabel(r"margin $\epsilon$")
        ax.set_title(f"PointGoal{task}")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("share of SCP+MPPI's steps")
    fig.tight_layout()
    top_legend(fig, axes, ncol=min(len(shown), 3))
    save(fig, "main_text", "fig_scp_stop_reasons")

def fig_horizon(rows, series, folder, name):
    # Violation rate and return against the planning horizon (PointGoal HORIZON_TASK, bound MAIN_BOUND, margin_eps
    # HORIZON_EPS), each time-matched sampler at its horizon's budget
    pts = horizon_points(rows)
    if len(pts) < 2:
        return skip(name, "no horizon tests in the results")
    Hs = sorted(pts)
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.5))
    for i, s in enumerate(series):
        got = [(H, pts[H][run_name(s, H)]) for H in Hs if run_name(s, H) in pts[H]]
        if not got:
            continue
        label, colour, ls, marker, filled = style(run_name(s, Hs[0]))
        x = np.array([H for H, _ in got], float) + (i - (len(series) - 1) / 2) * 0.35
        for ax, key in zip(axes, ("rate", "return")):
            y, lo, hi = (np.array([float(r[k]) for _, r in got]) for k in (key, f"{key}_lo", f"{key}_hi"))
            ax.errorbar(x, y, yerr=[y - lo, hi - y], color=colour, ls=ls, marker=marker,
                        mfc=colour if filled else "white", label=label)
    for ax, ylabel in zip(axes, ("violation rate", "episode return")):
        ax.set_xticks(Hs)
        ax.set_xlabel("planning horizon $N$")
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    top_legend(fig, axes, ncol=min(len(series), 4))
    save(fig, folder, name)

def fig_compute(rows):
    # Return against plan time per step at each horizon in COMPUTE_HORIZONS: each sampler at every budget it ran
    # (joined in budget order; each budget labelled once along the bottom, as both samplers take the same time for
    # it), SCP+MPPI and SCP as single points
    pts = horizon_points(rows)
    Hs = [H for H in COMPUTE_HORIZONS if H in pts]
    if not Hs:
        return skip("fig_return_vs_compute", "no horizon tests in the results")
    fig, axes = plt.subplots(1, len(Hs), figsize=(TEXT_WIDTH, 2.6), sharey=True, squeeze=False)
    for ax, H in zip(axes[0], Hs):
        times = {}
        for sampler in ("MPPI", "CEM"):
            runs = sorted((pl for pl in pts[H] if pl.partition("@")[0] == sampler), key=budget_of)
            if not runs:
                continue
            _, colour, _, marker, _ = style(f"{sampler}@65536")
            x = np.array([float(pts[H][pl]["plan_time_mean"]) for pl in runs])
            y, lo, hi = (np.array([float(pts[H][pl][k]) for pl in runs]) for k in ("return", "return_lo", "return_hi"))
            ax.errorbar(x, y, yerr=[y - lo, hi - y], color=colour, marker=marker, label=sampler)
            for pl, xi in zip(runs, x):
                times.setdefault(budget_label(pl), []).append(xi)
        below = blended_transform_factory(ax.transData, ax.transAxes)
        for label, xs in times.items():
            ax.text(np.exp(np.mean(np.log(xs))), 0.02, label, transform=below, ha="center", va="bottom", fontsize=7,
                    color="0.35")
        for pl in ("SCP+MPPI", "SCP"):
            if pl in pts[H]:
                label, colour, _, marker, _ = STYLE[pl]
                r = pts[H][pl]
                y, lo, hi = (float(r[k]) for k in ("return", "return_lo", "return_hi"))
                ax.errorbar([float(r["plan_time_mean"])], [y], yerr=[[y - lo], [hi - y]], color=colour, marker=marker,
                            ms=6, ls="none", label=label)
        ax.set_xscale("log")
        ax.set_xlabel("plan time per step (s)")
        ax.set_title(f"$N={H}$")
        lo_y, hi_y = ax.get_ylim()
        ax.set_ylim(lo_y - 0.12 * (hi_y - lo_y), hi_y) # room for the budget labels
    axes[0][0].set_ylabel("episode return")
    fig.tight_layout()
    top_legend(fig, axes)
    save(fig, "main_text", "fig_return_vs_compute")

def fig_latent_safety(rows):
    # How often a violation followed within one horizon of a plan that met the barrier in the model, and of one that
    # did not, pooled over every horizon-10 world model and margin_eps >= POOL_MIN_EPS, with 95% Wilson intervals
    if "steps_safe" not in rows[0]:
        return skip("fig_latent_safety", "summary.csv has no latent-safety counts; re-run analyse_results.py")
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.3))
    for i, pl in enumerate(MAIN_TEXT):
        sel = [r for r in rows if r["planner"] == pl and r["tag"].startswith("e1/") and int(r["horizon"]) == 10
               and float(r["eps"]) >= POOL_MIN_EPS and r["steps_safe"] != ""]
        _, colour, _, _, _ = STYLE[pl]
        for kind, dx, marker, mfc, label in (("safe", -0.1, "o", colour, "the plan met the barrier in the model"),
                                             ("unsafe", 0.1, "X", "white", "it did not")):
            k = sum(int(r[f"steps_{kind}_viol_soon"]) for r in sel)
            n = sum(int(r[f"steps_{kind}"]) for r in sel)
            if n == 0:
                continue
            p, (lo, hi) = max(k / n, 1e-5), wilson(k, n)
            ax.errorbar(i + dx, p, yerr=[[p - max(lo, 1e-5)], [hi - p]], color=colour, marker=marker, mfc=mfc, ms=6,
                        ls="none")
    ax.set_yscale("log")
    ax.set_xticks(range(len(MAIN_TEXT)), [STYLE[pl][0] for pl in MAIN_TEXT])
    ax.set_xlim(-0.5, len(MAIN_TEXT) - 0.5)
    ax.set_ylabel("share of steps followed by a\nviolation within one horizon")
    fig.tight_layout()
    fig.legend([Line2D([], [], color="0.3", marker="o", ls="none"),
                Line2D([], [], color="0.3", marker="X", mfc="white", ls="none")],
               ["after a plan that met the barrier in the model", "after one that did not"], loc="lower center",
               bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=2)
    save(fig, "main_text", "fig_latent_safety")

def fig_model_optimism(rows, episodes):
    # Model optimism, each executed plan's predicted return (discounted predicted rewards plus the value at its end,
    # which carries the goal bonus) minus the discounted return received from that step, for each planner and budget:
    # at horizon 10 over every world model (margin_eps >= POOL_MIN_EPS), at the longer horizons on the horizon tests'
    # models; mean over episodes with a 95% bootstrap interval
    if "optimism" not in episodes[0]:
        return skip("fig_model_optimism", "episodes.csv has no optimism; re-run analyse_results.py")
    rng = np.random.default_rng(0)
    pts = horizon_points(rows)
    panels = [("$N=10$, every world model", pooled_episodes(episodes))]
    for H in sorted(h for h in pts if h != 10):
        dirs = {r["data_dir"] for r in pts[H].values()}
        panels.append((f"$N={H}$", [r for r in episodes if r["data_dir"] in dirs and r["tag"] == f"e4/H_{H}"]))
    order = lambda pl: (["SCP+MPPI", "SCP", "MPPI", "CEM"].index(pl.partition("@")[0]), budget_of(pl))
    planners = [sorted({r["planner"] for r in eps if r["planner"].partition("@")[0] in
                        ("SCP+MPPI", "SCP", "MPPI", "CEM")}, key=order) for _, eps in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(TEXT_WIDTH, 2.6), sharey=True, squeeze=False,
                             gridspec_kw={"width_ratios": [max(len(p), 1) for p in planners]})
    for ax, (title, eps), shown in zip(axes[0], panels, planners):
        for i, pl in enumerate(shown):
            x = np.array([float(r["optimism"]) for r in eps if r["planner"] == pl and r["optimism"] not in ("", "nan")])
            x = x[np.isfinite(x)]
            if len(x) == 0:
                continue
            _, colour, _, marker, filled = style(pl)
            lo, hi = boot_ci(x, rng)
            ax.errorbar(i, x.mean(), yerr=[[x.mean() - lo], [hi - x.mean()]], color=colour, marker=marker,
                        mfc=colour if filled else "white", ms=6, ls="none")
        ax.axhline(0, color="0.6", lw=0.8, zorder=0)
        ax.set_xticks(range(len(shown)), ["SCP+\nMPPI" if pl == "SCP+MPPI" else pl if pl == "SCP" else
                                          f"{pl.partition('@')[0]}\n{budget_label(pl)}" for pl in shown], fontsize=7)
        ax.set_xlim(-0.6, len(shown) - 0.4)
        ax.set_title(title)
    axes[0][0].set_ylabel("predicted minus received\ndiscounted return")
    fig.tight_layout()
    save(fig, "ablation", "fig_model_optimism")

def fig_crash_types(episodes):
    # Each planner's crashes per 100 episodes, split into ones the model warned of (an executed plan in the last
    # horizon failed the barrier) and ones it did not, over every horizon-10 world model (margin_eps >= POOL_MIN_EPS)
    if "crash_unwarned" not in episodes[0]:
        return skip("fig_crash_types", "episodes.csv has no crash_unwarned; re-run analyse_results.py")
    eps = pooled_episodes(episodes)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.4))
    for i, pl in enumerate(EVERY):
        sel = [r for r in eps if r["planner"] == pl]
        if not sel:
            continue
        crashes = sum(r["violated"] == "True" for r in sel)
        unwarned = sum(r["crash_unwarned"] == "True" for r in sel)
        _, colour, _, _, _ = STYLE[pl]
        ax.bar(i, 100 * (crashes - unwarned) / len(sel), color=colour, alpha=0.45, width=0.7)
        ax.bar(i, 100 * unwarned / len(sel), bottom=100 * (crashes - unwarned) / len(sel), color=colour, width=0.7)
        ax.annotate(f"{unwarned}", (i, 100 * crashes / len(sel)), textcoords="offset points", xytext=(0, 2),
                    ha="center", fontsize=7)
    ax.set_xticks(range(len(EVERY)), [STYLE[pl][0].replace(" (", "\n(") for pl in EVERY], fontsize=7)
    ax.set_ylabel("crashes per 100 episodes")
    fig.tight_layout()
    fig.legend([Patch(color="0.3", alpha=0.45), Patch(color="0.3")],
               ["warned: a recent plan failed the barrier", "not warned: the model was wrong (count above)"],
               loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=2)
    save(fig, "ablation", "fig_crash_types")

def fig_lipschitz_convergence(rows):
    # SCP+MPPI's share of steps that stopped at the iteration cap, against margin_eps, for each Lipschitz bound
    bounds = sorted({float(r["bound"]) for r in rows if r["tag"].startswith("e1/") and int(r["horizon"]) == 10})
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.4), sharey=True)
    for ax, task in zip(axes, (1, 2)):
        for b in bounds:
            sweep = sweep_of(rows, task, b).get("SCP+MPPI", [])
            if sweep:
                ax.plot([e for e, _ in sweep], [float(r["max_iters"]) for _, r in sweep], color=bound_colour(b),
                        marker="o", label=bound_label(b))
        ax.set_xticks(EPS_TICKS)
        ax.set_xlabel(r"margin $\epsilon$")
        ax.set_title(f"PointGoal{task}")
    axes[0].set_ylabel("share of SCP+MPPI's steps\nat the iteration cap")
    fig.tight_layout()
    top_legend(fig, axes)
    save(fig, "appendix", "fig_lipschitz_convergence")

def fig_grid(rows, key, ylabel, name):
    # Every planner against margin_eps on every horizon-10 world model: tasks in rows, Lipschitz bounds in columns
    bounds = sorted({float(r["bound"]) for r in rows if r["tag"].startswith("e1/") and int(r["horizon"]) == 10})
    fig, axes = plt.subplots(2, len(bounds), figsize=(TEXT_WIDTH, 3.9), sharex=True, sharey=True, squeeze=False)
    for i, task in enumerate((1, 2)):
        for j, b in enumerate(bounds):
            plot_sweep(axes[i][j], sweep_of(rows, task, b), EVERY, key)
            axes[i][j].set_title(f"PointGoal{task}, {bound_label(b)}")
            if i == 1:
                axes[i][j].set_xlabel(r"margin $\epsilon$")
        axes[i][0].set_ylabel(ylabel)
    fig.tight_layout()
    top_legend(fig, axes, ncol=3)
    save(fig, "appendix", name)

def fig_survival(rows, episodes):
    # Episodes without a violation by each step, at each margin_eps, on one main world model. Episodes end early only
    # at a violation, so this is the Kaplan-Meier curve without censoring
    data_dir = next(r["data_dir"] for r in rows if r["tag"].startswith("e1/") and int(r["task"]) == SURVIVAL_TASK
                    and int(r["horizon"]) == 10 and np.isclose(float(r["bound"]), MAIN_BOUND))
    eps = sorted({float(r["eps"]) for r in episodes if r["data_dir"] == data_dir and r["tag"].startswith("e1/")})
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH, 3.9), sharex=True, sharey=True)
    t_max = max(int(r["steps"]) for r in episodes if r["data_dir"] == data_dir)
    t = np.arange(t_max + 1)
    for ax, e in zip(axes.flat, eps):
        for pl in SURVIVAL:
            ep = [r for r in episodes if r["data_dir"] == data_dir and r["tag"].startswith("e1/")
                  and np.isclose(float(r["eps"]), e) and r["planner"] == pl]
            if not ep:
                continue
            steps = np.array([int(r["steps"]) for r in ep])
            viol = np.array([r["violated"] == "True" for r in ep])
            label, colour, ls, _, _ = STYLE[pl]
            ax.step(t, 1 - np.mean(viol[None] & (steps[None] <= t[:, None]), axis=1), where="post", color=colour,
                    ls=ls, label=label)
        ax.set_title(rf"$\epsilon={e:g}$")
        ax.set_ylim(0, 1.02)
    for i, ax in enumerate(axes.flat[len(eps):], len(eps)):
        ax.axis("off")
        above = axes.flat[i - axes.shape[1]] # the panel above an empty cell keeps its step axis
        above.tick_params(labelbottom=True)
        above.set_xlabel("step")
    for ax in axes[:, 0]:
        ax.set_ylabel("episodes without a violation")
    for ax in axes[-1]:
        ax.set_xlabel("step")
    fig.tight_layout()
    top_legend(fig, axes, ncol=4)
    save(fig, "appendix", f"fig_survival_pg{SURVIVAL_TASK}_c{MAIN_BOUND:g}".replace(".", "_"))

def fig_trade_off(rows, episodes):
    # Each planner's violation rate and return pooled over margin_eps >= POOL_MIN_EPS on each task's main world model,
    # with 95% intervals (Wilson, bootstrap); arrows lead from each sampler at 512 sequences to the same sampler
    # time-matched: what more optimisation buys
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.6))
    for ax, task in zip(axes, (1, 2)):
        data_dir = next(r["data_dir"] for r in rows if r["tag"].startswith("e1/") and int(r["task"]) == task
                        and int(r["horizon"]) == 10 and np.isclose(float(r["bound"]), MAIN_BOUND))
        points = {}
        for pl in TRADE_OFF:
            ep = [r for r in episodes if r["data_dir"] == data_dir and r["tag"].startswith("e1/")
                  and float(r["eps"]) >= POOL_MIN_EPS and r["planner"] == pl]
            if not ep:
                continue
            k, ret = sum(r["violated"] == "True" for r in ep), np.array([float(r["return"]) for r in ep])
            (xlo, xhi), x, y = wilson(k, len(ep)), k / len(ep), ret.mean()
            ylo, yhi = boot_ci(ret, rng)
            label, colour, _, marker, filled = STYLE[pl]
            ax.errorbar(x, y, xerr=[[x - xlo], [xhi - x]], yerr=[[y - ylo], [yhi - y]], color=colour, marker=marker,
                        mfc=colour if filled else "white", ms=6, zorder=3, label=label)
            points[pl] = (x, y)
        for small, big in (("MPPI", "MPPI@65536"), ("CEM", "CEM@65536")):
            if small in points and big in points:
                ax.annotate("", points[big], points[small], zorder=2,
                            arrowprops={"arrowstyle": "->", "color": "0.5", "lw": 0.8, "shrinkA": 5, "shrinkB": 5})
        ax.set_title(f"PointGoal{task}")
        ax.set_xlabel(rf"violation rate ($\epsilon \geq {POOL_MIN_EPS:g}$)")
    axes[0].set_ylabel("episode return")
    fig.tight_layout()
    top_legend(fig, axes, ncol=3)
    save(fig, "ablation", "fig_return_vs_violations")

# Tables (LaTeX, booktabs), written beside the figures so the paper's numbers come from the same outputs

def latex_p(p):
    if not np.isfinite(p):
        return "--"
    if p >= 0.1:
        return f"{p:.2f}"
    if p >= 1e-3:
        return f"{p:.2g}"
    e = int(np.floor(np.log10(p)))
    m = round(p / 10 ** e)
    m, e = (1, e + 1) if m == 10 else (m, e)
    return f"${m}\\times 10^{{{e}}}$"

def latex_prob(p):
    # A probability, with the ends shown as bounds rather than as certainty
    if not np.isfinite(p):
        return "--"
    return "$>0.999$" if p > 0.999 else "$<0.001$" if p < 0.001 else f"{p:.3f}" if p > 0.99 or p < 0.01 else f"{p:.2f}"

def latex_ci(x, lo, hi, scale=1.0, digits=2, sign=False):
    f = (lambda v: f"${v * scale:+.{digits}f}$") if sign else (lambda v: f"${v * scale:.{digits}f}$")
    return f"{f(x)} [{f(lo)}, {f(hi)}]"

def write_table(folder, name, lines):
    (OUT / folder).mkdir(parents=True, exist_ok=True)
    (OUT / folder / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print(f"  {folder}/{name}.tex")

def table_pooled(bound_rows, bound):
    # SCP+MPPI minus each planner on the horizon-10 world models with one Lipschitz bound, pooled over both tasks and
    # margin_eps >= POOL_MIN_EPS, paired on task, margin_eps and seed, with intervals that resample whole seeds
    # (by_bound.csv). One table per bound: the main bound's in the main text, the others in the appendix
    # P(safer) and P(higher return): the probability, given the data, that SCP+MPPI's violation rate is lower and its
    # return higher than the baseline's on these models (Bayesian bootstrap over seeds)
    names = {"SCP": "SCP", "MPPI": "MPPI (512)", "CEM": "CEM (512)", "MPPI@65536": "MPPI (time-matched)",
             "CEM@65536": "CEM (time-matched)"}
    by = {r["other"]: r for r in bound_rows if np.isclose(float(r["bound"]), bound)}
    lines = [r"\begin{tabular}{lrcccc}", r"\toprule",
             r"Baseline & Pairs & $\Delta$ violations (pp) & $P$(safer) & $\Delta$ return & $P$(higher return) \\",
             r"\midrule"]
    for key in POOLED_ROWS:
        if key not in by:
            continue
        lines.append(f"{names.get(key, key)} & {comparison_cells(by[key])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("main_text" if np.isclose(bound, MAIN_BOUND) else "appendix",
                f"table_pooled_c{bound:g}".replace(".", "_"), lines)

def comparison_cells(row):
    # One comparison's cells, shared by the per-bound and horizon tables: pairs, the violation difference (points) and
    # the return difference with their 95% intervals, and the probabilities that SCP+MPPI is safer and earns more
    num = lambda k: float(row[k]) if row.get(k, "") not in ("", "nan") else np.nan
    return (f"{int(num('pairs'))} & "
            f"{latex_ci(num('viol_diff'), num('viol_diff_lo'), num('viol_diff_hi'), 100, 1, True)} & "
            f"{latex_prob(num('p_safer'))} & "
            f"{latex_ci(num('return_diff'), num('return_diff_lo'), num('return_diff_hi'), 1, 2, True)} & "
            f"{latex_prob(num('p_higher_return'))}")

def baseline_label(pl, H):
    # A baseline's name in the horizon tables: each sampler with its budget, "time-matched" at its horizon's
    base, _, budget = pl.partition("@")
    if base not in ("MPPI", "CEM"):
        return pl
    J = int(budget) if budget.isdigit() else SAMPLE_BUDGET
    return f"{base} (time-matched)" if J == TIME_MATCHED.get(H) else f"{base} ({J:,})"

def table_horizon(horizon_rows, series, folder, name):
    # SCP+MPPI minus each baseline at each planning horizon, on the horizon tests' model and margin (horizon.csv), the
    # same columns as the per-bound tables. series: the baselines to list (None: every planner run at that horizon)
    rows = [r for r in horizon_rows if int(r["task"]) == HORIZON_TASK and np.isclose(float(r["bound"]), MAIN_BOUND)
            and np.isclose(float(r["eps"]), HORIZON_EPS)]
    def order(pl):
        base, _, budget = pl.partition("@")
        return (["SCP", "MPPI", "CEM"].index(base) if base in ("SCP", "MPPI", "CEM") else 3,
                int(budget) if budget.isdigit() else SAMPLE_BUDGET)
    lines = [r"\begin{tabular}{@{}rlrcccc@{}}", r"\toprule",
             r"$N$ & Baseline & Pairs & $\Delta$ violations (pp) & $P$(safer) & $\Delta$ return & $P$(higher return) \\",
             r"\midrule"]
    first = True
    for H in sorted({int(r["horizon"]) for r in rows}):
        by = {r["other"]: r for r in rows if int(r["horizon"]) == H}
        keys = [run_name(sr, H) for sr in series] if series else sorted(by, key=order)
        keys = [k for k in keys if k in by]
        if not keys:
            continue
        if not first:
            lines.append(r"\midrule")
        first = False
        for j, key in enumerate(keys):
            lines.append(f"{H if j == 0 else ''} & {baseline_label(key, H)} & {comparison_cells(by[key])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(folder, name, lines)

def table_per_model(headline):
    # Every horizon-10 world model, each planner pooled over margin_eps >= POOL_MIN_EPS (headline.csv): violation rate
    # (Wilson) and return (bootstrap) with 95% intervals, and SCP+MPPI's paired McNemar p against it (Holm over models)
    lines = [r"\begin{tabular}{llrccc}", r"\toprule",
             r"Task, $c$ & Planner & Episodes & Violation rate & Return & $p$ vs SCP+MPPI \\", r"\midrule"]
    groups = {}
    for r in headline:
        groups.setdefault((int(r["task"]), float(r["bound"])), []).append(r)
    for i, ((task, b), rs) in enumerate(sorted(groups.items())):
        if i:
            lines.append(r"\midrule")
        rs = sorted(rs, key=lambda r: EVERY.index(r["planner"]) if r["planner"] in EVERY else len(EVERY))
        for j, r in enumerate(rs):
            f = {k: float(r[k]) if r.get(k, "") not in ("", "nan") else np.nan for k in
                 ("episodes", "rate", "rate_lo", "rate_hi", "return", "return_lo", "return_hi",
                  "p_mcnemar_vs_main_holm")}
            label = STYLE[r["planner"]][0] if r["planner"] in STYLE else r["planner"]
            head = f"PG{task}, {bound_label(b)}" if j == 0 else ""
            lines.append(f"{head} & {label} & {int(f['episodes'])} & "
                         f"{latex_ci(f['rate'], f['rate_lo'], f['rate_hi'])} & "
                         f"{latex_ci(f['return'], f['return_lo'], f['return_hi'])} & "
                         f"{latex_p(f['p_mcnemar_vs_main_holm']) if r['planner'] != 'SCP+MPPI' else ''} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("appendix", "table_per_model", lines)

def main():
    rows, episodes = load("summary.csv"), load("episodes.csv")
    caps = {r.get("seed_cap") for r in rows}
    if caps != {str(SEED_CAP)}:
        made = "before SEED_CAP existed" if None in caps else f"with SEED_CAP {', '.join(sorted(caps))}"
        raise SystemExit(f"{ANALYSIS} was made {made}, not {SEED_CAP}: set SEED_CAP in analyse_results.py and here to "
                         f"the same value and re-run analyse_results.py")
    print(f"Paper figures from {ANALYSIS} (seeds 0-{SEED_CAP - 1}), written to {OUT}:")
    fig_sweep(rows, MAIN_TEXT, "main_text", "fig_eps_sweep_c1_5")
    fig_stop_reasons(rows)
    fig_horizon(rows, HORIZON_MAIN, "main_text", "fig_horizon")
    fig_compute(rows)
    fig_latent_safety(rows)
    fig_sweep(rows, ABLATION, "ablation", "fig_sample_vs_time_matched")
    fig_horizon(rows, HORIZON_ALL, "ablation", "fig_horizon_all_budgets")
    fig_trade_off(rows, episodes)
    fig_model_optimism(rows, episodes)
    fig_crash_types(episodes)
    fig_lipschitz_convergence(rows)
    fig_grid(rows, "rate", "violation rate", "fig_eps_sweep_all_bounds_violations")
    fig_grid(rows, "return", "episode return", "fig_eps_sweep_all_bounds_return")
    fig_survival(rows, episodes)
    if (ANALYSIS / "by_bound.csv").exists():
        bound_rows = load("by_bound.csv")
        for bound in sorted({float(r["bound"]) for r in bound_rows}):
            table_pooled(bound_rows, bound)
    else:
        skip("pooled tables", "no by_bound.csv; re-run analyse_results.py")
    if (ANALYSIS / "horizon.csv").exists():
        horizon_rows = load("horizon.csv")
        table_horizon(horizon_rows, HORIZON_TABLE_ROWS, "main_text", "table_horizon")
        table_horizon(horizon_rows, None, "appendix", "table_horizon_all_budgets")
    else:
        skip("horizon tables", "no horizon.csv; re-run analyse_results.py")
    if (ANALYSIS / "headline.csv").exists():
        table_per_model(load("headline.csv"))
    else:
        skip("per-model table", "no headline.csv; re-run analyse_results.py")

if __name__ == "__main__":
    main()