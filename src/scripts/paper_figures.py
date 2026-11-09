import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

# The paper's figures, drawn from analyse_results.py's outputs (run that first): python paper_figures.py
# Saved as PDF (for the paper) and PNG (to preview) in OUT, one folder per place in the paper
ANALYSIS = ROOT / "src/save_data/experiments/analysis_results"
OUT = ROOT / "paper_graphs"
TEXT_WIDTH = 6.0 # inches, the paper's text width: every figure spans it
MAIN_BOUND = 1.5 # the main world models' Lipschitz bound
SURVIVAL_TASK = 1 # the main model whose survival curves go in the appendix
EPS_TICKS = [0.0, 0.1, 0.2, 0.3, 0.5]

# Each planner as it appears in the paper: label, colour, line style, marker, filled marker. The time-matched samplers
# are the main-text baselines (solid); the 512-sequence samplers appear only in the ablation and the appendix (dashed)
STYLE = {"SCP+MPPI": ("SCP+MPPI", "tab:purple", "-", "o", True),
         "SCP": ("SCP", "tab:blue", "-", "s", True),
         "MPPI@65536": ("MPPI (time-matched)", "tab:orange", "-", "^", True),
         "CEM@65536": ("CEM (time-matched)", "tab:green", "-", "v", True),
         "MPPI": ("MPPI (512 samples)", "tab:orange", "--", "^", False),
         "CEM": ("CEM (512 samples)", "tab:green", "--", "v", False)}
MAIN_TEXT = ["SCP+MPPI", "SCP", "MPPI@65536", "CEM@65536"]
ABLATION = ["SCP+MPPI", "MPPI", "CEM", "MPPI@65536", "CEM@65536"]
EVERY = ["SCP+MPPI", "SCP", "MPPI", "MPPI@65536", "CEM", "CEM@65536"]
TRADE_OFF = ["SCP+MPPI", "MPPI", "MPPI@65536", "CEM", "CEM@65536"]
POOL_MIN_EPS = 0.1 # the trade-off figure pools the margin_eps sweep from here up, as the report does
# How SCP+MPPI's steps ended: (summary column, label, colour). Certified is certified_tol where the summary has it
# (the rollout meets the barrier to SCP's convergence tolerance, 1e-4), else the logged certified (no tolerance);
# "converged_uncertified" is converged minus certified
STOPS = [("certified", "Certified: converged, rollout meets the barrier", "tab:green"),
         ("converged_uncertified", "Converged, rollout misses the barrier", "yellowgreen"),
         ("converged_infeasible", "Converged with slack: no safe plan nearby", "tab:orange"),
         ("max_iters", "Iteration cap", "tab:red")]

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9,
                     "axes.titlesize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "lines.linewidth": 1.2, "lines.markersize": 4, "errorbar.capsize": 2,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

def load(name):
    with open(ANALYSIS / name) as f:
        return list(csv.DictReader(f))

def sweep_of(rows, task, bound):
    # One horizon-10 world model's margin_eps sweep: {planner: [(eps, summary row)] in eps order}
    out = {}
    for r in rows:
        if r["tag"].startswith("e1/") and int(r["task"]) == task and int(r["horizon"]) == 10 \
                and np.isclose(float(r["bound"]), bound):
            out.setdefault(r["planner"], []).append((float(r["eps"]), r))
    return {pl: sorted(v, key=lambda x: x[0]) for pl, v in out.items()}

def bound_label(b):
    return r"$c=\infty$" if np.isinf(b) else f"$c={b:g}$"

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
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.4), sharey=True)
    for ax, task in zip(axes, (1, 2)):
        sweep = sweep_of(rows, task, MAIN_BOUND).get("SCP+MPPI", [])
        x, bottom = np.arange(len(sweep)), np.zeros(len(sweep))
        for key, label, colour in STOPS:
            if key == "converged_uncertified":
                h = np.array([max(0.0, float(r["converged"]) - float(r[cert])) for _, r in sweep])
            else:
                h = np.array([float(r[cert if key == "certified" else key]) for _, r in sweep])
            ax.bar(x, h, bottom=bottom, color=colour, width=0.7, label=label)
            bottom += h
        ax.set_xticks(x, [f"{e:g}" for e, _ in sweep])
        ax.set_xlabel(r"margin $\epsilon$")
        ax.set_title(f"PointGoal{task}")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("share of SCP+MPPI's steps")
    fig.tight_layout()
    top_legend(fig, axes, ncol=2)
    save(fig, "main_text", "fig_scp_stop_reasons")

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
    eps = [e for e in sorted({float(r["eps"]) for r in episodes if r["data_dir"] == data_dir
                              and r["tag"].startswith("e1/")})]
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH, 3.9), sharex=True, sharey=True)
    t_max = max(int(r["steps"]) for r in episodes if r["data_dir"] == data_dir)
    t = np.arange(t_max + 1)
    for ax, e in zip(axes.flat, eps):
        for pl in EVERY:
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
    top_legend(fig, axes, ncol=3)
    save(fig, "appendix", f"fig_survival_pg{SURVIVAL_TASK}_c{MAIN_BOUND:g}".replace(".", "_"))

def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z ** 2 / n
    centre, half = (p + z ** 2 / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return max(0.0, centre - half), min(1.0, centre + half)

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
            ylo, yhi = np.percentile(rng.choice(ret, (10_000, len(ret))).mean(axis=1), [2.5, 97.5])
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

def main():
    rows, episodes = load("summary.csv"), load("episodes.csv")
    print(f"Paper figures from {ANALYSIS}, written to {OUT}:")
    fig_sweep(rows, MAIN_TEXT, "main_text", "fig_eps_sweep_c1_5")
    fig_stop_reasons(rows)
    fig_sweep(rows, ABLATION, "ablation", "fig_sample_vs_time_matched")
    fig_trade_off(rows, episodes)
    fig_grid(rows, "rate", "violation rate", "fig_eps_sweep_all_bounds_violations")
    fig_grid(rows, "return", "episode return", "fig_eps_sweep_all_bounds_return")
    fig_survival(rows, episodes)

if __name__ == "__main__":
    main()
