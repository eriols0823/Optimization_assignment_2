"""Question 2: validation, base case and sensitivity of the pricing problem DP.

    python main.py --q2 validate      DP^cap against the central dispatch C (equivalence with the duals of C)
    python main.py --q2 steps         the three steps cap -> + budget balance -> + individual rationality, with the
                                      ways out of infeasibility (soft IR, price groups); CSV and figures in results/
    python main.py --q2 indifference  the member-hours where the best response at lambda^C is not unique, and the
                                      import if every member broke its ties against the manager
    python main.py --q2 sensitivity   the three steps along the tightness of the cap (CSV and figure in results/)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import CommunityModel
from src.pricing import PricingModel, best_response, pessimistic_import


def _check(ok: bool, text: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}")
    return bool(ok)


def validate(com, time_limit: float = 300) -> bool:
    """Validation checks of DP^cap. Returns True if all pass."""
    C = CommunityModel(com, "C").solve()
    ok = []
    print("1. DP^cap (no warm start, no welfare bound) reaches the welfare of C")
    dp = PricingModel(com, "cap", warm_start=False, welfare_bound=False, time_limit=time_limit)
    r = dp.solve()
    print(f"     size: {dp.size()}, solved in {r.runtime:.1f} s, status {r.status}")
    ok.append(_check(r.feasible and abs(r.objective - C.cost["total"]) < 1e-4,
                     f"J^DP = {r.objective:.4f} DKK, J^C = {C.cost['total']:.4f} DKK"))
    print("2. The reformulation is exact at the solution")
    ok.append(_check(r.checks["strong duality residual [DKK]"] < 1e-6,
                     f"strong duality residual {r.checks['strong duality residual [DKK]']:.1e} DKK"))
    ok.append(_check(r.checks["complementarity residual"] < 1e-6 and r.checks["pairs at big-M"] == 0,
                     f"complementarity residual {r.checks['complementarity residual']:.1e}, "
                     f"{r.checks['pairs at big-M']} pairs at a generic big-M"))
    gaps = dp.verify_best_responses(r)
    ok.append(_check(max(abs(v) for v in gaps.values()) < 1e-5,
                     f"every anticipated schedule is optimal in the member's own LP (largest gap "
                     f"{max(abs(v) for v in gaps.values()):.1e} DKK)"))
    print("3. The dual prices of C are optimal prices of DP^cap")
    gaps_C = {m.consumer_id: C.member_cost[m.consumer_id] - best_response(m, C.price)["cost"] for m in com.members}
    ok.append(_check(max(abs(v) for v in gaps_C.values()) < 1e-5,
                     "at x = lambda^C the schedule of C is a best response of every member (largest gap "
                     f"{max(abs(v) for v in gaps_C.values()):.1e} DKK)"))
    print(f"     prices are not unique: budget {r.surplus:.2f} DKK at the MILP's prices, "
          f"{C.surplus:.2f} DKK at lambda^C, same welfare")
    print(f"\n{sum(ok)}/{len(ok)} checks passed")
    return all(ok)


# ----------------------------------------------------------------------------------------------------- metrics
# The metrics of Question 2, with their reference and point of view (report, Table tab:DP_metrics):
#   loss vs C [DKK]        J^DP - J^C                      community, reference C, + = efficiency lost to a requirement
#   share of V^C [%]       (J^R - J^DP) / (J^R - J^C)      community, references R and C, 100 % = all the value of C
#   above cap X [kWh]      sum_t max(0, P_imp_t - cap_t)   DSO, reference 0
#   budget B [DKK]         sum_i Pi_i - D                  manager, reference 0 (balanced), + = surplus kept
#   worse off than R / I   members with J_i^DP > J_i^R / J_i^I                              members
#   IR violation [DKK]     sum_i s_i (soft variants)       members, reference 0: what would have to be given up
#   blocking members       members with s_i > 0
def _metrics(r, dp, R, I, C) -> dict:
    """One row of results for a solved variant r of DP (only the status if no solution was found)."""
    row = {"status": r.status, "time [s]": round(r.runtime, 1), "gap": r.gap}
    if not r.feasible:
        return row
    worse_R = [c for c in r.member_cost if r.member_cost[c] > R.member_cost[c] + 1e-4]
    worse_I = [c for c in r.member_cost if r.member_cost[c] > I.member_cost[c] + 1e-4]
    V_C = R.cost["total"] - C.cost["total"]
    row.update({
        "J [DKK]": r.cost["total"],
        "loss vs C [DKK]": r.cost["total"] - C.cost["total"],
        "share of V^C [%]": 100 * (R.cost["total"] - r.cost["total"]) / V_C if V_C > 1e-9 else None,  # only if C gains
        "above cap X [kWh]": float(np.maximum(0.0, r.P_imp - dp.cap).sum()),
        "budget B [DKK]": r.surplus,
        "worse off than R": len(worse_R), "worse off than I": len(worse_I),
        "IR violation [DKK]": sum(r.ir_slack.values()) if r.ir_slack else (0.0 if dp.requirements == "bb_ir" else None),
        "blocking members": " ".join(c for c, v in r.ir_slack.items() if v > 1e-4) if r.ir_slack else "",
    })
    return row


def _member_table(com, results: dict, R, I, C) -> pd.DataFrame:
    """Gain of every member against R (DKK) under I, under the dual settlement of C and in each solved variant."""
    table = pd.DataFrame({"type": {m.consumer_id: m.archetype for m in com.members},
                          "I": {c: R.member_cost[c] - I.member_cost[c] for c in R.member_cost},
                          "C (dual price)": {c: R.member_cost[c] - C.member_cost[c] for c in R.member_cost}})
    for name, r in results.items():
        if r is not None and r.feasible:
            table[name] = {c: R.member_cost[c] - r.member_cost[c] for c in r.member_cost}
    return table


STEP_VARIANTS = {
    "DP^cap": dict(requirements="cap"),
    "DP^BB": dict(requirements="bb"),
    "DP^BB,IR": dict(requirements="bb_ir"),
    "DP^BB,IR,soft": dict(requirements="bb_ir", ir_soft=True),
    "DP^BB,IR,asset,soft": dict(requirements="bb_ir", price_groups="asset", ir_soft=True),
    "DP^BB,IR,member": dict(requirements="bb_ir", price_groups="member"),
}


def steps(com, out_dir: Path, time_limit: float = 300, variants: dict | None = None) -> pd.DataFrame:
    """The three steps of the pricing study on the base case (cap -> + BB -> + IR), and the ways out of infeasibility.
    Saves q2_steps.csv (one row per variant), q2_members.csv (gain of every member), q2_prices.csv (posted prices) and
    the figures q2_prices and q2_members (PNG and PDF) in out_dir."""
    R, I, C = (CommunityModel(com, m).solve() for m in ("R", "I", "C"))
    results, rows, start = {}, [], None
    for name, kw in (variants or STEP_VARIANTS).items():
        print(f"Solving {name} ...", flush=True)
        dp = PricingModel(com, time_limit=time_limit, start_prices=start, **kw)
        r = results[name] = dp.solve()
        if name == "DP^BB" and r.feasible:
            start = r.prices["all"]                      # warm start of the uniform IR variants
        rows.append({"variant": name, **_metrics(r, dp, R, I, C)})
        print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    table = pd.DataFrame(rows).set_index("variant")
    members = _member_table(com, results, R, I, C)
    meter = CommunityModel(com, "C")                       # discounted meter prices of the contract
    prices = pd.DataFrame({"lambda^C": C.price, "p_imp discounted": meter.p_imp, "p_exp discounted": meter.p_exp,
                           **{f"{n} [{g}]": x for n, r in results.items() if r.feasible for g, x in r.prices.items()
                              if g == "all" or "asset" in n}})
    prices.index.name = "hour"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out_dir / "q2_steps.csv")
    members.round(3).to_csv(out_dir / "q2_members.csv")
    prices.round(4).to_csv(out_dir / "q2_prices.csv")
    _plot_prices(prices, C, out_dir)
    _plot_members(com, members, out_dir)
    print("\nThree steps\n", table.round(3).to_string(), "\n\nGain of each member against R [DKK]\n",
          members.round(2).to_string(), sep="")
    print(f"\nSaved to {out_dir}")
    return table


CAP_FACTORS = (1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0)   # f in cap = f E^R / 24 (base 1.3)


def sensitivity(com, out_dir: Path, cap_factors=CAP_FACTORS, time_limit: float = 240, tag: str = "") -> pd.DataFrame:
    """Sensitivity of the three steps to the tightness of the cap (Table 11 c). For every cap factor f: the value and
    the congestion rent of C, then DP^BB, DP^BB,IR,soft and DP^BB,IR,asset,soft (feasibility, efficiency cost, IR
    violation, blocking members). Saves q2_sensitivity<tag>.csv; plot_sensitivity() draws the figure."""
    c = com.contract
    R = CommunityModel(com, "R").solve()
    rows = []
    for f in cap_factors:
        cap = np.asarray(c.capacity_limit_kW, dtype=float) * f / c.capacity_limit_daily_energy_factor
        I = CommunityModel(com, "I", cap=cap).solve()
        C = CommunityModel(com, "C", cap=cap).solve()
        base = {"f": f, "cap [kW]": float(cap[0]), "V^C [DKK]": R.cost["total"] - C.cost["total"],
                "rent of C [DKK]": C.surplus, "capped hours in C": int((C.feeder_import >= cap - 1e-6).sum()),
                "worse off than R in C": sum(C.member_cost[i] > R.member_cost[i] + 1e-4 for i in C.member_cost)}
        start = None
        for name in ("DP^BB", "DP^BB,IR,soft", "DP^BB,IR,asset,soft"):
            print(f"f = {f}: solving {name} ...", flush=True)
            dp = PricingModel(com, cap=cap, time_limit=time_limit, start_prices=start, **STEP_VARIANTS[name])
            r = dp.solve()
            if name == "DP^BB" and r.feasible:
                start = r.prices["all"]
            rows.append({**base, "variant": name, **_metrics(r, dp, R, I, C)})
            print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    table = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out_dir / f"q2_sensitivity{tag}.csv", index=False)
    print(f"\nSaved to {out_dir / f'q2_sensitivity{tag}.csv'}")
    return table


# ----------------------------------------------------------------------------------------------------- figures
INK, BLUE, RED, GREY, LIGHT = "#0b0b0b", "#1c5cab", "#b8322f", "#52514e", "#c9c8c4"


def _save(fig, path: Path) -> None:
    """Save a figure as PNG (for viewing) and PDF (for the report), then close it."""
    import matplotlib.pyplot as plt
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=200)
    plt.close(fig)


def _plot_prices(prices: pd.DataFrame, C, out_dir: Path) -> None:
    """q2_prices: posted prices of the three steps against lambda^C and the discounted meter prices; capped hours shaded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hours = prices.index.values
    fig, ax = plt.subplots(figsize=(7.5, 3.4), layout="constrained")
    for t in hours[C.feeder_import >= C.cap - 1e-6]:
        ax.axvspan(t - 0.5, t + 0.5, color="#efeee9", lw=0)
    ax.step(hours, prices["p_imp discounted"], where="mid", color=GREY, ls="--", lw=1, label=r"$\tilde p^{\mathrm{imp}}_t$")
    ax.step(hours, prices["p_exp discounted"], where="mid", color=GREY, ls=":", lw=1, label=r"$\tilde p^{\mathrm{exp}}_t$")
    ax.step(hours, prices["lambda^C"], where="mid", color=INK, lw=2.2,
            label=r"$\lambda_t$ of $\mathcal{C}$ (optimal for $\mathcal{DP}^{\mathrm{cap}}$)")
    styles = {"DP^BB [all]": (BLUE, (0, (4, 2)), r"$\mathcal{DP}^{\mathrm{BB}}$ (close to $\lambda_t$ except in hours 19–20)"),
              "DP^BB,IR,soft [all]": (RED, "-", r"$\mathcal{DP}^{\mathrm{BB,IR,soft}}$")}
    for col, (color, ls, label) in styles.items():
        if col in prices:
            ax.step(hours, prices[col], where="mid", color=color, ls=ls, lw=1.6, label=label)
    ax.set_xlabel("Hour"); ax.set_ylabel("Price [DKK/kWh]"); ax.set_xlim(-0.5, 23.5)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    _save(fig, out_dir / "q2_prices")


def _plot_members(com, members: pd.DataFrame, out_dir: Path) -> None:
    """q2_members: gain of every member against R under the dual price of C and under the requirements of DP."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    assets = {m.consumer_id: (2 if m.has_pv else 0) + (1 if m.has_battery else 0) for m in com.members}
    slope = {m.consumer_id: m.slope_class for m in com.members}
    order = sorted(members.index, key=lambda c: (assets[c], slope[c]))
    y = np.array([assets[c] * 5 + slope[c] - 1 for c in order])
    fig, ax = plt.subplots(figsize=(6.5, 5.2), layout="constrained")
    ax.axvline(0, color=INK, lw=0.8)
    series = [("C (dual price)", dict(marker="o", ls="none", mfc="white", mec=INK), r"$\mathcal{C}$ with the dual price"),
              ("DP^BB", dict(marker="s", ls="none", color=BLUE), r"$\mathcal{DP}^{\mathrm{BB}}$"),
              ("DP^BB,IR,soft", dict(marker="D", ls="none", color=RED), r"$\mathcal{DP}^{\mathrm{BB,IR,soft}}$, one price"),
              ("DP^BB,IR,asset,soft", dict(marker="^", ls="none", color=GREY),
               r"$\mathcal{DP}^{\mathrm{BB,IR,soft}}$, one price per asset class")]
    for col, style, label in series:
        if col in members:
            ax.plot(members.loc[order, col], y, label=label, markersize=6, **style)
    ax.set_yticks(y, [f"{c} ({members.loc[c, 'type']})" for c in order], fontsize=8)
    ax.invert_yaxis(); ax.grid(axis="x", color="#e1e0dc", lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel(r"Gain against $\mathcal{R}$, $J^{\mathcal{R}}_i-J_i$ [DKK]")
    fig.legend(frameon=False, fontsize=8, loc="outside lower center", ncol=2)
    _save(fig, out_dir / "q2_members")


def plot_sensitivity(table: pd.DataFrame, out_dir: Path) -> None:
    """q2_sensitivity: along the cap factor, the efficiency cost of the requirements (top) and the IR violation with one
    price per hour and with one price per asset class, next to the congestion rent of C (bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(6.5, 4.8), layout="constrained")
    for name, color, marker, label in (("DP^BB", BLUE, "s", r"$\mathcal{DP}^{\mathrm{BB}}$"),
                                       ("DP^BB,IR,soft", RED, "D", r"$\mathcal{DP}^{\mathrm{BB,IR,soft}}$, one price"),
                                       ("DP^BB,IR,asset,soft", GREY, "o",
                                        r"$\mathcal{DP}^{\mathrm{BB,IR,soft}}$, one price per asset class")):
        sub = table[table["variant"] == name].sort_values("f")
        top.plot(sub["f"], sub["loss vs C [DKK]"], marker=marker, color=color, label=label)
        if name != "DP^BB":
            bottom.plot(sub["f"], sub["IR violation [DKK]"], marker=marker, color=color, label=label)
    rent = table.drop_duplicates("f").sort_values("f")
    bottom.plot(rent["f"], rent["rent of C [DKK]"] / 10, color=INK, ls=":", label=r"congestion rent of $\mathcal{C}$ / 10")
    # summing IR over the members under budget balance: sum_i s_i >= J^DP - J^R >= -V^C, for any price scheme
    bottom.plot(rent["f"], np.maximum(0.0, -rent["V^C [DKK]"]), color=INK, ls="--", lw=1,
                label=r"lower bound $\max(0,-V^{\mathcal{C}})$, any prices")
    for ax in (top, bottom):
        ax.axvline(1.3, color=LIGHT, lw=6, zorder=0)
        ax.grid(color="#e1e0dc", lw=0.8); ax.set_axisbelow(True)
        ax.set_xlim(1.05, 2.55)                    # beyond f = 2.5 every curve stays at zero (see the CSV)
    top.set_ylabel(r"Loss against $\mathcal{C}$ [DKK]")
    bottom.set_ylabel("IR violation [DKK]")
    bottom.set_xlabel(r"Cap factor $f$ (base case 1.3, shaded)")
    top.legend(frameon=False, fontsize=8); bottom.legend(frameon=False, fontsize=8)
    _save(fig, out_dir / "q2_sensitivity")


def indifference(com, out_dir: Path) -> pd.DataFrame:
    """Where the best response at lambda^C is a set: per member and hour, the range of its net consumption over its
    optimal responses, and the community import if every member broke its ties against the manager (hour by hour).
    Saves q2_indifference.csv (one row per hour) and q2_indifferent_member_hours.csv."""
    C = CommunityModel(com, "C").solve()
    lo, hi, ranges = pessimistic_import(com, C.price)
    cap = np.asarray(com.contract.capacity_limit_kW, dtype=float)
    net = sum(s["q"] for s in C.schedule.values())
    hours = pd.DataFrame({"lambda^C [DKK/kWh]": C.price, "net import C [kWh]": net,
                          "lowest import [kWh]": lo, "highest import [kWh]": hi, "cap [kW]": cap,
                          "indifferent members": [sum(ranges[c][1][t] - ranges[c][0][t] > 1e-5 for c in ranges)
                                                  for t in range(len(cap))]})
    hours.index.name = "hour"
    member_hours = pd.DataFrame({c: r[1] - r[0] for c, r in ranges.items()}).T
    member_hours.columns.name = "hour"
    n_ind = int((member_hours.values > 1e-5).sum())
    worst = float(np.maximum(0.0, hi - cap).sum())
    out_dir.mkdir(parents=True, exist_ok=True)
    hours.round(3).to_csv(out_dir / "q2_indifference.csv")
    member_hours.round(3).to_csv(out_dir / "q2_indifferent_member_hours.csv")
    print(hours.round(2).to_string())
    print(f"\nIndifferent member-hours at lambda^C: {n_ind} of {member_hours.size}; "
          f"distinct prices {sorted(set(np.round(C.price, 3)))}")
    print(f"Import above the cap if every member broke its ties against the manager, hour by hour: {worst:.1f} kWh "
          f"in {(hi > cap + 1e-6).sum()} hours (against 0 kWh in the optimistic solution)")
    with open(out_dir / "q2_indifference_summary.json", "w") as f:
        json.dump({"indifferent member-hours": n_ind, "member-hours": int(member_hours.size),
                   "worst-case exceedance [kWh]": worst}, f, indent=1)
    return hours
