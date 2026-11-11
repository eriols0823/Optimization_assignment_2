"""Question 2: validation and base case of the pricing problem DP.

    python main.py --q2 validate      DP^cap against the central dispatch C (equivalence with the duals of C)
    python main.py --q2 steps         the three steps cap -> + budget balance -> + individual rationality, with the
                                      ways out of infeasibility (soft IR, price groups); CSV and figures in results/
    python main.py --q2 indifference  the member-hours where the best response at lambda^C is not unique, and the
                                      import if every member broke its ties against the manager
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
                     f"{r.checks['pairs at big-M']} pairs at big-M"))
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


def _member_table(com, results: dict, R, I) -> pd.DataFrame:
    """Gain of every member against R (DKK) in each variant, with its archetype."""
    rows = {}
    for name, r in results.items():
        if r is not None and r.feasible:
            rows[name] = {cid: R.member_cost[cid] - r.member_cost[cid] for cid in r.member_cost}
    table = pd.DataFrame(rows)
    table.insert(0, "gain in I", {cid: R.member_cost[cid] - I.member_cost[cid] for cid in R.member_cost})
    table.insert(0, "type", {m.consumer_id: m.archetype for m in com.members})
    return table


def steps(com, out_dir: Path, time_limit: float = 600) -> pd.DataFrame:
    """The three steps of the pricing study on the base case, and the ways out of infeasibility.

    For each variant: feasibility, community cost and its loss against C (efficiency cost of the requirement), the
    manager's budget, the energy above the cap, the members worse off than R and I, the total IR violation and
    the blocking members (soft IR). Saves q2_steps.csv, q2_members.csv and q2_prices.csv in out_dir."""
    R, I, C = (CommunityModel(com, m).solve() for m in ("R", "I", "C"))
    variants = {
        "DP^cap": dict(requirements="cap"),
        "DP^BB": dict(requirements="bb"),
        "DP^BB,IR": dict(requirements="bb_ir"),
        "DP^BB,IR,soft": dict(requirements="bb_ir", ir_soft=True),
        "DP^BB,IR,asset": dict(requirements="bb_ir", price_groups="asset"),
        "DP^BB,IR,asset,soft": dict(requirements="bb_ir", price_groups="asset", ir_soft=True),
        "DP^BB,IR,member": dict(requirements="bb_ir", price_groups="member"),
    }
    results, rows = {}, []
    start = None
    for name, kw in variants.items():
        print(f"Solving {name} ...", flush=True)
        dp = PricingModel(com, time_limit=time_limit, start_prices=start, **kw)
        r = results[name] = dp.solve()
        row = {"variant": name, "status": r.status, "time [s]": round(r.runtime, 1), "gap": r.gap}
        if r.feasible:
            if name == "DP^BB" and "all" in r.prices:
                start = r.prices["all"]                    # warm start of the uniform IR variants
            worse_R = [c for c in r.member_cost if r.member_cost[c] > R.member_cost[c] + 1e-4]
            worse_I = [c for c in r.member_cost if r.member_cost[c] > I.member_cost[c] + 1e-4]
            row.update({
                "J [DKK]": r.cost["total"], "loss vs C [DKK]": r.cost["total"] - C.cost["total"],
                "V vs R [DKK]": R.cost["total"] - r.cost["total"], "budget B [DKK]": r.surplus,
                "above cap X [kWh]": float(np.maximum(0.0, r.P_imp - dp.cap).sum()),
                "worse off than R": len(worse_R), "worse off than I": len(worse_I),
                "IR violation [DKK]": sum(r.ir_slack.values()) if r.ir_slack else None,
                "blocking members": " ".join(c for c, s in r.ir_slack.items() if s > 1e-4) if r.ir_slack else "",
            })
        rows.append(row)
        print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    table = pd.DataFrame(rows).set_index("variant")
    members = _member_table(com, results, R, I)
    meter = CommunityModel(com, "C")                       # discounted meter prices of the contract
    prices = pd.DataFrame({"lambda^C": C.price, "p_imp discounted": meter.p_imp, "p_exp discounted": meter.p_exp,
                           **{f"{n} [{g}]": x for n, r in results.items() if r is not None and r.feasible
                              for g, x in r.prices.items() if g == "all" or n.endswith("asset")}})
    out_dir.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out_dir / "q2_steps.csv")
    members.round(3).to_csv(out_dir / "q2_members.csv")
    prices.round(4).to_csv(out_dir / "q2_prices.csv")
    print("\nThree steps\n", table.round(3).to_string(), "\n\nGain of each member against R [DKK]\n",
          members.round(2).to_string(), sep="")
    print(f"\nSaved to {out_dir}")
    return table


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
