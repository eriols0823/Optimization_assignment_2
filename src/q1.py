"""Question 1: validation and base case of the models R, I and C.

    python main.py --q1 validate    runs the validation checks of the models
    python main.py --q1 base        base-case table, saved to results/<community>/
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import CommunityModel


def _check(ok: bool, text: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}")
    return bool(ok)


def validate(com) -> bool:
    """Run the validation checks of Question 1 and print PASS or FAIL for each. Returns True if all pass."""
    models = {name: CommunityModel(com, name) for name in ("R", "I", "C")}
    res = {name: mod.solve() for name, mod in models.items()}
    R, I, C = res["R"], res["I"], res["C"]
    ref = com.contract.reference_model_R
    ok = []

    print("1. Model R reproduces the reference values of the contract file")
    ok.append(_check(abs(R.cost["total"] - ref["cost_DKK"]) < 0.01,
                     f"cost {R.cost['total']:.2f} DKK (reference {ref['cost_DKK']})"))
    ok.append(_check(abs(R.feeder_import.sum() - ref["daily_import_kWh"]) < 0.05,
                     f"daily import {R.feeder_import.sum():.2f} kWh (reference {ref['daily_import_kWh']})"))
    ok.append(_check(abs(R.peak - ref["peak_import_kW"]) < 0.01,
                     f"peak import {R.peak:.2f} kW (reference {ref['peak_import_kW']})"))
    # R has tied optima: other solver methods return other schedules with the same cost
    X = []
    for method in (0, 1, 2):
        mod = CommunityModel(com, "R")
        mod.m.Params.Method = method
        X.append(mod.solve().exceedance)
    ok.append(_check(min(X) - 0.05 <= ref["exceedance_kWh"] <= max(X) + 0.05,
                     f"energy above the cap {min(X):.2f} to {max(X):.2f} kWh over tied optima "
                     f"(reference {ref['exceedance_kWh']})"))
    if com.contract.net_import_in_R_kWh:
        net = {cid: max(0.0, s["q"].sum()) for cid, s in R.schedule.items()}
        dev = max(abs(net[cid] - v) for cid, v in com.contract.net_import_in_R_kWh.items())
        ok.append(_check(dev < 0.01, f"net import of every member (largest deviation {dev:.3f} kWh)"))

    print("2. Central dispatch without contract (one LP with own tariffs) equals the members solving alone")
    alone = sum(CommunityModel(replace(com, members=[mem]), "R").solve().cost["total"] for mem in com.members)
    ok.append(_check(abs(alone - R.cost["total"]) < 1e-6,
                     f"{R.cost['total']:.4f} DKK jointly and {alone:.4f} DKK alone"))

    print("3. Pooling never costs the community and both contracts reduce congestion")
    ok.append(_check(C.cost["total"] <= I.cost["total"] + 1e-6,
                     f"J^C = {C.cost['total']:.2f} DKK <= J^I = {I.cost['total']:.2f} DKK"))
    ok.append(_check(max(I.exceedance, C.exceedance) < R.exceedance,
                     f"energy above the cap: R {R.exceedance:.2f}, I {I.exceedance:.2f}, C {C.exceedance:.2f} kWh"))

    print("4. The internal price lambda_t of C matches the situation of each hour")
    mC = models["C"]
    imp, exp, lam = mC.P_imp.X, mC.P_exp.X, C.price
    exporting, importing = exp > 1e-6, imp > 1e-6
    below, above = importing & (imp < mC.cap - 1e-6), imp > mC.cap + 1e-6
    lo, hi = mC.p_exp.copy(), mC.p_imp.copy()             # no exchange: lambda between the two prices
    lo[exporting] = hi[exporting] = mC.p_exp[exporting]
    lo[importing], hi[importing] = mC.p_imp[importing], mC.p_imp[importing] + mC.alpha
    hi[below] = mC.p_imp[below]
    lo[above] = mC.p_imp[above] + mC.alpha
    at_cap = importing & ~below & ~above
    ok.append(_check(np.all((lam >= lo - 1e-6) & (lam <= hi + 1e-6)),
                     f"{exporting.sum()} h export, {below.sum()} h import below the cap, {at_cap.sum()} h at the cap, "
                     f"{above.sum()} h above the cap, {(~exporting & ~importing).sum()} h no exchange"))

    print("5. No member deviates up and down, charges and discharges, or imports and exports in the same hour")
    worst = 0.0
    for r in res.values():
        for s in r.schedule.values():
            for a, b in (("d_plus", "d_minus"), ("p_ch", "p_dis"), ("q_imp", "q_exp")):
                if a in s:
                    worst = max(worst, float(np.minimum(s[a], s[b]).max()))
    ok.append(_check(worst < 1e-6, f"largest overlap {worst:.1e} kWh"))

    print(f"\n{sum(ok)}/{len(ok)} checks passed")
    return all(ok)


def _drivers(com, r, R) -> dict:
    """What drives the result of one model: disutility per load class, penalties by PV ownership (I only),
    the hours above the cap and the tariff income the DSO gives up per kWh of congestion removed."""
    mem = {m.consumer_id: m for m in com.members}
    out = {f"disutility L{k} [DKK]": 0.0 for k in (1, 2, 3, 4)}
    for cid, s in r.schedule.items():
        out[f"disutility L{mem[cid].slope_class} [DKK]"] += mem[cid].linear_disutility * (s["d_plus"] + s["d_minus"]).sum()
    if "q_exc" in next(iter(r.schedule.values())):          # penalties are paid per member only in I
        alpha = com.contract.penalty_DKK_per_kWh
        for label, pv in (("with PV", True), ("without PV", False)):
            out[f"penalty {label} [DKK]"] = sum(alpha * s["q_exc"].sum()
                                                for cid, s in r.schedule.items() if mem[cid].has_pv == pv)
    income = r.cost["tariffs"] + r.cost["penalty"]
    removed = R.exceedance - r.exceedance
    out["DSO income [DKK]"] = income
    out["DSO cost per kWh removed [DKK/kWh]"] = (R.cost["tariffs"] + R.cost["penalty"] - income) / removed if removed > 0 else None
    out = {k: round(float(v), 2) if v is not None else None for k, v in out.items()}
    out["hours above cap"] = " ".join(str(h) for h in np.where(r.feeder_import > r.cap + 1e-6)[0])
    return out


def base_case(com, out_dir: Path) -> pd.DataFrame:
    """Base-case comparison of R, I and C. Prints four tables and saves them as CSV files in out_dir."""
    res = {name: CommunityModel(com, name).solve() for name in ("R", "I", "C")}
    R, I = res["R"], res["I"]

    def worse_off(r, ref):
        return sum(r.member_cost[cid] > ref.member_cost[cid] + 1e-6 for cid in r.member_cost)

    summary = pd.DataFrame({
        name: {
            "cost J [DKK]": r.cost["total"],
            "earnings V [DKK]": R.cost["total"] - r.cost["total"],
            "above cap X [kWh]": r.exceedance,
            "hours above cap": r.hours_above_cap,
            "peak import [kW]": r.peak,
            "surplus B [DKK]": r.surplus,
            "members worse off than R": worse_off(r, R) if name != "R" else None,
            "members worse off than I": worse_off(r, I) if name == "C" else None,
        }
        for name, r in res.items()
    }).T.round(2)
    parts = pd.DataFrame({name: r.cost for name, r in res.items()}).T.round(2)
    change = (parts - parts.loc["R"]).round(2)
    drivers = pd.DataFrame({name: _drivers(com, r, R) for name, r in res.items()})

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "q1_base_case.csv")
    parts.to_csv(out_dir / "q1_cost_parts.csv")
    change.to_csv(out_dir / "q1_cost_change.csv")
    drivers.to_csv(out_dir / "q1_drivers.csv")
    print("Base case\n", summary.to_string(), "\n\nCommunity cost by part [DKK]\n", parts.to_string(),
          "\n\nChange of each cost part compared with R [DKK]\n", change.to_string(),
          "\n\nWhat drives the results\n", drivers.to_string(), sep="")
    print(f"\nSaved to {out_dir}")
    return summary
