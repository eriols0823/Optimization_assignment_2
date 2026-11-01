"""Question 1: validation and base case of the models R, I and C.

    python main.py --q1 validate    runs the validation checks of the models
    python main.py --q1 base        base-case table, saved to results/<community>/
    python main.py --q1 sweep       sensitivity to the cap and the discount: CSV, summary and figure
    python main.py --q1 settlement  dual settlement of C per member, the manager's budget and figure
"""
from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import CommunityModel

DISCOUNTS = np.round(np.arange(0.0, 0.951, 0.05), 2)     # beta^DSO in the sensitivity analysis
CAP_FACTORS = np.round(np.arange(1.0, 4.001, 0.1), 2)    # f in cap = f * E^R / 24 (base case 1.3)


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
        mod.gurobi.Params.Method = method
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


def run_sweep(com, discounts=None, cap_factors=None, penalties=None) -> pd.DataFrame:
    """Solve I and C for every combination of the given contract terms. Used by every sensitivity analysis.

    discounts    values of the discount beta^DSO
    cap_factors  values of f in cap = f * E^R / 24; the cap is scaled as in Table 11 c): contract cap * f / 1.3
    penalties    values of the penalty alpha^DSO in DKK/kWh
    A term left as None keeps its contract value. Returns one row per combination with the earnings V_I and V_C
    compared with R (DKK), the energy above the cap X_R, X_I and X_C and the flexibility F_I and F_C (kWh).
    """
    c = com.contract
    base_factor = c.capacity_limit_daily_energy_factor                      # 1.3
    discounts = [c.tariff_discount_ratio] if discounts is None else discounts
    cap_factors = [base_factor] if cap_factors is None else cap_factors
    penalties = [c.penalty_DKK_per_kWh] if penalties is None else penalties
    combinations = list(product(cap_factors, discounts, penalties))
    print(f"Solving I and C for {len(combinations)} combinations of cap factor, discount and penalty ...")

    R = CommunityModel(com, "R").solve()                                    # R does not depend on the contract
    rows = []
    for f, beta, alpha in combinations:
        cap = np.asarray(c.capacity_limit_kW, dtype=float) * f / base_factor
        X_R = float(np.maximum(0.0, R.feeder_import - cap).sum())           # R measured against this cap
        I = CommunityModel(com, "I", discount=beta, penalty=alpha, cap=cap).solve()
        C = CommunityModel(com, "C", discount=beta, penalty=alpha, cap=cap).solve()
        rows.append({"f": f, "beta": beta, "alpha": alpha, "cap [kW]": cap[0],
                     "V_I [DKK]": R.cost["total"] - I.cost["total"],
                     "V_C [DKK]": R.cost["total"] - C.cost["total"],
                     "X_R [kWh]": X_R, "X_I [kWh]": I.exceedance, "X_C [kWh]": C.exceedance,
                     "F_I [kWh]": X_R - I.exceedance, "F_C [kWh]": X_R - C.exceedance})
    return pd.DataFrame(rows)


def sweep(com, out_dir: Path) -> pd.DataFrame:
    """Sensitivity to the cap and the discount: saves every point as CSV, prints the summary and draws the figure."""
    points = run_sweep(com, discounts=DISCOUNTS, cap_factors=CAP_FACTORS)
    out_dir.mkdir(parents=True, exist_ok=True)
    points.round(3).to_csv(out_dir / "q1_sweep.csv", index=False)
    _print_sweep_summary(points)
    _plot_sweep(points, out_dir)
    return points


def _print_sweep_summary(points: pd.DataFrame) -> None:
    """Per cap factor: the smallest discount at which C and I beat R, and the flexibility at the base discount."""
    def first_beta(group, column):
        positive = group[group[column] > 1e-6]
        return positive["beta"].min() if len(positive) else None

    at_base_beta = points[points["beta"] == 0.75].set_index("f")
    summary = pd.DataFrame({
        "cap [kW]": points.groupby("f")["cap [kW]"].first(),
        "X_R [kWh]": points.groupby("f")["X_R [kWh]"].first(),
        "C beats R from beta": points.groupby("f").apply(first_beta, "V_C [DKK]"),
        "I beats R from beta": points.groupby("f").apply(first_beta, "V_I [DKK]"),
        "V_C at beta 0.75 [DKK]": at_base_beta["V_C [DKK]"],
        "F_I at beta 0.75 [kWh]": at_base_beta["F_I [kWh]"],
        "F_C at beta 0.75 [kWh]": at_base_beta["F_C [kWh]"],
        "X_C at beta 0.75 [kWh]": at_base_beta["X_C [kWh]"],
    }).round(2)
    base = points[(points["f"] == 1.3) & (points["beta"] == 0.75)].iloc[0]
    print("\nSensitivity to the cap and the discount (None: never beats R up to beta 0.95)\n", summary.to_string(), sep="")
    print(f"\nRange over all points: V_C {points['V_C [DKK]'].min():.2f} to {points['V_C [DKK]'].max():.2f} DKK, "
          f"V_I {points['V_I [DKK]'].min():.2f} to {points['V_I [DKK]'].max():.2f} DKK")
    print(f"Base case (beta 0.75, f 1.3): V_C {base['V_C [DKK]']:.2f} DKK, F_C {base['F_C [kWh]']:.2f} kWh")
    print(f"Largest energy above the cap left in C: {points['X_C [kWh]'].max():.2f} kWh | "
          f"points where I beats C: {int((points['V_I [DKK]'] > points['V_C [DKK]'] + 1e-6).sum())}")
    spread = {m: points.groupby("f")[f"F_{m} [kWh]"].agg(np.ptp).max() for m in ("I", "C")}
    print(f"Largest change of the flexibility over the discount at one cap: F_I {spread['I']:.2f} kWh, F_C {spread['C']:.2f} kWh")


def _plot_sweep(points: pd.DataFrame, out_dir: Path) -> None:
    """Two separate figures, each saved as PNG and PDF:
    q1_sweep_earnings     community earnings V^C over discount and cap, with the break-even lines of C and I
    q1_sweep_flexibility  flexibility F^I and F^C for the DSO against the cap factor"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.lines import Line2D

    V_C = points.pivot(index="f", columns="beta", values="V_C [DKK]")    # rows: cap factor, columns: discount
    V_I = points.pivot(index="f", columns="beta", values="V_I [DKK]")
    beta, f = V_C.columns.values, V_C.index.values
    ink, blue, grey = "#0b0b0b", "#1c5cab", "#52514e"
    base_marker = dict(marker="o", markersize=8, markerfacecolor="white", markeredgecolor=ink, linestyle="none")

    # Figure 1: community earnings compared with R (red = loss, grey = 0, blue = gain)
    fig, ax = plt.subplots(figsize=(7, 4.8), layout="constrained")
    loss_gain = LinearSegmentedColormap.from_list("loss_gain", ["#b8322f", "#f0efec", blue])
    norm = TwoSlopeNorm(vcenter=0.0, vmin=min(V_C.values.min(), -1e-3), vmax=max(V_C.values.max(), 1e-3))
    mesh = ax.pcolormesh(beta, f, V_C.values, cmap=loss_gain, norm=norm, shading="nearest")
    fig.colorbar(mesh, ax=ax, label=r"Community earnings $V^{\mathcal{C}}$ [DKK]")
    handles = []
    for values, style, label in ((V_C, "-", r"$\mathcal{C}$ breaks even with $\mathcal{R}$"),
                                 (V_I, "--", r"$\mathcal{I}$ breaks even with $\mathcal{R}$")):
        if values.values.min() < 0 < values.values.max():
            ax.contour(beta, f, values.values, levels=[0.0], colors=ink, linewidths=1.5, linestyles=style)
            handles.append(Line2D([], [], color=ink, linestyle=style, label=label))
    ax.plot(0.75, 1.3, **base_marker)
    handles.append(Line2D([], [], label="base case", **base_marker))
    ax.set_xlabel(r"Tariff discount $\beta^{\mathrm{DSO}}$")
    ax.set_ylabel(r"Cap factor $f$ (smaller is tighter)")
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=9)
    _save(fig, out_dir / "q1_sweep_earnings")

    # Figure 2: flexibility for the DSO, i.e. the energy above the cap that each contract removes
    # (solid = C and dashed = I as in Figure 1)
    fig, ax = plt.subplots(figsize=(6, 3.8), layout="constrained")
    same = True
    for m, color, style, label in (("C", blue, "-", r"Community contract $\mathcal{C}$"),
                                   ("I", grey, "--", r"Individual contracts $\mathcal{I}$")):
        flexibility = points.groupby("f")[f"F_{m} [kWh]"]
        same = same and (flexibility.max() - flexibility.min()).max() < 0.01     # tied optima differ by ~0.001 kWh
        ax.plot(f, flexibility.max(), color=color, linestyle=style, linewidth=2, label=label)
        ax.plot(1.3, flexibility.max()[1.3], **base_marker)
    ax.plot([], [], label="base case", **base_marker)
    ax.set_xlabel(r"Cap factor $f$ (smaller is tighter)")
    ax.set_ylabel(r"Flexibility $F^m$ [kWh]")
    ax.set_ylim(bottom=0)
    ax.grid(color="#e1e0dc", linewidth=0.8)
    ax.legend(frameon=False, fontsize=9)
    if same:
        ax.set_title("The same for every discount", loc="left", fontsize=10, color=grey)
    _save(fig, out_dir / "q1_sweep_flexibility")
    print(f"\nFigures and CSV saved to {out_dir}")


ASSETS = ("no asset", "PV", "battery", "PV and battery")


def _asset(member) -> str:
    if member.has_pv:
        return "PV and battery" if member.has_battery else "PV"
    return "battery" if member.has_battery else "no asset"


def _payment_range(model, cost: float, consumption, eps: float = 1e-4) -> tuple[float, float]:
    """Lowest and highest value of sum_t lambda_t * consumption_t over all optimal dual solutions of model C.
    lambda_t is the marginal cost of consumption, so the range follows from re-solving with eps times the
    consumption added and removed (the one-sided derivatives of the community cost)."""
    with_more, with_less = [], []
    for sign, costs in ((1.0, with_more), (-1.0, with_less)):
        model.balance.RHS = sign * eps * np.asarray(consumption)
        model.gurobi.optimize()
        costs.append(model.gurobi.ObjVal)
    model.balance.RHS = np.zeros(model.n_hours)
    return (cost - with_less[0]) / eps, (with_more[0] - cost) / eps


def settlement(com, out_dir: Path) -> pd.DataFrame:
    """Dual settlement of C on the base case: what each member pays and gains against R and I, the flexibility it
    provides, the manager's budget and whether another optimal dual solution would change any payment.
    Saves q1_settlement.csv (one row per member), q1_internal_prices.csv (one row per hour) and the figure."""
    R, I = CommunityModel(com, "R").solve(), CommunityModel(com, "I").solve()
    model = CommunityModel(com, "C")
    C = model.solve()
    above = R.feeder_import > R.cap + 1e-6                     # hours T^R where R exceeds the cap
    rows = []
    for m in com.members:
        cid = m.consumer_id
        low, high = _payment_range(model, C.cost["total"], C.schedule[cid]["q"])
        rows.append({
            "member": cid, "type": m.archetype, "asset": _asset(m), "load class": m.slope_class,
            "payment Pi [DKK]": C.payment[cid],
            "own cost K [DKK]": C.member_cost[cid] - C.payment[cid],
            "V_C [DKK]": R.member_cost[cid] - C.member_cost[cid],             # gain against R in C
            "V_I [DKK]": R.member_cost[cid] - I.member_cost[cid],             # gain against R in I
            "F_C [kWh]": float((R.schedule[cid]["q"] - C.schedule[cid]["q"])[above].sum()),
            "payment range [DKK]": high - low,                                 # 0: the same for every optimal dual
        })
    table = pd.DataFrame(rows).set_index("member")
    table["not worse off than R"] = table["V_C [DKK]"] >= -1e-6
    table["not worse off than I"] = table["V_C [DKK]"] >= table["V_I [DKK]"] - 1e-6

    # internal price per hour and the range it can take over all optimal duals
    lam_low, lam_high = np.array([_payment_range(model, C.cost["total"], np.eye(model.n_hours)[t])
                                  for t in range(model.n_hours)]).T
    imp, exp = model.P_imp.X, model.P_exp.X
    state = np.where(exp > 1e-6, "export", np.where(imp >= C.cap - 1e-6, "at cap",
                                                    np.where(imp > 1e-6, "import", "no exchange")))
    hours = pd.DataFrame({"state": state, "lambda [DKK/kWh]": C.price, "gamma [DKK/kWh]": C.cap_price,
                          "p_imp discounted": model.p_imp, "p_exp discounted": model.p_exp,
                          "lambda lowest": lam_low, "lambda highest": lam_high})
    hours.index.name = "hour"

    out_dir.mkdir(parents=True, exist_ok=True)
    table.round(3).to_csv(out_dir / "q1_settlement.csv")
    hours.round(3).to_csv(out_dir / "q1_internal_prices.csv")
    _print_settlement(table, hours, R, C)
    _plot_settlement(table, out_dir)
    return table


def _print_settlement(table: pd.DataFrame, hours: pd.DataFrame, R, C) -> None:
    gains = ["V_C [DKK]", "V_I [DKK]", "F_C [kWh]"]
    V_C = R.cost["total"] - C.cost["total"]
    rent = float(C.cap_price @ C.cap)
    dso_bill = C.cost["energy"] + C.cost["tariffs"] + C.cost["penalty"]
    print("Settlement of C on the base case (V_C and V_I: gain of the member against R in C and in I)\n",
          table.drop(columns=["asset", "load class"]).round(2).to_string(), sep="")
    print("\nAverage per member by asset\n", table.groupby("asset")[gains].mean().reindex(ASSETS).round(2).to_string(),
          "\n\nAverage per member by load class\n", table.groupby("load class")[gains].mean().round(2).to_string(), sep="")
    print(f"\nMembers not worse off than R: {table['not worse off than R'].sum()} of {len(table)} | "
          f"not worse off than I: {table['not worse off than I'].sum()} of {len(table)}")
    print(f"Budget: payments {table['payment Pi [DKK]'].sum():.2f} DKK, DSO bill D {dso_bill:.2f} DKK, "
          f"surplus B {C.surplus:.2f} DKK | community gain V_C {V_C:.2f} DKK")
    print(f"Flexibility provided by the members {table['F_C [kWh]'].sum():.2f} kWh | F_C {R.exceedance - C.exceedance:.2f} kWh")
    print("\nInternal price per hour (lowest and highest over all optimal duals)\n", hours.round(3).to_string(), sep="")
    unique = (hours["lambda highest"] - hours["lambda lowest"]) < 1e-3
    print(f"\nlambda_t is unique in {unique.sum()} of {len(hours)} hours"
          + ("" if unique.all() else f"; the other hours: {list(hours.index[~unique])}"))
    print("\nChecks")
    _check(abs(C.surplus - rent) < 1e-6, f"surplus B {C.surplus:.2f} DKK equals the congestion rent "
                                         f"sum_t gamma_t * cap_t {rent:.2f} DKK")
    _check(abs(table["V_C [DKK]"].sum() - (V_C - C.surplus)) < 1e-6,
           f"member gains add up to V_C - B: {table['V_C [DKK]'].sum():.2f} = {V_C:.2f} - {C.surplus:.2f} DKK")
    _check(table["payment range [DKK]"].max() < 1e-3, "every payment is the same for all optimal duals "
                                                     f"(largest range {table['payment range [DKK]'].max():.2f} DKK)")


def _plot_settlement(table: pd.DataFrame, out_dir: Path) -> None:
    """q1_settlement: gain of every member against R in C and I (left) and the flexibility it provides (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ink, blue, grey, light = "#0b0b0b", "#1c5cab", "#52514e", "#c9c8c4"
    rows = table.assign(order=table["asset"].map(ASSETS.index)).sort_values(["order", "load class"])
    y = rows["order"] * 5 + (rows["load class"] - 1)              # four rows per asset, one empty row between
    fig, (left, right) = plt.subplots(1, 2, sharey=True, figsize=(7.5, 5.4), layout="constrained",
                                      gridspec_kw={"width_ratios": [2, 1]})
    left.axvline(0, color=ink, linewidth=0.8)
    left.hlines(y, rows["V_I [DKK]"], rows["V_C [DKK]"], color=light, linewidth=2)
    left.plot(rows["V_I [DKK]"], y, "o", markersize=7, markerfacecolor="white", markeredgecolor=grey)
    left.plot(rows["V_C [DKK]"], y, "o", markersize=7, color=blue)
    left.set_xlabel(r"Gain against $\mathcal{R}$, $V_i^m$ [DKK]")
    right.barh(y, rows["F_C [kWh]"], height=0.6, color=blue)
    right.set_xlabel(r"Flexibility provided $F_i^{\mathcal{C}}$ [kWh]")
    left.set_yticks(y, rows["type"])
    left.invert_yaxis()
    for ax in (left, right):
        ax.grid(axis="x", color="#e1e0dc", linewidth=0.8)
        ax.set_axisbelow(True)
    handles = [Line2D([], [], marker="o", linestyle="none", color=blue, label=r"Community contract $\mathcal{C}$"),
               Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=grey,
                      label=r"Individual contracts $\mathcal{I}$")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    _save(fig, out_dir / "q1_settlement")
    print(f"\nFigure and CSV files saved to {out_dir}")


def _save(fig, path: Path) -> None:
    """Save a figure as PNG (for viewing) and PDF (for the report), then close it."""
    import matplotlib.pyplot as plt
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=200)
    plt.close(fig)
