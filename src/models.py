"""The community models R, I and C as one parametrized linear program.

    R  no contract            each member alone, full tariffs on its own meter
    I  individual contracts   discounted tariffs and an own cap kappa_i * cap on each meter
    C  community contract     one meter and one cap, all members dispatched centrally

Symbols follow Section 1 of the report (Assignment 1 notation with the member index i) and the
comments give the equation numbers. Energies are in kWh per hour and prices in DKK/kWh.
"""
from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp
import numpy as np
from gurobipy import GRB

MODELS = ("R", "I", "C")
_ENV = None


def _env() -> gp.Env:
    """One silent Gurobi environment shared by all models."""
    global _ENV
    if _ENV is None:
        _ENV = gp.Env(empty=True)
        _ENV.setParam("OutputFlag", 0)
        _ENV.start()
    return _ENV


@dataclass
class Result:
    """Solution of one model. Hourly quantities are arrays of 24 values."""

    model: str
    cost: dict[str, float]                          # community cost J and its parts (DKK)
    member_cost: dict[str, float]                   # J_i (DKK)
    schedule: dict[str, dict[str, np.ndarray]]      # consumer_id -> variable name -> hourly values
    feeder_import: np.ndarray                       # import the DSO sees, P_t (kWh)
    cap: np.ndarray                                 # community cap (kW)
    price: np.ndarray | dict[str, np.ndarray]       # lambda_t (C) or lambda_i,t per member (R, I) (DKK/kWh)
    cap_price: np.ndarray | dict[str, np.ndarray] | None = None   # gamma_t (C) or gamma_i,t (I) (DKK/kWh)
    payment: dict[str, float] | None = None         # Pi_i (C only, DKK)
    surplus: float | None = None                    # B (C only, DKK)

    @property
    def exceedance(self) -> float:
        """X: energy imported above the cap (kWh)."""
        return float(np.maximum(0.0, self.feeder_import - self.cap).sum())

    @property
    def hours_above_cap(self) -> int:
        return int((self.feeder_import > self.cap + 1e-6).sum())

    @property
    def peak(self) -> float:
        """Highest hourly import at the feeder (kW)."""
        return float(self.feeder_import.max())


class CommunityModel:
    """Build and solve model R, I or C for one community on one day.

    com       CommunityData from load_community()
    model     "R", "I" or "C"
    discount  beta^DSO (default: the contract value; always 0 in R)
    penalty   alpha^DSO in DKK/kWh (default: the contract value)
    cap       hourly cap in kW (default: the contract cap); member i gets kappa_i * cap in I
    """

    def __init__(self, com, model: str, discount: float | None = None, penalty: float | None = None, cap=None):
        if model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        c = com.contract
        self.com, self.model = com, model
        self.n = len(com.energy_price)
        self.cap = np.asarray(c.capacity_limit_kW if cap is None else cap, dtype=float)
        self.alpha = c.penalty_DKK_per_kWh if penalty is None else penalty
        beta = 0.0 if model == "R" else (c.tariff_discount_ratio if discount is None else discount)
        # effective prices: p^imp and p^exp in R, the discounted prices of eq. (4) in I and C
        self.p_imp = com.energy_price + (1 - beta) * com.import_tariff
        self.p_exp = com.energy_price - (1 - beta) * com.export_tariff

        self.m = gp.Model(f"model_{model}", env=_env())
        self.m.Params.Method = 1            # dual simplex gives one reproducible dual solution
        self.var, self.con, self.K = {}, {}, {}
        for mem in com.members:
            self._add_member(mem)
        grid_cost = self._add_grid()
        self.m.setObjective(grid_cost + sum(self.K.values()), GRB.MINIMIZE)

    def _add_member(self, mem) -> None:
        """Feasible set X_i (1a)-(1h) and own cost K_i (2) of one member."""
        m, n, cid = self.m, self.n, mem.consumer_id
        v = {
            "l": m.addMVar(n, lb=mem.load_lower_kWh, ub=mem.load_upper_kWh),   # (1g)
            "d_plus": m.addMVar(n),
            "d_minus": m.addMVar(n),
            "q_pv": m.addMVar(n, ub=mem.pv_available),                         # (1b)
            "q": m.addMVar(n, lb=-GRB.INFINITY),
        }
        m.addConstr(v["l"] - mem.reference_load == v["d_plus"] - v["d_minus"])  # (1c)
        self.con[cid] = {"emin": m.addConstr(v["l"].sum() >= mem.min_daily_energy_kWh)}   # (1d)

        battery = 0
        if mem.has_battery:
            v["p_ch"] = m.addMVar(n, ub=mem.battery_max_charge_kW)            # (1f)
            v["p_dis"] = m.addMVar(n, ub=mem.battery_max_discharge_kW)
            v["e"] = m.addMVar(n, ub=mem.battery_capacity_kWh)
            e_init = mem.battery_initial_soc_kWh
            stored = mem.battery_charging_efficiency * v["p_ch"] - v["p_dis"] / mem.battery_discharging_efficiency
            m.addConstr(v["e"][0] == e_init + stored[0])                      # (1e) with e_{-1} = e^init
            m.addConstr(v["e"][1:] == v["e"][:-1] + stored[1:])               # (1e)
            m.addConstr(v["e"][n - 1] >= e_init)                              # (1h) cyclic end of day
            battery = v["p_ch"] - v["p_dis"]
        m.addConstr(v["q"] == v["l"] + battery - v["q_pv"])                   # (1a)

        self.var[cid] = v
        self.K[cid] = (mem.linear_disutility * (v["d_plus"].sum() + v["d_minus"].sum())
                       + mem.pv_marginal_cost * v["q_pv"].sum())              # (2)

    def _add_grid(self):
        """Grid side: own meters in R and I (3), (5) or one community meter in C (6). Returns its cost."""
        m, n = self.m, self.n
        if self.model == "C":
            self.P_imp, self.P_exp, self.P_exc = m.addMVar(n), m.addMVar(n), m.addMVar(n)
            # (6b) is written as supply minus demand: Gurobi's Pi is d(cost)/d(rhs), so Pi = lambda_t
            self.balance = m.addConstr(self.P_imp - self.P_exp - sum(v["q"] for v in self.var.values()) == 0)
            self.cap_con = m.addConstr(self.P_imp - self.P_exc <= self.cap)   # (6c), gamma_t = -Pi
            return self.p_imp @ self.P_imp - self.p_exp @ self.P_exp + self.alpha * self.P_exc.sum()

        cost = 0
        for cid, v in self.var.items():
            v["q_imp"], v["q_exp"] = m.addMVar(n), m.addMVar(n)
            self.con[cid]["balance"] = m.addConstr(v["q_imp"] - v["q_exp"] - v["q"] == 0)   # (3b), (5b)
            cost += self.p_imp @ v["q_imp"] - self.p_exp @ v["q_exp"]
            if self.model == "I":
                v["q_exc"] = m.addMVar(n)
                kappa = self.com.contract.share[cid]
                self.con[cid]["cap"] = m.addConstr(v["q_imp"] - v["q_exc"] <= kappa * self.cap)   # (5c)
                cost += self.alpha * v["q_exc"].sum()
        return cost

    def solve(self) -> Result:
        """Solve the LP and collect the quantities used in the report."""
        self.m.optimize()
        if self.m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"model {self.model} is not optimal (Gurobi status {self.m.Status})")
        sched = {cid: {name: x.X for name, x in v.items()} for cid, v in self.var.items()}
        K = {cid: float(k.getValue()) for cid, k in self.K.items()}

        # flows at the meter(s) that pay the grid
        if self.model == "C":
            imp, exp, exc = self.P_imp.X, self.P_exp.X, self.P_exc.X
        else:
            imp = sum(s["q_imp"] for s in sched.values())
            exp = sum(s["q_exp"] for s in sched.values())
            exc = sum(s["q_exc"] for s in sched.values()) if self.model == "I" else np.zeros(self.n)
        p = self.com.energy_price
        mem = {mm.consumer_id: mm for mm in self.com.members}
        cost = {
            "energy": float(p @ (imp - exp)),
            "tariffs": float((self.p_imp - p) @ imp + (p - self.p_exp) @ exp),
            "penalty": float(self.alpha * exc.sum()),
            "disutility": sum(mem[c].linear_disutility * (s["d_plus"] + s["d_minus"]).sum() for c, s in sched.items()),
            "pv": sum(mem[c].pv_marginal_cost * s["q_pv"].sum() for c, s in sched.items()),
        }
        cost["total"] = sum(cost.values())
        assert abs(cost["total"] - self.m.ObjVal) < 1e-6, "cost parts do not add up to the objective"

        if self.model == "C":
            lam, gamma = self.balance.Pi, -self.cap_con.Pi
            payment = {cid: float(lam @ s["q"]) for cid, s in sched.items()}                  # (7)
            member_cost = {cid: payment[cid] + K[cid] for cid in sched}                         # (7)
            grid_bill = cost["energy"] + cost["tariffs"] + cost["penalty"]                      # D in (8)
            return Result("C", cost, member_cost, sched, imp, self.cap, lam, gamma,
                          payment, sum(payment.values()) - grid_bill)                           # B in (8)

        lam = {cid: self.con[cid]["balance"].Pi for cid in sched}
        gamma = {cid: -self.con[cid]["cap"].Pi for cid in sched} if self.model == "I" else None
        member_cost = {cid: float(self.p_imp @ s["q_imp"] - self.p_exp @ s["q_exp"]
                                  + self.alpha * s.get("q_exc", np.zeros(self.n)).sum()) + K[cid]
                       for cid, s in sched.items()}                                             # (3a), (5a)
        return Result(self.model, cost, member_cost, sched, imp, self.cap, lam, gamma)
