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


def add_member(model: gp.Model, member, n_hours: int):
    """Add the feasible set X_i (1a)-(1h) of one member to a Gurobi model. Shared by all models.

    Returns three things:
      variables    the member's decision variables, each with one value per hour (kWh)
      constraints  the member's constraints by name, to read their multipliers later
      own_cost     the member's own cost K_i (2): disutility plus PV production cost (DKK)
    """
    variables = {
        "l": model.addMVar(n_hours, lb=member.load_lower_kWh, ub=member.load_upper_kWh),   # (1g)
        "d_plus": model.addMVar(n_hours),
        "d_minus": model.addMVar(n_hours),
        "q_pv": model.addMVar(n_hours, ub=member.pv_available),                           # (1b)
        "q": model.addMVar(n_hours, lb=-GRB.INFINITY),
    }
    load, d_plus, d_minus = variables["l"], variables["d_plus"], variables["d_minus"]
    constraints = {
        "deviation": model.addConstr(load - member.reference_load == d_plus - d_minus),     # (1c)
        "min_energy": model.addConstr(load.sum() >= member.min_daily_energy_kWh),            # (1d)
    }
    battery_flow = 0
    if member.has_battery:
        p_ch = variables["p_ch"] = model.addMVar(n_hours, ub=member.battery_max_charge_kW)       # (1f)
        p_dis = variables["p_dis"] = model.addMVar(n_hours, ub=member.battery_max_discharge_kW)
        e = variables["e"] = model.addMVar(n_hours, ub=member.battery_capacity_kWh)
        e_init = member.battery_initial_soc_kWh
        stored = member.battery_charging_efficiency * p_ch - p_dis / member.battery_discharging_efficiency
        constraints["soc_first_hour"] = model.addConstr(e[0] == e_init + stored[0])        # (1e) with e_{-1} = e^init
        constraints["soc"] = model.addConstr(e[1:] == e[:-1] + stored[1:])                 # (1e)
        constraints["end_of_day"] = model.addConstr(e[n_hours - 1] >= e_init)              # (1h) cyclic end of day
        battery_flow = p_ch - p_dis
    constraints["net_consumption"] = model.addConstr(
        variables["q"] == load + battery_flow - variables["q_pv"])                         # (1a)
    own_cost = (member.linear_disutility * (d_plus.sum() + d_minus.sum())
                + member.pv_marginal_cost * variables["q_pv"].sum())                       # (2)
    return variables, constraints, own_cost


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
        contract = com.contract
        self.com, self.model = com, model
        self.n_hours = len(com.energy_price)
        self.cap = np.asarray(contract.capacity_limit_kW if cap is None else cap, dtype=float)
        self.alpha = contract.penalty_DKK_per_kWh if penalty is None else penalty
        beta = 0.0 if model == "R" else (contract.tariff_discount_ratio if discount is None else discount)
        # effective prices: p^imp and p^exp in R, the discounted prices of eq. (4) in I and C
        self.p_imp = com.energy_price + (1 - beta) * com.import_tariff
        self.p_exp = com.energy_price - (1 - beta) * com.export_tariff

        self.gurobi = gp.Model(f"model_{model}", env=_env())
        self.gurobi.Params.Method = 1    # dual simplex gives one reproducible dual solution
        self.variables, self.constraints, self.own_cost = {}, {}, {}
        for member in com.members:
            member_id = member.consumer_id
            (self.variables[member_id], self.constraints[member_id],
             self.own_cost[member_id]) = add_member(self.gurobi, member, self.n_hours)
        grid_cost = self._add_grid()
        self.gurobi.setObjective(grid_cost + sum(self.own_cost.values()), GRB.MINIMIZE)

    def _add_grid(self):
        """Grid side: own meters in R and I (3), (5) or one community meter in C (6). Returns its cost."""
        model, n_hours = self.gurobi, self.n_hours
        if self.model == "C":
            self.P_imp, self.P_exp, self.P_exc = model.addMVar(n_hours), model.addMVar(n_hours), model.addMVar(n_hours)
            total_net_consumption = sum(variables["q"] for variables in self.variables.values())
            # (6b) is written as supply minus demand: Gurobi's Pi is d(cost)/d(rhs), so Pi = lambda_t
            self.balance = model.addConstr(self.P_imp - self.P_exp - total_net_consumption == 0)
            self.cap_constraint = model.addConstr(self.P_imp - self.P_exc <= self.cap)       # (6c), gamma_t = -Pi
            return self.p_imp @ self.P_imp - self.p_exp @ self.P_exp + self.alpha * self.P_exc.sum()

        grid_cost = 0
        for member_id, variables in self.variables.items():
            q_imp = variables["q_imp"] = model.addMVar(n_hours)
            q_exp = variables["q_exp"] = model.addMVar(n_hours)
            self.constraints[member_id]["balance"] = model.addConstr(q_imp - q_exp - variables["q"] == 0)   # (3b), (5b)
            grid_cost += self.p_imp @ q_imp - self.p_exp @ q_exp
            if self.model == "I":
                q_exc = variables["q_exc"] = model.addMVar(n_hours)
                own_cap = self.com.contract.share[member_id] * self.cap
                self.constraints[member_id]["cap"] = model.addConstr(q_imp - q_exc <= own_cap)          # (5c)
                grid_cost += self.alpha * q_exc.sum()
        return grid_cost

    def solve(self) -> Result:
        """Solve the LP and collect the quantities used in the report."""
        self.gurobi.optimize()
        if self.gurobi.Status != GRB.OPTIMAL:
            raise RuntimeError(f"model {self.model} is not optimal (Gurobi status {self.gurobi.Status})")
        schedule = {member_id: {name: var.X for name, var in variables.items()}
                    for member_id, variables in self.variables.items()}
        own_cost = {member_id: float(cost.getValue()) for member_id, cost in self.own_cost.items()}

        # energy through the meter(s) that pay the grid
        if self.model == "C":
            grid_import, grid_export, above_cap = self.P_imp.X, self.P_exp.X, self.P_exc.X
        else:
            grid_import = sum(hourly["q_imp"] for hourly in schedule.values())
            grid_export = sum(hourly["q_exp"] for hourly in schedule.values())
            above_cap = (sum(hourly["q_exc"] for hourly in schedule.values()) if self.model == "I"
                         else np.zeros(self.n_hours))
        p = self.com.energy_price
        members = {member.consumer_id: member for member in self.com.members}
        cost = {
            "energy": float(p @ (grid_import - grid_export)),
            "tariffs": float((self.p_imp - p) @ grid_import + (p - self.p_exp) @ grid_export),
            "penalty": float(self.alpha * above_cap.sum()),
            "disutility": sum(members[member_id].linear_disutility * (hourly["d_plus"] + hourly["d_minus"]).sum()
                              for member_id, hourly in schedule.items()),
            "pv": sum(members[member_id].pv_marginal_cost * hourly["q_pv"].sum()
                      for member_id, hourly in schedule.items()),
        }
        cost["total"] = sum(cost.values())
        assert abs(cost["total"] - self.gurobi.ObjVal) < 1e-6, "cost parts do not add up to the objective"

        if self.model == "C":
            price, cap_price = self.balance.Pi, -self.cap_constraint.Pi
            payment = {member_id: float(price @ hourly["q"]) for member_id, hourly in schedule.items()}   # (7)
            member_cost = {member_id: payment[member_id] + own_cost[member_id] for member_id in schedule}  # (7)
            dso_bill = cost["energy"] + cost["tariffs"] + cost["penalty"]                                  # D in (8)
            surplus = sum(payment.values()) - dso_bill                                                     # B in (8)
            return Result("C", cost, member_cost, schedule, grid_import, self.cap, price, cap_price, payment, surplus)

        price = {member_id: self.constraints[member_id]["balance"].Pi for member_id in schedule}
        cap_price = ({member_id: -self.constraints[member_id]["cap"].Pi for member_id in schedule}
                     if self.model == "I" else None)
        member_cost = {
            member_id: float(self.p_imp @ hourly["q_imp"] - self.p_exp @ hourly["q_exp"]
                             + self.alpha * hourly.get("q_exc", np.zeros(self.n_hours)).sum()) + own_cost[member_id]
            for member_id, hourly in schedule.items()
        }                                                                                                  # (3a), (5a)
        return Result(self.model, cost, member_cost, schedule, grid_import, self.cap, price, cap_price)
