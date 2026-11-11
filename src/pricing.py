"""Question 2: the dynamic pricing problem DP as one parametrized model (bilevel -> MPEC -> MILP).

The community manager (leader) posts one internal price x_t per hour. Every member (follower) then solves its own
linear program at these prices: the member model of Question 1 (feasible set X_i, own cost K_i) with the payment
sum_t x_t q_it in place of its grid bill. The manager pays the DSO at the connection point and maximizes the
community's welfare, anticipating the members' best responses.

Single-level reformulation (report, Section 2):
  1. every member's LP is replaced by its KKT conditions (exact: the LP is convex and its constraints are linear);
  2. every complementarity condition 0 <= a  _|_  b >= 0 is written as an SOS1 pair (default) or with big-M binaries;
  3. the bilinear payment sum_t x_t q_it, needed by budget balance and individual rationality, is replaced by the
     member's dual objective minus its own cost (strong duality of the member's LP), which is linear.

Variants (superscripts of the report), one switch at a time:
  requirements   "cap"    DP^cap     the contract only
                 "bb"     DP^BB      + budget balance:            sum_i Pi_i = D        (bb="eq")  or >= D (bb="geq")
                 "bb_ir"  DP^BB,IR   + individual rationality:    J_i^DP <= J_i^bench   for every member i
  ir_benchmark   "R", "I" or "RI" (the better of the two for each member)
  ir_soft        True: IR_i may be violated by s_i >= 0 at the price ir_penalty DKK per DKK (the blocking members)
  price_groups   "uniform" (one price per hour, the assignment's DP), "asset" (one per asset class), "member"
                 (one per member), or a dict {consumer_id: group}: price discrimination as a way out of infeasibility
  price_bounds   "wide" (0 <= x_t <= max_t p_imp_t + alpha) or "lambdaC" (p_exp_t <= x_t <= p_imp_t + alpha, the range of
                 the dual price of C), or a pair of arrays (lo, hi)

Units: energies in kWh per hour, prices in DKK/kWh, costs in DKK.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from src.models import CommunityModel, _env

REQUIREMENTS = ("cap", "bb", "bb_ir")


# ======================================================================================================================
# Followers: one member's best response, as a linear program or through its KKT conditions
# ======================================================================================================================
def best_response(member, prices, n_hours: int = 24) -> dict:
    """Solve member i's own LP at posted prices x_t (DKK/kWh). Returns its schedule, cost J_i and payment Pi_i.

    Used to check that the members' KKT system of the MILP describes the same optimum as the LP itself."""
    from src.models import add_member
    m = gp.Model("best_response", env=_env())
    variables, _, own_cost = add_member(m, member, n_hours)
    m.setObjective(np.asarray(prices) @ variables["q"] + own_cost, GRB.MINIMIZE)
    m.optimize()
    q = variables["q"].X
    return {"cost": m.ObjVal, "payment": float(np.asarray(prices) @ q), "own_cost": float(own_cost.getValue()), "q": q,
            "schedule": {k: v.X for k, v in variables.items()}}


def response_range(member, prices, n_hours: int = 24, tol: float = 1e-7) -> tuple[np.ndarray, np.ndarray]:
    """Lowest and highest net consumption q_it of member i over ALL its optimal responses to the prices (kWh).

    The best response of an LP is a set (a face of X_i). lo_t < hi_t marks an indifferent member-hour: the member
    can move energy into or out of hour t at no cost to itself. Solved as 2 x 24 small LPs with the cost fixed at
    its optimum (+ tol)."""
    from src.models import add_member
    prices = np.asarray(prices, dtype=float)
    m = gp.Model("response_range", env=_env())
    variables, _, own_cost = add_member(m, member, n_hours)
    cost = prices @ variables["q"] + own_cost
    m.setObjective(cost, GRB.MINIMIZE)
    m.optimize()
    m.addConstr(cost <= m.ObjVal + tol)
    lo, hi = np.zeros(n_hours), np.zeros(n_hours)
    q = variables["q"]
    for t in range(n_hours):
        for sense, out in ((GRB.MINIMIZE, lo), (GRB.MAXIMIZE, hi)):
            m.setObjective(q[t], sense)
            m.optimize()
            out[t] = m.ObjVal
    return lo, hi


def pessimistic_import(com, prices: dict[str, np.ndarray] | np.ndarray, groups: dict[str, str] | None = None):
    """Hourly import at the meter if every member breaks its ties against the manager, hour by hour.

    Returns (lo, hi, ranges): the lowest and highest possible community net import per hour over the members'
    optimal sets (members are independent, so the extremes of the sum are the sums of the extremes), and the
    per-member ranges {consumer_id: (lo, hi)}. Comparing hi with the cap gives the manager's exposure to the
    optimistic assumption of the MPEC."""
    ranges = {}
    for mem in com.members:
        x = prices if isinstance(prices, np.ndarray) else prices[(groups or {}).get(mem.consumer_id, "all")]
        ranges[mem.consumer_id] = response_range(mem, x, len(com.energy_price))
    lo = sum(r[0] for r in ranges.values())
    hi = sum(r[1] for r in ranges.values())
    return lo, hi, ranges


class _Complementarity:
    """Writes 0 <= a _|_ b >= 0 for lists of variables, as SOS1 pairs or with one binary and two big-M constants."""

    def __init__(self, model: gp.Model, method: str, big_m_dual: float):
        self.model, self.method, self.big_m_dual = model, method, big_m_dual
        self.pairs: list[tuple[gp.Var, gp.Var, str, float, bool]] = []

    def add(self, a, b, big_m_primal, name: str, big_m_dual=None):
        """big_m_primal, big_m_dual: valid upper bounds of a and b (scalars or one per pair). The dual bound defaults to
        the generic constant; tighter bounds derived from the KKT conditions give a stronger LP relaxation."""
        derived = big_m_dual is not None                  # a bound derived from the KKT conditions, valid by construction
        a, b = self._as_list(a), self._as_list(b)
        big_m_primal = np.broadcast_to(np.asarray(big_m_primal, dtype=float), (len(a),))
        big_m_dual = np.broadcast_to(np.asarray(self.big_m_dual if big_m_dual is None else big_m_dual, dtype=float),
                                     (len(a),))
        for k, (ak, bk) in enumerate(zip(a, b)):
            self.pairs.append((ak, bk, f"{name}[{k}]", float(big_m_dual[k]), derived))
            if self.method == "sos1":
                self.model.addSOS(GRB.SOS_TYPE1, [ak, bk])
            else:
                u = self.model.addVar(vtype=GRB.BINARY)
                self.model.addConstr(ak <= big_m_primal[k] * u)
                self.model.addConstr(bk <= big_m_dual[k] * (1 - u))

    @staticmethod
    def _as_list(v) -> list:
        return [v] if isinstance(v, gp.Var) else list(np.atleast_1d(v.tolist()))

    def worst_residual(self) -> float:
        """Largest product a * b over all pairs after solving (0 if every condition holds)."""
        return max((abs(a.X * b.X) for a, b, _, _, _ in self.pairs), default=0.0)

    def at_big_m(self, tol: float = 1e-6) -> list[str]:
        """Pairs whose dual side reached the generic big-M constant (M may be too small: increase it and re-solve).
        Pairs with a bound derived from the KKT conditions can sit at it legitimately (e.g. c^L + nu = 2 c^L when the
        load deviates downwards) and are not reported."""
        if self.method == "sos1":
            return []
        return [name for _, b, name, M, derived in self.pairs if not derived and b.X >= M - tol]


def add_member_kkt(model: gp.Model, member, price, comp: _Complementarity, n_hours: int = 24,
                   price_max: float | None = None) -> dict:
    """Add member i's best response to the posted prices `price` (24 Gurobi variables or numbers) as KKT conditions.

    Member i's LP (report eq. Q2-member):
        min  sum_t price_t q_t + K_i      s.t. the feasible set X_i of Question 1,  q_t = l_t + p_ch_t - p_dis_t - g_t
    Returns its primal variables, its dual variables, the net consumption q, its own cost K_i, and its dual objective
    (= its optimal cost J_i = payment + K_i by strong duality, linear in the dual variables).

    price_max: upper bound of the prices (DKK/kWh). With 0 <= x_t <= price_max the KKT conditions bound the duals of
    the load and the PV (report, Appendix): |nu_t| <= c^L, mu_E <= price_max + c^L, mu_l_lo <= price_max + c^L,
    mu_l_up <= price_max + 2 c^L, mu_g_lo <= c^PV, mu_g_up <= price_max - c^PV. These bounds are the big-M constants of
    those pairs; the battery's duals keep the generic constant.
    """
    T = n_hours
    cL, cPV = member.linear_disutility, member.pv_marginal_cost
    xm = comp.big_m_dual if price_max is None else float(price_max)
    l_ref, l_lo, l_up = member.reference_load, member.load_lower_kWh, member.load_upper_kWh
    E_min = member.min_daily_energy_kWh

    # ---------------- primal variables and primal feasibility
    l = model.addMVar(T, lb=l_lo, ub=l_up)
    d_plus, d_minus = model.addMVar(T), model.addMVar(T)
    model.addConstr(l - l_ref == d_plus - d_minus)                                  # : nu_t (free)
    primal = {"l": l, "d_plus": d_plus, "d_minus": d_minus}
    s_l_lo, s_l_up = model.addMVar(T), model.addMVar(T)                              # slacks of the load bounds
    model.addConstr(s_l_lo == l - l_lo)
    model.addConstr(s_l_up == l_up - l)
    s_E = model.addVar()                                                             # slack of the daily minimum
    model.addConstr(s_E == l.sum() - E_min)                                          # : mu_E >= 0

    pv_hours = np.where(member.pv_available > 1e-9)[0] if member.has_pv else np.array([], dtype=int)
    net = l
    if len(pv_hours):
        G = member.pv_available[pv_hours]
        g = model.addMVar(len(pv_hours), ub=G)
        s_g_up = model.addMVar(len(pv_hours))
        model.addConstr(s_g_up == G - g)
        primal["q_pv_hours"] = g
        pv_matrix = np.zeros((T, len(pv_hours))); pv_matrix[pv_hours, np.arange(len(pv_hours))] = 1.0
        net = net - pv_matrix @ g
    if member.has_battery:
        C_max, D_max, S_max = member.battery_max_charge_kW, member.battery_max_discharge_kW, member.battery_capacity_kWh
        eta_c, eta_d, e_init = member.battery_charging_efficiency, member.battery_discharging_efficiency, member.battery_initial_soc_kWh
        p_ch, p_dis, e = model.addMVar(T, ub=C_max), model.addMVar(T, ub=D_max), model.addMVar(T, ub=S_max)
        stored = eta_c * p_ch - p_dis / eta_d
        model.addConstr(e[0] == e_init + stored[0])                                  # : psi_0 (free)
        model.addConstr(e[1:] == e[:-1] + stored[1:])                                # : psi_t (free)
        s_end = model.addVar()
        model.addConstr(s_end == e[T - 1] - e_init)                                  # : mu_end >= 0 (end of day)
        s_ch_up, s_dis_up, s_e_up = model.addMVar(T), model.addMVar(T), model.addMVar(T)
        model.addConstr(s_ch_up == C_max - p_ch)
        model.addConstr(s_dis_up == D_max - p_dis)
        model.addConstr(s_e_up == S_max - e)
        primal.update(p_ch=p_ch, p_dis=p_dis, e=e)
        net = net + p_ch - p_dis
    q = model.addMVar(T, lb=-GRB.INFINITY)
    model.addConstr(q == net)
    primal["q"] = q
    own_cost = cL * (d_plus.sum() + d_minus.sum()) + (cPV * primal["q_pv_hours"].sum() if len(pv_hours) else 0.0)

    # ---------------- dual variables and dual feasibility (stationarity of the Lagrangian)
    nu = model.addMVar(T, lb=-cL, ub=cL)                                             # value of the deviation, |nu| <= c^L
    r_plus, r_minus = model.addMVar(T), model.addMVar(T)                             # reduced costs of d+ and d-
    model.addConstr(r_plus == cL + nu)
    model.addConstr(r_minus == cL - nu)
    mu_E = model.addVar()
    mu_l_lo, mu_l_up = model.addMVar(T), model.addMVar(T)
    model.addConstr(price - nu - mu_E - mu_l_lo + mu_l_up == 0)                     # d L / d l_t
    dual = {"nu": nu, "mu_E": mu_E, "mu_l_lo": mu_l_lo, "mu_l_up": mu_l_up}
    dual_obj = nu @ l_ref + mu_E * E_min + mu_l_lo @ l_lo - mu_l_up @ l_up

    comp.add(s_l_lo, mu_l_lo, l_up - l_lo, f"{member.consumer_id}.l_lo", xm + cL)
    comp.add(s_l_up, mu_l_up, l_up - l_lo, f"{member.consumer_id}.l_up", xm + 2 * cL)
    comp.add(d_plus, r_plus, member.max_hourly_deviation_kWh, f"{member.consumer_id}.d_plus", 2 * cL)
    comp.add(d_minus, r_minus, member.max_hourly_deviation_kWh, f"{member.consumer_id}.d_minus", 2 * cL)
    comp.add(s_E, mu_E, float(l_up.sum() - E_min), f"{member.consumer_id}.E_min", xm + cL)

    if len(pv_hours):
        mu_g_lo, mu_g_up = model.addMVar(len(pv_hours)), model.addMVar(len(pv_hours))
        price_pv = price[pv_hours] if isinstance(price, (gp.MVar, np.ndarray)) else np.asarray(price)[pv_hours]
        model.addConstr(-price_pv + cPV - mu_g_lo + mu_g_up == 0)                   # d L / d g_t
        comp.add(g, mu_g_lo, G, f"{member.consumer_id}.g_lo", cPV)
        comp.add(s_g_up, mu_g_up, G, f"{member.consumer_id}.g_up", max(xm - cPV, 1e-3))
        dual.update(mu_g_lo=mu_g_lo, mu_g_up=mu_g_up)
        dual_obj = dual_obj - mu_g_up @ G
    if member.has_battery:
        psi = model.addMVar(T, lb=-comp.big_m_dual, ub=comp.big_m_dual)               # SoC balance (free; bounded for M)
        mu_end = model.addVar()
        mu_ch_lo, mu_ch_up = model.addMVar(T), model.addMVar(T)
        mu_dis_lo, mu_dis_up = model.addMVar(T), model.addMVar(T)
        mu_e_lo, mu_e_up = model.addMVar(T), model.addMVar(T)
        model.addConstr(price + eta_c * psi - mu_ch_lo + mu_ch_up == 0)              # d L / d p_ch_t
        model.addConstr(-price - psi / eta_d - mu_dis_lo + mu_dis_up == 0)           # d L / d p_dis_t
        model.addConstr(-psi[:-1] + psi[1:] - mu_e_lo[:-1] + mu_e_up[:-1] == 0)      # d L / d e_t, t < 23
        model.addConstr(-psi[T - 1] - mu_end - mu_e_lo[T - 1] + mu_e_up[T - 1] == 0) # d L / d e_23
        comp.add(p_ch, mu_ch_lo, C_max, f"{member.consumer_id}.ch_lo")
        comp.add(s_ch_up, mu_ch_up, C_max, f"{member.consumer_id}.ch_up")
        comp.add(p_dis, mu_dis_lo, D_max, f"{member.consumer_id}.dis_lo")
        comp.add(s_dis_up, mu_dis_up, D_max, f"{member.consumer_id}.dis_up")
        comp.add(e, mu_e_lo, S_max, f"{member.consumer_id}.e_lo")
        comp.add(s_e_up, mu_e_up, S_max, f"{member.consumer_id}.e_up")
        comp.add(s_end, mu_end, S_max, f"{member.consumer_id}.end")
        dual.update(psi=psi, mu_end=mu_end, mu_ch_up=mu_ch_up, mu_dis_up=mu_dis_up, mu_e_up=mu_e_up)
        dual_obj = (dual_obj + psi[0] * e_init + mu_end * e_init
                    - C_max * mu_ch_up.sum() - D_max * mu_dis_up.sum() - S_max * mu_e_up.sum())
    return {"primal": primal, "dual": dual, "q": q, "own_cost": own_cost, "dual_obj": dual_obj}


# ======================================================================================================================
# Leader: the community manager
# ======================================================================================================================
@dataclass
class PricingResult:
    """Solution of one variant of DP. Hourly quantities are arrays of 24 values."""

    variant: str
    status: str                                   # "optimal", "time limit", "infeasible", ...
    objective: float | None = None                # community cost J^DP = D + sum_i K_i (+ penalty on IR slack) (DKK)
    gap: float | None = None
    runtime: float | None = None
    prices: dict[str, np.ndarray] = field(default_factory=dict)       # group -> x_t (DKK/kWh)
    member_price: dict[str, np.ndarray] = field(default_factory=dict) # consumer_id -> the price it faces
    q: dict[str, np.ndarray] = field(default_factory=dict)            # consumer_id -> net consumption q_it (kWh)
    schedule: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    member_cost: dict[str, float] = field(default_factory=dict)       # J_i^DP = Pi_i + K_i (DKK)
    payment: dict[str, float] = field(default_factory=dict)           # Pi_i = sum_t x_t q_it (DKK)
    own_cost: dict[str, float] = field(default_factory=dict)          # K_i (DKK)
    ir_slack: dict[str, float] = field(default_factory=dict)          # s_i (DKK), soft IR only
    P_imp: np.ndarray | None = None
    P_exp: np.ndarray | None = None
    P_exc: np.ndarray | None = None
    dso_bill: float | None = None                                     # D (DKK)
    cost: dict[str, float] = field(default_factory=dict)              # community cost by part (DKK)
    checks: dict[str, float] = field(default_factory=dict)            # validation residuals
    size: dict[str, int] = field(default_factory=dict)

    @property
    def surplus(self) -> float | None:
        """Manager's budget B = sum_i Pi_i - D (DKK); 0 under budget balance."""
        return None if self.dso_bill is None else sum(self.payment.values()) - self.dso_bill

    @property
    def feasible(self) -> bool:
        return self.objective is not None


class PricingModel:
    """Build and solve one variant of the pricing problem DP for one community on one day.

    com            CommunityData from load_community()
    requirements   "cap", "bb" or "bb_ir" (see the module docstring)
    bb             "eq" (payments collected = DSO bill) or "geq" (payments collected >= DSO bill)
    ir_benchmark   "R", "I" or "RI"; benchmarks: optional {consumer_id: J_i} to override the computed ones
    ir_soft        allow IR violations s_i >= 0 at ir_penalty DKK per DKK in the objective
    price_groups   "uniform", "asset", "member" or {consumer_id: group}
    price_bounds   "wide", "lambdaC" or (lo, hi)
    discount, penalty, cap   contract terms (defaults: the contract file), as in CommunityModel
    complementarity          "sos1" or "bigm" (big_m_dual: the constant on the dual side)
    no_waste       forbid simultaneous import and export and a penalty below the cap at the meter (binaries); needed
                   with bb="eq", otherwise the manager could balance its budget by wasting money at the meter
    fix_dispatch   None, or a Result of model C: every member's schedule is fixed to it. The complementarity pattern is
                   then known and the model is an LP in the prices and duals: "which prices make the dispatch of C a
                   best response of every member, and satisfy the requirements?" If it is feasible, the variant reaches
                   J^C and is optimal (J^C is a lower bound of every variant of DP).
    welfare_bound  add the valid inequality D + sum_i K_i >= J^C (the inducible region lies in the feasible set of C)
    start_prices   prices for the MIP start (default: the dual prices lambda^C of C, clipped to the bounds)
    """

    def __init__(self, com, requirements: str = "cap", bb: str = "eq", ir_benchmark: str = "R", ir_soft: bool = False,
                 ir_penalty: float = 1e3, price_groups="uniform", price_bounds="wide", discount=None, penalty=None,
                 cap=None, complementarity: str = "bigm", big_m_dual: float = 50.0, no_waste: bool | None = None,
                 benchmarks: dict[str, float] | None = None, time_limit: float = 300, mip_gap: float = 1e-6,
                 warm_start: bool = True, fix_dispatch=None, welfare_bound: bool = True, start_prices=None,
                 gurobi_params: dict | None = None, verbose: bool = False):
        if requirements not in REQUIREMENTS:
            raise ValueError(f"requirements must be one of {REQUIREMENTS}")
        self.com, self.requirements, self.bb, self.ir_benchmark = com, requirements, bb, ir_benchmark
        self.ir_soft, self.ir_penalty = ir_soft, ir_penalty
        self.T = T = len(com.energy_price)
        self.members = {m.consumer_id: m for m in com.members}

        # contract terms and discounted prices at the meter, as in model C
        self.C_model = CommunityModel(com, "C", discount=discount, penalty=penalty, cap=cap)
        self.p_imp, self.p_exp = self.C_model.p_imp, self.C_model.p_exp
        self.alpha, self.cap = self.C_model.alpha, self.C_model.cap
        self.terms = dict(discount=discount, penalty=penalty, cap=cap)

        # price groups and bounds
        self.group = self._groups(price_groups)
        lo, hi = self._bounds(price_bounds)
        self.price_lo, self.price_hi = lo, hi
        self.variant = self._name(price_groups, price_bounds)

        # benchmarks of individual rationality (same contract terms as DP)
        self.benchmark = benchmarks if benchmarks is not None else (
            self.benchmarks(com, ir_benchmark, **self.terms) if requirements == "bb_ir" else {})

        m = self.gurobi = gp.Model(f"DP_{self.variant}", env=_env())
        m.Params.TimeLimit, m.Params.MIPGap = time_limit, mip_gap
        m.Params.OutputFlag = int(verbose)
        m.Params.IntegralityFocus = 1                         # complementarity must hold, not almost hold
        for key, value in (gurobi_params or {}).items():
            m.setParam(key, value)
        self.comp = _Complementarity(m, complementarity, big_m_dual)

        # leader's prices: one vector per group
        self.x = {g: m.addMVar(T, lb=lo, ub=hi, name=f"x_{g}") for g in sorted(set(self.group.values()))}
        # followers: one KKT system per member
        price_max = float(hi.max()) if lo.min() >= 0 else None       # the tight big-M need 0 <= x_t <= price_max
        self.follower = {cid: add_member_kkt(m, mem, self.x[self.group[cid]], self.comp, T, price_max)
                         for cid, mem in self.members.items()}

        # connection point (as in C)
        self.P_imp, self.P_exp, self.P_exc = m.addMVar(T), m.addMVar(T), m.addMVar(T)
        total_q = sum(f["q"] for f in self.follower.values())
        m.addConstr(self.P_imp - self.P_exp == total_q)                                  # internal balance
        m.addConstr(self.P_imp - self.P_exc <= self.cap)                                 # soft cap
        self.no_waste = (requirements != "cap" and bb == "eq") if no_waste is None else no_waste
        if self.no_waste:
            big = sum(mem.load_max_kWh + (mem.battery_max_charge_kW or 0) + mem.pv_max_kW for mem in com.members)
            u, z = m.addMVar(T, vtype=GRB.BINARY), m.addMVar(T, vtype=GRB.BINARY)
            m.addConstr(self.P_imp <= big * u)
            m.addConstr(self.P_exp <= big * (1 - u))
            m.addConstr(self.P_exc <= big * z)
            m.addConstr(self.P_exc <= self.P_imp - self.cap + big * (1 - z))

        self.dso_bill = self.p_imp @ self.P_imp - self.p_exp @ self.P_exp + self.alpha * self.P_exc.sum()
        own = sum(f["own_cost"] for f in self.follower.values())
        payments = sum(f["dual_obj"] - f["own_cost"] for f in self.follower.values())   # sum_i Pi_i (strong duality)
        objective = self.dso_bill + own
        self.J_C = self.C_model.solve().cost["total"]
        if welfare_bound:
            m.addConstr(objective >= self.J_C - 1e-6, name="welfare_bound")             # valid: DP >= C
        if fix_dispatch is not None:
            self._fix(fix_dispatch)

        if requirements in ("bb", "bb_ir"):
            if bb == "eq":
                m.addConstr(payments == self.dso_bill, name="budget_balance")
            else:
                m.addConstr(payments >= self.dso_bill, name="budget_balance")
        self.s = {}
        if requirements == "bb_ir":
            for cid, f in self.follower.items():
                if ir_soft:
                    self.s[cid] = m.addVar()
                    m.addConstr(f["dual_obj"] <= self.benchmark[cid] + self.s[cid], name=f"IR_{cid}")
                else:
                    m.addConstr(f["dual_obj"] <= self.benchmark[cid], name=f"IR_{cid}")
            if ir_soft:
                objective = objective + ir_penalty * gp.quicksum(self.s.values())
        m.setObjective(objective, GRB.MINIMIZE)

        self.start_prices = start_prices
        if warm_start:
            self._warm_start()

    # ------------------------------------------------------------------ set-up helpers
    def _groups(self, price_groups) -> dict[str, str]:
        if isinstance(price_groups, dict):
            return dict(price_groups)
        if price_groups == "uniform":
            return {cid: "all" for cid in self.members}
        if price_groups == "member":
            return {cid: cid for cid in self.members}
        if price_groups == "asset":
            return {cid: ("PV" if m.has_pv else "") + ("B" if m.has_battery else "") or "none"
                    for cid, m in self.members.items()}
        raise ValueError("price_groups must be 'uniform', 'asset', 'member' or a dict")

    def _bounds(self, price_bounds):
        if isinstance(price_bounds, tuple):
            return (np.broadcast_to(np.asarray(price_bounds[0], float), (self.T,)).copy(),
                    np.broadcast_to(np.asarray(price_bounds[1], float), (self.T,)).copy())
        if price_bounds == "wide":
            return np.zeros(self.T), np.full(self.T, self.p_imp.max() + self.alpha)
        if price_bounds == "lambdaC":
            return self.p_exp.copy(), self.p_imp + self.alpha
        raise ValueError("price_bounds must be 'wide', 'lambdaC' or (lo, hi)")

    def _name(self, price_groups, price_bounds) -> str:
        parts = {"cap": "cap", "bb": "BB", "bb_ir": "BB,IR"}[self.requirements]
        if self.requirements != "cap" and self.bb == "geq":
            parts = parts.replace("BB", "BB>=")
        if self.requirements == "bb_ir":
            parts += f"({self.ir_benchmark}{',soft' if self.ir_soft else ''})"
        if price_groups != "uniform":
            parts += f",{price_groups if isinstance(price_groups, str) else 'groups'}"
        if price_bounds != "wide":
            parts += f",{price_bounds if isinstance(price_bounds, str) else 'bounds'}"
        return parts

    @staticmethod
    def benchmarks(com, rule: str = "R", discount=None, penalty=None, cap=None) -> dict[str, float]:
        """Cost J_i of every member in the benchmark of individual rationality: R, I or the better of the two (RI)."""
        J_R = CommunityModel(com, "R").solve().member_cost
        if rule == "R":
            return J_R
        J_I = CommunityModel(com, "I", discount=discount, penalty=penalty, cap=cap).solve().member_cost
        if rule == "I":
            return J_I
        return {cid: min(J_R[cid], J_I[cid]) for cid in J_R}

    def _warm_start(self):
        """MIP start: the given prices, or the dual prices of C (optimal for DP^cap): x_t = lambda_t^C, clipped."""
        if self.start_prices is not None:
            for g, x in self.x.items():
                x.Start = self.start_prices[g] if isinstance(self.start_prices, dict) else self.start_prices
            return
        lam = np.clip(self.C_model.solve().price, self.price_lo, self.price_hi)
        for x in self.x.values():
            x.Start = lam

    def _fix(self, C):
        """Fix every member's schedule and the exchange at the meter to the solution C of model C."""
        names = {"l": "l", "d_plus": "d_plus", "d_minus": "d_minus", "p_ch": "p_ch", "p_dis": "p_dis", "e": "e", "q": "q"}
        for cid, f in self.follower.items():
            s = C.schedule[cid]
            for mine, theirs in names.items():
                if mine in f["primal"]:
                    v = f["primal"][mine]
                    v.lb = v.ub = s[theirs]
            if "q_pv_hours" in f["primal"]:
                hours = np.where(self.members[cid].pv_available > 1e-9)[0]
                v = f["primal"]["q_pv_hours"]
                v.lb = v.ub = s["q_pv"][hours]
        net = sum(s["q"] for s in C.schedule.values())
        P_imp = C.feeder_import
        for mine, value in ((self.P_imp, P_imp), (self.P_exp, P_imp - net), (self.P_exc, np.maximum(0.0, P_imp - self.cap))):
            mine.lb = mine.ub = np.maximum(0.0, value)

    def size(self) -> dict[str, int]:
        m = self.gurobi
        m.update()
        return {"variables": m.NumVars, "binaries": m.NumBinVars, "SOS1": m.NumSOS,
                "complementarities": len(self.comp.pairs), "constraints": m.NumConstrs}

    # ------------------------------------------------------------------ solve
    def solve(self) -> PricingResult:
        """Solve the MILP and collect prices, schedules, payments, the budget and the validation residuals."""
        m = self.gurobi
        size = self.size()
        m.optimize()
        status = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time limit", GRB.INFEASIBLE: "infeasible",
                  GRB.INF_OR_UNBD: "infeasible"}.get(m.Status, f"gurobi status {m.Status}")
        if m.SolCount == 0:
            return PricingResult(self.variant, status, runtime=m.Runtime, size=size)

        res = PricingResult(self.variant, status, objective=m.ObjVal, gap=m.MIPGap, runtime=m.Runtime, size=size)
        res.prices = {g: x.X for g, x in self.x.items()}
        res.P_imp, res.P_exp, res.P_exc = self.P_imp.X, self.P_exp.X, self.P_exc.X
        res.dso_bill = float(self.dso_bill.getValue())
        sd_residual = 0.0
        for cid, f in self.follower.items():
            x_i = res.prices[self.group[cid]]
            q = f["q"].X
            own = float(f["own_cost"].getValue()) if not isinstance(f["own_cost"], float) else f["own_cost"]
            res.member_price[cid], res.q[cid] = x_i, q
            res.schedule[cid] = {k: v.X for k, v in f["primal"].items()}
            res.payment[cid] = float(x_i @ q)                                  # Pi_i evaluated directly
            res.own_cost[cid] = own
            res.member_cost[cid] = res.payment[cid] + own                     # J_i^DP
            sd_residual = max(sd_residual, abs(float(f["dual_obj"].getValue()) - res.member_cost[cid]))
        res.ir_slack = {cid: s.X for cid, s in self.s.items()}
        p = self.com.energy_price
        res.cost = {
            "energy": float(p @ (res.P_imp - res.P_exp)),
            "tariffs": float((self.p_imp - p) @ res.P_imp + (p - self.p_exp) @ res.P_exp),
            "penalty": float(self.alpha * res.P_exc.sum()),
            "disutility": sum(self.members[c].linear_disutility * (s["d_plus"] + s["d_minus"]).sum()
                              for c, s in res.schedule.items()),
            "pv": sum(self.members[c].pv_marginal_cost * s.get("q_pv_hours", np.zeros(1)).sum()
                      for c, s in res.schedule.items()),
        }
        res.cost["total"] = sum(res.cost.values())
        res.checks = {
            "strong duality residual [DKK]": sd_residual,                    # dual objective = primal cost, per member
            "complementarity residual": self.comp.worst_residual(),          # largest a * b
            "pairs at big-M": len(self.comp.at_big_m()),
            "simultaneous import/export [kWh]": float(np.minimum(res.P_imp, res.P_exp).max()),
        }
        return res

    def verify_best_responses(self, res: PricingResult, tol: float = 1e-5) -> dict[str, float]:
        """Re-solve every member's own LP at the posted prices: the cost in the MILP must equal the LP optimum.
        Returns {consumer_id: J_i in the MILP - J_i of the LP} (0 when the anticipated schedule is a best response)."""
        out = {}
        for cid, mem in self.members.items():
            br = best_response(mem, res.member_price[cid], self.T)
            out[cid] = res.member_cost[cid] - br["cost"]
        return out
