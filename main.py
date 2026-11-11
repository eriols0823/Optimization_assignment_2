"""Entry point of the Assignment 2 starter package: load a community and summarize it.

    python main.py                                   # base case (16 members), day d1
    python main.py --community my_mix                # your own data/params_community_my_mix.json
    python main.py --day d4                          # another day of the week (and that day's cap)
    python main.py --members                         # also print every member's Assignment 1 summary
    python main.py --demo                            # also run the scenario-generation demonstration (Question 3)
    python main.py --q1 validate                     # Question 1: validation checks of the models R, I and C
    python main.py --q1 base                         # Question 1: base-case table (results/<community>/)
    python main.py --q1 sweep                        # Question 1: sensitivity to the cap and the discount
    python main.py --q1 settlement                   # Question 1: dual settlement of C and the manager's budget
    python main.py --q2 validate                     # Question 2: DP^cap against C (equivalence with the duals of C)
    python main.py --q2 steps                        # Question 2: cap -> + budget balance -> + individual rationality
    python main.py --q2 indifference                 # Question 2: indifferent member-hours at the prices of C

No model is provided: build your Assignment 1 model class on the ``InputData`` of each member, then the community
models of the assignment (R, I, C, DP, ...) on top of it, one function per experiment, writing to ``results/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_community, list_communities

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default="base", help=f"base (shipped) or your own params_community_<name>.json; available: {list_communities()}")
    ap.add_argument("--day", default="d1", help="d1 (base case) ... d7")
    ap.add_argument("--members", action="store_true", help="print the Assignment 1 summary of every member")
    ap.add_argument("--demo", action="store_true", help="run the scenario-generation demonstration")
    ap.add_argument("--q1", choices=["validate", "base", "sweep", "settlement"], help="Question 1 experiment to run")
    ap.add_argument("--q2", choices=["validate", "steps", "indifference"], help="Question 2 experiment to run")
    args = ap.parse_args()

    com = load_community(args.community, day=args.day)
    print(com.summary(), "\n")
    if args.members:
        for m in com.members:
            print(m.summary(), "\n")
    print("Cap profile (kW):", " ".join(f"{x:.1f}" for x in com.contract.capacity_limit_kW))
    ic = com.contract.individual_capacity_limit_kW
    print("Individual caps in model I (kW), hour 0 / hour 11:",
          "  ".join(f"{cid} ({com.member(cid).archetype}) {ic[cid][0]:.2f}/{ic[cid][11]:.2f}" for cid in ("C1", "C1_PVB", "C4", "C4_PVB") if cid in com.member_ids))
    (RESULTS_DIR / args.community).mkdir(parents=True, exist_ok=True)
    if args.demo:
        import runpy
        runpy.run_module("src.scenario_generation", run_name="__main__")
    if args.q1:
        from src import q1
        print()
        if args.q1 == "validate":
            q1.validate(com)
        elif args.q1 == "base":
            q1.base_case(com, RESULTS_DIR / args.community)
        elif args.q1 == "settlement":
            q1.settlement(com, RESULTS_DIR / args.community)
        else:
            q1.sweep(com, RESULTS_DIR / args.community)
    if args.q2:
        from src import q2
        print()
        if args.q2 == "validate":
            q2.validate(com)
        elif args.q2 == "steps":
            q2.steps(com, RESULTS_DIR / args.community)
        else:
            q2.indifference(com, RESULTS_DIR / args.community)


if __name__ == "__main__":
    main()
