"""
Summarize a directory of `robo_nav.sim.nav --out` episode logs: success rate per hint mode, broken down
by route length (rooms on the plan), plus how the failures ended.

  python -m robo_nav.sim.summary sim_data/runs/batch2
"""

import argparse
import collections
import glob
import json
import os


def plan_hops(ep: dict) -> int:
    start = next(e for e in ep["log"] if e["event"] == "start")
    return len(start["plan"].split(" -> ")) - 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str)
    args = parser.parse_args()

    eps = []
    for path in sorted(glob.glob(os.path.join(args.run_dir, "*.json"))):
        with open(path) as f:
            ep = json.load(f)
        ep["file"] = os.path.basename(path)
        ep["hops"] = plan_hops(ep)
        eps.append(ep)

    for hint in sorted({e["hint"] for e in eps}):
        group = [e for e in eps if e["hint"] == hint]
        ok = [e for e in group if e["outcome"] == "success"]
        print(f"\nhint={hint}: {len(ok)}/{len(group)} success")
        by_hops = collections.defaultdict(list)
        for e in group:
            by_hops[e["hops"]].append(e)
        for hops in sorted(by_hops):
            g = by_hops[hops]
            s = [e for e in g if e["outcome"] == "success"]
            mean_t = sum(e["time_s"] for e in s) / len(s) if s else float("nan")
            print(f"  {hops} room(s) away: {len(s)}/{len(g)}   mean time (successes) {mean_t:5.1f}s")
        fails = collections.Counter(e["outcome"] for e in group if e["outcome"] != "success")
        if fails:
            print("  failures: " + ", ".join(f"{k} x{v}" for k, v in fails.most_common()))

    print("\nper episode:")
    for e in eps:
        print(f"  {e['file']:<32} {e['hops']} hops  {e['outcome']:<13} {e['time_s']:6.1f}s "
              f"{e['distance_m']:6.1f} m  {e['instructions']} instr")


if __name__ == "__main__":
    main()
