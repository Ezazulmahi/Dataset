r"""Step 8 - matched-detector comparison: train and test the same detector on every embedding method.

For each method (default: suniward wow hill mipod deds_v4) whose 0.4-bpp data is complete, trains
02_train.py --model <model> --data matched-<method> (resumable; restarted automatically after a crash),
then scores the paper's covers embedded with the same method (03_test.py, with and without TTA).
Curriculum (--curriculum 1.0, default): when matched_1.0\<method>\universal\ exists (06_build_matched.py
--payload 1.0 --sets universal), a first run <model>_matched-<method>_p1.0 is trained at 1.0 bpp
(--cur-epochs) and the 0.4 bpp run starts from its weights. Without that data the method is trained at 0.4 directly
(said in the output). Runs that did not learn (not_learned.txt, best val AUC < 0.6) are not tested or reported.
Finished methods are skipped. With --wait, methods whose data is still being built are waited for
(checked every 5 minutes) instead of skipped.
--pause <run>: before the first training, ask that run (e.g. wsernet_universal, started by run_all.bat) to
save and exit through its PAUSE file, so the GPU is free; after the comparison it is resumed here and its
test step run, exactly as run_all.bat would.
Results: results\<model>_matched-<method>[_tta]\, results\summary_all.csv, 08_summary_<model>.csv
    python 08_run_matched.py [--model srnet] [--methods ...] [--wait] [--pause wsernet_universal]
"""
import os, sys, subprocess, argparse, time, json
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C

def data_complete(method):
    for _, c, s, _ in C.matched_split(method):
        if not (c.exists() and s.exists()):
            return False
    return all(c.exists() and s.exists() for pairs in C.test_sets(f"matched-{method}").values() for c, s in pairs)

def run(args):
    print(">>", " ".join(args), flush=True)
    p = subprocess.Popen([sys.executable, "-X", "faulthandler", *args], cwd=HERE)
    while True:
        try:
            return p.wait()
        except KeyboardInterrupt:
            pass                                       # let the child save its state first

def curriculum_ready(method, payload):
    return all(s.exists() for _, _, s, sp in C.matched_split(method, payload) if sp in ("train", "val"))

def train_until_done(model, data, run_dir, extra=()):
    tries = 0
    while not (run_dir / "done.txt").exists():
        rc = run(["02_train.py", "--model", model, "--data", data, *extra])
        if rc == 130:
            return 130
        if not (run_dir / "done.txt").exists():
            tries += 1
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(run_dir / "crashes.log", "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} exit code {rc}\n")
            if tries > 5:
                print(f"[{run_dir.name}] crashed {tries} times - giving up", flush=True); return 1
            print(f"[{run_dir.name}] training stopped unexpectedly (exit {rc}); restarting from the last save in 15 s", flush=True)
            time.sleep(15)
    return 0

def pause(run_name):
    run_dir = C.RUNS / run_name
    if not run_dir.exists() or (run_dir / "done.txt").exists():
        return
    flag = run_dir / "PAUSE"; flag.touch()
    print(f"asking {run_name} to save and pause ...", flush=True)
    for _ in range(120):                               # up to 10 minutes; 02_train deletes the flag when it exits
        if not flag.exists():
            print(f"{run_name} paused", flush=True); return
        time.sleep(5)
    flag.unlink(missing_ok=True)                       # it was not running
    print(f"{run_name} was not running", flush=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="srnet_kv")
    ap.add_argument("--methods", nargs="+", default=["suniward", "wow", "hill", "mipod", "deds_v4"])
    ap.add_argument("--wait", action="store_true"); ap.add_argument("--pause", default=None)
    ap.add_argument("--curriculum", type=float, default=1.0, help="payload of the first stage (0: none)")
    ap.add_argument("--cur-epochs", type=int, default=30); a = ap.parse_args()
    paused = False
    while True:
        waiting = []
        for m in a.methods:
            data = f"matched-{m}"; run_dir = C.RUNS / f"{a.model}_{data}"
            tested = (C.RESULTS / f"{a.model}_{data}_tta" / "summary.csv").exists()
            if (run_dir / "done.txt").exists() and (tested or (run_dir / "not_learned.txt").exists()):
                continue
            if not data_complete(m):
                waiting.append(m); continue
            if a.pause and not paused:
                pause(a.pause); paused = True
            extra = []
            if a.curriculum and not run_dir.exists():  # the stage is decided once; an existing run just resumes
                if (C.ROOT / f"matched_{a.curriculum}" / m).exists() and not curriculum_ready(m, a.curriculum):
                    print(f"[{m}] {a.curriculum} bpp data only partly built - run 06_build_matched.py --payload "
                          f"{a.curriculum} --sets universal first", flush=True)
                    waiting.append(m); continue
                if curriculum_ready(m, a.curriculum):
                    c_dir = C.RUNS / f"{a.model}_{data}_p{a.curriculum}"
                    rc = train_until_done(a.model, data, c_dir, ["--payload", str(a.curriculum), "--epochs", str(a.cur_epochs)])
                    if rc == 130:
                        print("STOPPED - run this again to continue.", flush=True); return 130
                    if (c_dir / "not_learned.txt").exists() or not (c_dir / "best.pt").exists():
                        print(f"[{m}] the {a.curriculum} bpp stage did not learn - skipping {m}", flush=True); continue
                    extra = ["--init", c_dir.name]
                else:
                    print(f"[{m}] no {a.curriculum} bpp data - training at 0.4 bpp directly", flush=True)
            rc = train_until_done(a.model, data, run_dir, extra)
            if rc == 130:
                print("STOPPED - run this again to continue.", flush=True); return 130
            if (run_dir / "not_learned.txt").exists():
                print(f"[{m}] {run_dir.name} did not learn (not_learned.txt) - not tested, not reported", flush=True)
            elif (run_dir / "done.txt").exists():
                for extra in ([], ["--tta"]):
                    run(["03_test.py", "--run", run_dir.name, *extra])
        if not waiting or not a.wait:
            break
        print(f"{time.strftime('%H:%M')} waiting for data: {waiting} (checking every 5 min)", flush=True)
        time.sleep(300)
    rows = []
    for m in a.methods:
        f = C.RESULTS / f"{a.model}_matched-{m}_tta" / "summary.csv"
        if f.exists():
            d = pd.read_csv(f); d.insert(0, "method", m); rows.append(d)
    if rows:
        t = pd.concat(rows)[["method", "set", "pairs", "auc", "auc_ci_low", "auc_ci_high", "pe_min", "acc_at_val_thr"]]
        t.to_csv(HERE / f"08_summary_{a.model}.csv", index=False)
        t.to_csv(HERE / f"08_summary_{a.model}_all_sets.csv", index=False)
        # headline test: the 1,000 v5 fine-tuned stego of F:\scirep submission 2.0 (09_prepare_v5_test.py)
        t = t[t.set == "v5_paper_ft"]; t.to_csv(HERE / f"08_summary_{a.model}.csv", index=False)
        print("\nTest: 1,000 DEDS-v5 fine-tuned stego (scirep submission 2.0) vs their covers\n" + t.to_string(index=False), flush=True)
    if waiting:
        print(f"\nNot run yet (data incomplete): {waiting}. Run this again when their data is built.", flush=True)
    if a.pause and (C.RUNS / a.pause / "args.json").exists():   # resume the paused run and test it, as run_all.bat does
        cfg = json.load(open(C.RUNS / a.pause / "args.json"))
        print(f"\nresuming {a.pause}", flush=True)
        if train_until_done(cfg["model"], cfg["data"], C.RUNS / a.pause) == 130:
            return 130
        for extra in ([], ["--tta"]):
            run(["03_test.py", "--run", a.pause, *extra])
        run(["03_test.py", "--run", a.pause, "--ckpt", "last", "--tta"])
    print("ALL DONE", flush=True)

if __name__ == "__main__":
    sys.exit(main() or 0)
