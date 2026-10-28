r"""Step 2 - train a detector on the DEDS training pairs, resumable at any point.

    python 02_train.py --model effnet            # start, or continue where it stopped
    python 02_train.py --model srnet
    python 02_train.py --model effnet --run effnet_b --epochs 60    # a second, separate run

Everything for a run lives in runs/<run>/:
    last.pt     full state (weights, optimiser, scheduler, AMP scaler, RNG, epoch AND position inside the
                epoch). Written atomically every --save-min minutes, at the end of every epoch and on Ctrl+C.
    best.pt     weights of the epoch with the lowest validation P_E, plus the validation threshold
    checkpoints/epoch_NNN.pt   full state after every epoch (kept; copy one to last.pt to restart from it)
    PAUSE       create this file to make a running training save and exit (it is deleted on exit)
    log.csv     one row per epoch;  args.json, split.csv: the settings and the exact pairs used
Run the same command again to resume; it continues from the saved batch with the same data order.

Data (--data):
  universal  (default) the 5,000 universal-DCT pairs of F:\\Steganalyzer_UniversalDCT with their own fixed
             4,000 / 500 / 500 split. Generic DCT embedding only: no image made by the DEDS embedder is used,
             so the paper's stego images are a clean test (03_test.py).
  paper      5-fold cross-validation on the paper's own 1,000 pairs (04_cv_test.py).
  deds5000   DEDS stego of the 5,000 universal covers (step 01).
  matched-<method>  the 5,000 universal covers (luminance) and their stego from one method at 0.4 bpp (step 06/07);
             tested on the paper's covers embedded by the same method.
Curriculum (payload, then fine-tune):
    python 02_train.py --model srnet --data matched-suniward --payload 1.0 --epochs 30    # run srnet_matched-suniward_p1.0
    python 02_train.py --model srnet --data matched-suniward --init srnet_matched-suniward_p1.0
  For --data universal, --payload is the embedding rate of 00c_build_universal_curriculum.py (WSERNet curriculum):
    python 02_train.py --model wsernet --data universal --payload 1.0 --epochs 15 --run wsernet_universal_p1.0
  --init <run> starts a new run from that run's best.pt weights (optimiser and schedule start fresh).
A run whose best validation AUC stays below 0.6 gets not_learned.txt next to done.txt: the detector failed to
train, which is NOT evidence that the method is undetectable; 08_run_matched.py does not report it.
Runs with --init are exempt: their curriculum stage must have learned (else 08 skips the method), so a low AUC at
the real payload means the method is hard to detect, which is a result to report.
"""
import os, sys, csv, json, time, math, argparse, subprocess, shutil
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

def pe_min(y, s):
    fpr, tpr, thr = roc_curve(y, s); pe = 0.5 * (fpr + 1 - tpr); k = int(np.argmin(pe))
    return float(pe[k]), float(thr[k])

def make_split(data, fold, payload=0.4):
    """Rows (id, cover path, stego path, split)."""
    if data == "universal":                            # generic DCT embedding only; no DEDS image is used
        return C.universal_split(None if payload == 0.4 else payload)   # --payload = embedding rate (curriculum)
    if data.startswith("matched-"):                    # one embedding method, 4,000 universal covers, 0.4 bpp (or --payload)
        return C.matched_split(data.split("-", 1)[1], payload)
    if data == "deds5000":
        rows = list(csv.DictReader(open(C.ROOT / "embed_log.csv", newline="")))
        ok = sorted({r["image"] for r in rows if r["exact"] == "True" and (C.TRAIN_STEGO / r["image"]).exists()})
        return [(n, C.TRAIN_COVERS / n, C.TRAIN_STEGO / n, "val" if int(n[3:8]) % 10 == 0 else "train") for n in ok]
    # data == "paper": 5-fold cross-validation over the paper's 1,000 indices. The fine-tuned and baseline
    # pair of an index share prompt and seed, so they always fall in the same fold.
    ts = C.test_sets(); folds = C.paper_folds()
    out = []
    for gen, key in (("ft", "paper1000_finetuned"), ("bs", "paper1000_baseline")):
        for i, (c, s) in enumerate(ts[key]):
            if s is None:
                continue
            f = folds[i]
            sp = "test" if f == fold else ("val" if C.stable_hash(f"val{i}") % 10 == 0 else "train")
            out.append((f"{gen}_{i:03d}", c, s, sp))
    return out

def augment(cov, stg, rng):
    """Same random rotation/flip for a cover and its stego (the 8 symmetries of the square)."""
    cov = cov.copy(); stg = stg.copy()
    for i in range(len(cov)):
        k = int(rng.integers(4)); fl = bool(rng.integers(2))
        c, s = np.rot90(cov[i], k), np.rot90(stg[i], k)
        if fl:
            c, s = c[:, ::-1], s[:, ::-1]
        cov[i], stg[i] = c, s
    return cov, stg

@torch.no_grad()
def predict(model, imgs, dev, bs=64):
    model.eval(); out = []
    for i in range(0, len(imgs), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            o = model(C.to_tensor(imgs[i:i + bs], dev)).float(); out.append((o[:, 1] - o[:, 0]).cpu().numpy())   # logit margin
    return np.concatenate(out)

def save_atomic(obj, path):
    tmp = str(path) + ".tmp"; torch.save(obj, tmp); os.replace(tmp, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["srnet", "srnet_kv", "ucdfnet", "effnet", "wsernet"]); ap.add_argument("--run", default=None)
    ap.add_argument("--epochs", type=int); ap.add_argument("--pairs", type=int, help="pairs per batch")
    ap.add_argument("--lr", type=float); ap.add_argument("--wd", type=float)
    ap.add_argument("--patience", type=int, default=None); ap.add_argument("--save-min", type=float, default=5)
    ap.add_argument("--seed", type=int, default=2026); ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--data", default="universal", help="universal | matched-<method> (suniward, wow, hill, mipod, deds_v4) | paper | deds5000")
    ap.add_argument("--fold", default="all", help="paper data: 0-4, or 'all' to run the five folds in turn")
    ap.add_argument("--payload", type=float, default=0.4, help="matched data: training payload (curriculum stage)")
    ap.add_argument("--init", default=None, help="new run starts from runs/<init>/best.pt (curriculum)")
    a = ap.parse_args()
    if a.data == "paper" and a.fold == "all":          # run each fold as its own process; finished folds are skipped
        for k in range(C.N_FOLDS):
            argv = sys.argv[1:]
            if "--fold" in argv:
                j = argv.index("--fold"); del argv[j:j + 2]
            p = subprocess.Popen([sys.executable, os.path.abspath(__file__), *argv, "--fold", str(k)])
            while True:                                # on Ctrl+C keep waiting so the fold can save its state
                try:
                    rc = p.wait(); break
                except KeyboardInterrupt:
                    pass
            if rc == 130:
                print("\nSTOPPED during fold", k, "- run the same command to continue.", flush=True); return
            if rc != 0:
                print(f"fold {k} exited with code {rc}; stopping.", flush=True); return
        print("ALL FOLDS FINISHED", flush=True); return
    base = a.run or f"{a.model}_{a.data}" + (f"_p{a.payload}" if a.payload != 0.4 else "")
    run = C.RUNS / (f"{base}_f{a.fold}" if a.data == "paper" else base); run.mkdir(parents=True, exist_ok=True)
    if (run / "done.txt").exists() and not a.epochs:
        print(f"{run.name} already finished (delete runs/{run.name}/done.txt or pass --epochs to train longer)", flush=True); return
    dev = torch.device("cuda"); torch.backends.cudnn.benchmark = True
    last = run / "last.pt"; state = torch.load(last, map_location="cpu", weights_only=False) if last.exists() else None

    if state is None:                                  # new run: fix settings and split
        cfg = dict(C.DEFAULTS[a.model]); cfg.update({k: v for k, v in vars(a).items() if v is not None and k in cfg})
        cfg.update(model=a.model, seed=a.seed, patience=a.patience or cfg.get("patience", 15), pretrained=not a.no_pretrained, data=a.data,
                   fold=int(a.fold) if a.data == "paper" else None, payload=a.payload, init=a.init)
        split = make_split(a.data, cfg["fold"], a.payload)
        json.dump(cfg, open(run / "args.json", "w"), indent=1)
        with open(run / "split.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["id", "cover", "stego", "split"]); w.writerows(split)
    else:                                              # resume: settings and split come from the run folder
        cfg = json.load(open(run / "args.json"))
        if a.epochs:
            cfg["epochs"] = a.epochs; json.dump(cfg, open(run / "args.json", "w"), indent=1)
            (run / "done.txt").unlink(missing_ok=True)
        split = [(r["id"], r["cover"], r["stego"], r["split"]) for r in csv.DictReader(open(run / "split.csv", newline=""))]
    trp = [(c, s) for _, c, s, sp in split if sp == "train"]; vap = [(c, s) for _, c, s, sp in split if sp == "val"]
    tr = [i for i, _, _, sp in split if sp == "train"]; va = [i for i, _, _, sp in split if sp == "val"]
    print(f"run {run.name}: {cfg}\ntrain {len(tr)} pairs, val {len(va)} pairs, "
          f"test {sum(sp == 'test' for *_, sp in split)} pairs", flush=True)
    t0 = time.time()
    trc, trs = C.load_pairs(trp, C.CACHE / f"train_{C.stable_hash(','.join(map(str, trp))):08x}.npz")
    vac, vas = C.load_pairs(vap, C.CACHE / f"val_{C.stable_hash(','.join(map(str, vap))):08x}.npz")
    yv = np.r_[np.zeros(len(vac)), np.ones(len(vas))]; vimg = np.concatenate([vac, vas])
    print(f"images loaded in {time.time() - t0:.0f}s", flush=True)

    torch.manual_seed(cfg["seed"])
    model = C.build_model(cfg["model"], pretrained=cfg["pretrained"] and state is None).to(dev).to(memory_format=torch.channels_last)
    if state is None and cfg.get("init"):              # curriculum: start from the earlier stage's best weights
        src = torch.load(C.RUNS / cfg["init"] / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(src["model"])
        print(f"initialised from {cfg['init']} (epoch {src['epoch']}, val AUC {src['val_auc']:.4f})", flush=True)
    P = cfg["pairs"]; steps_ep = len(tr) // P; total = steps_ep * cfg["epochs"]; warm = steps_ep
    if cfg.get("optim") == "sgd":                  # paper recipe (WSERNet): no decay on biases / BN, step schedule
        decay = [p for p in model.parameters() if p.dim() > 1]; no_decay = [p for p in model.parameters() if p.dim() <= 1]
        opt = torch.optim.SGD([{"params": decay, "weight_decay": cfg["wd"]}, {"params": no_decay, "weight_decay": 0}],
                              lr=cfg["lr"], momentum=0.9)
        ms = [m * steps_ep for m in cfg["milestones"]]
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.1 ** sum(s >= m for m in ms))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler("cuda")
    ep, step0, best, bad = 0, 0, {"pe": 1.0, "epoch": -1}, 0
    if state is not None:
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"]); ep, step0, best, bad = state["epoch"], state["step"], state["best"], state["bad"]
        torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state(state["cuda_rng"])
        print(f"RESUMED at epoch {ep + 1}, batch {step0}/{steps_ep}; best so far P_E {best['pe']:.4f} (epoch {best['epoch'] + 1})", flush=True)

    def snapshot(epoch, step):
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": epoch, "step": step, "best": best, "bad": bad, "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state(), "cfg": cfg}, last)

    logf = run / "log.csv"; new_log = not logf.exists()
    y = torch.cat([torch.zeros(P), torch.ones(P)]).long().to(dev)
    try:
        while ep < cfg["epochs"] and bad < cfg["patience"]:
            order = np.random.default_rng(cfg["seed"] * 1000 + ep).permutation(len(tr))   # same order on resume
            model.train(); tl, tc, tn, te = 0.0, 0, 0, time.time(); last_save = time.time(); pos = step0
            for st in range(step0, steps_ep):
                idx = np.sort(order[st * P:(st + 1) * P])
                c, s = augment(trc[idx], trs[idx], np.random.default_rng([cfg["seed"], ep, st]))
                x = C.to_tensor(np.concatenate([c, s]), dev).contiguous(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.float16):
                    if hasattr(model, "train_loss"):   # networks with auxiliary losses (WSERNet)
                        loss, out = model.train_loss(x, y)
                    else:
                        out = model(x); loss = F.cross_entropy(out.float(), y)
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update(); sched.step()
                tl += loss.item() * len(y); tc += (out.argmax(1) == y).sum().item(); tn += len(y); pos = st + 1
                if (st + 1) % 50 == 0:
                    print(f"  ep {ep + 1} batch {st + 1}/{steps_ep} loss {tl / tn:.4f} acc {tc / tn:.4f} "
                          f"({(time.time() - te) / (st + 1 - step0):.2f}s/batch)", flush=True)
                if (run / "PAUSE").exists():         # pause switch (08_run_matched.py): save, then exit like Ctrl+C
                    raise KeyboardInterrupt
                if time.time() - last_save > 60 * a.save_min:
                    snapshot(ep, st + 1); last_save = time.time()
            sv = predict(model, vimg, dev); auc = roc_auc_score(yv, sv); pe, thr = pe_min(yv, sv)
            acc05 = float(((sv > 0) == yv).mean())
            improved = pe < best["pe"] - 1e-6 or (abs(pe - best["pe"]) <= 1e-6 and auc > best.get("auc", 0))
            if improved:
                best = {"pe": pe, "auc": auc, "thr": thr, "epoch": ep}; bad = 0
                save_atomic({"model": model.state_dict(), "cfg": cfg, "threshold": thr, "val_pe": pe, "val_auc": auc, "epoch": ep + 1}, run / "best.pt")
            else:
                bad += 1
            row = {"epoch": ep + 1, "train_loss": round(tl / max(tn, 1), 5), "train_acc": round(tc / max(tn, 1), 5), "val_auc": round(auc, 5),
                   "val_pe": round(pe, 5), "val_acc@0.5": round(acc05, 5), "lr": opt.param_groups[0]["lr"], "minutes": round((time.time() - te) / 60, 2),
                   "best": improved}
            with open(logf, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row)); (w.writeheader() if new_log else None); w.writerow(row); new_log = False
            print(f"EPOCH {ep + 1}/{cfg['epochs']}  train acc {row['train_acc']:.4f}  val AUC {auc:.4f}  val P_E {pe:.4f}"
                  f"{'  * best' if improved else ''}  ({row['minutes']} min)", flush=True)
            ep += 1; step0 = 0; snapshot(ep, 0)
            (run / "checkpoints").mkdir(exist_ok=True)                 # permanent copy of every epoch, never overwritten
            shutil.copyfile(last, run / "checkpoints" / f"epoch_{ep:03d}.pt")
    except KeyboardInterrupt:
        snapshot(ep, pos if "pos" in locals() else step0); (run / "PAUSE").unlink(missing_ok=True)
        print("\nSTOPPED - state saved. Run the same command to continue.", flush=True); sys.exit(130)
    snapshot(ep, 0)
    # a training failure, not an undetectable method. Not applied to a run started from a curriculum stage (--init):
    # that stage already proved the detector learns, so a low AUC at the real payload is a genuine result.
    if best.get("auc", 0) < 0.6 and not cfg.get("init"):
        (run / "not_learned.txt").write_text(f"best val AUC {best.get('auc', float('nan')):.4f} after epoch {ep}: "
                                             "the detector did not learn; do not report this run\n")
        print("WARNING: the detector did not learn (best val AUC < 0.6) - see not_learned.txt", flush=True)
    (run / "done.txt").write_text(f"finished after epoch {ep}\n")
    print(f"FINISHED. best epoch {best['epoch'] + 1}: val P_E {best['pe']:.4f}, AUC {best.get('auc', float('nan')):.4f}", flush=True)

if __name__ == "__main__":
    main()
