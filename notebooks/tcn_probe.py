"""TCN-only probe sweep: re-tune for the republished QLD data.

Run:
    ./.venv/bin/python notebooks/tcn_probe.py narrow
    ./.venv/bin/python notebooks/tcn_probe.py wide
    ./.venv/bin/python notebooks/tcn_probe.py both

Reuses ``seq_sweep.build_data`` so each probe sees exactly the same inputs as
the anchor configs already logged today. Probes deliberately push past the
anchor grid in the directions the noise-sensitivity story suggests: longer
``seq_len``, larger ``kernel_size``, heavier ``weight_decay`` / ``dropout``,
and a lower-LR + more-epochs config for stable convergence on the heavy-tailed
storm signal.

Runs are logged to ``experiments.jsonl`` under the ``tcnprobe_<set>`` prefix
so they stay distinguishable from the anchor sweep rows.
"""
import sys
import time
import warnings

import pandas as pd

import forecast as fc
from seq_sweep import build_data, FEATURE_SETS, PRIMARY_BUOY, FEATURE_MODE

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

BATCH_SIZE = 512
SEED = 42
SCALER = "robust"

# Each dict is one probe. Defaults baked into the script:
#   batch_size=512, scaler="robust", seed=42, device="cpu", residual=True.
# Best anchors on republished data (for reference):
#   narrow: channels=(64,), ep=2, do=0.1                  → +6.9% skill
#   wide:   channels=(128,)*4, ep=3, wd=1e-4, do=0.2     → +11.3% skill
TCN_PROBES: list[dict] = [
    # 1. Longer context, deep stack
    {"seq_len": 96, "channels": (128,) * 4, "kernel_size": 3, "epochs": 3,
     "weight_decay": 1e-4, "dropout": 0.30, "lr": 1e-3},
    # 2. Larger kernel at sl=48 (wider per-layer receptive field)
    {"seq_len": 48, "channels": (128,) * 4, "kernel_size": 5, "epochs": 3,
     "weight_decay": 1e-4, "dropout": 0.20, "lr": 1e-3},
    # 3. Longer context + larger kernel
    {"seq_len": 96, "channels": (128,) * 4, "kernel_size": 5, "epochs": 3,
     "weight_decay": 1e-4, "dropout": 0.30, "lr": 1e-3},
    # 4. Heavy regularisation at current-best capacity
    {"seq_len": 48, "channels": (128,) * 4, "kernel_size": 3, "epochs": 5,
     "weight_decay": 1e-3, "dropout": 0.40, "lr": 1e-3},
    # 5. Deeper stack, narrower channels, longer context
    {"seq_len": 96, "channels": (64,) * 6, "kernel_size": 3, "epochs": 3,
     "weight_decay": 1e-4, "dropout": 0.30, "lr": 1e-3},
    # 6. Larger kernel + heavy regularisation + more epochs
    {"seq_len": 48, "channels": (128,) * 4, "kernel_size": 5, "epochs": 5,
     "weight_decay": 1e-3, "dropout": 0.30, "lr": 1e-3},
    # 7. Longer context + larger kernel + lower LR + more epochs
    {"seq_len": 96, "channels": (128,) * 4, "kernel_size": 5, "epochs": 5,
     "weight_decay": 5e-4, "dropout": 0.30, "lr": 5e-4},
]

ALL_SETS = list(FEATURE_SETS)


def _name(feature_set: str, cfg: dict) -> str:
    L = len(cfg["channels"])
    h = cfg["channels"][0]
    wd_tag = (f"_wd{cfg['weight_decay']:.0e}".replace("e-0", "e-")
              if cfg["weight_decay"] else "")
    do_tag = f"_do{cfg['dropout']:g}" if cfg["dropout"] else ""
    k_tag = f"_k{cfg['kernel_size']}" if cfg["kernel_size"] != 3 else ""
    lr_tag = (f"_lr{cfg['lr']:.0e}".replace("e-0", "e-")
              if cfg["lr"] != 1e-3 else "")
    return (f"tcnprobe_{feature_set}_{FEATURE_MODE}_sl{cfg['seq_len']}"
            f"_h{h}_L{L}_ep{cfg['epochs']}{wd_tag}{do_tag}{k_tag}{lr_tag}")


def run_probe(d: dict, feature_set: str, cfg: dict) -> dict:
    name = _name(feature_set, cfg)
    m = fc.TCNForecaster(
        seq_len=cfg["seq_len"],
        channels=cfg["channels"],
        kernel_size=cfg["kernel_size"],
        dropout=cfg["dropout"],
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        batch_size=BATCH_SIZE,
        seed=SEED,
        device="cpu",
        verbose=False,
        scaler=SCALER,
    )
    t0 = time.time()
    m.fit(d["X_tr"], d["y_tr"])
    preds = m.predict(d["X_te"])
    elapsed = time.time() - t0

    metrics = fc.summarise(
        d["y_te"].to_numpy(), preds, y_pred_baseline=d["persist_preds"]
    )
    result = fc.EvaluationResult(name=name, metrics=metrics,
                                 predictions=preds, model=m)
    fc.log_run(
        result,
        data_sources=[PRIMARY_BUOY] + d["neighbours"] + d["wind_stations"],
        train_index=d["train_index"],
        test_index=d["test_index"],
        n_features=d["n_features"],
        extra={
            "feature_set": feature_set,
            "feature_mode": FEATURE_MODE,
            "seq_len": cfg["seq_len"],
            "hidden": cfg["channels"][0],
            "num_layers": len(cfg["channels"]),
            "epochs": cfg["epochs"],
            "lr": cfg["lr"],
            "batch_size": BATCH_SIZE,
            "scaler": SCALER,
            "device": "cpu",
            "imputation": "mean",
            "weight_decay": cfg["weight_decay"],
            "dropout": cfg["dropout"],
            "kernel_size": cfg["kernel_size"],
            "elapsed_min": round(elapsed / 60, 2),
            "probe": True,
        },
    )
    flag = "  <-- beats persistence" if metrics["SkillVsBaseline"] > 0 else ""
    print(f"  {name:78s}  RMSE {metrics['RMSE']:.4f}  "
          f"Skill {metrics['SkillVsBaseline']:+.4f}  ({elapsed:5.1f}s){flag}",
          flush=True)
    return {
        "set": feature_set,
        "name": name,
        "seq_len": cfg["seq_len"],
        "h": cfg["channels"][0],
        "L": len(cfg["channels"]),
        "k": cfg["kernel_size"],
        "ep": cfg["epochs"],
        "wd": cfg["weight_decay"],
        "do": cfg["dropout"],
        "lr": cfg["lr"],
        "RMSE": metrics["RMSE"],
        "Skill": metrics["SkillVsBaseline"],
        "secs": round(elapsed, 1),
    }


def main(sets: list[str]) -> None:
    builds: dict[str, dict] = {}
    if "narrow" in sets:
        print(f"\n###### feature set: NARROW ######", flush=True)
        print(f"building data ({FEATURE_MODE} mode)...", flush=True)
        builds["narrow"] = build_data("narrow")
        d = builds["narrow"]
        print(f"persistence RMSE: {d['persist_rmse']:.4f}   "
              f"(features={d['n_features']}, train={len(d['y_tr']):,}, "
              f"test={len(d['y_te']):,}, "
              f"test={d['test_index'][0]} → {d['test_index'][-1]})\n",
              flush=True)

    if "wide" in sets:
        if "narrow" in builds:
            anchor_ts = builds["narrow"]["test_index"][0]
        else:
            print("(deriving wide's anchor from narrow's natural 80/20 split)",
                  flush=True)
            anchor_ts = build_data("narrow")["test_index"][0]
        print(f"\n###### feature set: WIDE (anchored on narrow split @ "
              f"{anchor_ts}) ######", flush=True)
        print(f"building data ({FEATURE_MODE} mode)...", flush=True)
        builds["wide"] = build_data("wide", test_start_ts=anchor_ts)
        d = builds["wide"]
        print(f"persistence RMSE: {d['persist_rmse']:.4f}   "
              f"(features={d['n_features']}, train={len(d['y_tr']):,}, "
              f"test={len(d['y_te']):,}, "
              f"test={d['test_index'][0]} → {d['test_index'][-1]})\n",
              flush=True)

    rows: list[dict] = []
    for feature_set in sets:
        d = builds[feature_set]
        print(f"=== {feature_set.upper()} / TCN PROBE ===", flush=True)
        for cfg in TCN_PROBES:
            rows.append(run_probe(d, feature_set, cfg))
        print(flush=True)

    summary = pd.DataFrame(rows).sort_values(["set", "Skill"],
                                             ascending=[True, False])
    print("=== probe results, best skill first within each set ===")
    print(summary.drop(columns=["name"]).to_string(index=False))


def _parse_argv(argv: list[str]) -> list[str]:
    if not argv:
        return ALL_SETS
    arg = argv[0].lower()
    if arg == "both":
        return ALL_SETS
    if arg in ALL_SETS:
        return [arg]
    sys.exit(f"unknown feature set {arg!r}; choose narrow / wide / both")


if __name__ == "__main__":
    sets = _parse_argv(sys.argv[1:])
    main(sets)
