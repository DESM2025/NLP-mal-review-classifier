from __future__ import annotations

"""
Barrido (grid search) sobre hiperparametros del BiLSTM.

Reentrena entrenar_distilbert_bilstm.py varias veces, cada una con un
checkpoint y configuracion separados bajo models/sweeps/<run_id>/, y agrega
las metricas finales en models/sweeps/summary.csv.

Uso tipico (rapido, exploratorio):
    python scripts/sweep_bilstm.py \\
        --hidden 128 256 384 \\
        --layers 1 2 \\
        --dropout 0.20 0.30 0.40 \\
        --epochs-frozen 3 --epochs-unfrozen 2 --max-len 256

Uso con dry-run para ver los comandos sin ejecutar:
    python scripts/sweep_bilstm.py --dry-run
"""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "entrenar_distilbert_bilstm.py"
SWEEP_DIR = PROJECT_ROOT / "models" / "sweeps"


def parsear_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep de hiperparametros BiLSTM")
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 256, 384])
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--dropout", type=float, nargs="+", default=[0.20, 0.30, 0.40])
    parser.add_argument(
        "--epochs-frozen",
        type=int,
        default=3,
        help="Reducido vs entrenamiento final para sweeps mas rapidos",
    )
    parser.add_argument("--epochs-unfrozen", type=int, default=2)
    parser.add_argument(
        "--max-len",
        type=int,
        default=256,
        help="Conviene reducir el max_len para que el sweep sea factible en tiempo",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pretrained-model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-out", type=Path, default=SWEEP_DIR / "summary.csv")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="No abortar el sweep si una corrida falla",
    )
    return parser.parse_args()


def main() -> int:
    args = parsear_args()
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(args.hidden, args.layers, args.dropout))
    print(f"Total de combinaciones: {len(combos)}")

    resultados = []
    for i, (h, l, d) in enumerate(combos, 1):
        run_id = f"h{h}_l{l}_d{int(round(d * 100))}"
        run_dir = SWEEP_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.json"

        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--lstm-hidden", str(h),
            "--lstm-layers", str(l),
            "--dropout", str(d),
            "--epochs-frozen", str(args.epochs_frozen),
            "--epochs-unfrozen", str(args.epochs_unfrozen),
            "--max-len", str(args.max_len),
            "--batch-size", str(args.batch_size),
            "--pretrained-model", args.pretrained_model,
            "--seed", str(args.seed),
            "--model-out", str(run_dir / "model.pt"),
            "--config-out", str(config_path),
            "--errors-out", str(run_dir / "test_errors.csv"),
            "--tokenizer-out", str(run_dir / "tokenizer"),
        ]
        print(f"\n[{i}/{len(combos)}] {run_id}")
        print("  " + " ".join(cmd))
        if args.dry_run:
            continue

        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"[warn] Run {run_id} fallo (returncode={proc.returncode})")
            if not args.continue_on_error:
                return proc.returncode
            continue

        if config_path.exists():
            with config_path.open() as fh:
                cfg = json.load(fh)
            test = cfg.get("test", {})
            resultados.append(
                {
                    "run_id": run_id,
                    "lstm_hidden": h,
                    "lstm_layers": l,
                    "dropout": d,
                    "test_accuracy": test.get("accuracy"),
                    "test_loss": test.get("loss"),
                    "test_roc_auc": test.get("roc_auc"),
                    "test_f1_weighted": (test.get("weighted") or {}).get("f1"),
                    "test_f1_macro": (test.get("macro") or {}).get("f1"),
                }
            )

    if resultados:
        import pandas as pd

        df = pd.DataFrame(resultados).sort_values("test_accuracy", ascending=False)
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.summary_out, index=False, encoding="utf-8")
        print(f"\nResumen guardado en {args.summary_out}")
        print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
