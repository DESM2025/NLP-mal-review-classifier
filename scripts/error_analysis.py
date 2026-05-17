from __future__ import annotations

"""
Analisis de errores del modelo DistilBERT + BiLSTM ya entrenado.

Carga el checkpoint, reproduce el split de test (mismo seed) y produce:
- Resumen de metricas (accuracy / F1 por clase / AUC).
- Matriz de confusion en texto y opcionalmente en PNG (si matplotlib esta).
- CSV con las reseñas mal clasificadas, ordenadas por confianza descendente
  (los errores mas "seguros" del modelo, los mas interesantes para inspeccion).

Uso:
    python scripts/error_analysis.py
    python scripts/error_analysis.py --max-errores-csv 1000

No reentrena nada; es una herramienta de diagnostico sobre el checkpoint actual.
"""

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src.model_distilbert_lstm import DistilBertBiLSTMClassifier  # noqa: E402


DEFAULT_DATASET = PROJECT_ROOT / "data" / "anime_sentiment_balanceado.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "distilbert_bilstm_model.pt"
DEFAULT_TOKENIZER = PROJECT_ROOT / "models" / "distilbert_tokenizer"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models"


def parsear_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analisis de errores del modelo entrenado")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--text-column", type=str, default="texto_resena")
    parser.add_argument("--label-column", type=str, default="sentimiento")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-errores-csv", type=int, default=500)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def cargar_dataframe(ruta: Path, text_column: str, label_column: str) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    df = df.dropna(subset=[text_column, label_column]).copy()
    df[text_column] = df[text_column].astype(str).str.strip()
    df = df[df[text_column] != ""]
    df[label_column] = pd.to_numeric(df[label_column], errors="coerce")
    df = df[df[label_column].isin([0, 1])].copy()
    df[label_column] = df[label_column].astype(int)
    return df


def main() -> int:
    args = parsear_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Dispositivo: {device}")

    if not args.model.exists():
        print(f"[error] No existe el modelo: {args.model}")
        return 1
    if not args.tokenizer_dir.exists():
        print(f"[error] No existe el tokenizer: {args.tokenizer_dir}")
        return 1

    df = cargar_dataframe(args.dataset, args.text_column, args.label_column)
    textos = df[args.text_column].astype(str).tolist()
    etiquetas = df[args.label_column].to_numpy()

    _, x_test, _, y_test = train_test_split(
        textos,
        etiquetas,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=etiquetas,
    )
    print(f"Test split: {len(x_test)} muestras")

    checkpoint = torch.load(args.model, map_location="cpu")
    pretrained = checkpoint.get("pretrained_model_name", "distilbert-base-uncased")
    # Resolver ruta relativa contra la raiz del proyecto
    posible_ruta = Path(pretrained)
    if not posible_ruta.is_absolute():
        posible_ruta = PROJECT_ROOT / pretrained
    if posible_ruta.exists() and posible_ruta.is_dir():
        pretrained = str(posible_ruta)
    elif ("/" in pretrained) or ("\\" in pretrained):
        pretrained = "distilbert-base-uncased"
    max_len = int(checkpoint.get("max_len", 256))
    print(
        f"Checkpoint info: pretrained={pretrained} max_len={max_len} "
        f"lstm_hidden={checkpoint.get('lstm_hidden_size')} "
        f"layers={checkpoint.get('lstm_num_layers')} "
        f"best_val_acc={checkpoint.get('best_val_acc')}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    model = DistilBertBiLSTMClassifier(
        pretrained_model_name=pretrained,
        lstm_hidden_size=int(checkpoint.get("lstm_hidden_size", 256)),
        lstm_num_layers=int(checkpoint.get("lstm_num_layers", 1)),
        dropout=float(checkpoint.get("dropout", 0.30)),
        freeze_distilbert=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    scores: list[float] = []
    n = len(x_test)
    bs = args.batch_size
    with torch.no_grad():
        for i in range(0, n, bs):
            batch_texts = x_test[i : i + bs]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_ctx:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(logits).detach().cpu().float().numpy()
            scores.extend(probs.tolist())
            if (i // bs) % 50 == 0:
                print(f"  procesado {min(i + bs, n)}/{n}")

    y_true = np.array(y_test)
    y_score = np.array(scores)
    y_pred = (y_score >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")

    print("\n--- Metricas en test ---")
    print(f"accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"roc_auc:  {auc:.4f}")
    print(f"f1 negativa: {f1[0]:.4f} | f1 positiva: {f1[1]:.4f} | f1 ponderado: {f1_w:.4f}")
    print(f"precision negativa: {p[0]:.4f} | precision positiva: {p[1]:.4f}")
    print(f"recall negativa: {r[0]:.4f} | recall positiva: {r[1]:.4f}")
    print("\nMatriz de confusion (filas=real, cols=pred):")
    print(f"          pred=0   pred=1")
    print(f"real=0    {cm[0, 0]:>6d}  {cm[0, 1]:>6d}")
    print(f"real=1    {cm[1, 0]:>6d}  {cm[1, 1]:>6d}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    err_rows = []
    for texto, t, s, p_ in zip(x_test, y_true.tolist(), y_score.tolist(), y_pred.tolist()):
        if t != p_:
            err_rows.append(
                {
                    "label_real": int(t),
                    "label_pred": int(p_),
                    "score_positivo": float(s),
                    "confianza_error": float(s if p_ == 1 else 1.0 - s),
                    "longitud_caracteres": len(texto),
                    "texto_resena": texto,
                }
            )
    err_rows.sort(key=lambda r: -r["confianza_error"])
    df_err = pd.DataFrame(err_rows[: args.max_errores_csv])
    errores_path = args.output_dir / "test_errors.csv"
    df_err.to_csv(errores_path, index=False, encoding="utf-8")
    print(
        f"\nErrores totales: {len(err_rows)} | "
        f"top {len(df_err)} guardados en {errores_path}"
    )

    metrics_path = args.output_dir / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "roc_auc": auc,
                "weighted": {
                    "precision": float(p_w),
                    "recall": float(r_w),
                    "f1": float(f1_w),
                },
                "per_class": {
                    "negativa": {
                        "precision": float(p[0]),
                        "recall": float(r[0]),
                        "f1": float(f1[0]),
                    },
                    "positiva": {
                        "precision": float(p[1]),
                        "recall": float(r[1]),
                        "f1": float(f1[1]),
                    },
                },
                "confusion_matrix": cm.tolist(),
                "n_errores": len(err_rows),
                "n_test": int(len(y_true)),
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Metricas en JSON: {metrics_path}")

    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[info] matplotlib no instalado; se omite la generacion del PNG.")
        return 0

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred=neg", "pred=pos"])
    ax.set_yticklabels(["real=neg", "real=pos"])
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14,
            )
    ax.set_title("Matriz de confusion - test")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    png_path = args.output_dir / "confusion_matrix.png"
    fig.savefig(png_path, dpi=120)
    print(f"Heatmap guardado en: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
