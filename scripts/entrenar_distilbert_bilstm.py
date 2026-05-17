from __future__ import annotations

"""
Entrenamiento DistilBERT + BiLSTM con PyTorch.

Mejoras integradas:
- max_len por defecto 320 para cubrir mas texto de las reseñas largas.
- Weight decay diferenciado (no se aplica a bias ni LayerNorm).
- Scheduler con warmup lineal (10%) + decay lineal en cada fase.
- Layer-wise LR decay en la fase descongelada (capas inferiores con LR menor).
- Early stopping en la fase descongelada con paciencia configurable.
- Metricas extendidas en validacion y test: F1 por clase / macro / ponderado,
  precision, recall, ROC-AUC y matriz de confusion. Se guardan en el config JSON.
- Analisis de errores: top reseñas mal clasificadas exportadas a CSV.
- Soporte de --pretrained-model para usar un DistilBERT con continued pretraining.
"""

import argparse
import json
import random
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
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src.model_distilbert_lstm import DistilBertBiLSTMClassifier  # noqa: E402


DATASET_PATH = (PROJECT_ROOT / "data" / "anime_sentiment_balanceado.csv").resolve()
MODEL_PATH = (PROJECT_ROOT / "models" / "distilbert_bilstm_model.pt").resolve()
TOKENIZER_DIR = (PROJECT_ROOT / "models" / "distilbert_tokenizer").resolve()
CONFIG_PATH = (PROJECT_ROOT / "models" / "distilbert_bilstm_config.json").resolve()
ERRORS_PATH = (PROJECT_ROOT / "models" / "test_errors.csv").resolve()

NO_DECAY_KEYS = ("bias", "LayerNorm.weight", "LayerNorm.bias")


# =========================================================================== #
# Dataset                                                                     #
# =========================================================================== #


class SentimentDataset(Dataset):
    """Dataset de texto crudo + etiqueta, tokenizado on-the-fly."""

    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        tokenizer: AutoTokenizer,
        max_len: int,
    ) -> None:
        self.texts = texts
        self.labels = labels.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# =========================================================================== #
# CLI                                                                         #
# =========================================================================== #


def parsear_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento DistilBERT + BiLSTM con PyTorch (mejorado)"
    )
    # Datos
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--text-column", type=str, default="texto_resena")
    parser.add_argument("--label-column", type=str, default="sentimiento")
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)

    # Modelo
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="distilbert-base-uncased",
        help="HF model id o ruta local (p.ej. models/distilbert_anime_pretrained)",
    )
    parser.add_argument("--max-len", type=int, default=320)
    parser.add_argument("--lstm-hidden", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.30)

    # Optimizacion
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs-frozen", type=int, default=5)
    parser.add_argument("--epochs-unfrozen", type=int, default=8)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-full", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument(
        "--layer-lr-decay",
        type=float,
        default=0.9,
        help="Factor de decay LR por capa de DistilBERT en F2 (0.9 recomendado)",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=2,
        help="Epocas sin mejora en val_acc antes de cortar F2 (0 = desactiva)",
    )

    # Salidas
    parser.add_argument("--tokenizer-out", type=Path, default=TOKENIZER_DIR)
    parser.add_argument("--model-out", type=Path, default=MODEL_PATH)
    parser.add_argument("--config-out", type=Path, default=CONFIG_PATH)
    parser.add_argument("--errors-out", type=Path, default=ERRORS_PATH)
    parser.add_argument("--max-errores-csv", type=int, default=500)

    # Sistema
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolver_ruta_desde_root(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


# =========================================================================== #
# Datos                                                                       #
# =========================================================================== #


def cargar_datos(
    ruta_csv: Path, text_column: str, label_column: str
) -> tuple[list[str], np.ndarray]:
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No existe el dataset: {ruta_csv}")
    df = pd.read_csv(ruta_csv)
    columnas_requeridas = {text_column, label_column}
    faltantes = columnas_requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")
    df = df.dropna(subset=[text_column, label_column]).copy()
    df[text_column] = df[text_column].astype(str).str.strip()
    df = df[df[text_column] != ""]
    df[label_column] = pd.to_numeric(df[label_column], errors="coerce")
    df = df[df[label_column].isin([0, 1])].copy()
    return df[text_column].astype(str).tolist(), df[label_column].astype(int).to_numpy()


def dividir_datos(
    textos: list[str],
    etiquetas: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[list[str], list[str], list[str], np.ndarray, np.ndarray, np.ndarray]:
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        textos, etiquetas, test_size=test_size, random_state=seed, stratify=etiquetas
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=val_size,
        random_state=seed,
        stratify=y_train_full,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def fijar_semilla(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def obtener_dispositivo(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def crear_dataloader(
    textos: list[str],
    etiquetas: np.ndarray,
    tokenizer: AutoTokenizer,
    max_len: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    dataset = SentimentDataset(textos, etiquetas, tokenizer, max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# =========================================================================== #
# Optimizadores                                                               #
# =========================================================================== #


def construir_optimizador_solo_head(
    model: DistilBertBiLSTMClassifier, lr: float, weight_decay: float
) -> AdamW:
    """Optimizer para fase congelada con weight decay diferenciado."""
    decay_params: list[torch.Tensor] = []
    nodecay_params: list[torch.Tensor] = []
    for nombre, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in nombre for k in NO_DECAY_KEYS):
            nodecay_params.append(p)
        else:
            decay_params.append(p)
    return AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def _layer_id_distilbert_bilstm(name: str, num_layers_bert: int) -> int:
    """Profundidad relativa: 0 = head/lstm/classifier, mayor = mas cerca de embeddings."""
    if name.startswith("classifier") or name.startswith("lstm"):
        return 0
    if name.startswith("distilbert.embeddings"):
        return num_layers_bert + 1
    if name.startswith("distilbert.transformer.layer."):
        idx = int(name.split(".")[3])
        return num_layers_bert - idx
    return 0


def construir_optimizador_layerwise(
    model: DistilBertBiLSTMClassifier,
    lr_full: float,
    weight_decay: float,
    layer_decay: float,
) -> AdamW:
    """Optimizer con layer-wise LR decay para la fase descongelada."""
    num_layers_bert = len(model.distilbert.transformer.layer)
    grupos: dict[tuple[int, bool], dict] = {}
    for nombre, p in model.named_parameters():
        if not p.requires_grad:
            continue
        layer_id = _layer_id_distilbert_bilstm(nombre, num_layers_bert)
        es_no_decay = any(k in nombre for k in NO_DECAY_KEYS)
        clave = (layer_id, es_no_decay)
        if clave not in grupos:
            grupos[clave] = {
                "params": [],
                "lr": lr_full * (layer_decay ** layer_id),
                "weight_decay": 0.0 if es_no_decay else weight_decay,
            }
        grupos[clave]["params"].append(p)
    return AdamW(list(grupos.values()))


# =========================================================================== #
# Loops                                                                       #
# =========================================================================== #


def ejecutar_epoca_train(
    model: DistilBertBiLSTMClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    grad_clip: float,
) -> tuple[float, float]:
    model.train()
    losses: list[float] = []
    preds: list[int] = []
    labels_true: list[int] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        with amp_ctx:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        prev_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None and scaler.get_scale() >= prev_scale:
            scheduler.step()

        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        batch_preds = (probabilities >= 0.5).astype(int)
        batch_labels = labels.detach().cpu().numpy().astype(int)
        preds.extend(batch_preds.tolist())
        labels_true.extend(batch_labels.tolist())
        losses.append(float(loss.detach().cpu().item()))

    mean_loss = float(np.mean(losses)) if losses else 0.0
    acc = float(accuracy_score(labels_true, preds)) if labels_true else 0.0
    return mean_loss, acc


def evaluar_completo(
    model: DistilBertBiLSTMClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Evalua y devuelve metricas, etiquetas reales, scores y predicciones."""
    model.eval()
    losses: list[float] = []
    scores: list[float] = []
    labels_true: list[int] = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp
                else nullcontext()
            )
            with amp_ctx:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
            losses.append(float(loss.detach().cpu().item()))
            scores.extend(torch.sigmoid(logits).detach().cpu().float().numpy().tolist())
            labels_true.extend(labels.detach().cpu().numpy().astype(int).tolist())

    y_true = np.array(labels_true)
    y_score = np.array(scores)
    y_pred = (y_score >= 0.5).astype(int)

    metricas: dict = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true.size else 0.0,
    }
    if y_true.size and len(np.unique(y_true)) > 1:
        metricas["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        metricas["roc_auc"] = float("nan")

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metricas["per_class"] = {
        "negativa": {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f1[0])},
        "positiva": {"precision": float(p[1]), "recall": float(r[1]), "f1": float(f1[1])},
    }
    metricas["weighted"] = {"precision": float(p_w), "recall": float(r_w), "f1": float(f1_w)}
    metricas["macro"] = {"precision": float(p_m), "recall": float(r_m), "f1": float(f1_m)}
    metricas["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return metricas, y_true, y_score, y_pred


# =========================================================================== #
# Persistencia                                                                #
# =========================================================================== #


def guardar_checkpoint(
    model: DistilBertBiLSTMClassifier,
    output_path: Path,
    args: argparse.Namespace,
    best_val_acc: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "pretrained_model_name": args.pretrained_model,
        "max_len": args.max_len,
        "lstm_hidden_size": args.lstm_hidden,
        "lstm_num_layers": args.lstm_layers,
        "dropout": args.dropout,
        "best_val_acc": best_val_acc,
    }
    torch.save(checkpoint, output_path)


def guardar_configuracion(
    args: argparse.Namespace,
    output_path: Path,
    sizes: dict[str, int],
    val_metrics: dict,
    test_metrics: dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": str(args.dataset),
        "text_column": args.text_column,
        "label_column": args.label_column,
        "pretrained_model": args.pretrained_model,
        "max_len": args.max_len,
        "lstm_hidden": args.lstm_hidden,
        "lstm_layers": args.lstm_layers,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "epochs_frozen": args.epochs_frozen,
        "epochs_unfrozen": args.epochs_unfrozen,
        "lr_head": args.lr_head,
        "lr_full": args.lr_full,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "warmup_ratio": args.warmup_ratio,
        "layer_lr_decay": args.layer_lr_decay,
        "early_stop_patience": args.early_stop_patience,
        "seed": args.seed,
        "sizes": sizes,
        "validation": val_metrics,
        "test": test_metrics,
    }
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def guardar_errores(
    textos_test: list[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    max_filas: int,
) -> tuple[int, Path]:
    """Guarda las reseñas mal clasificadas ordenadas por confianza descendente."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for texto, t, s, p in zip(textos_test, y_true.tolist(), y_score.tolist(), y_pred.tolist()):
        if t != p:
            rows.append(
                {
                    "label_real": int(t),
                    "label_pred": int(p),
                    "score_positivo": float(s),
                    "confianza_error": float(s if p == 1 else 1.0 - s),
                    "longitud_caracteres": len(texto),
                    "texto_resena": texto,
                }
            )
    rows.sort(key=lambda r: -r["confianza_error"])
    df_err = pd.DataFrame(rows[:max_filas])
    df_err.to_csv(output_path, index=False, encoding="utf-8")
    return len(rows), output_path


def imprimir_resumen_test(metricas: dict) -> None:
    print("\n--- Metricas finales en test ---")
    print(f"loss:       {metricas['loss']:.4f}")
    print(f"accuracy:   {metricas['accuracy']:.4f}")
    print(f"roc_auc:    {metricas['roc_auc']:.4f}")
    print(
        f"f1 ponderado: {metricas['weighted']['f1']:.4f} | "
        f"precision: {metricas['weighted']['precision']:.4f} | "
        f"recall: {metricas['weighted']['recall']:.4f}"
    )
    print(f"f1 macro:    {metricas['macro']['f1']:.4f}")
    print(
        "per-class neg: "
        f"P={metricas['per_class']['negativa']['precision']:.4f} "
        f"R={metricas['per_class']['negativa']['recall']:.4f} "
        f"F1={metricas['per_class']['negativa']['f1']:.4f}"
    )
    print(
        "per-class pos: "
        f"P={metricas['per_class']['positiva']['precision']:.4f} "
        f"R={metricas['per_class']['positiva']['recall']:.4f} "
        f"F1={metricas['per_class']['positiva']['f1']:.4f}"
    )
    cm = np.array(metricas["confusion_matrix"])
    print("matriz de confusion (filas=real, cols=pred):")
    print(f"          pred=0   pred=1")
    print(f"real=0    {cm[0, 0]:>6d}  {cm[0, 1]:>6d}")
    print(f"real=1    {cm[1, 0]:>6d}  {cm[1, 1]:>6d}")


# =========================================================================== #
# Main                                                                        #
# =========================================================================== #


def main() -> int:
    args = parsear_args()
    args.dataset = resolver_ruta_desde_root(args.dataset)
    args.tokenizer_out = resolver_ruta_desde_root(args.tokenizer_out)
    args.model_out = resolver_ruta_desde_root(args.model_out)
    args.config_out = resolver_ruta_desde_root(args.config_out)
    args.errors_out = resolver_ruta_desde_root(args.errors_out)
    fijar_semilla(args.seed)

    device = obtener_dispositivo(force_cpu=args.cpu)
    use_amp = (device.type == "cuda") and (not args.disable_amp)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        print("[warn] CUDA no disponible; se usara CPU.")

    print("Cargando dataset...")
    textos, etiquetas = cargar_datos(args.dataset, args.text_column, args.label_column)
    print(f"Muestras cargadas: {len(textos)}")

    x_train, x_val, x_test, y_train, y_val, y_test = dividir_datos(
        textos, etiquetas, test_size=args.test_size, val_size=args.val_size, seed=args.seed
    )
    sizes = {"train": len(x_train), "val": len(x_val), "test": len(x_test)}
    print(
        f"Tamano | train={sizes['train']} | val={sizes['val']} | test={sizes['test']}"
    )

    print(f"Cargando tokenizer/base model: {args.pretrained_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    args.tokenizer_out.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.tokenizer_out)

    pin_memory = device.type == "cuda"
    train_loader = crear_dataloader(
        x_train, y_train, tokenizer, args.max_len, args.batch_size,
        args.num_workers, True, pin_memory,
    )
    val_loader = crear_dataloader(
        x_val, y_val, tokenizer, args.max_len, args.batch_size,
        args.num_workers, False, pin_memory,
    )
    test_loader = crear_dataloader(
        x_test, y_test, tokenizer, args.max_len, args.batch_size,
        args.num_workers, False, pin_memory,
    )

    model = DistilBertBiLSTMClassifier(
        pretrained_model_name=args.pretrained_model,
        lstm_hidden_size=args.lstm_hidden,
        lstm_num_layers=args.lstm_layers,
        dropout=args.dropout,
        freeze_distilbert=True,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_val_acc = -1.0
    last_val_metrics: dict = {}

    # ------------------ Fase 1: DistilBERT congelado ------------------ #
    if args.epochs_frozen > 0:
        print("\nFase 1: DistilBERT congelado (entrenando solo BiLSTM + clasificador)")
        optimizer = construir_optimizador_solo_head(
            model, lr=args.lr_head, weight_decay=args.weight_decay
        )
        total_steps_f1 = max(1, len(train_loader) * args.epochs_frozen)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_ratio * total_steps_f1),
            num_training_steps=total_steps_f1,
        )
        for epoch in range(1, args.epochs_frozen + 1):
            train_loss, train_acc = ejecutar_epoca_train(
                model, train_loader, criterion, optimizer, scheduler, scaler,
                device, use_amp, args.grad_clip,
            )
            val_metrics, _, _, _ = evaluar_completo(
                model, val_loader, criterion, device, use_amp
            )
            last_val_metrics = val_metrics
            print(
                f"[F1][Epoch {epoch}/{args.epochs_frozen}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1w={val_metrics['weighted']['f1']:.4f}"
            )
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                guardar_checkpoint(model, args.model_out, args, best_val_acc)

    # ------------------ Fase 2: DistilBERT descongelado + early stopping ------------------ #
    if args.epochs_unfrozen > 0:
        print(
            "\nFase 2: DistilBERT descongelado (fine-tuning completo, "
            "layer-wise LR + early stopping)"
        )
        model.unfreeze_distilbert()
        optimizer = construir_optimizador_layerwise(
            model, lr_full=args.lr_full, weight_decay=args.weight_decay,
            layer_decay=args.layer_lr_decay,
        )
        total_steps_f2 = max(1, len(train_loader) * args.epochs_unfrozen)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_ratio * total_steps_f2),
            num_training_steps=total_steps_f2,
        )

        epochs_sin_mejora = 0
        for epoch in range(1, args.epochs_unfrozen + 1):
            train_loss, train_acc = ejecutar_epoca_train(
                model, train_loader, criterion, optimizer, scheduler, scaler,
                device, use_amp, args.grad_clip,
            )
            val_metrics, _, _, _ = evaluar_completo(
                model, val_loader, criterion, device, use_amp
            )
            last_val_metrics = val_metrics
            print(
                f"[F2][Epoch {epoch}/{args.epochs_unfrozen}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1w={val_metrics['weighted']['f1']:.4f} val_auc={val_metrics['roc_auc']:.4f}"
            )

            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                guardar_checkpoint(model, args.model_out, args, best_val_acc)
                epochs_sin_mejora = 0
            else:
                epochs_sin_mejora += 1
                if (
                    args.early_stop_patience > 0
                    and epochs_sin_mejora >= args.early_stop_patience
                ):
                    print(
                        f"[early stop] Sin mejora por {epochs_sin_mejora} epocas en val_acc. "
                        "Detiene F2."
                    )
                    break

    if not args.model_out.exists():
        print("[warn] No hubo mejora durante entrenamiento; se guarda ultimo estado.")
        guardar_checkpoint(model, args.model_out, args, best_val_acc)

    # ------------------ Evaluacion final + analisis de errores ------------------ #
    print("\nCargando mejor checkpoint para evaluacion final de test...")
    checkpoint = torch.load(args.model_out, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, y_true, y_score, y_pred = evaluar_completo(
        model, test_loader, criterion, device, use_amp
    )
    n_errores, ruta_errores = guardar_errores(
        x_test, y_true, y_score, y_pred, args.errors_out, args.max_errores_csv,
    )

    imprimir_resumen_test(test_metrics)
    print(
        f"\nErrores totales en test: {n_errores} | "
        f"top {min(n_errores, args.max_errores_csv)} guardados en {ruta_errores}"
    )

    guardar_configuracion(args, args.config_out, sizes, last_val_metrics, test_metrics)
    print(f"\nMejor checkpoint guardado en: {args.model_out}")
    print(f"Configuracion/metricas guardadas en: {args.config_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
