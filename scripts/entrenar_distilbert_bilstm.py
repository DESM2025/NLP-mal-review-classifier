from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import random
from pathlib import Path
import sys
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src.model_distilbert_lstm import DistilBertBiLSTMClassifier


DATASET_PATH = (PROJECT_ROOT / "data" / "anime_sentiment_balanceado.csv").resolve()
MODEL_PATH = (PROJECT_ROOT / "models" / "distilbert_bilstm_model.pt").resolve()
TOKENIZER_DIR = (PROJECT_ROOT / "models" / "distilbert_tokenizer").resolve()
CONFIG_PATH = (PROJECT_ROOT / "models" / "distilbert_bilstm_config.json").resolve()


class SentimentDataset(Dataset):
    """Dataset de texto crudo + etiqueta, tokenizado on-the-fly con Hugging Face."""

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
        text = self.texts[idx]
        label = self.labels[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float32),
        }


def parsear_args() -> argparse.Namespace:
    """Parsea argumentos de linea de comandos para entrenamiento."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento DistilBERT + BiLSTM con PyTorch"
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--text-column", type=str, default="texto_resena")
    parser.add_argument("--label-column", type=str, default="sentimiento")
    parser.add_argument("--pretrained-model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--tokenizer-out", type=Path, default=TOKENIZER_DIR)
    parser.add_argument("--model-out", type=Path, default=MODEL_PATH)
    parser.add_argument("--config-out", type=Path, default=CONFIG_PATH)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--lstm-hidden", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs-frozen", type=int, default=3)
    parser.add_argument("--epochs-unfrozen", type=int, default=2)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-full", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolver_ruta_desde_root(path: Path) -> Path:
    """Resuelve rutas relativas contra PROJECT_ROOT y retorna ruta absoluta."""
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def cargar_datos(ruta_csv: Path, text_column: str, label_column: str) -> tuple[list[str], np.ndarray]:
    """Carga el dataset balanceado y devuelve textos y etiquetas binarias."""
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

    textos = df[text_column].astype(str).tolist()
    etiquetas = df[label_column].astype(int).to_numpy()
    return textos, etiquetas


def dividir_datos(
    textos: list[str],
    etiquetas: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[list[str], list[str], list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Divide en train/test y luego reserva validacion desde train."""
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        textos,
        etiquetas,
        test_size=test_size,
        random_state=seed,
        stratify=etiquetas,
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
    """Fija semillas para reproducibilidad en NumPy/PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def obtener_dispositivo(force_cpu: bool = False) -> torch.device:
    """Selecciona dispositivo de entrenamiento, priorizando CUDA."""
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


def ejecutar_epoca(
    model: DistilBertBiLSTMClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW | None,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    grad_clip: float,
) -> tuple[float, float]:
    """Ejecuta una epoca de entrenamiento o validacion y devuelve loss/accuracy."""
    entrenamiento = optimizer is not None
    model.train(entrenamiento)

    losses: list[float] = []
    preds: list[int] = []
    labels_true: list[int] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        if entrenamiento:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(entrenamiento):
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp
                else nullcontext()
            )
            with amp_context:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)

            if entrenamiento:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        batch_preds = (probabilities >= 0.5).astype(int)
        batch_labels = labels.detach().cpu().numpy().astype(int)

        preds.extend(batch_preds.tolist())
        labels_true.extend(batch_labels.tolist())
        losses.append(float(loss.detach().cpu().item()))

    mean_loss = float(np.mean(losses)) if losses else 0.0
    acc = float(accuracy_score(labels_true, preds)) if labels_true else 0.0
    return mean_loss, acc


def guardar_checkpoint(
    model: DistilBertBiLSTMClassifier,
    output_path: Path,
    args: argparse.Namespace,
    best_val_acc: float,
) -> None:
    """Guarda pesos y metadatos minimos para inferencia reproducible."""
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
    train_size: int,
    val_size: int,
    test_size: int,
    test_loss: float,
    test_acc: float,
) -> None:
    """Guarda hiperparametros y metricas finales en JSON."""
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
        "seed": args.seed,
        "sizes": {
            "train": train_size,
            "val": val_size,
            "test": test_size,
        },
        "test": {
            "loss": test_loss,
            "accuracy": test_acc,
        },
    }
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main() -> int:
    """Ejecuta entrenamiento DistilBERT + BiLSTM usando PyTorch y precision mixta."""
    args = parsear_args()
    args.dataset = resolver_ruta_desde_root(args.dataset)
    args.tokenizer_out = resolver_ruta_desde_root(args.tokenizer_out)
    args.model_out = resolver_ruta_desde_root(args.model_out)
    args.config_out = resolver_ruta_desde_root(args.config_out)
    fijar_semilla(args.seed)

    device = obtener_dispositivo(force_cpu=args.cpu)
    use_amp = (device.type == "cuda") and (not args.disable_amp)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU detectada por PyTorch: {gpu_name}")
    else:
        print("[warn] CUDA no disponible; se usara CPU.")

    print("Cargando dataset...")
    textos, etiquetas = cargar_datos(
        ruta_csv=args.dataset,
        text_column=args.text_column,
        label_column=args.label_column,
    )
    print(f"Muestras cargadas: {len(textos)}")

    print("Dividiendo datos en train/val/test...")
    x_train, x_val, x_test, y_train, y_val, y_test = dividir_datos(
        textos,
        etiquetas,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )
    print(
        "Tamano de conjuntos "
        f"| train={len(x_train)} | val={len(x_val)} | test={len(x_test)}"
    )

    print(f"Cargando tokenizer/base model: {args.pretrained_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    args.tokenizer_out.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.tokenizer_out)
    print(f"Tokenizer guardado en: {args.tokenizer_out}")

    pin_memory = device.type == "cuda"
    train_loader = crear_dataloader(
        textos=x_train,
        etiquetas=y_train,
        tokenizer=tokenizer,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = crear_dataloader(
        textos=x_val,
        etiquetas=y_val,
        tokenizer=tokenizer,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=pin_memory,
    )
    test_loader = crear_dataloader(
        textos=x_test,
        etiquetas=y_test,
        tokenizer=tokenizer,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=pin_memory,
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

    if args.epochs_frozen > 0:
        print("\nFase 1: DistilBERT congelado (entrenando solo BiLSTM + clasificador)")
        optimizer = AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=args.lr_head,
            weight_decay=args.weight_decay,
        )

        for epoch in range(1, args.epochs_frozen + 1):
            train_loss, train_acc = ejecutar_epoca(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                grad_clip=args.grad_clip,
            )
            val_loss, val_acc = ejecutar_epoca(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                optimizer=None,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                grad_clip=args.grad_clip,
            )
            print(
                f"[F1][Epoch {epoch}/{args.epochs_frozen}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                guardar_checkpoint(model, args.model_out, args, best_val_acc)

    if args.epochs_unfrozen > 0:
        print("\nFase 2: DistilBERT descongelado (fine-tuning completo)")
        model.unfreeze_distilbert()
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr_full,
            weight_decay=args.weight_decay,
        )

        for epoch in range(1, args.epochs_unfrozen + 1):
            train_loss, train_acc = ejecutar_epoca(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                grad_clip=args.grad_clip,
            )
            val_loss, val_acc = ejecutar_epoca(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                optimizer=None,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                grad_clip=args.grad_clip,
            )
            print(
                f"[F2][Epoch {epoch}/{args.epochs_unfrozen}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                guardar_checkpoint(model, args.model_out, args, best_val_acc)

    if not args.model_out.exists():
        print("[warn] No hubo mejora durante entrenamiento; se guarda ultimo estado.")
        guardar_checkpoint(model, args.model_out, args, best_val_acc)

    print("\nCargando mejor checkpoint para evaluacion final de test...")
    checkpoint = torch.load(args.model_out, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = ejecutar_epoca(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        optimizer=None,
        scaler=scaler,
        device=device,
        use_amp=use_amp,
        grad_clip=args.grad_clip,
    )

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Mejor checkpoint guardado en: {args.model_out}")

    guardar_configuracion(
        args=args,
        output_path=args.config_out,
        train_size=len(x_train),
        val_size=len(x_val),
        test_size=len(x_test),
        test_loss=test_loss,
        test_acc=test_acc,
    )
    print(f"Configuracion/metricas guardadas en: {args.config_out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
