from __future__ import annotations

"""
Continued pretraining (MLM) de DistilBERT sobre reseñas de anime.

Adapta el modelo base distilbert-base-uncased al dominio del anime entrenando
unicamente la objetivo de Masked Language Modeling sobre todas las reseñas
disponibles (sin etiquetas de sentimiento). El resultado se guarda en
models/distilbert_anime_pretrained/ y puede usarse como inicializacion del
fine-tuning supervisado:

    python scripts/pretrain_mlm.py
    python scripts/entrenar_distilbert_bilstm.py \\
        --pretrained-model models/distilbert_anime_pretrained
"""

import argparse
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DistilBertForMaskedLM,
    get_linear_schedule_with_warmup,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    PROJECT_ROOT / "data" / "anime_sentiment_limpio.csv",
    PROJECT_ROOT / "data" / "resenas_negativos_limpio.csv",
]
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "distilbert_anime_pretrained"


class TextOnlyDataset(Dataset):
    """Dataset que tokeniza textos crudos (sin etiquetas) para MLM."""

    def __init__(self, textos: list[str], tokenizer, max_len: int) -> None:
        self.textos = textos
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.textos)

    def __getitem__(self, idx: int) -> dict:
        encoded = self.tokenizer(
            self.textos[idx],
            truncation=True,
            max_length=self.max_len,
            return_special_tokens_mask=True,
        )
        return {k: torch.tensor(v) for k, v in encoded.items()}


def parsear_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continued pretraining MLM sobre reseñas de anime")
    parser.add_argument("--inputs", type=Path, nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--text-column", type=str, default="texto_resena")
    parser.add_argument("--pretrained-model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--mlm-prob", type=float, default=0.15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limita las muestras para pruebas rapidas (0 = todas)",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def fijar_semilla(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cargar_textos(rutas: list[Path], text_column: str, max_samples: int) -> list[str]:
    """Concatena reseñas crudas de varios CSV y deduplica por texto normalizado."""
    textos: list[str] = []
    for ruta in rutas:
        if not ruta.exists():
            print(f"[warn] No existe {ruta}, se omite.")
            continue
        df = pd.read_csv(ruta)
        if text_column not in df.columns:
            print(f"[warn] Columna {text_column} ausente en {ruta}, se omite.")
            continue
        col = df[text_column].astype(str).str.strip()
        col = col[col.str.len() > 30]
        textos.extend(col.tolist())

    vistos: set[str] = set()
    unicos: list[str] = []
    for t in textos:
        clave = t.lower()
        if clave in vistos:
            continue
        vistos.add(clave)
        unicos.append(t)

    if max_samples > 0:
        unicos = unicos[:max_samples]
    return unicos


def construir_optimizador(model, lr: float, weight_decay: float) -> AdamW:
    """Optimizador AdamW con weight decay diferenciado (sin decay para bias/LayerNorm)."""
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    decay_params, nodecay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n for k in no_decay):
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


def main() -> int:
    args = parsear_args()
    fijar_semilla(args.seed)

    if not args.cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[warn] CUDA no disponible; se usara CPU.")

    use_amp = (device.type == "cuda") and not args.disable_amp

    print("Cargando textos de dominio...")
    textos = cargar_textos(args.inputs, args.text_column, args.max_samples)
    if not textos:
        print("[error] No se pudieron cargar textos para MLM.")
        return 1
    print(f"Textos disponibles para MLM: {len(textos)}")

    print(f"Cargando tokenizer/modelo: {args.pretrained_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    model = DistilBertForMaskedLM.from_pretrained(args.pretrained_model).to(device)
    model.train()

    dataset = TextOnlyDataset(textos, tokenizer, args.max_len)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )

    total_steps = max(1, len(loader) * args.epochs)
    optimizer = construir_optimizador(model, args.lr, args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(
        f"Pasos totales: {total_steps} | warmup: {int(args.warmup_ratio * total_steps)}"
        f" | batch={args.batch_size} | epochs={args.epochs}"
    )
    log_every = max(1, total_steps // 50)
    step = 0

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        n_batches = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp
                else nullcontext()
            )
            with amp_ctx:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= prev_scale:
                scheduler.step()

            running_loss += float(loss.detach().cpu().item())
            n_batches += 1
            step += 1
            if step % log_every == 0:
                print(
                    f"[Epoch {epoch}/{args.epochs}] step={step}/{total_steps} "
                    f"loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                )

        avg_loss = running_loss / max(1, n_batches)
        try:
            ppl = math.exp(avg_loss)
        except OverflowError:
            ppl = float("inf")
        print(f"[Epoch {epoch}] train_loss={avg_loss:.4f} perplexity={ppl:.2f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModelo MLM guardado en: {args.output_dir}")
    print(
        "Para usarlo en fine-tuning supervisado:\n"
        f"  python scripts/entrenar_distilbert_bilstm.py --pretrained-model {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
