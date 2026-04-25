from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POS_INPUT_PATH = PROJECT_ROOT / "data" / "anime_sentiment_limpio.csv"
DEFAULT_NEG_INPUT_PATH = PROJECT_ROOT / "data" / "resenas_negativos_limpio.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "anime_sentiment_balanceado.csv"


def normalizar_texto_para_clave(texto: str) -> str:
    return re.sub(r"\s+", " ", str(texto).strip().lower())


def parsear_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crear dataset balanceado 50/50")
    parser.add_argument("--entrada-general", type=Path, default=DEFAULT_POS_INPUT_PATH)
    parser.add_argument("--entrada-negativos", type=Path, default=DEFAULT_NEG_INPUT_PATH)
    parser.add_argument("--salida", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def cargar_y_validar_csv(ruta: Path, nombre: str) -> pd.DataFrame:
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {nombre}: {ruta}")

    df = pd.read_csv(ruta)
    columnas_requeridas = {"anime_id", "texto_resena", "sentimiento"}
    faltantes = columnas_requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en {nombre}: {sorted(faltantes)}")

    df = df.dropna(subset=["anime_id", "texto_resena", "sentimiento"]).copy()
    df["texto_resena"] = df["texto_resena"].astype(str).str.strip().str.lower()
    df = df[df["texto_resena"] != ""]
    df["sentimiento"] = pd.to_numeric(df["sentimiento"], errors="coerce")
    df = df[df["sentimiento"].isin([0, 1])].copy()
    df["sentimiento"] = df["sentimiento"].astype(int)
    return df


def deduplicar_por_clave(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_texto_norm"] = out["texto_resena"].map(normalizar_texto_para_clave)
    out = out.drop_duplicates(subset=["anime_id", "_texto_norm"], keep="first")
    return out


def verificar_sin_duplicados(df: pd.DataFrame) -> int:
    out = df.copy()
    out["_texto_norm"] = out["texto_resena"].map(normalizar_texto_para_clave)
    n_duplicados = int(
        out.duplicated(subset=["anime_id", "_texto_norm"], keep=False).sum()
    )
    return n_duplicados


def main() -> int:
    args = parsear_args()

    df_general = cargar_y_validar_csv(args.entrada_general, "dataset general limpio")
    df_neg = cargar_y_validar_csv(args.entrada_negativos, "dataset negativos limpio")

    df_pos = df_general[df_general["sentimiento"] == 1].copy()
    df_neg = df_neg[df_neg["sentimiento"] == 0].copy()

    df_pos = deduplicar_por_clave(df_pos)
    df_neg = deduplicar_por_clave(df_neg)

    objetivo = len(df_neg)
    if objetivo == 0:
        print("[error] No hay reseñas negativas para balancear.")
        return 1

    if len(df_pos) < objetivo:
        print(
            f"[error] No hay suficientes positivas: disponibles={len(df_pos)} | requeridas={objetivo}"
        )
        return 1

    df_pos_sample = df_pos.sample(n=objetivo, random_state=args.seed)

    df_balanceado = pd.concat([df_neg, df_pos_sample], ignore_index=True)
    df_balanceado = df_balanceado.sample(frac=1.0, random_state=args.seed).reset_index(
        drop=True
    )

    n_duplicados = verificar_sin_duplicados(df_balanceado)
    if n_duplicados > 0:
        print(f"[error] Se detectaron duplicados en dataset final: {n_duplicados}")
        return 1

    if "_texto_norm" in df_balanceado.columns:
        df_balanceado = df_balanceado.drop(columns=["_texto_norm"])

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    df_balanceado.to_csv(args.salida, index=False, encoding="utf-8")

    n_neg = int((df_balanceado["sentimiento"] == 0).sum())
    n_pos = int((df_balanceado["sentimiento"] == 1).sum())

    print("Dataset balanceado generado")
    print(f"Entrada general: {args.entrada_general}")
    print(f"Entrada negativos: {args.entrada_negativos}")
    print(f"Salida: {args.salida}")
    print(f"Total filas: {len(df_balanceado)}")
    print(f"Negativas (0): {n_neg}")
    print(f"Positivas (1): {n_pos}")
    print(f"Duplicados detectados: {n_duplicados}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
