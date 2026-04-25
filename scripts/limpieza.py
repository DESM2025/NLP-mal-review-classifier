from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "anime_sentiment.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "anime_sentiment_limpio.csv"
DEFAULT_NEG_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "resenas_negativos.csv"
DEFAULT_NEG_OUTPUT_PATH = PROJECT_ROOT / "data" / "resenas_negativos_limpio.csv"


def normalizar_texto(texto: str) -> str:
    return " ".join(str(texto).strip().lower().split())


def parsear_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Limpieza de reseñas de anime")
    parser.add_argument("--entrada", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--salida", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--entrada-negativos", type=Path, default=DEFAULT_NEG_INPUT_PATH)
    parser.add_argument("--salida-negativos", type=Path, default=DEFAULT_NEG_OUTPUT_PATH)
    parser.add_argument("--min-chars", type=int, default=25)
    return parser.parse_args()


def porcentaje(parte: int, total: int) -> float:
    if total == 0:
        return 0.0
    return (parte / total) * 100.0


def limpiar_dataset(
    entrada: Path,
    salida: Path,
    min_chars: int,
    nombre: str,
    obligatorio: bool,
) -> bool:
    if not entrada.exists():
        if obligatorio:
            print(f"[error] No existe el archivo de entrada ({nombre}): {entrada}")
            return False
        print(f"[warn] No existe archivo para limpieza ({nombre}), se omite: {entrada}")
        return True

    df = pd.read_csv(entrada)
    n_inicial = len(df)

    columnas_requeridas = {"anime_id", "texto_resena", "sentimiento"}
    faltantes = columnas_requeridas - set(df.columns)
    if faltantes:
        print(f"[error] Faltan columnas requeridas ({nombre}): {sorted(faltantes)}")
        return False

    df = df.dropna(subset=["anime_id", "texto_resena", "sentimiento"]).copy()

    df["texto_resena"] = df["texto_resena"].astype(str).str.strip().str.lower()
    df = df[df["texto_resena"] != ""]

    df["sentimiento"] = pd.to_numeric(df["sentimiento"], errors="coerce")
    df = df[df["sentimiento"].isin([0, 1])].copy()
    df["sentimiento"] = df["sentimiento"].astype(int)

    df = df[df["texto_resena"].str.len() >= min_chars].copy()

    df["_texto_norm"] = df["texto_resena"].map(normalizar_texto)
    df = df.drop_duplicates(subset=["anime_id", "_texto_norm"], keep="first")

    if "recomendacion_original" in df.columns:
        df["recomendacion_original"] = (
            df["recomendacion_original"].astype(str).str.strip().str.lower()
        )

    df = df.drop(columns=["_texto_norm"])

    n_final = len(df)
    n_eliminadas = n_inicial - n_final

    odiados = int((df["sentimiento"] == 0).sum())
    amados = int((df["sentimiento"] == 1).sum())

    p_odiados = porcentaje(odiados, n_final)
    p_amados = porcentaje(amados, n_final)

    salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(salida, index=False, encoding="utf-8")

    print(f"Limpieza completada ({nombre})")
    print(f"Entrada: {entrada}")
    print(f"Salida: {salida}")
    print(f"Filas iniciales: {n_inicial}")
    print(f"Filas finales: {n_final}")
    print(f"Filas eliminadas: {n_eliminadas}")
    print(f"Amados (1): {amados} ({p_amados:.6f}%)")
    print(f"Odiados (0): {odiados} ({p_odiados:.6f}%)")

    return True


def main() -> int:
    args = parsear_args()

    ok_general = limpiar_dataset(
        entrada=args.entrada,
        salida=args.salida,
        min_chars=args.min_chars,
        nombre="general",
        obligatorio=True,
    )
    ok_negativos = limpiar_dataset(
        entrada=args.entrada_negativos,
        salida=args.salida_negativos,
        min_chars=args.min_chars,
        nombre="negativos",
        obligatorio=False,
    )

    return 0 if ok_general and ok_negativos else 1


if __name__ == "__main__":
    raise SystemExit(main())
