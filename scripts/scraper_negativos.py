from __future__ import annotations

import argparse
import csv
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "resenas_negativos.csv"
DEFAULT_BACKUP_PATH = PROJECT_ROOT / "data" / "raw" / "resenas_negativos_backup.csv"


def normalizar_texto_para_clave(texto: str) -> str:
    texto = re.sub(r"\s+", " ", texto).strip().lower()
    return texto


def clave_resena(anime_id: str, texto_resena: str) -> tuple[str, str]:
    return str(anime_id), normalizar_texto_para_clave(str(texto_resena))


def construir_sesion() -> requests.Session:
    sesion = requests.Session()
    sesion.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://myanimelist.net/",
        }
    )
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.2,
        status_forcelist=[403, 405, 408, 429, 500, 502, 503, 504],
        allowed_methods={"GET"},
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    sesion.mount("https://", adapter)
    sesion.mount("http://", adapter)
    return sesion


def obtener_html(
    sesion: requests.Session,
    url: str,
    timeout: int = 30,
    max_intentos: int = 5,
) -> str | None:
    for intento in range(1, max_intentos + 1):
        try:
            respuesta = sesion.get(url, timeout=timeout)
        except requests.RequestException as exc:
            espera = min(30.0, (2**intento) + random.uniform(0.5, 2.0))
            print(
                f"[warn] fallo request intento {intento}/{max_intentos}: {url} | {exc}"
            )
            time.sleep(espera)
            continue

        status = respuesta.status_code
        if status == 200:
            return respuesta.text

        if status in (403, 405, 429):
            espera = min(90.0, (2**intento) * 2 + random.uniform(1.0, 4.0))
            print(
                f"[warn] status {status} intento {intento}/{max_intentos}: {url} | cooldown {espera:.1f}s"
            )
            time.sleep(espera)
            continue

        if status in (500, 502, 503, 504):
            espera = min(30.0, (2**intento) + random.uniform(0.5, 2.0))
            print(f"[warn] status {status} intento {intento}/{max_intentos}: {url}")
            time.sleep(espera)
            continue

        print(f"[warn] status {status}: {url}")
        return None

    print(f"[warn] agotados intentos: {url}")
    return None


def extraer_anime_id(href: str) -> str | None:
    match = re.search(r"/anime/(\d+)", href)
    return match.group(1) if match else None


def obtener_top_ids(
    sesion: requests.Session,
    max_anime: int,
    min_delay: float,
    max_delay: float,
) -> list[str]:
    ids: list[str] = []

    for offset in range(0, max_anime, 50):
        pagina = offset // 50 + 1
        total_paginas = (max_anime + 49) // 50
        print(f"Recolectando top anime: pagina {pagina}/{total_paginas}")

        url = f"https://myanimelist.net/topanime.php?limit={offset}"
        html = obtener_html(sesion, url)
        if html is None:
            continue

        soup = BeautifulSoup(html, "html.parser")
        enlaces = soup.select("h3.anime_ranking_h3 a[href*='/anime/']")
        if not enlaces:
            enlaces = soup.select("a[href*='/anime/']")

        for enlace in enlaces:
            href = enlace.get("href", "")
            anime_id = extraer_anime_id(href)
            if anime_id and anime_id not in ids:
                ids.append(anime_id)
                if len(ids) >= max_anime:
                    return ids

        time.sleep(random.uniform(min_delay, max_delay))

    return ids[:max_anime]


def extraer_recomendacion(caja_review: BeautifulSoup) -> str | None:
    etiquetas = caja_review.select("span.tag, a.tag, div.tag")
    for etiqueta in etiquetas:
        texto_etiqueta = normalizar_texto_para_clave(etiqueta.get_text(" ", strip=True))
        if "not recommended" in texto_etiqueta:
            return "not_recommended"
        if "recommended" in texto_etiqueta:
            return "recommended"

    encabezado = normalizar_texto_para_clave(caja_review.get_text(" ", strip=True)[:260])
    if re.search(r"\bnot recommended\b", encabezado):
        return "not_recommended"
    if re.search(r"\brecommended\b", encabezado):
        return "recommended"

    return None


def extraer_texto_review(caja_review: BeautifulSoup) -> str | None:
    nodo = caja_review.select_one("div.text")
    if nodo is None:
        return None

    texto = nodo.get_text(" ", strip=True)
    texto = texto.replace("*Spoiler Warning*", "")
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto or None


def obtener_base_reviews_url(sesion: requests.Session, anime_id: str) -> str:
    """Intenta resolver la URL canonica con slug para acceder al listado completo de reviews."""
    html = obtener_html(sesion, f"https://myanimelist.net/anime/{anime_id}")
    if html is None:
        return f"https://myanimelist.net/anime/{anime_id}/reviews"

    match_canonical = re.search(
        rf'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']https://myanimelist\.net/anime/{anime_id}/([^"\'/?#]+)',
        html,
        flags=re.IGNORECASE,
    )
    if match_canonical:
        slug = match_canonical.group(1)
        return f"https://myanimelist.net/anime/{anime_id}/{slug}/reviews"

    return f"https://myanimelist.net/anime/{anime_id}/reviews"


def extraer_resenas_anime(
    sesion: requests.Session,
    anime_id: str,
    paginas_por_anime: int,
    min_delay: float,
    max_delay: float,
) -> list[dict]:
    filas: list[dict] = []
    claves_vistas_anime: set[tuple[str, str]] = set()
    base_reviews_url = obtener_base_reviews_url(sesion, anime_id)

    for pagina in range(1, paginas_por_anime + 1):
        if pagina == 1:
            urls = [
                base_reviews_url,
                f"{base_reviews_url}?p=1",
                f"{base_reviews_url}/?p=1",
                f"https://myanimelist.net/anime/{anime_id}/reviews",
                f"https://myanimelist.net/anime/{anime_id}/reviews?p=1",
            ]
        else:
            urls = [
                f"{base_reviews_url}?p={pagina}",
                f"{base_reviews_url}/?p={pagina}",
                f"https://myanimelist.net/anime/{anime_id}/reviews?p={pagina}",
                f"https://myanimelist.net/anime/{anime_id}/reviews/?p={pagina}",
            ]

        html = None
        for url in urls:
            html = obtener_html(sesion, url)
            if html is not None:
                break

        if html is None:
            print(f"[warn] no se pudo obtener pagina {pagina} para anime {anime_id}")
            break

        soup = BeautifulSoup(html, "html.parser")
        cajas = soup.find_all("div", class_=re.compile(r"\breview-element\b"))
        if not cajas:
            if pagina == 1 and re.search(r"captcha|access denied|cloudflare", html, re.IGNORECASE):
                print(f"[warn] posible bloqueo anti-bot en anime {anime_id}")
            break

        nuevos = 0
        for caja in cajas:
            recomendacion = extraer_recomendacion(caja)
            texto = extraer_texto_review(caja)
            if recomendacion != "not_recommended" or texto is None:
                continue

            clave = clave_resena(anime_id, texto)
            if clave in claves_vistas_anime:
                continue
            claves_vistas_anime.add(clave)

            filas.append(
                {
                    "anime_id": anime_id,
                    "texto_resena": texto,
                    "recomendacion_original": recomendacion,
                    "sentimiento": 0,
                }
            )
            nuevos += 1

        print(f"  pagina {pagina}: {nuevos} resenas negativas validas")
        time.sleep(random.uniform(min_delay, max_delay))

    return filas


def extraer_resenas_worker(
    anime_id: str,
    paginas_por_anime: int,
    min_delay: float,
    max_delay: float,
) -> tuple[str, list[dict]]:
    sesion = construir_sesion()
    time.sleep(random.uniform(0.15, 0.65))
    filas = extraer_resenas_anime(
        sesion,
        anime_id=anime_id,
        paginas_por_anime=paginas_por_anime,
        min_delay=min_delay,
        max_delay=max_delay,
    )
    return anime_id, filas


def guardar_csv(ruta: Path, filas: list[dict]) -> None:
    filas_unicas: list[dict] = []
    claves_vistas: set[tuple[str, str]] = set()
    for fila in filas:
        clave = clave_resena(fila.get("anime_id", ""), fila.get("texto_resena", ""))
        if clave in claves_vistas:
            continue
        claves_vistas.add(clave)
        filas_unicas.append(fila)

    ruta.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas_unicas).to_csv(
        ruta,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_ALL,
    )


def parsear_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scraper de resenas negativas (not_recommended) de MyAnimeList"
    )
    parser.add_argument("--max-anime", type=int, default=1000)
    parser.add_argument("--paginas-por-anime", type=int, default=35)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--backup-cada", type=int, default=50)
    parser.add_argument("--min-delay", type=float, default=2.0)
    parser.add_argument("--max-delay", type=float, default=4.0)
    parser.add_argument("--salida", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--backup", type=Path, default=DEFAULT_BACKUP_PATH)
    return parser.parse_args()


def main() -> int:
    args = parsear_args()
    print("Iniciando extraccion de resenas negativas de MyAnimeList...")

    if args.workers < 1:
        print("[warn] workers menor a 1, se ajusta a 1")
        args.workers = 1
    if args.workers > 8:
        print("[warn] workers muy alto para MAL, se limita a 8")
        args.workers = 8

    sesion = construir_sesion()
    top_ids = obtener_top_ids(sesion, args.max_anime, args.min_delay, args.max_delay)
    if not top_ids:
        print("No se pudieron obtener IDs del top anime.")
        return 1

    filas_totales: list[dict] = []
    claves_vistas_globales: set[tuple[str, str]] = set()
    procesados = 0
    total = len(top_ids)

    print(f"Ejecucion paralela segura | workers={args.workers} | animes={total}")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futuros = {
            executor.submit(
                extraer_resenas_worker,
                anime_id,
                args.paginas_por_anime,
                args.min_delay,
                args.max_delay,
            ): anime_id
            for anime_id in top_ids
        }

        for futuro in as_completed(futuros):
            anime_id = futuros[futuro]
            procesados += 1
            try:
                anime_id_resultado, filas = futuro.result()
                nuevas_unicas: list[dict] = []
                descartadas = 0
                for fila in filas:
                    clave = clave_resena(
                        fila.get("anime_id", ""),
                        fila.get("texto_resena", ""),
                    )
                    if clave in claves_vistas_globales:
                        descartadas += 1
                        continue
                    claves_vistas_globales.add(clave)
                    nuevas_unicas.append(fila)

                filas_totales.extend(nuevas_unicas)
                print(
                    f"Completado {procesados}/{total} | ID {anime_id_resultado} | nuevas {len(nuevas_unicas)} | descartadas {descartadas} | acumulado {len(filas_totales)}"
                )
            except Exception as exc:
                print(f"[warn] error en anime {anime_id}: {exc}")

            if procesados % args.backup_cada == 0 and filas_totales:
                guardar_csv(args.backup, filas_totales)
                print(f"  backup guardado: {args.backup}")

    if not filas_totales:
        print("No se extrajeron resenas negativas validas. Revisa conectividad o selectores.")
        return 1

    guardar_csv(args.salida, filas_totales)
    print(f"Extraccion completa: {len(filas_totales)} resenas negativas en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
