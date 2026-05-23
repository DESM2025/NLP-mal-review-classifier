from __future__ import annotations

import sys
import os
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from contextlib import nullcontext
import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer

from src.model_distilbert_lstm import DistilBertBiLSTMClassifier

MODEL_PATH = PROJECT_ROOT / "models" / "distilbert_bilstm_model.pt"
TOKENIZER_DIR = PROJECT_ROOT / "models" / "distilbert_tokenizer"
DEFAULT_MAX_LEN = 256

# Modelos del free tier de Gemini (Google AI Studio).
# flash es el mejor balance; flash-lite tiene más cuota gratis; pro es el más capaz.
MODELOS_GEMINI = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"]


def configurar_pagina() -> None:
    """Configura metadatos y estilo visual principal de la app."""
    st.set_page_config(
        page_title="Modelo de análisis de Sentimiento en reseñas de Anime",
        page_icon="🌸",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

            :root {
                --bg-1: #0b1220;
                --bg-2: #1e293b;
                --card: rgba(15, 23, 42, 0.78);
                --text: #e2e8f0;
                --muted: #94a3b8;
                --ok: #22c55e;
                --bad: #ef4444;
                --accent: #38bdf8;
            }

            html, body, [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(1200px 500px at 15% -10%, #1d4ed8 0%, rgba(29, 78, 216, 0) 65%),
                    radial-gradient(900px 500px at 90% 10%, #0ea5e9 0%, rgba(14, 165, 233, 0) 55%),
                    linear-gradient(145deg, var(--bg-1) 0%, var(--bg-2) 100%);
                color: var(--text);
                font-family: "Manrope", sans-serif;
            }

            [data-testid="stHeader"] {
                background: transparent;
            }

            [data-testid="stTextArea"] textarea {
                border-radius: 14px;
                border: 1px solid rgba(148, 163, 184, 0.35);
                background: rgba(15, 23, 42, 0.85);
                color: var(--text);
                font-size: 1rem;
                line-height: 1.45;
            }

            [data-testid="stTextArea"] textarea:focus {
                border: 1px solid var(--accent);
                box-shadow: 0 0 0 0.15rem rgba(56, 189, 248, 0.25);
            }

            .app-card {
                background: var(--card);
                border: 1px solid rgba(148, 163, 184, 0.28);
                border-radius: 16px;
                padding: 1.1rem 1rem 0.8rem 1rem;
                backdrop-filter: blur(8px);
            }

            .subtitle {
                color: var(--muted);
                margin-top: -0.35rem;
                margin-bottom: 0.9rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Cargando modelo y tokenizer...")
def cargar_modelo_y_tokenizer() -> tuple[DistilBertBiLSTMClassifier, AutoTokenizer, int]:
    """Carga y cachea el modelo PyTorch + tokenizer para inferencia eficiente."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")
    if not TOKENIZER_DIR.exists():
        raise FileNotFoundError(f"No se encontró el tokenizer en: {TOKENIZER_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    pretrained_name = checkpoint.get("pretrained_model_name", "distilbert-base-uncased")
    # Si es una ruta relativa, resolverla contra la raíz del proyecto
    posible_ruta = Path(pretrained_name)
    if not posible_ruta.is_absolute():
        posible_ruta = PROJECT_ROOT / pretrained_name
    if posible_ruta.exists() and posible_ruta.is_dir():
        pretrained_name = str(posible_ruta)
    elif ("/" in pretrained_name) or ("\\" in pretrained_name):
        # Era una ruta local pero no existe: caer al base de HF
        # (los pesos finales vendrán del checkpoint vía load_state_dict)
        pretrained_name = "distilbert-base-uncased"
    max_len = int(checkpoint.get("max_len", DEFAULT_MAX_LEN))

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    model = DistilBertBiLSTMClassifier(
        pretrained_model_name=pretrained_name,
        lstm_hidden_size=int(checkpoint.get("lstm_hidden_size", 256)),
        lstm_num_layers=int(checkpoint.get("lstm_num_layers", 1)),
        dropout=float(checkpoint.get("dropout", 0.30)),
        freeze_distilbert=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, tokenizer, max_len


def inferir_probabilidad_positiva(
    texto: str,
    model: DistilBertBiLSTMClassifier,
    tokenizer: AutoTokenizer,
    max_len: int,
) -> float:
    """Recibe texto crudo, tokeniza con DistilBERT y devuelve probabilidad positiva."""
    device = next(model.parameters()).device
    encoded = tokenizer(
        texto,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.type == "cuda"
            else nullcontext()
        )
        with amp_context:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            prob = torch.sigmoid(logits).item()

    return float(np.clip(prob, 0.0, 1.0))


def inferir_probabilidad_sliding(
    texto: str,
    model: DistilBertBiLSTMClassifier,
    tokenizer: AutoTokenizer,
    max_len: int,
    stride: int = 64,
) -> tuple[float, int]:
    """
    Inferencia por ventana deslizante para reseñas más largas que max_len.

    Tokeniza el texto completo, lo divide en chunks de max_len con solape de
    `stride` tokens y promedia las probabilidades. Para reseñas que caben en
    una sola ventana, se comporta igual que la inferencia normal.

    Devuelve (probabilidad_promedio, n_chunks_usados).
    """
    device = next(model.parameters()).device
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Tokenizar sin truncar y sin tokens especiales para controlarlos manualmente
    ids_completos = tokenizer.encode(
        texto, add_special_tokens=False, truncation=False
    )
    ventana = max_len - 2  # reservamos espacio para [CLS] y [SEP]
    if ventana <= 0:
        return inferir_probabilidad_positiva(texto, model, tokenizer, max_len), 1

    # Construir chunks con stride
    chunks: list[list[int]] = []
    if len(ids_completos) <= ventana:
        chunks.append(ids_completos)
    else:
        paso = max(1, ventana - stride)
        i = 0
        while i < len(ids_completos):
            chunk = ids_completos[i : i + ventana]
            chunks.append(chunk)
            if i + ventana >= len(ids_completos):
                break
            i += paso

    probabilidades: list[float] = []
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), amp_context:
        for chunk in chunks:
            ids = [cls_id] + chunk + [sep_id]
            attn = [1] * len(ids)
            # Padding a max_len
            faltante = max_len - len(ids)
            if faltante > 0:
                ids = ids + [pad_id] * faltante
                attn = attn + [0] * faltante
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            attention_mask = torch.tensor([attn], dtype=torch.long, device=device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilidades.append(torch.sigmoid(logits).item())

    prob_media = float(np.clip(np.mean(probabilidades), 0.0, 1.0))
    return prob_media, len(chunks)


def renderizar_resultado(
    prob_positiva: float,
    umbral_negativo: float,
    umbral_positivo: float,
) -> None:
    """Muestra resultado con zona incierta y niveles de confianza."""
    porcentaje_positivo = float(np.clip(prob_positiva, 0.0, 1.0)) * 100.0
    st.progress(int(round(porcentaje_positivo)))
    st.caption(f"Nivel de positividad estimado: {porcentaje_positivo:.2f}%")
    st.caption(f"Score del modelo (sigmoid): {prob_positiva:.6f}")

    if prob_positiva >= umbral_positivo:
        st.success(
            f"Reseña Positiva / Recomendada. Confianza: {porcentaje_positivo:.2f}%"
        )
    elif prob_positiva <= umbral_negativo:
        confianza_negativa = (1.0 - float(np.clip(prob_positiva, 0.0, 1.0))) * 100.0
        st.error(
            "Reseña Negativa / No Recomendada. "
            f"Confianza: {confianza_negativa:.2f}%"
        )
    else:
        st.warning(
            "Reseña Mixta / Incierta. "
            "El texto contiene señales contradictorias o ambiguas."
        )


def obtener_api_key() -> str | None:
    """Busca la API key de Gemini en st.secrets y luego en variables de entorno."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            valor = str(st.secrets["GEMINI_API_KEY"]).strip()
            if valor:
                return valor
    except Exception:
        # st.secrets puede lanzar si no existe secrets.toml: lo ignoramos.
        pass
    valor_env = os.environ.get("GEMINI_API_KEY", "").strip()
    return valor_env or None


def _parsear_respuesta_gemini(crudo: str) -> dict:
    """Extrae sentimiento/confianza/razón de la respuesta de Gemini de forma robusta."""
    resultado = {
        "sentimiento": "incierta",
        "confianza": None,
        "razon": "",
        "crudo": crudo,
    }

    data = {}
    try:
        data = json.loads(crudo)
    except Exception:
        match = re.search(r"\{.*\}", crudo, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                data = {}

    if isinstance(data, dict) and data:
        sent = str(data.get("sentimiento", "")).strip().lower()
        if "pos" in sent:
            resultado["sentimiento"] = "positiva"
        elif "neg" in sent:
            resultado["sentimiento"] = "negativa"
        conf = data.get("confianza")
        try:
            resultado["confianza"] = float(conf)
        except (TypeError, ValueError):
            resultado["confianza"] = None
        resultado["razon"] = str(data.get("razon", "")).strip()
        if resultado["sentimiento"] != "incierta":
            return resultado

    # Fallback: heurística sobre texto plano si el JSON falló o vino incompleto.
    bajo = crudo.lower()
    if "positiv" in bajo and "negativ" not in bajo:
        resultado["sentimiento"] = "positiva"
    elif "negativ" in bajo and "positiv" not in bajo:
        resultado["sentimiento"] = "negativa"
    return resultado


def clasificar_con_gemini(
    texto: str,
    api_key: str,
    model_name: str = "gemini-2.5-flash",
) -> dict:
    """
    Clasifica una reseña con Gemini (zero-shot) como baseline de LLM general.

    Devuelve dict con: sentimiento, confianza (0-100 o None), razón y crudo.
    Lanza ImportError si falta google-genai, u otras excepciones de red/API
    que el caller debe capturar.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    prompt = (
        "Eres un clasificador de sentimiento de reseñas de anime. "
        "Clasifica la reseña del usuario como 'positiva' (recomendada) o "
        "'negativa' (no recomendada). La reseña puede estar en inglés o español. "
        "Responde ÚNICAMENTE con JSON válido con estas claves exactas: "
        "sentimiento (valor 'positiva' o 'negativa'), "
        "confianza (número entero 0-100), "
        "razon (texto breve, máximo 20 palabras).\n\n"
        f'Reseña:\n"""\n{texto}\n"""'
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )
    crudo = (getattr(response, "text", None) or "").strip()
    return _parsear_respuesta_gemini(crudo)


def renderizar_gemini(res: dict) -> None:
    """Muestra el veredicto de Gemini con confianza, razon y respuesta cruda."""
    sent = res.get("sentimiento", "incierta")
    conf = res.get("confianza")
    razon = res.get("razon", "")

    if conf is not None:
        st.progress(int(round(float(np.clip(conf, 0.0, 100.0)))))
        st.caption(f"Confianza declarada por Gemini: {conf:.0f}%")

    if sent == "positiva":
        st.success("Gemini: Positiva / Recomendada")
    elif sent == "negativa":
        st.error("Gemini: Negativa / No recomendada")
    else:
        st.warning("Gemini: No se pudo determinar el sentimiento.")

    if razon:
        st.caption(f"Razón de Gemini: {razon}")

    with st.expander("Ver respuesta cruda de Gemini"):
        st.code(res.get("crudo") or "(vacío)", language="json")


def main() -> None:
    """Punto de entrada de la aplicación Streamlit."""
    configurar_pagina()

    with st.sidebar:
        st.subheader("Configuración")
        umbral_negativo = st.slider(
            "Umbral negativo",
            min_value=0.20,
            max_value=0.50,
            value=0.45,
            step=0.01,
        )
        umbral_positivo = st.slider(
            "Umbral positivo",
            min_value=0.50,
            max_value=0.85,
            value=0.60,
            step=0.01,
        )
        if umbral_negativo >= umbral_positivo:
            st.error("El umbral negativo debe ser menor que el positivo.")
            return

        usar_sliding = st.checkbox(
            "Sliding window para reseñas largas",
            value=True,
            help=(
                "Si la reseña excede el max_len del modelo, se divide en ventanas y se promedian las probabilidades. "
            ),
        )

        st.divider()
        st.markdown("**Comparar con un LLM**")
        comparar_gemini = st.checkbox(
            "Comparar con Gemini",
            value=False,
            help=(
                "Consulta a Gemini (LLM general, zero-shot) y muestra ambos resultados lado a lado. "
            ),
        )
        gemini_model = MODELOS_GEMINI[0]
        api_key_manual = None
        if comparar_gemini:
            gemini_model = st.selectbox(
                "Modelo Gemini",
                MODELOS_GEMINI,
                index=0,
                help="flash: buen balance. flash-lite: más cuota gratis. pro: el más capaz.",
            )
            if obtener_api_key() is None:
                api_key_manual = st.text_input(
                    "API key de Gemini",
                    type="password",
                    help=(
                        "Pégala aquí para esta sesión, o guárdala en "
                        ".streamlit/secrets.toml como GEMINI_API_KEY. "
                        "Obténla gratis en aistudio.google.com/apikey"
                    ),
                )
            else:
                st.caption("API key detectada en secrets/entorno.")

        if st.button("Recargar modelo/tokenizer", use_container_width=True):
            cargar_modelo_y_tokenizer.clear()
            st.rerun()

    st.title("Modelo de análisis de Sentimiento en reseñas de Anime")
    st.markdown(
        """
        <p class="subtitle">
        Demo en vivo de un modelo DistilBERT + BiLSTM entrenado con reseñas de MyAnimeList.<br>
        Escriba una reseña en inglés y el modelo estimará si es recomendada o no recomendada.
        </p>
        """,
        unsafe_allow_html=True,
    )

    texto_usuario = st.text_area(
        "Reseña de anime",
        placeholder=(
            "Ejemplo: The animation was excellent, but the pacing felt too slow "
            "in the middle episodes..."
        ),
        height=220,
    )

    analizar = st.button("Analizar Reseña", type="primary", use_container_width=True)

    if not analizar:
        return

    if not texto_usuario.strip():
        st.warning("Ingresa una reseña antes de analizar.")
        return

    try:
        modelo, tokenizer, max_len = cargar_modelo_y_tokenizer()
    except Exception as exc:
        st.error(f"No se pudo cargar el modelo/tokenizer: {exc}")
        return

    with st.spinner("Analizando reseña..."):
        if usar_sliding:
            prob_positiva, n_chunks = inferir_probabilidad_sliding(
                texto=texto_usuario,
                model=modelo,
                tokenizer=tokenizer,
                max_len=max_len,
            )
        else:
            prob_positiva = inferir_probabilidad_positiva(
                texto=texto_usuario,
                model=modelo,
                tokenizer=tokenizer,
                max_len=max_len,
            )
            n_chunks = 1

    st.subheader("Resultado")
    if n_chunks > 1:
        st.caption(f"Reseña larga: se procesaron {n_chunks} ventanas de {max_len} tokens.")

    if not comparar_gemini:
        renderizar_resultado(
            prob_positiva=prob_positiva,
            umbral_negativo=float(umbral_negativo),
            umbral_positivo=float(umbral_positivo),
        )
        return

    # Modo comparación: tu modelo vs Gemini, lado a lado.
    res_gemini = None
    col_modelo, col_gemini = st.columns(2)

    with col_modelo:
        st.markdown("**Modelo DistilBERT + BiLSTM**")
        renderizar_resultado(
            prob_positiva=prob_positiva,
            umbral_negativo=float(umbral_negativo),
            umbral_positivo=float(umbral_positivo),
        )

    with col_gemini:
        st.markdown(f"**Gemini ({gemini_model})**")
        api_key = obtener_api_key() or (api_key_manual or "").strip()
        if not api_key:
            st.info(
                "Ingresa tu API key de Gemini en la barra lateral para comparar."
            )
        else:
            with st.spinner("Consultando a Gemini..."):
                try:
                    res_gemini = clasificar_con_gemini(
                        texto=texto_usuario,
                        api_key=api_key,
                        model_name=gemini_model,
                    )
                except ImportError:
                    st.error(
                        "Falta la librería google-genai. "
                        "Instala con: pip install google-genai"
                    )
                except Exception as exc:
                    st.error(f"No se pudo consultar a Gemini: {exc}")
            if res_gemini is not None:
                renderizar_gemini(res_gemini)


if __name__ == "__main__":
    main()
