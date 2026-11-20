"""Genera la aplicación de la Entrega 4: docs/index.html (+ docs/data/*.json).

Toma lo que produce el pipeline (Silver y auditoría del DAG) y lo que dejó el notebook de modelado
(include/data/models/underdog_modelo.json y resultados_e3.json), y arma una página estática con gráficos
de Altair. GitHub Pages la publica desde la carpeta docs/ del repositorio.

    .venv/Scripts/python app/build_app.py                    # desde cero: necesita Silver, la auditoría y los modelos
    .venv/Scripts/python app/build_app.py --desde-publicado  # sólo con lo que ya está en docs/data/

La página se arma en dos pasos:
1. Lo que sale del Silver y del notebook (embudo, tablas de la Entrega 3, la fórmula, los gráficos de «Los datos» y
   «Lo que encontramos») se resume en docs/data/silver_resumen.json.
2. Todo lo demás (métricas, calibración, matriz de confusión, la regla de apuesta, las estrategias de comparación y la
   ganancia acumulada) se calcula sobre las predicciones fuera de muestra de docs/data/partidos.json.
Así, con --desde-publicado, cualquiera puede regenerar la página desde el repositorio sin tener el Silver, y el resultado
es el mismo que el de la corrida completa.

Si los datos cambian: correr el DAG, re-ejecutar el notebook y correr este script sin argumentos.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from include.config import AUDIT_DIR, DATA_DIR, SILVER_DATASET_PATH, SILVER_MANIFEST_PATH  # noqa: E402
from include.src.columns import LEAKAGE_COLUMNS, TARGET_COLUMN  # noqa: E402
from include.src.features import logit  # noqa: E402

SALIDA = RAIZ / "docs"
DATOS = SALIDA / "data"
MODELOS_DIR = DATA_DIR / "models"
TEST_TEMPORADA = "2015/2016"
APUESTA = 100
SEMILLA = 42

# Estrategias de comparación («dummies»): cada regla se elige mirando sólo la validación, igual que el modelo
UMBRALES_SIN_REGLA = [0.15, 0.20, 0.25, 0.30, 0.35]   # «modelo sin la regla»: apostar si la prob. del modelo supera un umbral fijo
CORTES_RANGO = [0, 0.10, 0.20, 0.30, 0.40, 1]          # «rango de cuota»: apostar si la cuota le da al underdog entre dos cortes
MIN_APUESTAS_VALIDACION = 500                          # un rango con menos apuestas que esto es pura suerte
SORTEOS = 300                                          # apostar al azar la misma cantidad de veces que el modelo
REMUESTREOS = 2000                                     # bootstrap del intervalo de confianza del retorno
MIN_APUESTAS_LINEA = 80                                # el retorno acumulado se dibuja desde que hay esta cantidad de apuestas
UMBRAL_MATRIZ = 0.20                                   # la matriz de confusión corta en la tasa base, no en 50%

# El partido que se usa de ejemplo en toda la página
EJEMPLO = {"local": "Bayer 04 Leverkusen", "visitante": "Eintracht Frankfurt", "fecha": "16/04/2016"}

# Versiones exactas de la librería que dibuja los gráficos de Altair en el navegador
VEGA, VEGA_LITE, VEGA_EMBED = "6.2.0", "6.4.1", "7.0.2"

# Colores con el mismo significado en toda la aplicación. Pensados para daltonismo: no hay rojo contra verde ni marrón
# contra verde, y en los gráficos con varias líneas cada una tiene su rótulo y su trazo (lleno o punteado).
COLOR = {"modelo": "#0f7f73", "sin_regla": "#69c2b6", "mercado": "#e08a1e", "todos": "#b4508c", "jugadores": "#2563a8",
         "nunca": "#8b929c", "azar": "#dfe2e6", "gano": "#2563a8", "no_gano": "#2f3542", "gano_suave": "#a9c1e0",
         "favorito": "#4a5563", "empate": "#c3c9d0", "fondo_puntos": "#d5dae1", "tinta": "#1e2430", "tinta_suave": "#6b7380"}
RESULTADO = alt.Scale(domain=["Ganó el underdog", "No ganó"], range=[COLOR["gano"], COLOR["no_gano"]])

NOMBRES_GAPS = {
    "nofav_xi_overall_mean_gap": "Rating medio del XI", "nofav_top3_overall_gap": "Rating de los 3 mejores",
    "nofav_best_overall_gap": "Mejor jugador", "nofav_worst_overall_gap": "Peor titular",
    "nofav_def_overall_gap": "Rating de la defensa", "nofav_mid_overall_gap": "Rating del mediocampo",
    "nofav_att_overall_gap": "Rating del ataque", "nofav_best_finishing_gap": "Mejor definición",
    "nofav_best_reactions_gap": "Mejores reacciones", "nofav_best_marking_gap": "Mejor marca",
}
LIGAS = {
    "England Premier League": "Inglaterra", "France Ligue 1": "Francia", "Germany 1. Bundesliga": "Alemania",
    "Italy Serie A": "Italia", "Spain LIGA BBVA": "España", "Netherlands Eredivisie": "Países Bajos",
    "Portugal Liga ZON Sagres": "Portugal", "Scotland Premier League": "Escocia", "Belgium Jupiler League": "Bélgica",
}


# ----------------------------------------------------------------------------- formatos en castellano
def entero(n: float) -> str:
    return f"{int(round(n)):,}".replace(",", ".")


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}".replace(".", ",") + "%"


def signo(x: float, d: int = 1) -> str:
    """Retornos: siempre con signo."""
    return f"{100 * x:+.{d}f}".replace(".", ",").replace("-", "−") + "%"


def num(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace(".", ",").replace("-", "−")


def pesos(x: float) -> str:
    return ("+" if x >= 0 else "−") + "$" + entero(abs(x))


# ----------------------------------------------------------------------------- datos y modelo (sólo en la corrida completa)
def cargar() -> tuple[pd.DataFrame, dict, dict]:
    ficha = json.loads((MODELOS_DIR / "underdog_modelo.json").read_text(encoding="utf-8"))
    resultados = json.loads((MODELOS_DIR / "resultados_e3.json").read_text(encoding="utf-8"))
    df = (pd.read_parquet(SILVER_DATASET_PATH)
            .merge(pd.read_parquet(AUDIT_DIR / "match_audit.parquet"), on="match_id")
            .sort_values(["fecha", "match_id"]).reset_index(drop=True))
    df["cuota_underdog"] = np.where(df["equipo_favorito"] == "local", df["odds_away"], df["odds_home"])
    df["prob_modelo"] = predecir(df, ficha)
    return df, ficha, resultados


def predecir(df: pd.DataFrame, ficha: dict) -> np.ndarray:
    """La fórmula del modelo, tal cual la escribió el notebook (y la misma que corre la página en JavaScript)."""
    z = np.full(len(df), ficha["intercepto"])
    for col, v in ficha["probabilidades"].items():
        z += v["coef"] * logit(df[col].fillna(v["mediana"]))
    for col, v in ficha["numericas"].items():
        z += v["coef"] * (df[col].fillna(v["mediana"]) - v["media"]) / v["desvio"]
    return 1 / (1 + np.exp(-z))


def verificar_contra_el_pipeline(df: pd.DataFrame) -> float:
    """Lo que muestra la app tiene que coincidir con el modelo del notebook."""
    import joblib
    modelo = joblib.load(MODELOS_DIR / "underdog_modelo.joblib")
    return float(np.max(np.abs(modelo.predict_proba(df)[:, 1] - df["prob_modelo"].values)))


def fuera_de_muestra(df: pd.DataFrame, resultados: dict) -> pd.DataFrame:
    """Los partidos que se predijeron sin haberlos visto, con esa predicción en `p`:
    la validación temporal (cada uno lo predice un modelo entrenado sólo con los anteriores, exportado por el notebook)
    y el test (el modelo final, que nunca vio 2015/16). Los de 2008–2010 sólo sirvieron para entrenar y quedan afuera."""
    oof = pd.Series(list(resultados["oof_modelo"].values()), index=resultados["oof_indices_match_id"])
    val = df[df["match_id"].isin(oof.index)].assign(p=lambda d: d["match_id"].map(oof), conjunto="validación temporal (2010–2015)")
    test = df[df["temporada"] == TEST_TEMPORADA].assign(p=lambda d: d["prob_modelo"], conjunto="test 2015/16 (modelo final)")
    return pd.concat([val, test]).sort_values(["fecha", "match_id"]).reset_index(drop=True)


def partidos_para_explorar(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por partido fuera de muestra, con lo que habría dicho y hecho el modelo con lo que sabía hasta ese momento."""
    favorito_local = df["equipo_favorito"] == "local"
    valor = df["p"] * df["cuota_underdog"] - 1
    apuesta = valor > 0
    gano = df[TARGET_COLUMN].astype(int)
    return pd.DataFrame({
        "fecha": df["fecha"].dt.strftime("%Y-%m-%d"),
        "temporada": df["temporada"],
        "liga": df["liga"].map(LIGAS).fillna(df["liga"]),
        "conjunto": df["conjunto"],
        "favorito": np.where(favorito_local, df["equipo_local"], df["equipo_visitante"]),
        "underdog": np.where(favorito_local, df["equipo_visitante"], df["equipo_local"]),
        "partido": df["equipo_local"] + " " + df["goles_local"].astype(str) + "–" + df["goles_visitante"].astype(str) + " " + df["equipo_visitante"],
        "resultado": np.where(gano == 1, "Ganó el underdog", "No ganó"),
        "gano": gano,
        "prob_mercado": df["prob_no_favorito"].round(4),
        "prob_modelo": df["p"].round(4),
        "cuota": df["cuota_underdog"].round(2),
        "valor": valor.round(4),
        "decision": np.where(apuesta, "Apuesta", "No apuesta"),
        "apuesta": apuesta.astype(int),
        "ganancia": np.where(apuesta, APUESTA * (df["cuota_underdog"] * gano - 1), 0).round(1),
        "gap_xi": df["nofav_xi_overall_mean_gap"].round(2),
    })


def tabla_datos(df: pd.DataFrame) -> list[dict]:
    """Cuánto gana el underdog según qué tan favorito es el favorito (las barras de «Los datos»)."""
    cortes = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 1]
    etiquetas = ["hasta 40%", "40–50%", "50–60%", "60–70%", "70–80%", "más de 80%"]
    t = (df.assign(tramo=pd.cut(df["prob_favorito"], cortes, labels=etiquetas))
           .groupby("tramo", observed=True)[TARGET_COLUMN].agg(tasa="mean", partidos="size").reset_index())
    t["tramo"] = t["tramo"].astype(str)
    return t.to_dict("records")


def tabla_hallazgo(df: pd.DataFrame) -> list[dict]:
    """Victorias reales del underdog y lo que esperaba la cuota, por quintil de diferencia de rating del XI."""
    q = pd.qcut(df["nofav_xi_overall_mean_gap"], 5)
    etiquetas = [f"{i.left:+.1f} a {i.right:+.1f}".replace(".", ",") for i in q.cat.categories]
    etiquetas[0] = f"{df['nofav_xi_overall_mean_gap'].min():+.1f} a {q.cat.categories[0].right:+.1f}".replace(".", ",")
    t = (df.assign(quintil=q.cat.rename_categories(etiquetas))
           .groupby("quintil", observed=True)
           .agg(real=(TARGET_COLUMN, "mean"), cuota=("prob_no_favorito", "mean"), partidos=("match_id", "size")).reset_index())
    t["quintil"] = t["quintil"].astype(str)
    return t.to_dict("records")


# ----------------------------------------------------------------------------- tablas de los «Cómo lo sabemos»
def tabla_html(registros: list[dict], columnas: list[str] | None = None, formatos: dict | None = None,
               destacar: int | None = None) -> str:
    formatos = formatos or {}
    columnas = columnas or list(registros[0])
    filas = []
    for i, r in enumerate(registros):
        celdas = []
        for c in columnas:
            v = r.get(c)
            if c in formatos and v not in (None, "", "nan"):
                try:
                    v = formatos[c].format(float(v))
                except (TypeError, ValueError):
                    pass
            celdas.append(f"<td>{html.escape('' if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))}</td>")
        clase = " class='destacada'" if i == destacar else ""
        filas.append(f"<tr{clase}>" + "".join(celdas) + "</tr>")
    cab = "".join(f"<th>{html.escape('' if c == 'fila' else c)}</th>" for c in columnas)
    return f"<div class='tbl'><table><tr>{cab}</tr>{''.join(filas)}</table></div>"


def embudo() -> str:
    ex = pd.read_parquet(AUDIT_DIR / "excluded_matches.parquet")["motivo"].value_counts()
    pasos = [("liga_sin_cuotas", "Se sacan las 2 ligas sin cuotas (Polonia y Suiza)"),
             ("alineacion_incompleta", "Se exige la alineación titular completa (22 jugadores)"),
             ("sin_cuotas_1x2", "Se exigen cuotas 1X2 completas de al menos una casa"),
             ("cuota_empate_incoherente", "Se sacan cuotas incoherentes (empate implícito > 40%)"),
             ("sin_underdog_identificable", "Se sacan partidos con las dos probabilidades idénticas"),
             ("partido_parejo", "Se exige un underdog claro: diferencia de probabilidad > 0,05")]
    quedan = 25_979
    filas = [{"paso": "Partidos en la fuente (European Soccer Database)", "se sacan": "", "quedan": entero(quedan)}]
    for motivo, texto in pasos:
        n = int(ex.get(motivo, 0))
        quedan -= n
        filas.append({"paso": texto, "se sacan": entero(n), "quedan": entero(quedan)})
    return tabla_html(filas)


# ----------------------------------------------------------------------------- casos para «Probá el modelo»
def casos_ejemplo(df: pd.DataFrame) -> list[dict]:
    test = df[df["temporada"] == TEST_TEMPORADA].copy()
    test["valor"] = test["prob_modelo"] * test["cuota_underdog"] - 1
    sorpresas = test[test[TARGET_COLUMN] == 1].nsmallest(2, "prob_modelo")      # las dos sorpresas más grandes del test
    mas_valor = test.nlargest(2, "valor")                                        # las apuestas que el modelo más quería
    apuesta_ganada = test[(test["valor"] > 0) & (test[TARGET_COLUMN] == 1)].nlargest(1, "valor")
    elegidos = [(r, "la apuesta con más valor") for _, r in mas_valor.iterrows()]
    elegidos += [(r, "apuesta que salió") for _, r in apuesta_ganada.iterrows()]
    elegidos += [(r, "sorpresa que nadie vio") for _, r in sorpresas.iterrows()]
    elegidos += [(r, "") for _, r in test.sample(3, random_state=42).iterrows()]
    return [caso(r, etiqueta) for r, etiqueta in elegidos]


def caso(r: pd.Series, etiqueta: str = "") -> dict:
    """Un partido real, con todo lo que «Probá el modelo» necesita para cargarlo."""
    return {
        "etiqueta": etiqueta, "liga": LIGAS.get(r["liga"], r["liga"]), "fecha": r["fecha"].strftime("%d/%m/%Y"),
        "local": r["equipo_local"], "visitante": r["equipo_visitante"],
        "resultado": f"Terminó {r['goles_local']}–{r['goles_visitante']}",
        "cuotas": [float(r["odds_home"]), float(r["odds_draw"]), float(r["odds_away"])],
        "gaps": {c: (None if pd.isna(r[c]) else round(float(r[c]), 4)) for c in NOMBRES_GAPS},
    }


def partidos_del_test(df: pd.DataFrame) -> list[dict]:
    """Todos los partidos de la temporada de test (los que el modelo final nunca vio), para elegir por liga y equipo."""
    test = df[df["temporada"] == TEST_TEMPORADA].sort_values(["fecha", "match_id"])
    return [caso(r) for _, r in test.iterrows()]


def resumen_del_silver(df: pd.DataFrame, ficha: dict, resultados: dict) -> dict:
    """Lo que sólo se puede calcular con el Silver, la auditoría y el notebook. Se guarda en docs/data/silver_resumen.json."""
    return {
        "fecha_datos": json.loads(SILVER_MANIFEST_PATH.read_text(encoding="utf-8")).get("built_at_utc", "")[:10],
        "partidos": int(len(df)), "ganados": int(df[TARGET_COLUMN].sum()),
        "tabla_datos": tabla_datos(df), "tabla_hallazgo": tabla_hallazgo(df),
        "html": {
            "EMBUDO": embudo(), "FUGA": ", ".join(f"<code>{c}</code>" for c in LEAKAGE_COLUMNS),
            "T_COMPARACION": tabla_html(resultados["comparacion"], formatos={"AUC media": "{:.3f}", "AUC desvío": "{:.3f}", "AUC train": "{:.3f}",
                                                                             "log-loss media": "{:.4f}", "Δ AUC % vs logística": "{:+.2f}%"}),
            "T_GRUPOS": tabla_html(resultados["grupos"], formatos={"AUC media": "{:.3f}", "AUC desvío": "{:.3f}",
                                                                   "Δ AUC vs prob. sola": "{:+.4f}", "desvío de Δ": "{:.4f}"}),
            "T_PLATA": tabla_html(resultados["plata"], formatos={"% de partidos": "{:.1f}%", "ROI %": "{:+.2f}%"}),
            "T_PEDAZOS": tabla_html(resultados["plata_por_pedazo"], formatos={k: "{:+.1f}%" for k in resultados["plata_por_pedazo"][0] if k not in ("fila", "casa")}),
            "T_TEST": tabla_html(resultados["test"], formatos={"AUC": "{:.3f}", "log-loss": "{:.4f}", "ROI %": "{:+.2f}%"}),
            "T_COEF": tabla_html(resultados["coeficientes"], formatos={"coeficiente": "{:+.3f}"}),
        },
        "C": str(ficha["C"]), "intercepto": ficha["intercepto"],
        "coef_prob": ficha["probabilidades"]["prob_no_favorito"]["coef"],
        "modelo": modelo_para_la_pagina(df, ficha),
        "casos": casos_ejemplo(df),
    }


def modelo_para_la_pagina(df: pd.DataFrame, ficha: dict) -> dict:
    rangos = {c: {"min": v["min_entrenamiento"], "max": v["max_entrenamiento"], "nombre": NOMBRES_GAPS[c]}
              for c, v in ficha["numericas"].items()}
    entrenamiento = df[df["temporada"] != TEST_TEMPORADA]
    rangos.update({c: {"min": float(entrenamiento[c].min()), "max": float(entrenamiento[c].max()), "nombre": c}
                   for c in ficha["probabilidades"]})
    return {"intercepto": ficha["intercepto"], "probabilidades": ficha["probabilidades"], "numericas": ficha["numericas"], "rangos": rangos}


# ----------------------------------------------------------------------------- etapa 1: ¿predice? (sin plata)
def auc(y: np.ndarray, p: np.ndarray) -> float:
    r = pd.Series(p).rank().values
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def metricas(test: pd.DataFrame, tasa_base: float) -> list[dict]:
    """Las tres formas de predecir, medidas en el test sin hablar de plata."""
    y = test["gano"].values
    filas = []
    for nombre, p in [(f"Piso: decir siempre {pct(tasa_base, 0)} (la tasa base)", np.full(len(y), tasa_base)),
                      ("La cuota sola (probabilidad sin margen)", test["prob_mercado"].values),
                      ("Nuestro modelo (cuota + jugadores)", test["prob_modelo"].values)]:
        filas.append({"": nombre, "AUC": auc(y, p), "log-loss": log_loss(y, p), "Brier": float(np.mean((y - p) ** 2))})
    return filas


def grafico_calibracion(test: pd.DataFrame) -> alt.Chart:
    """Probabilidad dicha contra proporción que ganó de verdad, en diez grupos de partidos de igual tamaño."""
    filas = []
    for quien, col in [("Nuestro modelo", "prob_modelo"), ("La cuota", "prob_mercado")]:
        g = test.assign(grupo=pd.qcut(test[col], 10, labels=False, duplicates="drop")).groupby("grupo")
        t = g.agg(dijo=(col, "mean"), paso=("gano", "mean"), partidos=("gano", "size")).reset_index()
        t["quien"] = quien
        filas.append(t)
    t = pd.concat(filas)
    escala = alt.Scale(domain=["Nuestro modelo", "La cuota"], range=[COLOR["modelo"], COLOR["mercado"]])
    trazo = alt.Scale(domain=["Nuestro modelo", "La cuota"], range=[[1, 0], [6, 4]])
    eje = alt.Scale(domain=[0, 0.5])
    x = alt.X("dijo:Q", title="Probabilidad que dio (promedio del grupo)", axis=alt.Axis(format="%", tickCount=5), scale=eje)
    y = alt.Y("paso:Q", title="Proporción que ganó de verdad", axis=alt.Axis(format="%", tickCount=5), scale=eje)
    diag = alt.Chart(pd.DataFrame({"dijo": [0, 0.5], "paso": [0, 0.5]})).mark_line(color="#b9bfc7", strokeWidth=1.5).encode(x=x, y=y)
    rot = alt.Chart(pd.DataFrame({"dijo": [0.36], "paso": [0.43], "t": ["sobre la diagonal: lo que dice es lo que pasa"]})).mark_text(
        fontSize=12, fontStyle="italic", color=COLOR["tinta_suave"], angle=315).encode(x=x, y=y, text="t:N")
    lineas = alt.Chart(t).mark_line(point=alt.OverlayMarkDef(size=70, filled=True), strokeWidth=2.5).encode(
        x=x, y=y,
        color=alt.Color("quien:N", scale=escala, legend=alt.Legend(title=None, orient="top-left", symbolStrokeWidth=3, labelFontSize=12)),
        strokeDash=alt.StrokeDash("quien:N", scale=trazo, legend=alt.Legend(title=None, orient="top-left")),
        tooltip=[alt.Tooltip("quien:N", title=""), alt.Tooltip("dijo:Q", title="Dijo", format=".1%"),
                 alt.Tooltip("paso:Q", title="Ganó de verdad", format=".1%"), alt.Tooltip("partidos:Q", title="Partidos")])
    return (diag + rot + lineas).properties(width="container", height=330)


def matriz(test: pd.DataFrame, col: str, umbral: float) -> dict:
    y = test["gano"].values == 1
    dice = test[col].values >= umbral
    return {"aciertos": int((dice & y).sum()), "falsas_alarmas": int((dice & ~y).sum()),
            "se_le_escapan": int((~dice & y).sum()), "bien_descartados": int((~dice & ~y).sum())}


def grafico_matrices(test: pd.DataFrame, umbral: float) -> alt.Chart:
    """Matriz de confusión del modelo y de la cuota con el mismo umbral, una al lado de la otra."""
    paneles = []
    for quien, col, color in [("Nuestro modelo", "prob_modelo", COLOR["modelo"]), ("La cuota", "prob_mercado", COLOR["mercado"])]:
        m = matriz(test, col, umbral)
        dice_si, dice_no = f"Dice «gana» (≥ {pct(umbral, 0)})", "Dice «no gana»"
        t = pd.DataFrame([
            {"dice": dice_si, "real": "Ganó el underdog", "n": m["aciertos"], "que": "acierto"},
            {"dice": dice_no, "real": "Ganó el underdog", "n": m["se_le_escapan"], "que": "se le escapa"},
            {"dice": dice_si, "real": "No ganó", "n": m["falsas_alarmas"], "que": "falsa alarma"},
            {"dice": dice_no, "real": "No ganó", "n": m["bien_descartados"], "que": "bien descartado"}])
        t["bien"] = t["que"].isin(["acierto", "bien descartado"])
        t["texto"] = t["n"].map(entero)
        x = alt.X("dice:N", title=None, sort=[dice_si, dice_no], axis=alt.Axis(orient="top", labelAngle=0, labelFontSize=11, labelLimit=150))
        y = alt.Y("real:N", title=None, sort=["Ganó el underdog", "No ganó"], axis=alt.Axis(labelFontSize=11))
        base = alt.Chart(t).encode(x=x, y=y)
        celdas = base.mark_rect(stroke="white", strokeWidth=3, cornerRadius=6).encode(
            color=alt.condition("datum.bien", alt.value(color), alt.value("#eef0f2")),
            tooltip=[alt.Tooltip("real:N", title="Lo que pasó"), alt.Tooltip("dice:N", title="Lo que dijo"),
                     alt.Tooltip("n:Q", title="Partidos", format=",")])
        numero = base.mark_text(fontSize=20, fontWeight="bold", dy=-7).encode(
            text="texto:N", color=alt.condition("datum.bien", alt.value("white"), alt.value(COLOR["tinta"])))
        rotulo = base.mark_text(fontSize=11, dy=13).encode(
            text="que:N", color=alt.condition("datum.bien", alt.value("white"), alt.value(COLOR["tinta_suave"])))
        paneles.append((celdas + numero + rotulo).properties(
            width=230, height=170, title=alt.TitleParams(quien, color=color, fontSize=14, anchor="start")))
    return alt.hconcat(*paneles, spacing=30)


def grafico_roc(test: pd.DataFrame) -> alt.Chart:
    filas = []
    y = test["gano"].values
    for quien, col in [("Nuestro modelo", "prob_modelo"), ("La cuota", "prob_mercado")]:
        orden = np.argsort(-test[col].values)
        tp = np.cumsum(y[orden]) / y.sum()
        fp = np.cumsum(1 - y[orden]) / (len(y) - y.sum())
        paso = max(1, len(tp) // 160)
        idx = np.r_[0, np.arange(0, len(tp), paso), len(tp) - 1]
        filas.append(pd.DataFrame({"fp": np.r_[0, fp[idx]], "tp": np.r_[0, tp[idx]], "quien": quien}))
    filas.append(pd.DataFrame({"fp": [0, 1], "tp": [0, 1], "quien": "Adivinar (AUC 0,5)"}))
    t = pd.concat(filas)
    dominio = ["Nuestro modelo", "La cuota", "Adivinar (AUC 0,5)"]
    return alt.Chart(t).mark_line(strokeWidth=2.5).encode(
        x=alt.X("fp:Q", title="Falsas alarmas (de los que no ganaron, qué parte marcó)", axis=alt.Axis(format="%")),
        y=alt.Y("tp:Q", title="Aciertos (de los que ganaron, qué parte marcó)", axis=alt.Axis(format="%")),
        color=alt.Color("quien:N", scale=alt.Scale(domain=dominio, range=[COLOR["modelo"], COLOR["mercado"], "#aab1ba"]),
                        legend=alt.Legend(title=None, orient="bottom-right", symbolStrokeWidth=3)),
        strokeDash=alt.StrokeDash("quien:N", scale=alt.Scale(domain=dominio, range=[[1, 0], [6, 4], [2, 3]]),
                                  legend=alt.Legend(title=None, orient="bottom-right")),
        order="fp:Q",
    ).properties(width="container", height=320)


# ----------------------------------------------------------------------------- etapa 2: ¿sirve para apostar?
def ganancia_si_apuesta(e: pd.DataFrame) -> pd.Series:
    return APUESTA * (e["cuota"] * e["gano"] - 1)


def retorno(e: pd.DataFrame, apuesta: pd.Series) -> tuple[int, float]:
    n = int(apuesta.sum())
    return n, (float(ganancia_si_apuesta(e)[apuesta].sum() / (APUESTA * n)) if n else 0.0)


def intervalo(e: pd.DataFrame, apuesta: pd.Series, rng: np.random.Generator) -> tuple[float, float]:
    g = ganancia_si_apuesta(e)[apuesta].values / APUESTA
    if len(g) == 0:
        return 0.0, 0.0
    medias = rng.choice(g, size=(REMUESTREOS, len(g)), replace=True).mean(axis=1)
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def nombre_rango(a: float, b: float) -> str:
    return f"{pct(a, 0)} o más" if b >= 1 else (f"menos de {pct(b, 0)}" if a == 0 else f"entre {pct(a, 0)} y {pct(b, 0)}")


def elegir_reglas(val: pd.DataFrame) -> dict:
    """Los parámetros de las estrategias de comparación, elegidos sólo con la validación (2010–2015)."""
    umbrales = {u: retorno(val, val["prob_modelo"] > u)[1] for u in UMBRALES_SIN_REGLA}
    umbral = max(umbrales, key=umbrales.get)
    rangos = {}
    for i, a in enumerate(CORTES_RANGO):
        for b in CORTES_RANGO[i + 1:]:
            if (a, b) == (0, 1):
                continue
            n, r = retorno(val, (val["prob_mercado"] >= a) & (val["prob_mercado"] < b))
            if n >= MIN_APUESTAS_VALIDACION:
                rangos[(a, b)] = r
    rango = max(rangos, key=rangos.get)
    return {"umbral": umbral, "umbrales": umbrales, "rango": rango}


def estrategias(e: pd.DataFrame, reglas: dict) -> dict[str, tuple[pd.Series, str, str]]:
    """Cada estrategia: (en qué partidos apuesta, nombre corto para los gráficos, explicación)."""
    u = reglas["umbral"]
    a, b = reglas["rango"]
    pm = e["prob_mercado"]
    return {
        "modelo": (e["apuesta"] == 1, "Modelo con la regla", "prob. del modelo × cuota − 1 > 0 (valor esperado positivo)"),
        "sin_regla": (e["prob_modelo"] > u, f"Modelo sin la regla (> {pct(u, 0)})",
                      f"la prob. del modelo supera {pct(u, 0)} (el mejor umbral fijo en validación), sin mirar la cuota"),
        "todos": (pd.Series(True, index=e.index), "Apostar a todos", "siempre, a ciegas"),
        "rango": ((pm >= a) & (pm < b), f"Cuota le da {nombre_rango(a, b)}",
                  f"la cuota le da al underdog {nombre_rango(a, b)} (el mejor rango en validación)"),
        "once": (e["gap_xi"] > 0, "Once mejor que el del favorito", "su once titular tiene mejor rating medio que el del favorito"),
        "parejos": (pm >= 0.30, "Partidos parejos (30% o más)", "la cuota le da al underdog 30% o más"),
        "batacazos": (pm < 0.10, "Batacazos (menos de 10%)", "la cuota le da al underdog menos de 10%"),
        "nunca": (pd.Series(False, index=e.index), "No apostar nunca", "nunca (es lo que hace la cuota usada como modelo)"),
    }


def tabla_estrategias(val: pd.DataFrame, test: pd.DataFrame, reglas: dict, rng: np.random.Generator) -> list[dict]:
    ev, et = estrategias(val, reglas), estrategias(test, reglas)
    filas = []
    for k in ev:
        nv, rv = retorno(val, ev[k][0])
        nt, rt = retorno(test, et[k][0])
        bajo, alto = intervalo(test, et[k][0], rng)
        filas.append({"clave": k, "nombre": ev[k][1], "cuando": ev[k][2], "val_n": nv, "val_r": rv, "test_n": nt, "test_r": rt,
                      "test_bajo": bajo, "test_alto": alto})
    return filas


def sorteos(e: pd.DataFrame, n: int, rng: np.random.Generator) -> tuple[np.ndarray, pd.DataFrame]:
    """Apostar $100 al azar en n partidos, SORTEOS veces: el retorno final de cada sorteo y la banda del 90% central
    de la ganancia acumulada día a día."""
    e = e.sort_values("fecha")
    g = ganancia_si_apuesta(e).values
    finales = np.empty(SORTEOS)
    acumulados = np.empty((SORTEOS, len(e)))
    for s in range(SORTEOS):
        elegidos = np.zeros(len(e), dtype=bool)
        elegidos[rng.choice(len(e), size=n, replace=False)] = True
        jugada = np.where(elegidos, g, 0.0)
        finales[s] = jugada.sum() / (APUESTA * n)
        acumulados[s] = np.cumsum(jugada)
    fechas = e["fecha"].values
    ultimo = pd.Series(np.arange(len(e))).groupby(fechas).max()
    banda = pd.DataFrame({"fecha": ultimo.index, "bajo": np.percentile(acumulados[:, ultimo.values], 5, axis=0),
                          "alto": np.percentile(acumulados[:, ultimo.values], 95, axis=0)})
    return finales, banda


def grafico_regla(test: pd.DataFrame, ejemplo: pd.Series) -> alt.Chart:
    """La regla de apuesta vista como un gráfico: lo que paga la cuota contra lo que cree el modelo."""
    t = test.assign(paga=(1 / test["cuota"]).round(4), decide=np.where(test["apuesta"] == 1, "El modelo apuesta", "No apuesta"))
    t = t[["partido", "cuota", "paga", "prob_modelo", "apuesta", "decide"]]   # sólo lo que usa el gráfico
    eje = alt.Scale(domain=[0, 0.45])
    x = alt.X("paga:Q", title="Probabilidad que paga la cuota (1 / cuota)", axis=alt.Axis(format="%", tickCount=5), scale=eje)
    y = alt.Y("prob_modelo:Q", title="Probabilidad según el modelo", axis=alt.Axis(format="%", tickCount=5), scale=eje)
    puntos = alt.Chart(t).mark_circle(clip=True).encode(
        x=x, y=y,
        color=alt.Color("decide:N", scale=alt.Scale(domain=["El modelo apuesta", "No apuesta"], range=[COLOR["modelo"], COLOR["fondo_puntos"]]),
                        legend=alt.Legend(title=None, orient="top-left", labelFontSize=12)),
        size=alt.condition("datum.apuesta == 1", alt.value(34), alt.value(14)),
        opacity=alt.condition("datum.apuesta == 1", alt.value(0.85), alt.value(0.55)),
        tooltip=[alt.Tooltip("partido:N", title="Partido"), alt.Tooltip("cuota:Q", title="Cuota del underdog", format=".2f"),
                 alt.Tooltip("paga:Q", title="Paga como si tuviera", format=".1%"),
                 alt.Tooltip("prob_modelo:Q", title="El modelo le da", format=".1%")])
    linea = alt.Chart(pd.DataFrame({"paga": [0, 0.45], "prob_modelo": [0, 0.45]})).mark_line(
        color=COLOR["tinta"], strokeWidth=2, strokeDash=[6, 4]).encode(x=x, y=y)
    rot_linea = alt.Chart(pd.DataFrame({"paga": [0.405], "prob_modelo": [0.425], "t": ["lo que paga la cuota"]})).mark_text(
        fontSize=12, fontWeight="bold", color=COLOR["tinta"], angle=315).encode(x=x, y=y, text="t:N")
    zonas = alt.Chart(pd.DataFrame({"paga": [0.03, 0.30], "prob_modelo": [0.33, 0.08],
                                    "t": ["arriba: el modelo ve más chances de las que se pagan → apuesta",
                                          "abajo: se paga de menos → no apuesta"]})).mark_text(
        align="left", fontSize=12, fontStyle="italic", color=COLOR["tinta_suave"]).encode(x=x, y=y, text="t:N")
    ej = pd.DataFrame([{"paga": 1 / ejemplo["cuota"], "prob_modelo": ejemplo["prob_modelo"],
                        "t": f"{ejemplo['underdog'].split()[-1]}: paga {pct(1 / ejemplo['cuota'])}, el modelo cree {pct(ejemplo['prob_modelo'])}"}])
    marca = alt.Chart(ej).mark_point(size=260, strokeWidth=2.5, color=COLOR["tinta"], filled=False).encode(x=x, y=y)
    # el rótulo del ejemplo va arriba a la izquierda, donde no hay puntos, con una línea que lo une al partido
    ej2 = ej.assign(tx=0.02, ty=0.24)
    union = alt.Chart(ej2).mark_rule(color=COLOR["tinta"], strokeWidth=1).encode(x="tx:Q", y="ty:Q", x2="paga:Q", y2="prob_modelo:Q")
    texto = alt.Chart(ej2).mark_text(align="left", dy=-8, fontSize=12, fontWeight="bold", color=COLOR["tinta"]).encode(
        x=alt.X("tx:Q", scale=eje), y=alt.Y("ty:Q", scale=eje), text="t:N")
    return (puntos + linea + rot_linea + zonas + union + marca + texto).properties(width="container", height=380)


def grafico_con_y_sin_regla(filas: list[dict]) -> alt.Chart:
    """El mismo modelo con y sin la regla, contra apostar a todos, en los dos períodos."""
    elegidas = [f for f in filas if f["clave"] in ("modelo", "sin_regla", "todos")]
    t = pd.DataFrame([{"estrategia": f["nombre"], "periodo": p, "roi": 100 * f[c], "n": f[n]}
                      for f in elegidas for p, c, n in [("Validación 2010–2015", "val_r", "val_n"), ("Test 2015/16", "test_r", "test_n")]])
    t["texto"] = t["roi"].map(lambda v: f"{v:+.1f}%".replace(".", ",").replace("-", "−"))
    orden = [f["nombre"] for f in elegidas]
    periodos = ["Validación 2010–2015", "Test 2015/16"]
    y = alt.Y("estrategia:N", title=None, sort=orden, axis=alt.Axis(labelFontSize=12, labelFontWeight="bold", labelLimit=220))
    x = alt.X("roi:Q", title="Retorno por cada $100 apostados (%)", scale=alt.Scale(domain=[-10, 10]))
    color = alt.Color("periodo:N", scale=alt.Scale(domain=periodos, range=["#9aa3ad", COLOR["tinta"]]),
                      legend=alt.Legend(title=None, orient="top", labelFontSize=12))
    off = alt.YOffset("periodo:N", sort=periodos)
    barras = alt.Chart(t).mark_bar(height=16, cornerRadius=3).encode(
        y=y, x=x, color=color, yOffset=off,
        tooltip=[alt.Tooltip("estrategia:N", title="Estrategia"), alt.Tooltip("periodo:N", title="Período"),
                 alt.Tooltip("roi:Q", title="Retorno %", format="+.1f"), alt.Tooltip("n:Q", title="Apuestas", format=",")])
    texto = (alt.Chart(t[t["roi"] >= 0]).mark_text(fontSize=12, fontWeight="bold", align="left", dx=5).encode(
                 y=y, yOffset=off, x=x, text="texto:N", color=alt.value(COLOR["tinta"]))
             + alt.Chart(t[t["roi"] < 0]).mark_text(fontSize=12, fontWeight="bold", align="right", dx=-5).encode(
                 y=y, yOffset=off, x=x, text="texto:N", color=alt.value(COLOR["tinta"])))
    cero = alt.Chart(pd.DataFrame({"roi": [0]})).mark_rule(color=COLOR["tinta_suave"]).encode(x="roi:Q")
    return (barras + texto + cero).properties(width="container", height=230)


def grafico_test(test: pd.DataFrame, reglas: dict, banda: pd.DataFrame) -> alt.Chart:
    """Ganancia acumulada en el test, en pesos, con la misma plata en juego para todas las estrategias: cada una reparte en sus
    apuestas lo mismo que apostó el modelo en total ($100 por cada apuesta del modelo). Así no gana la que más veces apuesta."""
    e = test.sort_values("fecha")
    est = estrategias(e, reglas)
    claves = ["modelo", "sin_regla", "todos", "rango", "once", "nunca"]
    g = ganancia_si_apuesta(e)
    n_modelo = int(est["modelo"][0].sum())
    en_juego = APUESTA * n_modelo
    # lo que gana cada estrategia apostando $100 por partido, escalado para que su total apostado sea el del modelo
    escala_k = {k: (n_modelo / est[k][0].sum() if est[k][0].sum() else 0.0) for k in claves}
    diario = (pd.DataFrame({est[k][1]: np.where(est[k][0], g, 0.0) * escala_k[k] for k in claves})
                .groupby(e["fecha"].values).sum().cumsum())
    inicio = pd.DataFrame(0.0, index=[(pd.Timestamp(diario.index[0]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")], columns=diario.columns)
    diario = pd.concat([inicio, diario])
    largo = diario.rename_axis("fecha").reset_index().melt(id_vars="fecha", var_name="estrategia", value_name="ganancia")
    nombres = [est[k][1] for k in claves]
    colores = [COLOR[k] if k in COLOR else c for k, c in zip(claves, [None, None, None, COLOR["mercado"], COLOR["jugadores"], None])]
    trazos = [[1, 0], [6, 4], [1, 0], [6, 4], [2, 3], [1, 0]]
    gruesos = alt.condition(f"datum.estrategia == '{nombres[0]}'", alt.value(4), alt.value(2.2))
    x = alt.X("fecha:T", title=None, axis=alt.Axis(format="%b %Y", tickCount=6, grid=False))
    y = alt.Y("ganancia:Q", title=f"Ganancia acumulada ($), apostando {pesos(en_juego)[1:]} en total cada una",
              axis=alt.Axis(tickCount=6, format=",.0f",
                            labelExpr="(datum.value > 0 ? '+$' : datum.value < 0 ? '−$' : '$') + replace(datum.label, '−', '')"))
    # leyenda arriba (con el mismo trazo que cada línea) y, al final de cada línea, cuánto terminó: así entra también en el celular
    leyenda = alt.Legend(title=None, orient="top", columns=2, labelFontSize=12, labelLimit=150, symbolStrokeWidth=3, symbolSize=420,
                         symbolType="stroke")
    color = alt.Color("estrategia:N", scale=alt.Scale(domain=nombres, range=colores), legend=leyenda)
    banda = banda.assign(fecha=lambda d: pd.to_datetime(d["fecha"]).dt.strftime("%Y-%m-%d"))
    area = alt.Chart(banda).mark_area(color=COLOR["azar"], opacity=0.9).encode(
        x=x, y="bajo:Q", y2="alto:Q",
        tooltip=[alt.Tooltip("fecha:T", title="Fecha", format="%d/%m/%Y"), alt.Tooltip("bajo:Q", title="Azar, 5% ($)", format="+,.0f"),
                 alt.Tooltip("alto:Q", title="Azar, 95% ($)", format="+,.0f")])
    lineas = alt.Chart(largo).mark_line(interpolate="step-after").encode(
        x=x, y=y, color=color, strokeWidth=gruesos,
        strokeDash=alt.StrokeDash("estrategia:N", scale=alt.Scale(domain=nombres, range=trazos), legend=leyenda),
        tooltip=[alt.Tooltip("fecha:T", title="Fecha", format="%d/%m/%Y"), alt.Tooltip("estrategia:N", title="Estrategia"),
                 alt.Tooltip("ganancia:Q", title="Acumulado ($)", format="+,.0f")])
    # rótulos al final de cada línea, separados para que no se pisen
    fin = diario.iloc[-1]
    ultimo = diario.index[-1]
    rango_y = max(diario.max().max(), banda["alto"].max()) - min(diario.min().min(), banda["bajo"].min())
    sep = rango_y / 16
    pos = sorted(((float(fin[nm]), nm) for nm in nombres), reverse=True)
    ys = []
    for v, _ in pos:
        ys.append(min(v, ys[-1] - sep) if ys else v)
    rot = pd.DataFrame([{"fecha": ultimo, "ganancia": yy, "estrategia": nm, "t": pesos(v)} for (v, nm), yy in zip(pos, ys)])
    rotulos = alt.Chart(rot).mark_text(align="left", dx=8, fontSize=12, fontWeight="bold").encode(x=x, y=y, text="t:N", color=color)
    piso = min(diario.min().min(), banda["bajo"].min())
    azar = pd.DataFrame({"fecha": [banda["fecha"].iloc[len(banda) // 2]], "ganancia": [piso - sep * 0.8],
                         "t": ["gris: 90% de los sorteos al azar"]})
    rot_azar = alt.Chart(azar).mark_text(align="center", fontSize=11, fontStyle="italic", color=COLOR["tinta_suave"]).encode(x=x, y=y, text="t:N")
    return (area + lineas + rotulos + rot_azar).properties(width="container", height=400)


# ----------------------------------------------------------------------------- 0 · el problema, 1 · los datos, 2 · lo que encontramos
def grafico_problema(ej: dict) -> tuple[alt.Chart, dict]:
    """Las tres cuotas de un partido real pasadas a probabilidad: suman más de 100%, y lo que sobra es el margen de la casa."""
    local, visit = ej["local"].split()[-1], ej["visitante"].split()[-1]
    inv = [1 / c for c in ej["cuotas"]]
    suma = sum(inv)
    # cada tramo se nombra en la leyenda (con su porcentaje): los rótulos adentro de la barra no entran en el celular
    t = pd.DataFrame({"que": [f"Gana {local}: {pct(inv[0])}", f"Empate: {pct(inv[1])}", f"Gana {visit}: {pct(inv[2])}"],
                      "p": inv, "cuota": ej["cuotas"]})
    t["fin"] = t["p"].cumsum()
    t["inicio"] = t["fin"] - t["p"]
    t["fin"] = t["fin"].clip(upper=1.0)   # lo que pasa del 100% se pinta aparte, como margen
    margen = f"Margen de la casa: {pct(suma - 1)}"
    t = pd.concat([t, pd.DataFrame([{"que": margen, "p": suma - 1, "cuota": np.nan, "inicio": 1.0, "fin": suma}])])
    x = alt.X("inicio:Q", title=None, scale=alt.Scale(domain=[0, 1.04]), axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1]))
    color = alt.Color("que:N", sort=None, scale=alt.Scale(domain=list(t["que"]),
                                                         range=[COLOR["favorito"], COLOR["empate"], COLOR["gano"], COLOR["mercado"]]),
                      legend=alt.Legend(title=None, orient="top", columns=2, labelFontSize=12, labelLimit=220, symbolType="square", symbolSize=180))
    barra = alt.Chart(t).mark_bar(height=46, stroke="white", strokeWidth=1).encode(
        x=x, x2="fin:Q", color=color,
        tooltip=[alt.Tooltip("que:N", title=""), alt.Tooltip("cuota:Q", title="Cuota", format=".2f"),
                 alt.Tooltip("p:Q", title="1 / cuota", format=".1%")])
    cien = alt.Chart(pd.DataFrame({"x": [1.0], "t": ["100%"]}))
    regla = cien.mark_rule(color=COLOR["tinta"], strokeWidth=2, strokeDash=[4, 3]).encode(x="x:Q")
    return (barra + regla).properties(width="container", height=60), {"suma": suma, "inv": inv}


def grafico_datos(t: pd.DataFrame, tasa: float) -> alt.Chart:
    t = t.copy()
    t["por100"] = 100 * t["tasa"]
    t["destacado"] = [i in (0, len(t) - 1) for i in range(len(t))]
    t["valor"] = t["por100"].round().astype(int).astype(str)
    etiquetas = list(t["tramo"])
    # debajo de cada tramo, cuántos partidos tiene (dos renglones en el eje)
    expr = " : ".join(f"datum.label == '{r.tramo}' ? ['{r.tramo}', '{entero(r.partidos)} partidos']" for r in t.itertuples()) + " : datum.label"
    x = alt.X("tramo:O", sort=etiquetas, title="Probabilidad de ganar del favorito, según la cuota",
              axis=alt.Axis(labelAngle=0, labelFontSize=12, labelExpr=expr, labelLineHeight=16))
    y = alt.Y("por100:Q", title="De cada 100 partidos, cuántos ganó el underdog", axis=alt.Axis(tickCount=4),
              scale=alt.Scale(domain=[0, 45]))
    base = alt.Chart(t).encode(x=x, y=y, tooltip=[
        alt.Tooltip("tramo:O", title="Favorito"), alt.Tooltip("tasa:Q", title="Gana el underdog", format=".1%"),
        alt.Tooltip("partidos:Q", title="Partidos", format=",")])
    barras = base.mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, width={"band": 0.72}).encode(
        color=alt.condition("datum.destacado", alt.value(COLOR["gano"]), alt.value(COLOR["gano_suave"])))
    numero = base.mark_text(dy=-10, fontSize=15, fontWeight="bold", color=COLOR["tinta"]).encode(text="valor:N")
    media = alt.Chart(pd.DataFrame({"por100": [100 * tasa], "t": [f"promedio: {round(100 * tasa)} de cada 100"]}))
    regla = media.mark_rule(color=COLOR["tinta_suave"], strokeDash=[5, 4], strokeWidth=1.5).encode(y="por100:Q")
    rot_media = media.mark_text(align="right", x="width", dx=-2, dy=-8, fontSize=12, color=COLOR["tinta_suave"]).encode(y="por100:Q", text="t:N")
    notas = pd.DataFrame([
        {"tramo": etiquetas[0], "por100": t["por100"].iloc[0] + 9.5, "t": f"partido parejo:\n{round(10 * t['tasa'].iloc[0])} de cada 10"},
        {"tramo": etiquetas[-1], "por100": t["por100"].iloc[-1] + 9.5, "t": f"favorito abrumador:\n{round(100 * t['tasa'].iloc[-1])} de cada 100"}])
    nota = alt.Chart(notas).mark_text(fontSize=12, fontStyle="italic", color=COLOR["gano"], lineBreak="\n", baseline="bottom").encode(
        x=x, y=y, text="t:N")
    return (barras + regla + rot_media + numero + nota).properties(width="container", height=340)


def grafico_hallazgo(t: pd.DataFrame) -> alt.Chart:
    real, cuota = "Victorias reales del underdog", "Lo que esperaba la cuota"
    etiquetas = list(t["quintil"])
    largo = (t.rename(columns={"real": real, "cuota": cuota})
              .melt(id_vars=["quintil", "partidos"], value_vars=[real, cuota], var_name="serie", value_name="valor"))
    escala = alt.Scale(domain=[real, cuota], range=[COLOR["gano"], COLOR["mercado"]])
    # el trazo va codificado (no fijado a mano), así la leyenda muestra la línea punteada igual que el gráfico
    trazo = alt.Scale(domain=[real, cuota], range=[[1, 0], [7, 5]])
    leyenda = dict(title=None, orient="top-left", direction="vertical", labelFontSize=13, symbolStrokeWidth=3.5, symbolSize=500,
                   symbolType="stroke", labelFontWeight="bold", labelLimit=400)
    x = alt.X("quintil:O", sort=etiquetas, title="Rating medio del XI: underdog − favorito (cinco grupos de igual tamaño)",
              axis=alt.Axis(labelAngle=0, labelFontSize=12))
    y = alt.Y("valor:Q", title="Proporción", axis=alt.Axis(format="%", tickCount=4), scale=alt.Scale(domain=[0, 0.34]))
    lineas = alt.Chart(largo).mark_line(point=alt.OverlayMarkDef(size=90, filled=True), strokeWidth=3.5).encode(
        x=x, y=y, color=alt.Color("serie:N", scale=escala, legend=alt.Legend(**leyenda)),
        strokeDash=alt.StrokeDash("serie:N", scale=trazo, legend=alt.Legend(**leyenda)),
        tooltip=[alt.Tooltip("quintil:O", title="Diferencia de rating"), alt.Tooltip("serie:N", title="Serie"),
                 alt.Tooltip("valor:Q", title="Valor", format=".1%"), alt.Tooltip("partidos:Q", title="Partidos", format=",")])
    nota = alt.Chart(pd.DataFrame({"quintil": [etiquetas[3]], "valor": [0.07],
                                   "t": ["las dos líneas casi se pisan: el precio ya traía la calidad del once"]})).mark_text(
        fontSize=13, fontStyle="italic", color=COLOR["tinta_suave"]).encode(x=x, y=y, text="t:N")
    return (lineas + nota).properties(width="container", height=320)


# ----------------------------------------------------------------------------- la copa
def grafico_explorar(ligas: list[str], reglas: dict) -> alt.Chart:
    """Ancho fijo de referencia: la página lo ajusta al ancho de la pantalla antes de dibujarlo (ver plantilla)."""
    datos = alt.UrlData("data/partidos.json")
    liga = alt.param(name="liga", value=None, bind=alt.binding_select(options=[None] + ligas, labels=["Todas las ligas"] + ligas, name="Liga"))
    zona = alt.selection_interval(name="zona", encodings=["x", "y"])
    # sólo el test (lo que el modelo final nunca vio), igual que la sección 4; la validación queda en los «Cómo lo sabemos»
    filtro = f"datum.temporada == '{TEST_TEMPORADA}' && (liga == null || datum.liga == liga)"
    x = alt.X("prob_mercado:Q", title="Probabilidad del underdog según la cuota", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.36]))
    y = alt.Y("valor:Q", title="Valor esperado de apostarle, según el modelo", axis=alt.Axis(format="+%"), scale=alt.Scale(domain=[-0.4, 0.3]))
    detalle = [alt.Tooltip("fecha:T", title="Fecha", format="%d/%m/%Y"), alt.Tooltip("liga:N", title="Liga"),
               alt.Tooltip("partido:N", title="Partido"), alt.Tooltip("underdog:N", title="Underdog"),
               alt.Tooltip("cuota:Q", title="Cuota del underdog", format=".2f"),
               alt.Tooltip("prob_mercado:Q", title="Prob. según la cuota", format=".1%"),
               alt.Tooltip("prob_modelo:Q", title="Prob. según el modelo", format=".1%"),
               alt.Tooltip("decision:N", title="El modelo")]

    fondo = alt.Chart(datos).transform_filter(filtro).transform_filter("datum.apuesta == 0").mark_circle(
        size=10, opacity=0.35, color=COLOR["fondo_puntos"], clip=True).encode(x=x, y=y, tooltip=detalle)
    apuestas = alt.Chart(datos).transform_filter(filtro).transform_filter("datum.apuesta == 1").mark_circle(
        size=38, opacity=0.85, stroke="white", strokeWidth=0.6, clip=True).encode(
        x=x, y=y, tooltip=detalle, order=alt.Order("resultado:N", sort="descending"),  # los que ganaron se dibujan encima
        color=alt.Color("resultado:N", scale=RESULTADO, title="Apuestas del modelo",
                        legend=alt.Legend(orient="top-left", direction="horizontal", titleFontSize=12, labelFontSize=12)))
    cero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=COLOR["modelo"], strokeWidth=1.5, strokeDash=[6, 4]).encode(y="y:Q")
    rotulo = alt.Chart(pd.DataFrame({"y": [0], "t": ["arriba de esta línea, el modelo apuesta"]})).mark_text(
        align="left", x=6, dy=14, color=COLOR["modelo"], fontWeight="bold", fontSize=12).encode(y="y:Q", text="t:N")
    nube = (fondo + apuestas + cero + rotulo).add_params(liga, zona).properties(width=760, height=400)

    u = reglas["umbral"]
    a, b = reglas["rango"]
    n_rango = f"Cuota le da {nombre_rango(a, b)}"
    n_sin = f"Modelo sin la regla (> {pct(u, 0)})"
    orden = ["Modelo con la regla", n_sin, "Apostar a todos", n_rango, "Once mejor que el del favorito"]
    escala = alt.Scale(domain=orden, range=[COLOR["modelo"], COLOR["sin_regla"], COLOR["todos"], COLOR["mercado"], COLOR["jugadores"]])
    g = f"{APUESTA} * (datum.cuota * datum.gano - 1)"
    plata = alt.Chart(datos).transform_filter(filtro).transform_filter(zona).transform_calculate(
        g_todos=g,
        a_sin=f"datum.prob_modelo > {u} ? 1 : 0", g_sin=f"datum.prob_modelo > {u} ? {g} : 0",
        a_rango=f"datum.prob_mercado >= {a} && datum.prob_mercado < {b} ? 1 : 0",
        g_rango=f"datum.prob_mercado >= {a} && datum.prob_mercado < {b} ? {g} : 0",
        a_once="datum.gap_xi > 0 ? 1 : 0", g_once=f"datum.gap_xi > 0 ? {g} : 0",
    ).transform_aggregate(
        partidos="count()", suma_todos="sum(g_todos)", suma_modelo="sum(ganancia)", apuestas="sum(apuesta)",
        n_sin="sum(a_sin)", suma_sin="sum(g_sin)", n_rango="sum(a_rango)", suma_rango="sum(g_rango)",
        n_once="sum(a_once)", suma_once="sum(g_once)",
    ).transform_calculate(**{
        "Apostar a todos": f"100 * datum.suma_todos / ({APUESTA} * datum.partidos)",
        "Modelo con la regla": f"datum.apuestas > 0 ? 100 * datum.suma_modelo / ({APUESTA} * datum.apuestas) : 0",
        n_sin: f"datum.n_sin > 0 ? 100 * datum.suma_sin / ({APUESTA} * datum.n_sin) : 0",
        n_rango: f"datum.n_rango > 0 ? 100 * datum.suma_rango / ({APUESTA} * datum.n_rango) : 0",
        "Once mejor que el del favorito": f"datum.n_once > 0 ? 100 * datum.suma_once / ({APUESTA} * datum.n_once) : 0",
    }).transform_fold(orden, as_=["estrategia", "roi"])
    escala_roi = alt.Scale(domain=[-40, 40], clamp=True)
    barras = plata.mark_bar(height=20, cornerRadius=3).encode(
        y=alt.Y("estrategia:N", title=None, sort=orden, axis=alt.Axis(labelFontSize=12, labelFontWeight="bold", labelLimit=240)),
        x=alt.X("roi:Q", title="Retorno por cada $100 apostados (%), en lo filtrado", scale=escala_roi),
        color=alt.Color("estrategia:N", scale=escala, legend=None),
        tooltip=[alt.Tooltip("estrategia:N", title="Estrategia"), alt.Tooltip("roi:Q", title="Retorno %", format="+.1f"),
                 alt.Tooltip("partidos:Q", title="Partidos filtrados", format=","),
                 alt.Tooltip("apuestas:Q", title="Apuestas del modelo", format=",")])
    pos_texto = dict(y=alt.Y("estrategia:N", sort=orden), x=alt.X("roi:Q", scale=escala_roi), text=alt.Text("roi:Q", format="+.1f"))
    texto = (plata.transform_filter("datum.roi >= 0").mark_text(align="left", dx=6, fontWeight="bold", fontSize=13).encode(**pos_texto)
             + plata.transform_filter("datum.roi < 0").mark_text(align="right", dx=-6, fontWeight="bold", fontSize=13).encode(**pos_texto))
    resumen = alt.Chart(datos).transform_filter(filtro).transform_filter(zona).transform_aggregate(
        partidos="count()", apuestas="sum(apuesta)").transform_calculate(
        t="format(datum.partidos, ',') + ' partidos en la selección · el modelo apuesta en ' + format(datum.apuestas, ',')"
    ).mark_text(align="left", fontSize=13, fontWeight="bold", color=COLOR["tinta"]).encode(text="t:N").properties(width=760, height=20)
    cero_roi = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=COLOR["tinta_suave"]).encode(x="x:Q")
    return alt.vconcat(nube, resumen, (barras + texto + cero_roi).properties(width=760, height=170),
                       spacing=14).resolve_scale(color="independent")


def grafico_probar() -> alt.Chart:
    # fuentes con nombre y sin valores: la página les carga el caso que arma el usuario (view.data en JavaScript)
    caso_ = alt.NamedData(name="caso")
    escala = alt.Scale(domain=["Según la cuota", "Según el modelo"], range=[COLOR["mercado"], COLOR["modelo"]])
    barras = alt.Chart(caso_).mark_bar(height=34).encode(
        y=alt.Y("quien:N", title=None, sort=["Según la cuota", "Según el modelo"]),
        x=alt.X("p:Q", title="Probabilidad de que gane el underdog", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.6])),
        color=alt.Color("quien:N", scale=escala, legend=None),
        tooltip=[alt.Tooltip("quien:N", title=""), alt.Tooltip("p:Q", title="Probabilidad", format=".1%")])
    texto = barras.mark_text(align="left", dx=6, fontWeight="bold", color="#222").encode(text=alt.Text("p:Q", format=".1%"))
    equilibrio = alt.NamedData(name="equilibrio")
    regla = alt.Chart(equilibrio).mark_rule(color=COLOR["tinta"], strokeDash=[5, 3], strokeWidth=2).encode(x="p:Q")
    rotulo = alt.Chart(equilibrio).mark_text(align="left", dx=5, dy=-48, color=COLOR["tinta"]).encode(
        x="p:Q", text=alt.value("lo que paga la cuota (1 / cuota)"))
    return (barras + texto + regla + rotulo).properties(width="container", height=120)


# ----------------------------------------------------------------------------- página
def spec(chart: alt.Chart) -> str:
    return chart.configure_view(stroke=None).configure_axis(labelFontSize=12, titleFontSize=12, titleColor="#444",
                                                            gridColor="#eceef1").to_json(indent=None)


def armar_pagina(silver: dict, explorar: pd.DataFrame, test_casos: list[dict]) -> str:
    """Todo lo que sale de las predicciones fuera de muestra, más el resumen del Silver, metido en la plantilla."""
    rng = np.random.default_rng(SEMILLA)
    e = explorar.copy()
    val = e[~e["conjunto"].str.startswith("test")]
    test = e[e["conjunto"].str.startswith("test")]
    tasa_base = float(val["gano"].mean())

    # etapa 1
    met = metricas(test, tasa_base)
    m_mod, m_cuota = matriz(test, "prob_modelo", UMBRAL_MATRIZ), matriz(test, "prob_mercado", UMBRAL_MATRIZ)
    m50 = matriz(test, "prob_modelo", 0.5)

    # etapa 2
    reglas = elegir_reglas(val)
    filas = tabla_estrategias(val, test, reglas, rng)
    f = {r["clave"]: r for r in filas}
    fin_val, _ = sorteos(val, f["modelo"]["val_n"], rng)
    fin_test, banda = sorteos(test, f["modelo"]["test_n"], rng)

    # el partido de ejemplo
    ej = next(c for c in test_casos if c["local"] == EJEMPLO["local"] and c["visitante"] == EJEMPLO["visitante"] and c["fecha"] == EJEMPLO["fecha"])
    fila_ej = test[(test["fecha"] == "-".join(reversed(EJEMPLO["fecha"].split("/")))) & (test["underdog"] == EJEMPLO["visitante"])].iloc[0]
    g_problema, n_problema = grafico_problema(ej)
    inv = n_problema["inv"]
    valor_ej = float(fila_ej["prob_modelo"] * fila_ej["cuota"] - 1)

    t_datos = pd.DataFrame(silver["tabla_datos"])
    t_hallazgo = pd.DataFrame(silver["tabla_hallazgo"])
    tasa = silver["ganados"] / silver["partidos"]
    ligas = sorted(test["liga"].unique())
    graficos = {
        "g_problema": spec(g_problema), "g_datos": spec(grafico_datos(t_datos, tasa)), "g_hallazgo": spec(grafico_hallazgo(t_hallazgo)),
        "g_calibracion": spec(grafico_calibracion(test)), "g_matrices": spec(grafico_matrices(test, UMBRAL_MATRIZ)),
        "g_roc": spec(grafico_roc(test)), "g_regla": spec(grafico_regla(test, fila_ej)),
        "g_con_sin": spec(grafico_con_y_sin_regla(filas)), "g_test": spec(grafico_test(test, reglas, banda)),
        "g_explorar": spec(grafico_explorar(ligas, reglas)),
        "g_probar": spec(grafico_probar()),
    }

    t_met = tabla_html([{"": r[""], "AUC (más es mejor)": num(r["AUC"]), "log-loss (menos es mejor)": num(r["log-loss"], 4),
                         "Brier (menos es mejor)": num(r["Brier"], 4)} for r in met], destacar=2)
    principales = ["modelo", "sin_regla", "todos", "rango", "once", "nunca"]
    t_estr_test = tabla_html([{"Apostar $100 al underdog cuando…": r["cuando"], "apuestas": entero(r["test_n"]),
                               "retorno por $100": signo(r["test_r"]) if r["test_n"] else "—",
                               "IC 95%": f"[{signo(r['test_bajo'])}; {signo(r['test_alto'])}]" if r["test_n"] else "—"}
                              for r in (f[k] for k in principales)], destacar=0)
    t_estr_todo = tabla_html([{"Estrategia": r["nombre"], "Apostar cuando…": r["cuando"],
                               "validación: apuestas": entero(r["val_n"]), "validación: retorno": signo(r["val_r"]) if r["val_n"] else "—",
                               "test: apuestas": entero(r["test_n"]), "test: retorno": signo(r["test_r"]) if r["test_n"] else "—"}
                              for r in filas] + [{"Estrategia": f"Al azar ({SORTEOS} sorteos)", "Apostar cuando…": "en la misma cantidad de partidos que el modelo, elegidos al azar",
                                                  "validación: apuestas": entero(f["modelo"]["val_n"]),
                                                  "validación: retorno": f"{signo(np.median(fin_val))} (mediana)",
                                                  "test: apuestas": entero(f["modelo"]["test_n"]),
                                                  "test: retorno": f"{signo(np.median(fin_test))} (mediana)"}], destacar=0)
    # la tabla corta de la sección 4: retorno por cada $100 en los dos períodos, y si gana en los dos
    t_resumen = tabla_html([{"Estrategia": r["nombre"], "Validación 2010–2015": signo(r["val_r"]), "Test 2015/16": signo(r["test_r"]),
                             "¿Gana en los dos?": "Sí" if r["val_r"] > 0 and r["test_r"] > 0 else "No"}
                            for r in (f[k] for k in ["modelo", "sin_regla", "todos", "rango", "once"])], destacar=0)
    t_umbrales = tabla_html([{"umbral fijo": f"prob. del modelo > {pct(u, 0)}", "retorno en validación": signo(r)}
                             for u, r in reglas["umbrales"].items()])

    h = silver["html"]
    a, b = reglas["rango"]
    contexto = {
        # portada y números generales
        "TASA": pct(tasa), "N": entero(silver["partidos"]), "GANADOS": entero(silver["ganados"]),
        "DE_CADA_10": str(round(10 * tasa)), "N_EXPLORAR": entero(len(e)), "N_TEST": entero(len(test)), "N_VAL": entero(len(val)),
        "AUC_TEST_2": num(met[2]["AUC"], 2), "AUC_MERCADO_2": num(met[1]["AUC"], 2),
        "AUC_TEST": num(met[2]["AUC"]), "AUC_MERCADO": num(met[1]["AUC"]),
        "ROI_TODOS": signo(f["todos"]["val_r"]), "ROI_MODELO": signo(f["modelo"]["val_r"]),
        "ROI_TODOS_TEST": signo(f["todos"]["test_r"]), "ROI_TEST": signo(f["modelo"]["test_r"]),
        "APUESTAS_TEST": entero(f["modelo"]["test_n"]), "APUESTAS_VAL": entero(f["modelo"]["val_n"]),
        "PCT_APUESTAS_VAL": pct(f["modelo"]["val_n"] / len(val), 0),
        "GANANCIA_TEST": pesos(f["modelo"]["test_r"] * APUESTA * f["modelo"]["test_n"]),
        # 0 · el problema (el partido de ejemplo)
        "EJ_LOCAL": ej["local"], "EJ_VISIT": ej["visitante"], "EJ_FECHA": ej["fecha"], "EJ_RESULTADO": ej["resultado"],
        "EJ_C_LOCAL": num(ej["cuotas"][0], 2), "EJ_C_EMPATE": num(ej["cuotas"][1], 2), "EJ_C_VISIT": num(ej["cuotas"][2], 2),
        "EJ_P_LOCAL": pct(inv[0]), "EJ_P_EMPATE": pct(inv[1]), "EJ_P_VISIT": pct(inv[2]),
        "EJ_SUMA": pct(sum(inv)), "EJ_MARGEN": pct(sum(inv) - 1),
        "EJ_PN_LOCAL": pct(inv[0] / sum(inv)), "EJ_PN_VISIT": pct(inv[2] / sum(inv)),
        "EJ_GAP": num(100 * (inv[0] - inv[2]) / sum(inv), 0),
        "EJ_GANA_100": entero(100 * ej["cuotas"][2]), "EJ_LOCAL_100": entero(100 * ej["cuotas"][0]), "EJ_PAGA": pct(1 / fila_ej["cuota"]),
        "EJ_MODELO": pct(fila_ej["prob_modelo"]), "EJ_MERCADO": pct(fila_ej["prob_mercado"]),
        "EJ_P_MODELO_NUM": num(fila_ej["prob_modelo"], 3), "EJ_VALOR": signo(valor_ej), "EJ_VALOR_PESOS": entero(100 * valor_ej),
        # 1 · los datos y 2 · lo que encontramos
        "DATOS_ALTO": pct(t_datos["tasa"].iloc[0], 0), "DATOS_BAJO": pct(t_datos["tasa"].iloc[-1], 0),
        "XI_BAJO": pct(t_hallazgo["real"].iloc[0], 0), "XI_ALTO": pct(t_hallazgo["real"].iloc[-1], 0),
        "FECHA_DATOS": silver["fecha_datos"],
        # 3 · ¿el modelo predice?
        "T_METRICAS": t_met, "TASA_BASE": pct(tasa_base, 0), "TASA_TEST": pct(test["gano"].mean()),
        "LL_MODELO": num(met[2]["log-loss"], 4), "LL_CUOTA": num(met[1]["log-loss"], 4), "LL_PISO": num(met[0]["log-loss"], 4),
        "M_ACIERTOS": entero(m_mod["aciertos"]), "M_FALSAS": entero(m_mod["falsas_alarmas"]), "M_ESCAPAN": entero(m_mod["se_le_escapan"]),
        "M_DESCARTA": entero(m_mod["bien_descartados"]), "M_GANADOS": entero(m_mod["aciertos"] + m_mod["se_le_escapan"]),
        "MC_ACIERTOS": entero(m_cuota["aciertos"]), "MC_FALSAS": entero(m_cuota["falsas_alarmas"]),
        "M50_DICE": entero(m50["aciertos"] + m50["falsas_alarmas"]), "M50_ACC": pct((m50["bien_descartados"] + m50["aciertos"]) / len(test), 0),
        "UMBRAL_MATRIZ": pct(UMBRAL_MATRIZ, 0),
        # 4 · ¿sirve para apostar?
        "T_ESTRATEGIAS_TEST": t_estr_test, "T_RESUMEN_ESTRATEGIAS": t_resumen, "T_ESTRATEGIAS_TODO": t_estr_todo, "T_UMBRALES": t_umbrales,
        "UMBRAL_SIN": pct(reglas["umbral"], 0), "RANGO": nombre_rango(a, b),
        "SIN_VAL": signo(f["sin_regla"]["val_r"]), "SIN_TEST": signo(f["sin_regla"]["test_r"]),
        "SIN_N_VAL": entero(f["sin_regla"]["val_n"]), "SIN_N_TEST": entero(f["sin_regla"]["test_n"]),
        "RANGO_VAL": signo(f["rango"]["val_r"]), "RANGO_TEST": signo(f["rango"]["test_r"]),
        "ONCE_VAL": signo(f["once"]["val_r"]), "ONCE_TEST": signo(f["once"]["test_r"]),
        "IC_TEST": f"[{signo(f['modelo']['test_bajo'])}; {signo(f['modelo']['test_alto'])}]",
        "AZAR_P95_TEST": signo(np.percentile(fin_test, 95)), "AZAR_P95_VAL": signo(np.percentile(fin_val, 95)),
        "AZAR_SUPERA_VAL": pct(float(np.mean(fin_val < f["modelo"]["val_r"])), 0),
        "AZAR_SUPERA_TEST": pct(float(np.mean(fin_test < f["modelo"]["test_r"])), 0),
        "SORTEOS": str(SORTEOS),
        # «Cómo lo sabemos» que vienen del notebook
        **h, "C": silver["C"], "INTERCEPTO": num(silver["intercepto"]), "COEF_PROB": num(silver["coef_prob"]),
        "GRAFICOS": json.dumps(graficos, ensure_ascii=False),
        "MODELO": json.dumps(silver["modelo"], ensure_ascii=False),
        "CASOS": json.dumps(silver["casos"], ensure_ascii=False),
        "VEGA": VEGA, "VEGA_LITE": VEGA_LITE, "VEGA_EMBED": VEGA_EMBED,
    }
    pagina = (Path(__file__).parent / "plantilla.html").read_text(encoding="utf-8")
    for clave, valor in contexto.items():
        pagina = pagina.replace("{{" + clave + "}}", valor)
    assert "{{" not in pagina, "quedó un marcador sin reemplazar en la plantilla: " + pagina[pagina.index("{{"):pagina.index("{{") + 40]
    return pagina


def construir(desde_publicado: bool = False) -> None:
    DATOS.mkdir(parents=True, exist_ok=True)
    if desde_publicado:
        silver = json.loads((DATOS / "silver_resumen.json").read_text(encoding="utf-8"))
        explorar = pd.read_json(DATOS / "partidos.json", dtype={"fecha": str, "temporada": str})
        test_casos = json.loads((DATOS / "partidos_test.json").read_text(encoding="utf-8"))
        origen = "lo publicado en docs/data/"
    else:
        df, ficha, resultados = cargar()
        dif = verificar_contra_el_pipeline(df)
        assert dif < 1e-9, f"La fórmula de la app no coincide con el modelo del notebook ({dif})"
        explorar = partidos_para_explorar(fuera_de_muestra(df, resultados))
        test_casos = partidos_del_test(df)
        silver = resumen_del_silver(df, ficha, resultados)
        explorar.to_json(DATOS / "partidos.json", orient="records", force_ascii=False)
        (DATOS / "partidos_test.json").write_text(json.dumps(test_casos, ensure_ascii=False), encoding="utf-8")
        (DATOS / "silver_resumen.json").write_text(json.dumps(silver, ensure_ascii=False, indent=1), encoding="utf-8")
        origen = f"el Silver y el notebook (fórmula = notebook, dif. {dif:.1e})"
    (SALIDA / ".nojekyll").write_text("", encoding="utf-8")
    pagina = armar_pagina(silver, explorar, test_casos)
    (SALIDA / "index.html").write_text(pagina, encoding="utf-8")
    print(f"App generada en {SALIDA / 'index.html'} ({len(pagina) / 1024:.0f} KB) desde {origen}: {len(explorar):,} partidos fuera de muestra.")


if __name__ == "__main__":
    construir(desde_publicado="--desde-publicado" in sys.argv[1:])
