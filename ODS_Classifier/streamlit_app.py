import os.path as osp

import pandas as pd
import streamlit as st

import Definitions
from src.ModelController import ModelController, ODS_NOMBRES

st.set_page_config(page_title="Clasificador de ODS", page_icon="🌍", layout="centered")

RUTA_TEST = osp.join(Definitions.ROOT_DIR, "resources", "data", "test_data.xlsx")


@st.cache_resource(show_spinner="Cargando el modelo…")
def cargar_controlador():
    return ModelController()


@st.cache_data
def cargar_test():
    return pd.read_excel(RUTA_TEST) if osp.exists(RUTA_TEST) else pd.DataFrame(columns=["textos", "ODS"])


ctrl = cargar_controlador()
test_data = cargar_test()


def mostrar_resultado(res):
    """Muestra el ODS predicho y el top-3 de una predicción individual."""
    if res["sin_vocabulario"]:
        st.warning("El texto no comparte ningún término con el vocabulario del modelo (¿está en otro idioma?). "
                   "No se puede clasificar de forma confiable.")
        return
    st.success(f"**ODS {res['ods']} · {res['nombre']}**")
    c1, c2 = st.columns(2)
    c1.metric("ODS predicho", res["ods"])
    c2.metric("Margen sobre el 2.º ODS", f"{res['margen']:.2f}")
    if res["n_terminos"] < 3 or res["margen"] < 0.1:
        st.info("El modelo tiene poca certeza: el texto tiene pocos términos conocidos o puede tratar varios ODS a la vez. "
                "Revisa también las otras opciones.")
    st.subheader("Los 3 ODS con mayor puntaje")
    top = res["ranking"].copy()
    st.bar_chart(top.assign(ODS=top.apply(lambda r: f"ODS {r['ODS']} · {r['Nombre']}", axis=1)).set_index("ODS")["Puntaje"])
    st.dataframe(top.assign(Puntaje=top["Puntaje"].round(3)), hide_index=True)
    st.caption("El puntaje es la distancia del texto a la frontera de decisión del LinearSVC (más alto = más cerca de ese ODS). "
               "**No es una probabilidad.**")


# ───────────────────────────── Barra lateral ─────────────────────────────
with st.sidebar:
    st.header("Acerca del modelo")
    st.write("Clasifica un texto según el **Objetivo de Desarrollo Sostenible (ODS)** de la Agenda 2030 "
             "con el que está más relacionado.")
    st.markdown("**Pipeline:** preprocesamiento (tokenización, palabras vacías y *stemming* en español) → "
                "TF-IDF (1-2 gramas) → SVD truncada (300 componentes) → `LinearSVC`.")
    st.caption("El modelo reconoce los ODS 1 al 16; el ODS 17 no está representado en los datos de entrenamiento.")

st.title("🌍 Clasificador de textos por ODS")
tab_libre, tab_lote = st.tabs(["✍️ Texto libre", "📄 Archivo por lotes"])

# ───────────────────────────── Pestaña 1: texto libre ─────────────────────────────
with tab_libre:
    st.write("Escribe o pega un texto en español y el modelo indicará con qué ODS se relaciona.")

    if "texto" not in st.session_state:
        st.session_state["texto"] = ""

    def _cargar_ejemplo():
        i = st.session_state["ejemplo"]
        if i != "— escribir mi propio texto —":
            st.session_state["texto"] = test_data.loc[int(i.split(".")[0]) - 1, "textos"]

    if len(test_data):
        opciones = ["— escribir mi propio texto —"] + [
            f"{i + 1}. (ODS real {r.ODS}) {r.textos[:70]}…" for i, r in test_data.head(10).iterrows()]
        st.selectbox("¿Quieres probar con un texto de ejemplo del conjunto de prueba?", opciones,
                     key="ejemplo", on_change=_cargar_ejemplo)

    st.text_area("Texto a clasificar", key="texto", height=180,
                 placeholder="Pega aquí un párrafo, un aporte ciudadano, un fragmento de un informe…")

    if st.button("Clasificar texto", type="primary"):
        valido, mensaje = ctrl.validate_text(st.session_state["texto"])
        if not valido:
            st.warning(mensaje)
        else:
            mostrar_resultado(ctrl.predict(st.session_state["texto"]))

# ───────────────────────────── Pestaña 2: archivo por lotes ─────────────────────────────
with tab_lote:
    st.write("Sube un archivo **Excel (.xlsx) o CSV** con varios textos para clasificarlos de una vez.")
    archivo = st.file_uploader("Archivo con los textos", type=["xlsx", "csv"])
    usar_test = st.checkbox("O usar los textos de prueba incluidos con la aplicación (200 textos con su ODS real)",
                            disabled=archivo is not None or not len(test_data))

    df_in = None
    if archivo is not None:
        df_in = pd.read_csv(archivo) if archivo.name.lower().endswith(".csv") else pd.read_excel(archivo)
    elif usar_test:
        df_in = test_data.copy()

    if df_in is not None and len(df_in):
        columnas = list(df_in.columns)
        por_defecto = columnas.index("textos") if "textos" in columnas else 0
        col_texto = st.selectbox("Columna que contiene el texto", columnas, index=por_defecto)
        st.write(f"{len(df_in):,} filas cargadas.")

        if st.button("Clasificar archivo", type="primary"):
            textos = df_in[col_texto].fillna("").astype(str)
            res = ctrl.predict_batch(textos)
            res.insert(0, "Texto", textos.str.slice(0, 100).values)
            if "ODS" in df_in.columns:
                real = pd.to_numeric(df_in["ODS"], errors="coerce").values
                res["ODS real"] = real
                res["¿Acierto?"] = (res["ODS predicho"].values == real)
            st.session_state["lote"] = {"resultados": res, "textos": textos.tolist()}

    lote = st.session_state.get("lote")
    if lote is not None:
        res, textos = lote["resultados"], lote["textos"]
        st.subheader("Resultados")
        if "¿Acierto?" in res.columns:
            st.metric("Exactitud sobre el archivo (usando la columna ODS como ODS real)", f"{res['¿Acierto?'].mean():.1%}")
        st.dataframe(res.assign(Puntaje=res["Puntaje"].round(3)), hide_index=True)
        st.download_button("Descargar resultados (CSV)", res.to_csv(index=False).encode("utf-8-sig"),
                           file_name="clasificacion_ods.csv", mime="text/csv")

        st.subheader("Detalle de un texto")
        i = st.selectbox("Elige un texto", range(len(textos)), format_func=lambda k: f"{k + 1}. {textos[k][:70]}…")
        st.write(textos[i])
        mostrar_resultado(ctrl.predict(textos[i]))
