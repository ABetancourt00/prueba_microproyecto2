import streamlit as st

from src.ModelController import ModelController

st.title("Clasificador de textos por ODS")
st.write("Escribe un texto en español y el modelo indica con qué Objetivo de Desarrollo Sostenible se relaciona.")


@st.cache_resource
def cargar_modelo():
    return ModelController()


modelo = cargar_modelo()

texto = st.text_area("Texto a clasificar", height=200)

if st.button("Clasificar"):
    if len(texto.split()) < 5:
        st.warning("Escribe un texto de al menos 5 palabras.")
    else:
        resultado = modelo.predict(texto)

        if resultado["sin_vocabulario"]:
            st.warning("El modelo no reconoce ninguna palabra del texto (¿está en otro idioma?).")
        else:
            st.success(f"ODS predicho: {resultado['ods']} - {resultado['nombre']}")

            st.write("Los 3 ODS con mayor puntaje:")
            for ods, nombre, puntaje in resultado["top3"]:
                st.write(f"ODS {ods} - {nombre}: {puntaje:.2f}")

            st.caption("El puntaje es la distancia a la frontera de decisión del LinearSVC, no es una probabilidad.")
