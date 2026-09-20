"""Controlador del modelo: carga los 3 artefactos entrenados en el notebook
(vectorizer, svd y model) y los encadena para clasificar textos."""
import os.path as osp
import sys

import joblib
import nltk
import numpy as np
import pandas as pd

import Definitions

# 1) DataPrePreprocessing usa las stopwords de NLTK al importarse. En Streamlit Cloud el
#    entorno inicia vacío, así que se descargan si no existen (antes de cargar los artefactos).
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# 2) joblib guardó el vectorizador con una referencia al módulo "DataPrePreprocessing":
#    src/ debe estar en el path para poder deserializarlo.
SRC_DIR = osp.join(Definitions.ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

ODS_NOMBRES = {
    1: "Fin de la pobreza", 2: "Hambre cero", 3: "Salud y bienestar", 4: "Educación de calidad",
    5: "Igualdad de género", 6: "Agua limpia y saneamiento", 7: "Energía asequible y no contaminante",
    8: "Trabajo decente y crecimiento económico", 9: "Industria, innovación e infraestructura",
    10: "Reducción de las desigualdades", 11: "Ciudades y comunidades sostenibles",
    12: "Producción y consumo responsables", 13: "Acción por el clima", 14: "Vida submarina",
    15: "Vida de ecosistemas terrestres", 16: "Paz, justicia e instituciones sólidas",
    17: "Alianzas para lograr los objetivos",
}


class ModelController:
    """Encadena TF-IDF -> SVD -> LinearSVC, igual que el Pipeline del notebook."""

    MIN_PALABRAS = 5

    def __init__(self):
        modelos = osp.join(Definitions.ROOT_DIR, "resources", "models")
        self.vectorizer = joblib.load(osp.join(modelos, "vectorizer.joblib"))
        self.svd = joblib.load(osp.join(modelos, "svd.joblib"))
        self.model = joblib.load(osp.join(modelos, "model.joblib"))
        self.clases = [int(c) for c in self.model.classes_]

    def validate_text(self, texto):
        """Devuelve (es_valido, mensaje)."""
        if texto is None or not str(texto).strip():
            return False, "Escribe un texto para clasificar."
        if len(str(texto).split()) < self.MIN_PALABRAS:
            return False, f"El texto es muy corto: escribe al menos {self.MIN_PALABRAS} palabras."
        return True, ""

    def _puntajes(self, textos):
        """Puntajes del LinearSVC (distancia a cada frontera) y nº de términos reconocidos por texto."""
        X = self.vectorizer.transform(textos)
        n_terminos = X.getnnz(axis=1)
        puntajes = self.model.decision_function(self.svd.transform(X))
        return puntajes, n_terminos

    def predict(self, texto, top_n=3):
        """Clasifica UN texto. Devuelve el ODS predicho y los top_n ODS con mayor puntaje."""
        puntajes, n_terminos = self._puntajes([texto])
        puntajes, n_terminos = puntajes[0], int(n_terminos[0])
        orden = np.argsort(puntajes)[::-1][:top_n]
        ranking = pd.DataFrame({
            "ODS": [self.clases[i] for i in orden],
            "Nombre": [ODS_NOMBRES.get(self.clases[i], "") for i in orden],
            "Puntaje": [float(puntajes[i]) for i in orden],
        })
        return {
            "ods": int(ranking.loc[0, "ODS"]),
            "nombre": ranking.loc[0, "Nombre"],
            "ranking": ranking,
            "margen": float(ranking.loc[0, "Puntaje"] - ranking.loc[1, "Puntaje"]),
            "n_terminos": n_terminos,
            "sin_vocabulario": n_terminos == 0,   # ningún término del texto está en el vocabulario
        }

    def predict_batch(self, textos):
        """Clasifica varios textos a la vez. Devuelve un DataFrame con una fila por texto."""
        textos = pd.Series(textos).fillna("").astype(str).tolist()
        puntajes, n_terminos = self._puntajes(textos)
        idx = puntajes.argmax(axis=1)
        ods = [self.clases[i] for i in idx]
        return pd.DataFrame({
            "ODS predicho": ods,
            "Nombre ODS": [ODS_NOMBRES.get(o, "") for o in ods],
            "Puntaje": puntajes[np.arange(len(idx)), idx],
            "Términos reconocidos": n_terminos,
        })
