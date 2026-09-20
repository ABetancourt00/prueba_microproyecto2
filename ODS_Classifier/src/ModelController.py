import os
import sys

import joblib
import nltk

import Definitions

# Descargar las stopwords (en Streamlit Cloud el entorno empieza vacío)
nltk.download("stopwords", quiet=True)

# Agregar la carpeta src al path para que joblib encuentre DataPrePreprocessing
sys.path.append(os.path.join(Definitions.ROOT_DIR, "src"))

nombres_ods = {
    1: "Fin de la pobreza", 2: "Hambre cero", 3: "Salud y bienestar", 4: "Educación de calidad",
    5: "Igualdad de género", 6: "Agua limpia y saneamiento", 7: "Energía asequible y no contaminante",
    8: "Trabajo decente y crecimiento económico", 9: "Industria, innovación e infraestructura",
    10: "Reducción de las desigualdades", 11: "Ciudades y comunidades sostenibles",
    12: "Producción y consumo responsables", 13: "Acción por el clima", 14: "Vida submarina",
    15: "Vida de ecosistemas terrestres", 16: "Paz, justicia e instituciones sólidas",
}


class ModelController:
    def __init__(self):
        ruta = os.path.join(Definitions.ROOT_DIR, "resources", "models")
        self.vectorizer = joblib.load(os.path.join(ruta, "vectorizer.joblib"))
        self.svd = joblib.load(os.path.join(ruta, "svd.joblib"))
        self.model = joblib.load(os.path.join(ruta, "model.joblib"))

    def predict(self, texto):
        # Mismos pasos del pipeline: TF-IDF -> SVD -> LinearSVC
        X = self.vectorizer.transform([texto])

        # Si ninguna palabra del texto está en el vocabulario no se puede clasificar
        if X.nnz == 0:
            return {"sin_vocabulario": True}

        puntajes = self.model.decision_function(self.svd.transform(X))[0]

        # Los 3 ODS con mayor puntaje
        orden = puntajes.argsort()[::-1][:3]
        top3 = []
        for i in orden:
            ods = int(self.model.classes_[i])
            top3.append((ods, nombres_ods[ods], puntajes[i]))

        return {"sin_vocabulario": False, "ods": top3[0][0], "nombre": top3[0][1], "top3": top3}
