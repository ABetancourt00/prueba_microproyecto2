"""
Modulo de preprocesamiento de texto para el proyecto de clasificacion de
textos por Objetivo de Desarrollo Sostenible (ODS).

Se aisla en un modulo propio -en vez de definirse dentro del notebook- para
que la misma funcion pueda:
  - usarse como `preprocessor` del `TfidfVectorizer` durante el entrenamiento
    (en el notebook), y
  - importarse en la aplicacion de despliegue (`src/ModelController.py`) para
    aplicar exactamente el mismo procesamiento al texto que ingresa el
    usuario.

Esto evita inconsistencias entre entrenamiento y produccion, y es necesario
para que `joblib` pueda serializar y deserializar correctamente el
`TfidfVectorizer` (que guarda una referencia a esta funcion por nombre de
modulo: si el nombre del modulo cambia entre entrenamiento y despliegue,
la carga del artefacto falla).
"""
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

_spanish_stopwords = stopwords.words("spanish")
_tokenizer = RegexpTokenizer(r"[^\W\d_]+")
_stemmer = SnowballStemmer("spanish")


def text_preprocess(text):
    """Normaliza, tokeniza, elimina palabras vacias y aplica stemming
    en espanol a un texto de entrada."""
    text = text.lower()
    tokens = _tokenizer.tokenize(text)
    tokens = [t for t in tokens if t not in _spanish_stopwords and len(t) > 2]
    tokens = [_stemmer.stem(t) for t in tokens]
    return " ".join(tokens)
