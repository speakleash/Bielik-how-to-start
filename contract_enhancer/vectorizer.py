from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

query_prefix = "zapytanie: "
answer_prefix = ""

model = SentenceTransformer("sdadas/mmlw-retrieval-roberta-large")

def vectorize(text):
    if isinstance(text, str):
        text = [text]
    return model.encode(text, normalize_embeddings=True)