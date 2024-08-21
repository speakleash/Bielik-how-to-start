import fitz
from vectorizer import vectorize
import re
import numpy as np
from transformers import AutoTokenizer

from logging import getLogger
logger = getLogger(__name__)


def parse_pdf(fname, callback=None):
    with open(fname, "rb") as f:
        all_text = ""
        reader = fitz.open(stream=f.read(), filetype="pdf")

        for page in reader:
            # callback(page.number/reader.page_count)
            text = page.get_text()
            if text is not None:
                all_text += text

        return all_text


EXTENSIONS = {
    "pdf": parse_pdf,
}


def parse_file(fname):
    extension = fname.split(".")[-1]
    if extension not in EXTENSIONS:
        raise NotImplementedError(f"Unsupported file extension {extension}")

    text = EXTENSIONS[extension](fname)
    return text


def vectorize_text(text):
    """ Vectorize the text and append it to the library """
    # We will split the text into paragraphs, with a naive assumption
    # that each paragraph is divided by a number followed by a dot.
    text = text.replace("\n", " ")
    stacked = re.split(r" \d+\.", text)
    stacked = [i.strip() for i in stacked if i.strip() and len(i.strip()) > 10]
    vec = vectorize(stacked)
    
    return vec, stacked
    
    
def prepare_data(fname):
    text = parse_file(fname)
    vec, split_text = vectorize_text(text)
    return vec, split_text

def find_in_db(query, library, texts):
    query_vec = vectorize(query)
    scores = np.dot(library, query_vec.T)
    best_match = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    best_matches = []
    for i in best_match[:10]:
        best_matches.append(texts[i])
        
    return best_matches

def prepare_prompt(samples, text, tokenizer):
    samples = "\n - ".join([i[0] for i in samples])
    msg = [
        {
            "role": "user", "content": (
                "Poprawmy fragment umowy tak, żeby był zgodny z naszą biblioteką umów. Oto paragrafy podobne do tego, co piszemy. Wybierz jeden z nich."
                "\n<relevant_info>\n"
                f"{samples}"
                "\n</relevant_info>\n"
                "To jest paragraf, który właśnie napisaliśmy:\n"
                f"{text}\n"
                "Jak przepisałbyś go, żeby był zgodny z naszą biblioteką umów? "
                "Nie przepisuj go słowo w słowo z biblioteki, ale dostosuj go do naszych potrzeb. "
                "Napisz tylko i wyłącznie treść nowego paragrafu. Bądź tak zwięzły, jak to możliwe. "
                "Pamiętaj, żeby nie zmieniać znaczenia paragrafu, a jedynie dostosować go do naszych potrzeb. "
            )
        }
    ]
    
    chat_applied = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    print(chat_applied)
    return chat_applied

if __name__ == "__main__":
    vec, texts = prepare_data("data/Umowa IT Wykonawca Markup.pdf")
    print(find_in_db("Umowa podlega prawu polskiemu", vec, texts))