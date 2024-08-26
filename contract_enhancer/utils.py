import re
import logging

import fitz
import numpy as np
from vectorizer import vectorize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_pdf(fname, callback=None):
    """Parses a PDF file and returns its text content."""
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
    """Determines the file type and parses it using the appropriate function."""
    extension = fname.split(".")[-1].lower()
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
    """Prepares the data for processing by vectorizing the text."""
    text = parse_file(fname)
    vec, split_text = vectorize_text(text)
    return vec, split_text


def find_in_db(query, library, texts):
    """Finds the best matching paragraphs in the library for the given query."""
    query_vec = vectorize([query])  # Dodano [] aby query było listą
    scores = np.dot(library, query_vec.T)
    best_match = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    best_matches = []
    for i in best_match[:10]:
        best_matches.append(texts[i])

    return best_matches


def prepare_prompt(samples, text, tokenizer):
    """Prepares the prompt for the chatbot model."""
    samples = "\n - ".join([i for i in samples])
    msg = [
            {
                    "role": "user",
                    "content": (
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
    logger.info(chat_applied)
    return chat_applied


if __name__ == "__main__":
    vec, texts = prepare_data("data/Umowa IT Wykonawca Markup.pdf")
    print(find_in_db("Umowa podlega prawu polskiemu", vec, texts))
