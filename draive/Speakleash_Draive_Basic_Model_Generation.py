# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "draive[ollama]~=0.37",
# ]
# ///
#

from asyncio import run
from typing import Sequence

from draive import DataModel, ctx, generate_model, setup_logging
from draive.ollama import OllamaChatConfig, ollama_lmm

setup_logging("model_generation")


# przygotowujemy modele danych do wygenerowania opis pól będzie użyty
# jako schemat danych przedstawiony dla modelu do wygenerowania
# natomiast wynik zostnie zweryfikowany tak aby odpowiadał opisowi
class Ingredient(DataModel):
    name: str
    amount: str | float


class Formula(DataModel):
    name: str
    description: str | None
    preparation: str
    ingredients: Sequence[Ingredient]


async def main(provided_model, provided_temp) -> None:
    async with ctx.scope(  # przygotowanie nowego kontekstu
        "model_generation",
        ollama_lmm(),  # użycie ollama jako llm w kontekście
        OllamaChatConfig(
            model=provided_model,
            temperature=provided_temp,
        ),
    ):
        result: Formula = await generate_model(
            # Podajemy jaki model danych chcemy wyprodukować
            Formula,
            # Należy podać instrukcje / systemowy prompt, aby poinstruować model
            # jak wygenerować dane, schemat modelu danych zostanie dodany automatycznie
            # na podstawie zadeklarowanego modelu danych
            instruction="Prepare the formula according to the request",
            # wejście jest podawane oddzielnie od instrukcji
            input="Podaj przepis na sękacza.",
        )

        print(f"RESULT {provided_model} | temperature {provided_temp}:\n{result}")


# Wybież jeden z poniższych modeli do generowania odpowiedzi
MODEL_V1 = "SpeakLeash/bielik-7b-instruct-v0.1-gguf:Q4_K_S"
MODEL_V2 = "SpeakLeash/bielik-11b-v2.2-instruct:Q4_K_M"
TEMPERATURE = 0.2
run(main=main(MODEL_V2, TEMPERATURE))
