# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "draive[ollama]~=0.87.5",
# ]
# ///
#

from asyncio import run
from typing import Sequence

from draive import DataModel, ModelGeneration, ctx, setup_logging
from draive.ollama import Ollama, OllamaChatConfig

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


async def main() -> None:
    async with ctx.scope(  # przygotowanie nowego kontekstu
        "model_generation",
        OllamaChatConfig(
            model="SpeakLeash/bielik-4.5b-v3.0-instruct:FP16",
            temperature=0.7,
        ),
        disposables=(Ollama(),),  # użycie ollama jako llm w kontekście
    ):
        result: Formula = await ModelGeneration.generate(
            # Podajemy jaki model danych chcemy wyprodukować
            Formula,
            # Należy podać instrukcje / systemowy prompt, aby poinstruować model
            # jak wygenerować dane, schemat modelu danych zostanie dodany automatycznie
            # na podstawie zadeklarowanego modelu danych
            instruction="Prepare the formula according to the request",
            # wejście jest podawane oddzielnie od instrukcji
            input="Podaj przepis na sękacza.",
        )

        print("RESULT:\n", result)


run(main=main())
