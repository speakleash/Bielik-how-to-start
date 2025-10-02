# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "draive[ollama]~=0.87.5",
# ]
# ///
#


from asyncio import run

from draive import TextGeneration, ctx, setup_logging
from draive.ollama import Ollama, OllamaChatConfig

setup_logging("text_completion")


async def main() -> None:
    async with ctx.scope(  # przygotowanie nowego kontekstu
        "text_completion",
        OllamaChatConfig(
            model="SpeakLeash/bielik-4.5b-v3.0-instruct:FP16",
            temperature=0.7,
        ),
        disposables=(Ollama(),),  # użycie ollama jako llm w kontekście
    ):
        # TextGeneration to prosty interfejs przeznaczony do generowania tekstu
        result: str = await TextGeneration.generate(
            # Należy podać instrukcje / systemowy prompt, aby poinstruować model
            instruction="Prepare the simplest completion of a given text",
            # wejście jest podawane oddzielnie od instrukcji
            input="Z jakich składników robi się sękacza?.",
        )

        print("RESULT:\n", result)


run(main=main())
