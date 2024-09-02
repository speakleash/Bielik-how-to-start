"""
Poniższy kod jest adaptacją przykładu z biblioteki Draive znajdującego się pod linkiem:
https://github.com/miquido/draive/blob/main/guides/BasicUsage.ipynb
"""
from asyncio import run
from draive import LMM, ctx, generate_text
from draive import setup_logging
from draive.ollama import OllamaClient, ollama_lmm_invocation, OllamaChatConfig

setup_logging("text_completion")


async def main(provided_model, provided_temp) -> None:
    async def text_completion(text: str) -> str:
        # generate_text to prosty interfejs przeznaczony do generowania tekstu
        return await generate_text(
                # Należy podać instrukcje / systemowy prompt, aby poinstruować model
                instruction="Prepare the simplest completion of a given text",
                # wejście jest podawane oddzielnie od instrukcji
                input=text,
        )

    async with ctx.new(  # przygotowanie nowego kontekstu
            "text_completion",
            state=[
                    LMM(invocation=ollama_lmm_invocation),
                    OllamaChatConfig(
                            model=provided_model,
                            temperature=provided_temp,
                    )
            ],
            dependencies=[OllamaClient],
    ):
        result: str = await text_completion(
                text="Z jakich składników robi się sękacza?.",
        )

        print(f'RESULT {provided_model} | temperature {provided_temp}:\n{result}')

# Wybież jeden z poniższych modeli do generowania odpowiedzi
MODEL_V1 = 'SpeakLeash/bielik-7b-instruct-v0.1-gguf:Q4_K_S'
MODEL_V2 = 'SpeakLeash/bielik-11b-v2.2-instruct:Q4_K_M'
TEMPERATURE = 0.2
run(main=main(MODEL_V2, TEMPERATURE))
