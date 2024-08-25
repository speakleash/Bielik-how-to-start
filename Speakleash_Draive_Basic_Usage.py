from asyncio import run
from draive import LMM, ctx, generate_text
from draive import setup_logging
from draive.ollama import OllamaClient, ollama_lmm_invocation, OllamaChatConfig

setup_logging("text_completion")

async def main(provided_model, provided_temp) -> None:

    async def text_completion(text: str) -> str:
        # generate_text is a simple interface for generating text
        return await generate_text(
            # We have to provide instructions / system prompt to instruct the model
            instruction="Prepare the simplest completion of a given text",
            # input is provided separately
            input=text,
        )

    async with ctx.new(  # prepare new context
        "text_completion",
        state=[
                LMM(invocation=ollama_lmm_invocation),
                OllamaChatConfig(
                        model=provided_model,
                        temperature=provided_temp,
                )
               ],  # set currently used LMM to OpenAI
        dependencies=[OllamaClient],  # use OpenAIClient dependency for accessing OpenAI services
    ):
        result: str = await text_completion(
            text="Roses are red...",
        )

        print(f'RESULT {provided_model} | temperature {provided_temp}:\n{result}')

model = 'SpeakLeash/bielik-7b-instruct-v0.1-gguf'
temperature = 0.2
run(main=main(model, temperature))