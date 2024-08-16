from asyncio import run
from draive import LMM, ctx, generate_text
from draive.ollama import OllamaClient, ollama_lmm_invocation, OllamaChatConfig

async def main() -> None:

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
                OllamaChatConfig(model='SpeakLeash/bielik-7b-instruct-v0.1-gguf')
               ],  # set currently used LMM to OpenAI
        dependencies=[OllamaClient],  # use OpenAIClient dependency for accessing OpenAI services
    ):
        result: str = await text_completion(
            text="Roses are red...",
        )

        print(result)
run(main=main())