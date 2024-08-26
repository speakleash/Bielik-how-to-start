import re

import numpy as np
import gradio as gr
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import utils

sampling_params = SamplingParams(temperature=1.0, max_tokens=512)
tokenizer = AutoTokenizer.from_pretrained("speakleash/Bielik-7B-Instruct-v0.1")
llm = LLM(model="speakleash/Bielik-7B-Instruct-v0.1")

with gr.Blocks(fill_width=True) as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("# Poprawianie treści umowy")
            gr.Markdown(
                    "W tym narzędziu Bielik pokaże swoją moc w poprawianiu treści umów. Dodaj do biblioteki umowy"
                    " na których model powinien się wzorować, a następnie wprowadź paragraf który chcesz poprawić.")
            gr.Markdown("### Instrukcja:")
            gr.Markdown("1. Dodaj umowy do biblioteki.")
            gr.Markdown("2. Wprowadź treść umowy, którą chcesz poprawić.")
            gr.Markdown("3. Kliknij przycisk 'Znajdź paragrafy'.")
            vec_library = gr.State(None)
            texts = gr.State([])


            def add_to_library(files, vec_library, texts):
                for f in files:
                    vec, split_text = utils.prepare_data(f)

                    if vec_library is None:
                        vec_library = vec
                    else:
                        vec_library = np.vstack([vec_library, vec])
                    texts = texts + split_text

                return vec_library, texts, [i.name for i in files]


            file_output = gr.Files(interactive=False, label="Biblioteka")
            upload_button = gr.UploadButton(label="Dodaj umowę do biblioteki", file_count="multiple",
                                            file_types=["pdf"])
            upload_button.upload(add_to_library, [upload_button, vec_library, texts], [vec_library, texts, file_output])

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Treść paragrafu do poprawki", lines=10)
            search_button = gr.Button(value="Znajdź paragrafy")

        with gr.Column():
            suggestion = gr.Textbox(label="Sugestia", interactive=False)


        def search_in_library(text_input, vec_library, texts):
            text = text_input.replace("\n", " ")
            stacked = re.split(r" \d+\.", text)

            latest_paragraph = stacked[-1]

            print(f"Searching for: {latest_paragraph}")

            matching_paragraphs = utils.find_in_db(latest_paragraph, vec_library, texts)
            print(f"Found: {matching_paragraphs}")

            matching_paragraphs = [[i] for i in matching_paragraphs]

            yield "Trwa generowanie..."

            prompts = utils.prepare_prompt(matching_paragraphs, text_input, tokenizer)
            outputs = llm.generate(prompts, sampling_params)

            for output in outputs:
                generated_text = output.outputs[0].text
                yield generated_text


        search_button.click(search_in_library, [text_input, vec_library, texts], [suggestion])

if __name__ == "__main__":
    demo.launch()
