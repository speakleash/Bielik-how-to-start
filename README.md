<h1 align="center">
<img src="https://huggingface.co/speakleash/Bielik-7B-Instruct-v0.1/raw/main/speakleash_cyfronet.png">
</h1><br>

# Jak zacząć pracę z Bielikiem

Repozytorium zawiera skrypty oraz notatniki, które przedstawiają różne przykłady użycia LLM Bielika.

## Wymagania:

1. Python<br>
   Aby móc pracować z kodem, zalecana jest instalacja Pythona w wersji `>=3.9`.
   Instrukcje instalacji można znaleźć na oficjalnej stronie Pythona:<br> https://www.python.org/downloads/

2. Jupyter Notebook<br>
   Zalecana jest praca w środowisku Jupyter Notebook.
   Instrukcje instalacji i uruchomienia Jupyter Notebook: <br>
   https://jupyter.org/install
   <br>Do każdego notebooka załączony jest link prowadzący do przestrzeni Google Colab zawierającej kod z danym przykładem użycia.

## Rozpoczęcie pracy

1. Sklonuj to repozytorium na swój lokalny komputer:<br>
   `git clone https://github.com/speakleash/Bielik-how-to-start.git`
2. Uruchom Jupyter Notebook i otwórz wybrany notatnik z przykładami.
3. W przypadku przykładów znajdujących się osobno w folderach (draive, contract_enhancer) należy uprzednio zainstalować wymagane zależności:<br>
   `pip install -r requirements.txt`

W przypadku problemów lub pytań, sprawdź sekcję "Issues" w repozytorium lub skontaktuj się z autorami projektu.

## Examples

| Notebook                                                   | Collab                                                                                                                                                                                                                                                                                                                                                                                                                                      | Description                                                   |
|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| Bielik_(4_bit)_RAG.ipynb                                   | V1: <a target="_blank" href="https://colab.research.google.com/drive/13XCBuJQsaeGi6HvfMc1MDZn0RNsrP8yp?authuser=1"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="V1 Open In Colab"/></a> <br>V2: <a target="_blank" href="https://colab.research.google.com/drive/1ZdYsJxLVo9fW75uonXE5PCt8MBgvyktA?authuser=1"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="V2 Open In Colab"/></a> | RAG with HuggingFace transformers                             |
| Bielik_(4_bit)_simple_examples.ipynb                       | V1: <a target="_blank" href="https://colab.research.google.com/drive/1eBVXla_41L7koAufmjp8K65MPGBajZio?authuser=1"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="V1 Open In Colab"/></a> <br>V2: <a target="_blank" href="https://colab.research.google.com/drive/1bGYkzfeDL8rdj8qYAsjV7c84ocZfKUzn?authuser=1"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="V2 Open In Colab"/></a> | Work with text, docs, inference                               |
| Bielik_Streamlit_simple_app_tunnel_GGUF_Q4.ipynb           | <a target="_blank" href="https://colab.research.google.com/drive/1qUzPhx2uckvciuq9_pMJgoypmnkrk1nT?authuser=1"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                                                                                                                                                                                                                | Inference with streaming using Streamlit with Bielik (GGUF Q4) | 
| Bielik_Data_Generation_and_Fewshot_Prompting_(4_bit).ipynb | <a target="_blank" href="https://colab.research.google.com/drive/1DXTdzFRbLb1VrlvCzeFTI2nd5oFBi0QF?authuser=1"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                                                                                                                                                                                                                | Data Generation, Few-shot prompting                           |
| Bielik_Ollama_integration.ipynb                            | <a target="_blank" href="https://colab.research.google.com/drive/1XguCvlZ6oestH_AerzEkMc5WjLqSsICt?authuser=1"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                                                                                                                                                                                                                | Ollama CLI/API tutorial                                       |
| Bielik_Streamlit_simple_app_tunnel_4bit.ipynb              | <a target="_blank" href="https://colab.research.google.com/drive/1Pkb_4svxy6AxRePCVqW5q1hieuhgf605?authuser=1"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                                                                                                                                                                                                                | Inference with streaming using Streamlit with Bielik 4bit     |
| Bielik_Instruct_QUANT_Tests.ipynb                          | <a target="_blank" href="https://colab.research.google.com/drive/1bsU6C4X0RMRRzsrMAvzGoaqioaqo_p29?authuser=1"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                                                                                                                                                                                                                | e.g. RAG, function calling                                    |
| draive                                                     |                                                                                                                                                                                                                                                                                                                                                                                                                                             | Inference using draive lib                                    | Inference using draive lib                                                                          |
| contract_enhancer                                          |                                                                                                                                                                                                                                                                                                                                                                                                                                             | RAG for contract enhancement                                  | RAG for contract enhancement  |
