# Bielik + Weaviate + Ollama

🎯 Informacje
-------------

Ten folder zawiera zestaw przykładów w jaki sposób mozna uruchomić lokalnie [Bielika](https://bielik.ai/) z użyciem [Ollama](https://ollama.com/) wraz z bazą wektorową [Weaviate](https://weaviate.io/).

📦 Wymagania
------------

W celu uruchomienia przykładów nalezy mieć skonfigurowane:

1. Docker
2. Python3

💡 Rozpoczęcie pracy
--------------------

Przed uruchomieniem notebooków należy przygotować środowisko:

1. Uruchomić kontenery z bazą Weaviate, wektoryzerami i modułem generatywnym Ollama:

```sh
docker compose up
```

2. Zanim przystąpisz do pracy ściągnij lokalnie Bielika wewnątrz kontenera Ollama:

```sh
docker exec -i generative_ollama ollama pull SpeakLeash/bielik-7b-instruct-v0.1-gguf
```

3. (opcjonalnie) Skonfiguruj osobne środowisko python:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

📖 Przykłady
------------

1. [0-import.ipynb](./notebooks/0-import.ipynb) - zaimportuj i zwektoryzuj dane
2. [1-rag.ipynb](./notebooks/1-rag.ipynb) - odpytaj lokalnie swoje dane z wykorzystaniem Bielika

🔗 Przydatne odnośniki
----------------------

- [Dataset użyty w notebookach](https://huggingface.co/datasets/allegro/summarization-polish-summaries-corpus)
