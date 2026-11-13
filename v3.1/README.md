# Bielik 11B v3.1

Ten katalog zbiera informacje potrzebne do rozpoczęcia pracy z modelem
[Bielik-11B-v3.1-Instruct](https://huggingface.co/speakleash/Bielik-11B-v3.1-Instruct),
udostępnionym 10 listopada 2026 przez SpeakLeash i ACK Cyfronet AGH.

Jest to aktualizacja linii Bielik 11B v3: szersze wielojęzyczne trenowanie
(56 języków, z naciskiem na języki europejskie i polski) oraz mocniejsze
dopasowanie do instrukcji, użycia narzędzi i rozumowania.

- Karta modelu: https://huggingface.co/speakleash/Bielik-11B-v3.1-Instruct
- Oficjalna kolekcja (instruct i kwantyzacje): https://huggingface.co/collections/speakleash/bielik-11b-v31-6af30e07709eddbf23806633
- Parsery vLLM (tool calling i reasoning): https://github.com/speakleash/bielik-tools

## Parametry modelu

| Pole | Wartość |
|---|---|
| Identyfikator | `speakleash/Bielik-11B-v3.1-Instruct` |
| Licencja | Apache 2.0 |
| Typ | causal decoder-only, 11 mld parametrów (architektura Llama) |
| Kontekst | 32 768 tokenów |
| Temperatura (parametr z karty modelu) | 0,3 |
| Szablon czatu | dostarczany z modelem (`chat_template.jinja`); przy vLLM nie trzeba podawać `--chat-template` |

Długość kontekstu 32 768 pochodzi z `max_position_embeddings` w konfiguracji oficjalnych kwantyzacji (AWQ, FP8-Dynamic, NVFP4) oraz z przykładu vLLM w karcie modelu (`--max-model-len 32768`).

Pobranie pełnych wag Instruct wymaga akceptacji zgody na stronie modelu na Hugging Face. Warianty skwantyzowane są w osobnych repozytoriach (tabela niżej).

## Szablon czatu, rozumowanie i tool calling

Model używa formatu ChatML (`<|im_start|>` / `<|im_end|>`) oraz własnego formatu wywołań narzędzi i rozumowania:

- narzędzia: `<tool_call>{"name": …, "arguments": …}</tool_call>`
- rozumowanie: `<think>…</think>`

Poziom rozumowania steruje się parametrem `reasoning_effort`: `none` (domyślnie), `low`, `medium`, `high`, `max`. Przy `none` model odpowiada bezpośrednio. Flaga `enable_thinking` znana z Bielika v3.0 jest przez ten szablon ignorowana.

Szablon dopisuje wbudowany prompt systemowy „Model Guidelines” (tożsamość modelu, data graniczna wiedzy, bieżąca data, zasady bezpieczeństwa; ok. 1000 tokenów). Własna wiadomość systemowa trafia do sekcji „Additional Instructions”. Datę na potrzeby powtarzalnych uruchomień można ustawić przez `current_date` i `current_day_of_week` w `chat_template_kwargs`. Wbudowany prompt można wyłączyć flagą `provider_system_prompt` — szczegóły są w karcie modelu.

Aby w API zgodnym z OpenAI otrzymać ustrukturyzowane `tool_calls` oraz osobne pole `reasoning`, vLLM należy uruchomić z parserami z [bielik-tools](https://github.com/speakleash/bielik-tools). Wymagany jest vLLM **0.24 lub nowszy** (sprawdzany na wersji 0.32).

### Uruchomienie vLLM

Przykład z karty modelu:

```bash
git clone https://github.com/speakleash/bielik-tools.git

vllm serve speakleash/Bielik-11B-v3.1-Instruct \
    --enable-auto-tool-choice \
    --tool-parser-plugin ./bielik-tools/tools/bielik_vllm_tool_parser.py \
    --tool-call-parser bielik \
    --reasoning-parser-plugin ./bielik-tools/tools/bielik_vllm_reasoning_parser.py \
    --reasoning-parser bielik \
    --exclude-tools-when-tool-choice-none \
    --max-model-len 32768
```

- Należy załadować dokładnie jeden parser narzędzi i jeden parser rozumowania (oba rejestrują nazwę `bielik`).
- `--exclude-tools-when-tool-choice-none` usuwa definicje narzędzi z promptu, gdy żądanie ma `tool_choice="none"`; w przeciwnym razie model może mimo to zapisać wywołanie narzędzia jako zwykły tekst.

### Rozumowanie

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="speakleash/Bielik-11B-v3.1-Instruct",
    messages=[{"role": "user", "content": "Ile jest liczb pierwszych mniejszych od 50?"}],
    reasoning_effort="high",          # or extra_body={"chat_template_kwargs": {"reasoning_effort": "high"}}
    max_tokens=4096,
)
msg = resp.choices[0].message
print("reasoning:", getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None))
print("answer:", msg.content)
```

Zostaw wystarczający `max_tokens` na rozumowanie i odpowiedź: jeśli generacja zatrzyma się w trakcie myślenia, pole `content` będzie puste.

### Tool calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Returns the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

resp = client.chat.completions.create(
    model="speakleash/Bielik-11B-v3.1-Instruct",
    messages=[{"role": "user", "content": "Jaka jest pogoda w Krakowie?"}],
    tools=tools,
    tool_choice="auto",               # "required", "none" or a named function also work
)
print(resp.choices[0].message.tool_calls)
```

Argumenty narzędzi należy traktować jako niezaufane dane wejściowe. Szczegóły dotyczące `tool_choice`, `response_format` i `chat_template_kwargs` są opisane w [karcie modelu](https://huggingface.co/speakleash/Bielik-11B-v3.1-Instruct).

## Oficjalne kwantyzacje

Wszystkie poniższe repozytoria pochodzą z organizacji `speakleash` na Hugging Face i bazują na `speakleash/Bielik-11B-v3.1-Instruct`.

| Repozytorium | Format |
|---|---|
| [Bielik-11B-v3.1-Instruct-GGUF](https://huggingface.co/speakleash/Bielik-11B-v3.1-Instruct-GGUF) | GGUF: `bf16`, `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0` (llama.cpp lub Ollama) |
| [Bielik-11B-v3.1-Instruct-AWQ](https://huggingface.co/speakleash/Bielik-11B-v3.1-Instruct-AWQ) | AWQ, wagi 4-bit |
| [Bielik-11B-v3.1-Instruct-FP8-Dynamic](https://huggingface.co/speakleash/Bielik-11B-v3.1-Instruct-FP8-Dynamic) | FP8, dynamiczne aktywacje FP8 |
| [Bielik-11B-v3.1-Instruct-NVFP4](https://huggingface.co/speakleash/Bielik-11B-v3.1-Instruct-NVFP4) | NVFP4, wagi i aktywacje |
| [Bielik-11B-v3.1-Instruct-NVFP4A16](https://huggingface.co/speakleash/Bielik-11B-v3.1-Instruct-NVFP4A16) | NVFP4, wagi + aktywacje BF16 |
| [Bielik-11B-v3.1-Instruct-MLX-4bit](https://huggingface.co/speakleash/Bielik-11B-v3.1-Instruct-MLX-4bit) | MLX, wagi 4-bit |
| [Bielik-11B-v3.1-Instruct-MLX-8bit](https://huggingface.co/speakleash/Bielik-11B-v3.1-Instruct-MLX-8bit) | MLX, wagi 8-bit |

## DFlash — modele draft do dekodowania spekulatywnego

[Bielik-11B-v3.1-DFlash](https://huggingface.co/speakleash/Bielik-11B-v3.1-DFlash) i [Bielik-11B-v3.1-DFlash2](https://huggingface.co/speakleash/Bielik-11B-v3.1-DFlash2) **nie są samodzielnymi modelami czatowymi**. Działają wyłącznie jako draft w parze z modelem docelowym `Bielik-11B-v3.1-Instruct`.

### DFlash (vLLM > 0.20.0, SGLang > 0.5.6)

```bash
vllm serve speakleash/Bielik-11B-v3.1-Instruct \
  --speculative-config '{"method": "dflash", "model": "speakleash/Bielik-11B-v3.1-DFlash", "num_speculative_tokens": 15}' \
  --attention-backend flash_attn \
  --max-num-batched-tokens 32768
```

### DFlash2 (vLLM >= 0.28.0, SGLang >= 0.5.19)

```bash
vllm serve speakleash/Bielik-11B-v3.1-Instruct \
  --speculative-config '{"method": "dflash", "model": "speakleash/Bielik-11B-v3.1-DFlash2", "num_speculative_tokens": 15}' \
  --attention-backend flash_attn \
  --max-num-batched-tokens 32768
```

Dostępne są też warianty draft: GGUF (`bf16`, `Q8_0`, `Q4_K_M`) oraz FP8-Dynamic. Przykład llama.cpp z karty GGUF:

```bash
llama-server -m Bielik-11B-v3.1-Instruct.Q4_K_M.gguf \
  -md Bielik-11B-v3.1-DFlash.Q4_K_M.gguf \
  --spec-type draft-dflash --spec-draft-n-max 15 --jinja
```

Dla DFlash2 użyj analogicznie pliku `Bielik-11B-v3.1-DFlash2.Q4_K_M.gguf`. Przykłady SGLang oraz warianty FP8 są na kartach poszczególnych modeli draft.

## Relacja do przykładu V3.0 w tym repozytorium

Notatnik [`Bielik_v3_0_unsloth.ipynb`](../Bielik_v3_0_unsloth.ipynb) pozostaje przykładem dla **Bielik-4.5B-v3.0-Instruct** i flagi `enable_thinking`. Dla v3.1 należy użyć `reasoning_effort` oraz szablonu dostarczonego z modelem (albo parserów z `bielik-tools`), a nie polegać na `enable_thinking`.

Przykłady V1 i V2 w głównym README dotyczą starszych modeli (7B v0.1, 11B v2.2 itd.) i nie były tu przepisywane.
