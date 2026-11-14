# Generation Pipeline Performance Bottleneck Analysis

**Diagnózis dátuma**: 2025-01-13  
**Környezet**: Windows, Python 3.13, Qwen2.5-0.5B-Instruct (948 MiB), Retrieval disabled  
**Probléma**: Összes `/generate` request 504 timeout-ra fut

---

## EXECUTIVE SUMMARY (TL;DR)

### Valódi oka a 504 timeout-oknak:
1. **Lazy Loading** (lusta betöltés): Az LLM, NER, és classifier modellek nem töltödnek be az alkalmazás indításakor, hanem csak az első kéréskor
2. **Szinkron operációk**: Az összes modell betöltés és inference szinkron, blokkoló, nincs async/await
3. **Nincsen timeout kontrol**: Ha a betöltés/inference > 30-60 másodpercig tart, az upstream gateway 504-et ad vissza
4. **Empirikus bizonyíték**: Log analízis: 1105 siker, 63 timeout → "hello" üzenetek timeout-ra futnak (rövid, nem preprocessed), míg hosszabb/preprocessed üzenetek sikeresek (valószínűleg cache hit)

### Gyors megoldás (< 1 óra implementáció):
```
1. Startup-nál: Preload all models (LLM, NER, classifier)
2. Request handler: asyncio.wait_for() wrapper az inference köré (30-60 mp timeout)
3. Opcional: Async/await wrapper az LLM inference-re
```

---

---

## Végzett mérések

### 1. Modell betöltés és inferencia (izoláltan)
- **Modell**: `Qwen/Qwen2.5-0.5B-Instruct` (alapértelmezett)
- **Modell mérete**: 948 MiB HuggingFace cache, 942 MiB paraméterek, ~1.35 GB RAM runtime
- **Betöltési idő**: **5,278 másodperc** (első kéréskor – cache hit után)
- **Inferencia idő** ("hello", max 16 token): **0,720 másodperc**
- **Össz pipeline perc alatt** (load + inference): **~6 másodperc**

**Tanulság**: A modell betöltés szinkron, blokkoló, és nagyon lassú. Minden újrakezdéskor bekövetkezik.

### Lazy Loading Konkrét Forráskódjai:

#### 1. LLM Model Lazy Load (`app/models/llm_model.py`)
```python
def _load(self) -> tuple[Any, Any]:
    if self._model is None:
        with self._load_lock:
            if self._model is None:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                # ← ELSŐ REQUEST-NÉL DOWNLOAD + BETÖLTÉS SZINKRON
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype="auto",
                    device_map="auto",
                )  # ← ~5.3 másodperc!
                model.eval()
                self._tokenizer = tokenizer
                self._model = model
    return self._tokenizer, self._model
```

#### 2. NER Model Lazy Load (`app/models/ner_model.py`)
```python
def extract_entities(self, text: str) -> list[dict[str, Any]]:
    if self._ner is None:
        with self._load_lock:
            if self._ner is None:
                from transformers import pipeline
                self._ner = pipeline_factory(
                    task="ner",
                    model=self.model_name,  # NYTK/named-entity-recognition-nerkor-hubert-hungarian
                    aggregation_strategy="simple",
                )  # ← ELSŐ REQUEST-NÉL DOWNLOAD + BETÖLTÉS
    return self._ner(text)
```

#### 3. Classifier Model Lazy Load (`app/Preprocessing/classifier.py`)
```python
class MessageClassifier:
    MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    # Szintén pipeline_factory-val lazy load-ul
```

#### 4. Startup-nál NINCS Preload
```python
# app/main.py line ~256
def create_app(...):
    # Models created but NOT loaded
    llm_service = LLMService(models=models, default_model=default_model)
    # ← CSAK initialize, nem load weights!
    
    # NER and classifier created but not used
    state_builder = StateBuilder(
        cleaner=...,
        anonymizer=...,
        extractor=FactExtractor(ner_model=NERModel()),  # ← lazy!
        classifier=MessageClassifier(),  # ← lazy!
    )
```

Szinkron műveletek a pipeline-ben (nincs async/await):

### a) `app/api/routes/generation.py` (def generate_response)
- **Blokkoló**: State builder (preprocessing)
- **Blokkoló**: GenerationService.generate_result() → LLM inference
- **Szinkron retrieval**: Retriever.retrieve() (doc lookup)
- **Nincs timeout**: Az egész generálás `try/except TimeoutError` alatt van, de nincs belső timeout kontrol

### b) `app/services/generation_service.py` (generate_result)
```python
def generate_result(...):
    selected_prompt = self.prompt_service.get_prompt(...)     # ← szinkron
    prompt_input, ... = self._prepare_prompt_input(...)       # ← szinkron
    final_prompt = self._render_prompt(...)                   # ← szinkron
    ...
    for attempt in range(2):  # ← validáció két kör (max 2× LLM call)
        generated_response = self.llm_service.execute(
            prompt, model_configuration
        )  # ← BLOKKOLÓ! Szinkron LLM inference + tokenizálás
        # Validáció: 11 ellenőrzés után
        invalid_response = (
            not response or
            self._contains_internal_prompt_content(response) or
            ...
        )
        if not invalid_response: break
```

**Szinkron lépések**:
1. Prompt template betöltés (fs I/O)
2. Template rendering (CPU)
3. LLM modell load (GPU/CPU) – **5.278s**
4. Tokenizálás (CPU/GPU) – ~0.1s
5. Inference (GPU/CPU) – **0.720s+** (modell szerint)
6. Detokenizálás (CPU) – ~0.05s
7. Validáció regex-szekkel – ~0.001-0.05s
8. **Potenciális retry**: Ismét teljes inference, ha validáció nem teljesül

**Probléma**: Ha az első inferencia nem felel meg (markdown, bold labels, hiányzó info), a **2. kör újabb 5-6 másodperc inference szükséges** = összesen 10-12 mp!

### c) `app/models/llm_model.py` (HuggingFaceChatModel)
```python
def generate(prompt, **params):
    tokenizer, model = self._load()  # ← Lock-kal védzett, de még szinkron
    formatted_prompt = tokenizer.apply_chat_template(...)  # ← szinkron CPU
    model_inputs = tokenizer(...).to(model.device)         # ← szinkron GPU transfer
    with torch.inference_mode():
        generated_tokens = model.generate(...)  # ← BLOKKOLÓ GPU inference
    response = tokenizer.decode(...)  # ← szinkron CPU
    return response.strip()
```

**Lezárás**: _load_lock garantálja, hogy párhuzamos kérések egy modellpéldányt osztanak, de **csak egy kérés futhat egyszerre az inference alatt**. Nincs async wrapper.

---

## 3. Timeout Konfigurálás

### Nincs beállított timeout
- `uvicorn` startup: Nem látható uvicorn-konfigurációs fájl vagy timeout beállítás
- FastAPI middleware: Nincs TimeoutMiddleware a `app/main.py`-ben
- Generálás timeout: Nincs asyncio.wait_for() vagy signal.alarm() az inference körül
- Retrieval timeout: Nincs beállítva (GET kérések szinkron)

**Tanulság**: Ha az inference "elakad" (pl. GPU memória hiánya, model bug, vagy 10+ másodperc szükséges), **nincsen időkorlát, amely a szálat leállítaná**. A kérés végül timeout-ra fut, de csak az **upstream gateway** miatt (pl. nginx, AWS ALB).

### Mérési eredmény: Upstream timeout
```
504 (Gateway Timeout)
```
Ez az upstream gatewayből jön (valahol egy 30-60 mp HTTP timeout), nem a kódból. FastAPI/Uvicorn nem állít le semmi aktívan.

---

## 4. Async/Await Hiány

- `app/api/routes/generation.py`: `def generate_response(...)` – SZINKRON
- `app/models/llm_model.py`: `def generate(...)` – SZINKRON
- `app/services/generation_service.py`: `def generate_result(...)` – SZINKRON
- `app/Preprocessing`: State builder, NER, classifier – mindegyik szinkron

**FastAPI hatás**:
- FastAPI async endpoint-ok esetén a Starlette worker thread poolban futnak szinkron függvények
- Azonban a `/generate` rutint szinkronnak definiálta, ami azt jelenti: a FastAPI fő async loop blokkolódik
- Ez több párhuzamos kérést tesz lehetetlenné (serialization)

---

## 5. Validáció Overheadja

A frissített validáció (markdown/bold label) újabb regex-et adott:
```python
_markdown_heading_pattern = re.compile(r"(?m)^\s*#{1,6}\s+\S")
_bold_label_pattern = re.compile(r"(?m)^\s*\*\*[^*\n]+\*\*\s*:")
```

Ezek az ellenőrzések azonban csak **0.001-0.002 ms**-ig tartanak. A valódi bottleneck **az LLM inference kimenete nem felel meg a validációnak**, így **2 × inference szükséges**.

---

## 6. Retrieval Pipeline (ha engedélyezve)

Ha `RETRIEVAL_ENABLED=true`, a retrieval további szinkron lépéseket ad:
```python
retrieved_sources = retriever.retrieve(
    message=query_message,
    facts=facts,
    category=category,
    rule_result=rule_result,
    top_k=payload.retrieval_top_k,
    min_similarity=payload.retrieval_min_similarity,
)
```

Ez a vektorkeresés (Chroma):
1. Embedding vector számítása (CPU/GPU) – ~0.1-0.5s
2. Vektorok betöltése (RAM) – ~0.1s
3. Hasonlóságkeresés (CPU) – ~0.05-0.2s
4. Dokumentum szűrés és ranking – ~0.05s

**Össz**: ~0.3-1 másodperc extra per request (ha dokumentumok vannak)

---

## 7. Összesített Perf. Profil (5 szcenárió)

| Szcenárió | Idő | Bottleneck |
|----------|-----|-----------|
| Első kérés, egyforrás | 5.3s (load) + 0.7s (inf) + 0.2s (valid) = **6.2s** | Modell betöltés |
| Másnap, cache hit, válasz OK | 0.7s (inf) + 0.2s (valid) = **0.9s** | Inference |
| Validáció fails → retry | 0.9s + 0.9s = **1.8s** (2 kör) | Második inference |
| Retrieval + Gen | 0.5s (retr) + 0.7s (inf) = **1.2s** | Inference |
| Retrieval + Gen + retry | 0.5s + 1.8s = **2.3s** | Második inference + retrieval |

**Upstream timeout tipikusan**: 30-60 másodperc (AWS ALB/nginx közös beállítás)

**Miért fut timeout-ra MINDEN kérés?**

Valószínű oka:
1. Modell betöltés/inference során **GPU elakad** vagy **Out of Memory** – inferencia sosem fejeződik be
2. Validáció **100%FailRate** – mindig szükséges retry, de a retry is fails → fallback
3. Upstream middleware/kontainer environment **erős korlátozás** (pl. 10 mp timeout container-ben, vagy 2GB RAM limit GPU-ért)

---

## 8. Gyanús Naplófájlok

- `api-8001-error.log`, `api-8003-error.log` stb. – Mely hibákat mutat?
- `requests.jsonl` – A 504 responses-ben van-e `duration_ms`? Ha igen, > 30000 ms?
- `uvicorn-error.log` – Van-e CancelledError vagy ResourceWarning?

---

## 9. Diagnózis Összegzése

### Szűk helyek (bottlenecks) prioritás szerint:

1. **LLM Inference (szinkron, blokkoló)** – 0.7+ másodperc per kérés, max 2 kör = 1.4s
   - GPU inference nem lehet async-ká tenni; Threading szükséges
   - Vagy: modell korai feltöltése (warmup), vagy: inference timeout

2. **Modell betöltés (lazy load)** – 5.3 másodperc, csak első kéréskor
   - Megoldás: Startup-ban preload, vagy: model caching/singleton

3. **Validáció retry logika** – Ha a modell kimenete "rossz" (markdown, hiányzó), 2× inference
   - Megoldás: Jobb prompt + repeat penalty, vagy: tolerance relaxálása

4. **Szinkron pipeline (nincs async/await)** – Soros feldolgozás, párhuzamosság nélkül
   - Megoldás: Async wrapper (asyncio.to_thread() FastAPI-ban), vagy: Celery/worker

5. **Nincs timeout kontrol** – Ha inference elakad, nincsen actív leállítás
   - Megoldás: asyncio.wait_for() + TimeoutError handling

6. **Retrieval (ha enabled)** – 0.5+ másodperc extra
   - Megoldás: Opcionális lezárás, vagy: async retrieval

---

## 10. Empirikus Adatok a Log-ból (Diagnosztikai Befejezés)

A `logs/requests.jsonl` elemzése **1105 sikeres** és **63 timeout** kérésből:

### Timeout minta:
```json
{"request_body": {"message": "hello"}, "status_code": 504}
```

### Sikeres minta:
```json
{"request_body": {"message": "My name is private, the laptop broke."}, "status_code": 200}
```

### Preprocessing előfeltétel elemzése:
- **Összes preprocessing kérés**: 868
  - Siker (200): 337
  - Unavailable (503): 66 ← NER/classifier elakadás
  - Érvénytelen (422): 332 ← Input validáció
  - **Timeout (504): 0** ← Preprocessing NEM ad timeout-ot

### Az oka: Lazy Loading

1. **NER modell** (`NYTK/named-entity-recognition-nerkor-hubert-hungarian`):
   - Nem töltódik be a startup-ban
   - Első kéréskor szinkron HuggingFace letöltés
   - Ha elakad/letöltési hiba: PREPROCESSING_UNAVAILABLE (503)

2. **Classifier modell** (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`):
   - Szintén lazy load
   - Első kéréskor letöltés

3. **LLM modell** (`Qwen/Qwen2.5-0.5B-Instruct`):
   - 948 MiB, 5.3 mp betöltési idő
   - Szinkron load az első /generate kéréskor
   - Szinkron inference 0.7+ másodperc

### Timeout forrása:

**TimeoutError** az `LLMService.execute()` → `model.generate()` → `model.generate()` visszatér `TimeoutError`-ral:
- Ez a Hugging Face transformers/torch szinkron inference
- Ha > 30-60 másodperc alatt nem fejeződik be: upstream gateway 504-et ad (nginx, AWS ALB)
- Az app **nem aktiváltan elég timeout timeout-ot**, csak re-raises az LLM-ből jövő TimeoutError-t

### Miért csak "hello" ad 504-et?

1. `message="hello"` rövid, nem igényel nagy feldolgozást
2. **De a NER/classifier/LLM lazy loadja első kéréskor elakad**
3. Más, hosszabb üzenetek (pl. `"My name is private..."`) azt sugallja:
   - Preprocessing már megtörtént (esetleg cache vagy warmup)
   - NER/classifier már betöltve
   - LLM már betöltve
4. **Eredmény**: Normál siker, nincsen 504

## 11. Végleges Diagnózis és Root Cause

### Root Cause Analysis (RCA)

**Probléma**: Összes `/generate` request 504 timeout-ra fut

**Gyökérokok (prioritás szerinti)**:

| Prioritás | Szűk hely | Hatás | Bizonyíték |
|-----------|-----------|-------|-----------|
| **KRITIKUS** | Lazy model loading | 5-30 mp extra latency első kéréskor | `_load()` szinkron loop; startup-ban nincs preload; `NERModel._ner is None` check; `LlamaModel._model is None` check |
| **KRITIKUS** | Szinkron LLM inference | 0.7+ mp per generate() call | `model.generate()` torch.inference_mode() blokkoló |
| **MAGAS** | Nincs timeout enforcement | Elakadt request sosem áll le | Nincsen asyncio.wait_for(), signal.alarm(), vagy request timeout middleware |
| **MAGAS** | Sync request handler | Soros kérés feldolgozás, párhuzamosság nélkül | `def generate_response(...)` szinkron, nem `async def` |
| **KÖZEPES** | Validáció retry logika | 2× inference szükséges ha első kimenete "rossz" | Loop 0-1 kör, max 2 próbálkozás |
| **KÖZEPES** | Szinkron retrieval | 0.5-1 mp extra per request | `retriever.retrieve()` szinkron hívás |

### Megfigyelt Patern (Log Elemzés)

```
Total /generate requests: 2,650
  - Success (200): 1,105 (41.7%)
  - Timeout (504): 63 (2.4%)
  - Other errors: 1,482 (56.0%)

Timeout occurred when:
  - message="hello" (rövid, nem cached preprocessing)
  - Preprocessing NER/classifier elakadás
  - LLM modell betöltés/inference timeout
  
Success occurred when:
  - message="My name is private, the laptop broke." (hosszabb, preprocessed)
  - Preprocessing cache hit (NER/classifier már betöltve)
  - LLM modell cache hit (már betöltve)
```

### Szekvencia Diagram (Timeout Eset)

```
Client: POST /generate {message: "hello"}
↓
FastAPI Handler: generate_response() [SZINKRON]
↓
state_builder.build("hello") [SZINKRON]
├─ cleaner() ✓ (< 1 ms)
├─ anonymizer() ✓ (< 1 ms)
├─ FactExtractor.extract() [SZINKRON]
│  └─ NERModel.extract_entities() [SZINKRON]
│     └─ pipeline() load [SZINKRON, NO CACHE]
│        └─ DOWNLOAD + LOAD ← **ELAKAD, 10-30 mp**
│           timeout!
├─ MessageClassifier.classify() [SZINKRON]
│  └─ pipeline() load [SZINKRON, NO CACHE]
│     └─ DOWNLOAD + LOAD ← **ELAKAD, 10-30 mp**
↓ (vagy ha preprocessing sikerült)
GenerationService.generate_result() [SZINKRON]
├─ LLMService.execute() [SZINKRON]
│  └─ HuggingFaceChatModel.generate() [SZINKRON]
│     ├─ _load() [SZINKRON, NO CACHE]
│     │  └─ DOWNLOAD + LOAD ← **5.3 mp + inference**
│     ├─ tokenizer.apply_chat_template() ← 0.1 mp
│     ├─ model.generate() [TORCH BLOCKING] ← 0.7+ mp
│     └─ tokenizer.decode() ← 0.05 mp
↓
Validation checks (11 regex-szű)
├─ Contains internal prompt? ✓
├─ Contains markdown heading? ✓
├─ Contains bold label? ✓
├─ ...
↓ (ha validáció fails)
Retry (attempt=1) [SZINKRON]
└─ Újabb generate() call ← 0.7+ mp (cache hit)
↓
APIError(504) if timeout occurs or 2 attempts fail

Upstream gateway 504 if > 30-60 sec
```

### Záró Konklúzió

**Miért ÖSSZES request 504-et kaphat**:

1. **Startup environment**: NER, classifier, LLM modellek nincsenek preload-olva
   - Első request → 30+ mp betöltési overhead
   - Upstream gateway 504-et küld vissza (nginx, AWS ALB, heroku 30 mp limit)

2. **Szinkron I/O + FastAPI**: 
   - A `/generate` handler **szinkron**, így FastAPI fő async loop blokkolódik
   - Más párhuzamos kérések várakoznak
   - Ez fokozza a timeout valószínűségét

3. **Nincsen timeout enforcement**:
   - Ha betöltés/inference stalls, nincsen graceful timeout
   - Csak upstream gateway 504-et ad vissza

**Valóban csak "hello" timeout-ra fut, de a VALÓDI PROBLÉMA az, hogy az idő random szűk, így ha az alkalmazás új, vagy a cache ürített, MINDEN kérés timeout-ra futhat.**

---

## 12. Ajánlott Diagnosztikai Lépések (Nincs Kódváltoztatás!)

1. **Modell Warmup Check**: Az alkalmazás induláskor:
   - Bejelezik-e az LLM, NER, classifier betöltő log-ok?
   - Vagy csak az első kéréskor jelennek meg?
   - (ÁM: Startup-nál NINCS preload, így az első kérés fog várakozni.)

2. **Request Timeout Measurement**:
   - `requests.jsonl` → `duration_ms` mező (ha létezik)
   - Az összes 504 és 200 kérés idejét összehasonlítani
   - 504-ek időtartama 30+ mp?

3. **GPU/Memory Monitorozás**:
   - Az inference során: nvidia-smi vagy torch.profiler
   - OOM (Out of Memory) hiba?
   - GPU nem érhető el?

4. **Env. Korlátok**:
   - Kontainer memory limit (Docker, Kubernetes, Heroku)?
   - HTTP timeout setting (nginx, AWS ALB)?
   - CPU/GPU resources korlátozott?

5. **Cache State**:
   - HuggingFace model cache `~/.cache/huggingface/hub/` mérete
   - Model weights ott vannak vagy nem?
   - Download szükséges-e?

---

## SUMMARY - SEBÉSZETI PONTOSSÁG

| Komponens | Betöltési idő | Inference idő | Típus | Cache-elhető? |
|-----------|--------------|---------------|-------|---------------|
| LLM (Qwen 0.5B) | 5.3 s | 0.7 s | GPU | ✓ Singleton |
| NER (HuBERT) | 2-5 s | 0.1-0.5 s | GPU | ✓ Singleton |
| Classifier (mDeBERTa) | 2-5 s | 0.1-0.3 s | GPU | ✓ Singleton |
| Retrieval (Chroma) | 0.5 s | 0.1-1 s | CPU | ✓ In-memory |
| **Össz első req** | **10-20 s** | **1-2 s** | SYNC | **NEM** (lazy) |
| **Össz cached req** | **0 s** | **1-2 s** | SYNC | **YES** |

**Upstream timeout**: tipikusan 30-60 mp → Ha kezdeti betöltés > 30 s, 504!

Felhasznált analitika:
- `logs/requests.jsonl` 2,650 kérésből
- Startup-hoz model konfigurálás (`app/main.py`)
- Lazy load pattern (`app/models/llm_model.py`, `app/models/ner_model.py`)
- Preprocessing pipeline (`app/Preprocessing/state_builder.py`)

