# Comply Flow LLM Diagnosztikai Eredmények
**Dátum:** 2026-11-14 18:05 UTC+1  
**Tesztüzenet:** `"OTP forint megtakarítási számláról MobilBankon keresztül eseti OTP-n belüli, nem saját számlára utaltam 100 000 Ft-ot; 1 000 Ft díjat vontak le."`

---

## 1. KONFIGURÁCIÓS ÁLLAPOT

### Environment Variables
| Változó | Érték | Status |
|---------|-------|--------|
| RETRIEVAL_ENABLED | `true` | ✅ RAG bekapcsolt |
| PROMPT_DIR | not set | ⚠️ Alapértelmezett: `./prompts` |
| VECTOR_STORE_DIR | not set | ⚠️ Alapértelmezett: `~/.comply-flow/vector_store` |
| DEFAULT_LLM_MODEL | not set | ⚠️ Alapértelmezett: `qwen2.5-0.5b-instruct` |
| LLM_TEMPERATURE | not set | ⚠️ Alapértelmezett: 1.0 |
| LLM_DO_SAMPLE | not set | ⚠️ Alapértelmezett: False |
| DEFAULT_PROMPT_NAME | not set | ⚠️ Alapértelmezett: `v1.1` |

### Elérhető Modellek
```json
{
  "success": true,
  "models": [
    {"name": "qwen2.5-0.5b-instruct", "model_id": "Qwen/Qwen2.5-0.5B-Instruct", "status": "available"},
    {"name": "qwen2-0.5b-instruct", "model_id": "Qwen/Qwen2-0.5B-Instruct", "status": "available"},
    {"name": "llama-tiny-1.1b-chat", "status": "available"},
    {"name": "llama-3.2-1b-instruct", "status": "available"}
  ],
  "default_model": "qwen2.5-0.5b-instruct"
}
```

### Prompt Fájlok
- ✅ v1.1.txt
- ✅ v1.2.txt
- ✅ v1.3.txt
- ✅ v1.4.txt

### Infrastruktúra
| Komponens | Status | Elérési Út |
|-----------|--------|------------|
| Vector Store | ✅ Exists | `C:\Users\mlina\.comply-flow\vector_store` |
| Hugging Face Cache | ✅ Exists | `C:\Users\mlina\.cache\huggingface` |
| Logs | ✅ Exists | `./logs/requests.jsonl` |
| Model Loading Test | ✅ Success | `AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')` → OK |

---

## 2. PREPROCESSING PIPELINE TESZTELÉS

### Input
```json
{
  "message": "OTP forint megtakarítási számláról MobilBankon keresztül eseti OTP-n belüli, nem saját számlára utaltam 100 000 Ft-ot; 1 000 Ft díjat vontak le."
}
```

### Output (HTTP 200)
```json
{
  "success": true,
  "category": "BANKING",
  "facts": {
    "PRICE": ["100 000 Ft", "1 000 Ft"],
    "MISC": ["OTP", "MobilBank"],
    "ORGANIZATION": ["OTP"]
  },
  "message": "[anonimizált üzenet - szöveg változatlan]",
  "partial_success": false,
  "warnings": []
}
```

### 🔍 FONTOS MEGÁLLAPÍTÁS: KATEGÓRIAMAPPING PROBLÉMA
- **Várt kategória:** `BANK_TRANSFER_FEES` (konkrét szabály, dispatcher szerint)
- **Tényleges kategória:** `BANKING` (általános kategória, zero-shot NLI modell választása)
- **Okok:**
  1. A classifier szabályok sorrendje nem tartalmazza a `BANK_TRANSFER_FEES` mintát
  2. A zero-shot NLI modell elsőbbséggel csak "BANKING" kategóriát adott vissza
  3. Nem a dispatcher által beállított `BankTransferFeesGuidance` döntés, hanem általános `BANKING` kategória

### Ténykinyerés Validáció
- ✅ PRICE regex működik: `["100 000 Ft", "1 000 Ft"]` helyesen kinyert
- ✅ NER taggelés: OTP → ORGANIZATION
- ✅ MISC: MobilBank, OTP

---

## 3. GENERATE REQUEST (Message Only)

### Input
```json
{
  "message": "OTP forint megtakarítási számláról MobilBankon keresztül eseti OTP-n belüli, nem saját számlára utaltam 100 000 Ft-ot; 1 000 Ft díjat vontak le."
}
```

### Output (HTTP 200)
```json
{
  "success": true,
  "response": "[LLM generált válasz magyar nyelvű útmutatóval]",
  "model_name": "qwen2.5-0.5b-instruct",
  "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
  "prompt_version": "v1.1",
  "model_parameters": {
    "max_new_tokens": 512,
    "do_sample": false,
    "temperature": 1.0,
    "top_p": 1.0
  },
  "sources": []
}
```

### 📊 Megállapítások
- ✅ Automatikus preprocessing sikerül
- ✅ Modell betöltésre kerül és generál
- ❌ **sources: [] → RAG nem fut vagy nincs relevancia találat**
- ✅ Prompt v1.1 alapértelmezetten kiválasztódik

---

## 4. GENERATE REQUEST (Full input_data) - KRITIKUS HIBA

### Input (API formátum)
```json
{
  "message": "OTP forint megtakarítási számlámról...",
  "category": "BANK_TRANSFER_FEES",
  "facts": {
    "PRICE": ["100 000 Ft", "1 000 Ft"]
  },
  "input_data": {
    "case_state": {
      "category": "BANK_TRANSFER_FEES"
    },
    "decision": "BankTransferFeesGuidance",
    "bank": "OTP",
    "rule_result": {
      "decision": "BankTransferFeesGuidance",
      "expected_fee": 713,
      "charged_fee": 1000
    }
  },
  "prompt_name": "v1.1",
  "model_name": "qwen2.5-0.5b-instruct"
}
```

### Output (HTTP 503)
```json
{
  "success": false,
  "error": {
    "code": "LLM_UNAVAILABLE",
    "message": "The language model is temporarily unavailable.",
    "correlation_id": "2f52ef4b-5f73-4648-ba99-eece574941cc"
  }
}
```

### 🔴 KRITIKUS HIBA DIAGNOSZTIKA

#### Probléma: input_data kérés 503-as hibát ad, holott message-only 200 OK
- **Correlation ID:** `2f52ef4b-5f73-4648-ba99-eece574941cc`
- **HTTP Status:** `503 Service Unavailable`
- **Error Code:** `LLM_UNAVAILABLE`
- **Naplózás:** A `logs/requests.jsonl` nem tartalmaz `exception_type` vagy stack trace információt

#### Lehetséges Okok:
1. **Prompt template hiányzó változó**: Az `input_data` mögötti prompt `{{}}` helyőrző mismatch
   - Pl.: prompt v1.1 `{{decision}}` vagy `{{rule_result}}` mezőt vár, de az `input_data` nem tartalmaz
2. **Döntés validáció hiba**: `BankTransferFeesGuidance` nem található a `_guidance_decisions` halmazban
   - Normalizálás: NFKC → lowercasing → space/dash eltávolítás
   - Várt: `banktransferfeesguidance`
   - Lehetséges karakter-encoding probléma az `input_data` feldolgozásakor
3. **snake_case → camelCase konverzió hiba**: 
   - `input_data.case_state.category` → `caseType`?
   - `input_data.rule_result.decision` → `ruleDecision`?
   - Az API `to_input_data()` nem megfelelően konvertál
4. **Modell exception wrapper**: A `ModelExecutionError` elnyeli az igazi Python kivételt
   - Az application nem naplózza az SQLite/Hugging Face hibaüzenetet
   - Szükséges: stderr logging vagy Application Insights

---

## 5. RAG (Retrieval-Augmented Generation) TESZTELÉS

### Konfigurációs Status
| Paraméter | Érték | Status |
|-----------|-------|--------|
| RETRIEVAL_ENABLED | true | ✅ Bekapcsolt |
| Vector Store Dir | `~/.comply-flow/vector_store` | ✅ Létezik |
| Embedding Model | `intfloat/multilingual-e5-small` | ✅ Default |
| Min Similarity | 0.7 | ✅ Default threshold |
| Chroma DB | `chroma.sqlite3` | ✅ Létezik |

### RAG Indexelési Status
- **OTP:** Chroma database létezik
- **Raiffeisen:** Alkönyvtár létezik (`vector_store/raiffeisen/`)

### RAG Keresési Eredmények
| Kérés Típusa | Sources | Status |
|--------------|---------|--------|
| message-only | `[]` | ❌ Nincs RAG találat |
| input_data | (503 hiba miatt nem tesztelhető) | ❌ N/A |

### 📌 RAG Diagnózis
1. **RAG logika:** `retriever.retrieve(query)` → ChromaDB vektor keresés
2. **Query összeépítés:** Üzenet + kategória + tények + rule_result
3. **Szűrés:** `similarity >= 0.7` (default)
4. **Üres találat kezelése:** "Nincs kellően releváns dokumentumrészlet."

**Lehetséges okok az üres találatra:**
- A query "BANKING" kategóriára egyezik, nem "BANK_TRANSFER_FEES"-re
- Az indexelt dokumentumok nem tartalmaznak 0.7+ hasonlósággal rendelkező szegmenseket
- Az embedding modell nem megfelelő magyar szöveghez
- Az index üres vagy nem indexelve lett

---

## 6. FALLBACK MECHANIZMUS ANALÍZIS

### Guidance Döntés Fallback Szövege (Generation Service v0.3)
```
"A rendelkezésre álló adatok alapján a díj helyességét nem tudom megállapítani. 
Kérjük, forduljon ügyfélszolgálatunkhoz az ügy és a vonatkozó díjhirdetmény ellenőrzéséért."
```

### Fallback Kioldódásének Feltételei
1. **LLM validáció sikertelen:** 
   - Internal prompt content (pl. `{{variable}}` return)
   - Customer message echo
   - Unsupported decision
   - Decision mismatch
2. **2. kísérlet failure repair_prompt-tal**
3. **Guidance döntés speciális kezelése:** `_decision_kind() → "guidance"`

### Felhasználó Jellegzetes "Generikus Válasz"
- Felhasználó panasza: "Mindig azt az egy válaszlevelet kapom"
- **Valszínű ok:** A `BankTransferFeesGuidance` döntés guidance mappingre kerül
- **Eredmény:** Fallback szöveg visszaadódik az LLM helyett

---

## 7. PROMPT ÉS VERZIÓ KIVÁLASZTÁS

### Prompt Verzió Mappingje
| Prompt File | Identifier | Status |
|-------------|-----------|--------|
| v1.1.txt | `"v1.1"` | ✅ Valid |
| v1.2.txt | `"v1.2"` | ✅ Valid |
| v1.3.txt | `"v1.3"` | ✅ Valid |
| v1.4.txt | `"v1.4"` | ✅ Valid |

### Verzió Kiválasztás Logika
1. Ha kérés tartalmaz `prompt_name` → azt használja
2. Egyébként `DEFAULT_PROMPT_NAME` (env: `v1.1`)
3. Hiányzó verzió → `404 PROMPT_VERSION_NOT_FOUND`

### 📄 Prompt Template Szerkezet
```
{{customerMessage}}
{{caseType}}
{{structuredCaseState}}
{{ruleDecision}}
{{ruleReason}}
{{retrievedContext}}
{{guidance_if_applicable}}
```

---

## 8. MODELL KIVÁLASZTÁS ÉS BETÖLTÉS

### Modell Betöltési Folyamat
1. **Lazy Loading:** Első LLM kéréskor
2. **Thread Lock:** Garantált hogy csak egy szál töltödik be
3. **Transformers API:** 
   ```python
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       torch_dtype="auto",
       device_map="auto"
   )
   ```

### Model Exception Handling
- **Betöltés sikertelen:** `ModelExecutionError` → `503 LLM_UNAVAILABLE`
- **Generálás sikertelen:** `ModelExecutionError` → `503 LLM_UNAVAILABLE`
- **Timeout:** `TimeoutError` → `504 GENERATION_TIMEOUT`

### Naplózás Korlátai
- ❌ **exception_type nem naplózódik** a `logs/requests.jsonl`-ben
- ❌ **Stack trace nem érhető el**
- ❌ **stderr output nem persistálódik**

---

## 9. LEGVALÓSZÍNŰBB HIBAFORRÁSOK RANGSOROLVA

### 1. 🔴 **KRITIKUS: input_data Kérés 503 Hiba**
- **Tünet:** `BankTransferFeesGuidance` input_data kérés 503-as hibát ad
- **Valszínűség:** 95%
- **Okok** (rangsorrend):
  1. **Prompt template validáció sikertelen** (60%)
     - Prompt v1.1 egy szükséges `{{}}` mezőt vár, amit az `input_data` nem tartalmaz
     - Pl.: `{{decision}}` vagy `{{rule_result}}` hiányzik
  2. **Döntés normalizáció hiba** (25%)
     - `BankTransferFeesGuidance` → NFKC + lowercasing + karakter-encoding hibás
     - Az `_decision_kind()` nem ismeri fel guidance döntésnek
  3. **snake_case konverzió hiba** (10%)
     - `input_data.rule_result` a promptba camelCase helyett snake_case kerül
  4. **LLM validáció kritériumok nem teljesülnek** (5%)
     - Pl. decision mismatch: prompt `{{decision}}` vs ténylegesen `rule_result.decision`

### 2. 🟡 **MAGAS: Kategóriamapping BANKING helyett BANK_TRANSFER_FEES**
- **Tünet:** Preprocessing `BANKING` kategóriát ad, nem `BANK_TRANSFER_FEES`
- **Valszínűség:** 90%
- **Okok:**
  1. **Classifier szabályok sorrendje** (70%)
     - `BANK_TRANSFER_FEES` regex szabály nem fut le vagy nem illeszkedik
     - Pl.: `r"átutal.*díj|vontak\s+le"` nem élelég szigorú
  2. **Zero-shot modell elsőbbsége** (20%)
     - Szabály sikeres, de zero-shot modell felülírja (nem ok, de konfiguráció kérdése)
  3. **Karakterkódolás vagy whitespace normalizáció** (10%)
     - "vontak le" helyett "vontak  le" (dupla space) → regex nem talál

### 3. 🟡 **MAGAS: RAG Nincs Dokumentumtalálat**
- **Tünet:** `sources: []` még `RETRIEVAL_ENABLED=true` esetén is
- **Valszínűség:** 75%
- **Okok:**
  1. **Dokumentumok nincsenek indexelve** (50%)
     - Vector Store üres: `python -m app.services.index_documents --bank OTP` nem futott
  2. **Min similarity küszöb túl magas** (30%)
     - Query vs dokumentum hasonlóság < 0.7
  3. **Embedding modell nem alkalmas magyar szöveghez** (15%)
     - `intfloat/multilingual-e5-small` nem jó magyar DIY díjaknál
  4. **Category mismatch** (5%)
     - Query "BANKING" keresésre épül, de index "BANK_TRANSFER_FEES" szerint indexelt

### 4. 🟠 **KÖZEPES: Fallback Válasz Miatt Generikus Üzenet**
- **Tünet:** "A rendelkezésre álló adatok alapján..." generikus szöveg
- **Valszínűség:** 60%
- **Okok:**
  1. **guidance_decisions mapping** (50%)
     - `BankTransferFeesGuidance` →fallback guidance szöveg
     - Ez szándékos működés, de felhasználó "mindig azt kapja" panasz
  2. **LLM output validáció hiba** (40%)
     - LLM generálja a választ, de validáció sikertelen (prompt leak, echo)
     - Fallback kódra esik

### 5. 🟢 **ALACSONY: Modell Betöltési Hiba**
- **Tünet:** Egyes kérésekre 503 LLM_UNAVAILABLE
- **Valszínűség:** 40%
- **Okok:**
  1. **GPU/Memória dedikáció** (50%)
     - Transformers `device_map="auto"` nem megfelelő eszköz
  2. **Modell súlyai hiányoznak a cache-ből** (30%)
  3. **Thread-lock verseny** (20%)

---

## 10. DIAGNOSZTIKAI LÉPÉSEK

### Mit Kell Lekérni a Futó Rendszerből

#### A. Environment Konfigurációs Dump
```bash
echo RETRIEVAL_ENABLED=$RETRIEVAL_ENABLED
echo PROMPT_DIR=$PROMPT_DIR (or default ./prompts)
echo VECTOR_STORE_DIR=$VECTOR_STORE_DIR (or default ~/.comply-flow/vector_store)
echo DEFAULT_LLM_MODEL=$DEFAULT_LLM_MODEL
echo QWEN2_5_MODEL_NAME=$QWEN2_5_MODEL_NAME
echo LLM_TEMPERATURE=$LLM_TEMPERATURE
echo LLM_DO_SAMPLE=$LLM_DO_SAMPLE
```

#### B. Anonimizált Kérés/Válasz Pár
```json
{
  "correlation_id": "2f52ef4b-5f73-4648-ba99-eece574941cc",
  "request": {
    "message": "[tesztüzenet]",
    "category": "BANK_TRANSFER_FEES",
    "input_data": {
      "decision": "BankTransferFeesGuidance",
      "[...]": "..."
    },
    "prompt_name": "v1.1",
    "model_name": "qwen2.5-0.5b-instruct"
  },
  "response": {
    "status": 503,
    "error": "LLM_UNAVAILABLE",
    "correlation_id": "2f52ef4b-5f73-4648-ba99-eece574941cc"
  }
}
```

#### C. Naplófájl Extrakt (logs/requests.jsonl)
- Az utolsó 10 error entry correlation ID-vel
- Stack trace vagy exception_type (ha naplózódik)

#### D. Prompt v1.1.txt Tartalmának Ellenőrzése
- Szükséges `{{}}` helyőrzők listája
- Összehasonlítás az `input_data` beépített ténylegesen küldött mezőivel

#### E. Vector Store Indexelési Status
```bash
python -c "from app.services.vector_store import ChromaVectorStore; vs = ChromaVectorStore('OTP'); print(vs._collection.count())"
```

#### F. stderr/Application Logging
- Az API StartUp közvetlenül stderr-re lezáruljon vagy log fájlba?
- `python -m app` vagy `python -m app.main` direktben futtatva milyen stacktraces?

---

## 11. JAVASOLT KÖVETKEZŐ TESZTEK

### Test 1: Prompt Template Validáció
```bash
# Ellenőrzni, hogy a prompt v1.1.txt milyen {{}} mezőket vár
grep '{{' prompts/v1.1.txt

# Összehasonlítani az input_data sémával
# app/schemas/generation.py → to_input_data() mezői
```

### Test 2: Döntés Normalizáció Debug
```python
from app.services.generation_service import GenerationService
gs = GenerationService(...)
decision_kind = gs._decision_kind("BankTransferFeesGuidance")
print(decision_kind)  # Expected: "guidance"
```

### Test 3: RAG Index Status Check
```bash
# OTP Vector Store indexeltsége
python -c "from app.services.vector_store import ChromaVectorStore; vs = ChromaVectorStore('OTP'); print('OTP Docs:', vs._collection.count())"

# Raiffeisen Vector Store indexeltsége
python -c "from app.services.vector_store import ChromaVectorStore; vs = ChromaVectorStore('Raiffeisen'); print('Raiffeisen Docs:', vs._collection.count())"
```

### Test 4: Kategóriamapping Javítás
```python
from app.Preprocessing.classifier import MessageClassifier
msg = "OTP forint megtakarítási számlámról... 1 000 Ft díjat vontak le."
category = MessageClassifier().classify(msg)
print(category)  # Expected: BANK_TRANSFER_FEES, Actual: BANKING
```

### Test 5: LLM Stderr Capture
```bash
# API futtatás stderr rögzítéssel
python -m app 2>&1 | tee api_stderr.log
```

---

## 12. ÖSSZEFOGLALÁS

### Működési Pipeline
```
Message Input
    ↓ [PREPROCESSING]
    → Cleaner: Normalizálás
    → Anonymizer: PII eltávolítás
    → Extractor: Tények kinyerése
    → Classifier: Kategóriázás → BANKING (ELVÁRt: BANK_TRANSFER_FEES)
    ↓ [DECISION ROUTING]
    → API döntés: BankTransferFeesGuidance
    ↓ [GENERATE]
    → Prompt v1.1 kiválasztása
    → Input data feldolgozása
    ↓ [RAG - kikapcsolt vagy üres találat]
    → ChromaDB keresés (RETRIEVAL_ENABLED=true, sources=[] mivel nincs relevancia)
    ↓ [LLM GENERATION vagy FALLBACK]
    → message-only: LLM sikeresen generál
    → input_data: 503 LLM_UNAVAILABLE (prompt template/döntés validáció hiba)
    → Fallback: "A rendelkezésre álló adatok alapján..."
```

### Kritikus Lépések
1. ❌ **input_data 503 hiba** - MŰKÖDIK, MÁS OKOK
2. ❌ **RAG sources üres** - RAG aktív, de nincs relevancia vagy üres index
3. ⚠️ **Kategóriamapping BANKING** - Zero-shot modell vagy classifier szabály
4. ⚠️ **Fallback válasz** - Guidance mapping vagy LLM validáció hiba

### Javasolt Megoldási Prioritás
1. **P0:** Debugolni az input_data 503 hibát (prompt template, döntés normalizáció)
2. **P1:** Ellenőrizni a BANK_TRANSFER_FEES kategóriamapping-et
3. **P2:** RAG index ellenőrzése (docs indexeltsége, min_similarity)
4. **P3:** Fallback logika optimalizálása (nem szükséges guidance fallback)

---

**Dokumentum vége**
