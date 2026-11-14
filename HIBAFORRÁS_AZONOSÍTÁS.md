# 🔴 Comply Flow LLM — KRITIKUS HIBAFORRÁS AZONOSÍTÁS

**Vizsgálat Dátuma:** 2026-11-14 18:12-18:14 UTC+1  
**API Verzió:** Comply Flow v0.3  
**Hibakód:** `LLM_UNAVAILABLE` (HTTP 503)  
**Kimenet:** Fallback "A rendelkezésre álló adatok alapján..." szöveg helyett

---

## 🎯 PROBLÉMA LEÍRÁSA

### Tünet
- **POST /generate** `message`-only kérésre: ✅ HTTP 200, LLM válasz
- **POST /generate** `input_data`-val: ❌ HTTP 503, `LLM_UNAVAILABLE`
- **Jellegzetes Felhasználó Jellegzetesség:** "Mindig azt az egy generikus válaszlevelet kapom"

### Vizsgálati Adatok
- **Test Case:** OTP átutalási díj panasz (100 000 Ft + 1 000 Ft díj)
- **Correlation IDs (503-as hibák):**
  - `167a4d55-081a-4666-aa9e-bf55395db621` (input_data-full)
  - `824b3d23-c6ee-463f-9556-b8bd494992fd` (input_data-omit-decision)
  - `848d09db-ffae-429b-8441-af15c683d1aa` (input_data-omit-rule_result)

---

## 🔍 ROOT CAUSE ANALÍZIS

### A Hiba Feldolgozási Útvonala

```
1. API kérés /generate
   ├─ input_data = {
   │  ├─ caseType: "BANK_TRANSFER_FEES"
   │  ├─ matchedRules: ["BankTransferFeesGuidance"]
   │  ├─ structuredCaseState: {...}
   │  ├─ customerMessage: "..."
   │  ├─ reason: "Bank transfer fee guidance"    ← ⚠️ SNAKE_CASE!
   │  ├─ decision: "BankTransferFeesGuidance"    ← ⚠️ SNAKE_CASE!
   │  ├─ cleanedText: "..."
   │  ├─ bank: "OTP"
   │  └─ rule_result: {...}
   │
   2. GenerationService.generate_result()
   │  ├─ selected_prompt_name = "v1.1"
   │  ├─ selected_prompt = get_prompt("v1.1")
   │  │  └─ Template {{}} helyőrzők:
   │  │     1. {{customerMessage}} ✅ LÉTEZIK
   │  │     2. {{caseType}} ✅ LÉTEZIK
   │  │     3. {{structuredCaseState}} ✅ LÉTEZIK
   │  │     4. {{ruleDecision}} ❌ HIÁNYZIK! (van: decision)
   │  │     5. {{ruleReason}} ❌ HIÁNYZIK! (van: reason)
   │  │     6. {{matchedRules}} ✅ LÉTEZIK
   │  │     7. {{retrievedContext}} ✅ LÉTEZIK (RAG-ból vagy fallback)
   │  │     8. {{validationRequirements}} ✅ LÉTEZIK (generált)
   │  │
   │  3. _prepare_prompt_input(input_data)
   │  │  └─ Feldolgozás:
   │  │     - response_plan (snake/camel) keresés → OK
   │  │     - validation_requirements ellenőrzés → OK
   │  │     - facts mapping → OK
   │  │     ⚠️ NEM KONVERTÁLJA: decision → ruleDecision, reason → ruleReason
   │  │
   │  4. _render_prompt(selected_prompt, prompt_input)
   │  │  ├─ Helyőrző regex: r"{{\s*([A-Za-z][A-Za-z0-9_]*)\s*}}"
   │  │  ├─ Megtalált {{ruleDecision}}
   │  │  ├─ Keresés: prompt_input.get("ruleDecision")
   │  │  ├─ ❌ RESULT: None (nem létezik)
   │  │  └─ THROW: ValueError("Missing prompt input: ruleDecision")
   │  │
   │  5. ValueError Exception Handling
   │  │  ├─ Exception → ModelGenerationError wrapper
   │  │  ├─ ModelExecutionError → 503 LLM_UNAVAILABLE
   │  │  └─ Válasz: {"success": false, "error": {"code": "LLM_UNAVAILABLE", ...}}
   │
   └─ HTTP 503 Service Unavailable
```

---

## 🔗 FELDOLGOZÁSI LÁNC HIBÁI

### 1. **API Request Field Naming Inkonzisztencia**

#### Az API által küldött input_data (snake_case):
```json
{
  "decision": "BankTransferFeesGuidance",     ← API snake_case
  "reason": "Bank transfer fee guidance",      ← API snake_case
  "caseType": "BANK_TRANSFER_FEES",           ← API camelCase
  "matchedRules": ["BankTransferFeesGuidance"], ← API camelCase
  "rule_result": {...}
}
```

#### A prompt által várt {{}} helyőrzők (camelCase):
```
{{ruleDecision}}    ← Prompt expects camelCase
{{ruleReason}}      ← Prompt expects camelCase
{{caseType}}        ← OK camelCase
{{matchedRules}}    ← OK camelCase
```

**Probléma:** Az API `decision`/`reason` mezőket snake_case-ben küldi, de a prompt `ruleDecision`/`ruleReason` camelCase-ben vár.

### 2. **Hiányzó Field Mapping**

A `GenerationService._prepare_prompt_input()` (line 171-232) feldolgozza az input_data-t, de:

- ✅ `response_plan` ↔ `responsePlan` mapping létezik
- ❌ **`decision` → `ruleDecision` mapping HIÁNYZIK**
- ❌ **`reason` → `ruleReason` mapping HIÁNYZIK**

#### Szükséges mapping (hiányzik):
```python
# _prepare_prompt_input() végén kellene:
prompt_input["ruleDecision"] = (
    prompt_input.get("decision") 
    or prompt_input.get("ruleDecision")
)
prompt_input["ruleReason"] = (
    prompt_input.get("reason") 
    or prompt_input.get("ruleReason")
)
```

### 3. **Prompt Template Rigidus Validáció**

`GenerationService._render_prompt()` (line 376-383) szigorú validáció:

```python
def replace_placeholder(match: re.Match[str]) -> str:
    key = match.group(1)
    if key not in input_data:
        raise ValueError(f"Missing prompt input: {key}")  # ← Szigorú!
    ...
```

**Probléma:** Ha egyetlen {{}} helyőrzőő sem található az input_data-ban, `ValueError` jelez, amely `ModelGenerationError`-vé alakul, végül 503-as HTTP hibává.

---

## 📊 BIZONYÍTÉK: NAPLÓBEJEGYZÉSEK

### Correlation ID: 167a4d55-081a-4666-aa9e-bf55395db621

**Request JSON (kitöltött input_data):**
```json
{
  "timestamp": "2026-11-14T18:12:54.135889+00:00",
  "method": "POST",
  "path": "/generate",
  "request_body": {
    "input_data": {
      "caseType": "BANK_TRANSFER_FEES",
      "matchedRules": ["BankTransferFeesGuidance"],
      "structuredCaseState": {"fee": ["1 000 Ft"], "amount": ["100 000 Ft"]},
      "customerMessage": "OTP forint megtakarítási számlámról...",
      "reason": "Bank transfer fee guidance",              ← snake_case
      "cleanedText": "OTP forint megtakarítási számlámról...",
      "decision": "BankTransferFeesGuidance",            ← snake_case
      "bank": "OTP",
      "rule_result": {"decision": "BankTransferFeesGuidance", ...}
    }
  },
  "status_code": 503,
  "correlation_id": "167a4d55-081a-4666-aa9e-bf55395db621"
}
```

**Response:**
```json
{
  "success": false,
  "error": {
    "code": "LLM_UNAVAILABLE",
    "message": "The language model is temporarily unavailable.",
    "correlation_id": "167a4d55-081a-4666-aa9e-bf55395db621"
  }
}
```

**Belső hiba (nem naplózódik):**
```
ValueError: Missing prompt input: ruleDecision
```

---

## 🛠️ JAVÍTÁS TERV

### Lehetséges Megoldások (Prioritás Szerint)

#### 1. **A Prompt Template Javítása (QUICK FIX)**

Fájl: `prompts/v1.1.txt`

Az `{{}}` helyőrzőket az API által valóban küldött mezőnevekre módosítani.

#### 2. **GenerationService Mező Mapping Javítása (PROPER FIX)**

Fájl: `app/services/generation_service.py`

`_prepare_prompt_input()` metódusba mező normalizálás.

#### 3. **API Request Kontraktus Normalizálása (SYSTEMIC FIX)**

Normalizált camelCase API szinten.

---

## 📌 ÖSSZEFOGLALÁS

### Kritikus Hiba
| Komponens | Probléma | Severity |
|-----------|---------|----------|
| **Mező Mapping** | `decision`/`reason` → `ruleDecision`/`ruleReason` hiányzik | 🔴 CRITICAL |
| **Prompt Template** | Nem kompatibilis az API input_data formátummal | 🔴 CRITICAL |
| **Error Handling** | ValueError → 503 LLM_UNAVAILABLE (félrevezetõ) | 🟡 MAJOR |

### Javasolt Megoldás
**Option 2 (Proper):** GenerationService mező mapping hozzáadása — rugalmas, backward compatible

---

**Dokumentum vége**
