# Comply Flow LLM — P0 Hiba Javítási Javaslat

## 🔴 Probléma
`POST /generate` input_data kérésre HTTP 503 LLM_UNAVAILABLE hibát ad.

## ✅ Megoldás

### 1. Fájl: `app/services/generation_service.py`

**Helyei:** `_prepare_prompt_input()` metódus, a `response_plan` feldolgozása után (kb. 231. sor)

**Current Code (171-232. sorok):**
```python
@classmethod
def _prepare_prompt_input(
    cls,
    input_data: Mapping[str, object],
) -> tuple[dict[str, object], list[str], list[str], list[str]]:
    prompt_input = dict(input_data)
    snake_case_plan = prompt_input.get("response_plan")
    camel_case_plan = prompt_input.get("responsePlan")
    if (
        snake_case_plan is not None
        and camel_case_plan is not None
        and snake_case_plan != camel_case_plan
    ):
        raise ValueError("response_plan and responsePlan must match.")

    response_plan = snake_case_plan if snake_case_plan is not None else camel_case_plan
    if response_plan is None:
        response_plan = {}
    if not isinstance(response_plan, Mapping):
        raise ValueError("response_plan must be an object.")
    # ... resto of method ...
```

**Javasolt Fix (a `response_plan` feldolgozása után):**
```python
@classmethod
def _prepare_prompt_input(
    cls,
    input_data: Mapping[str, object],
) -> tuple[dict[str, object], list[str], list[str], list[str]]:
    prompt_input = dict(input_data)
    snake_case_plan = prompt_input.get("response_plan")
    camel_case_plan = prompt_input.get("responsePlan")
    if (
        snake_case_plan is not None
        and camel_case_plan is not None
        and snake_case_plan != camel_case_plan
    ):
        raise ValueError("response_plan and responsePlan must match.")

    response_plan = snake_case_plan if snake_case_plan is not None else camel_case_plan
    if response_plan is None:
        response_plan = {}
    if not isinstance(response_plan, Mapping):
        raise ValueError("response_plan must be an object.")

    # === NEW: Field name normalization (snake_case -> camelCase) ===
    # Handle decision / ruleDecision mapping
    snake_decision = prompt_input.get("decision")
    camel_decision = prompt_input.get("ruleDecision")
    if snake_decision is not None and camel_decision is None:
        prompt_input["ruleDecision"] = snake_decision
    elif snake_decision is not None and camel_decision is not None and snake_decision != camel_decision:
        raise ValueError("decision and ruleDecision must match.")

    # Handle reason / ruleReason mapping
    snake_reason = prompt_input.get("reason")
    camel_reason = prompt_input.get("ruleReason")
    if snake_reason is not None and camel_reason is None:
        prompt_input["ruleReason"] = snake_reason
    elif snake_reason is not None and camel_reason is not None and snake_reason != camel_reason:
        raise ValueError("reason and ruleReason must match.")
    # === END NEW ===

    validation_requirements = response_plan.get(
        "validation_requirements",
        response_plan.get("validationRequirements", {}),
    )
    if validation_requirements is None:
        validation_requirements = {}
    if not isinstance(validation_requirements, Mapping):
        raise ValueError("response_plan.validation_requirements must be an object.")

    # ... rest of method unchanged ...
```

---

## 📊 Hatás Analízis

### Fix Előnyei
✅ Backward compatible (ha már camelCase van, nem másolódik felül)  
✅ Flexible (snake vagy camel input fogadja)  
✅ Gyors implementáció (< 1 perc)  
✅ Testelhető (egyszerű unit test)

### Potenciális Side Effects
- Nincs (snake/camel mezőket hozzáadja, nem módosít meglévőt)

### Regressziós Teszt
```bash
# Test 1: message-only (existing behavior, must still work)
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "OTP átutalási díj panasz"
  }'
# Esperátum: HTTP 200 OK

# Test 2: input_data snake_case (new fix)
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "input_data": {
      "decision": "BankTransferFeesGuidance",
      "reason": "Bank transfer fee guidance",
      "caseType": "BANK_TRANSFER_FEES",
      "structuredCaseState": {"fee": ["1 000 Ft"]},
      "customerMessage": "OTP forint megtakarítási szám...",
      "matchedRules": ["BankTransferFeesGuidance"],
      "retrievedContext": "No relevant documents",
      "validationRequirements": "No special requirements"
    }
  }'
# Esperátum: HTTP 200 OK (nem 503!)

# Test 3: input_data camelCase (existing behavior, must still work)
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "input_data": {
      "ruleDecision": "BankTransferFeesGuidance",
      "ruleReason": "Bank transfer fee guidance",
      "caseType": "BANK_TRANSFER_FEES",
      "[...]": "..."
    }
  }'
# Esperátum: HTTP 200 OK
```

---

## 🔧 Alkalmazás Lépések

1. **File Módosítás:** `app/services/generation_service.py` 
   - Add field mapping code (14 sor)
   - Nem módosít más részt

2. **Test Futtatás:**
   ```bash
   pytest tests/test_api_workflow.py -v
   pytest tests/test_generation_service.py -v
   ```

3. **Manual Verification:**
   - Futtassa a regressziós teszteket

4. **Deployment:**
   - Verzió bump (pl. v0.3.1)
   - Commit message: "Fix: normalize decision/reason field names in prompt input"

---

## 📝 Git Commit

```
Fix: normalize snake_case to camelCase in generate input_data

When the API sends decision/reason in snake_case but prompts expect
ruleDecision/ruleReason in camelCase, GenerationService._render_prompt()
fails with ValueError("Missing prompt input: ruleDecision"), causing 503.

Add field normalization in _prepare_prompt_input() to map:
- decision -> ruleDecision
- reason -> ruleReason

This maintains backward compatibility (camelCase still works) while
supporting the API's current snake_case output format.

Fixes: input_data requests returning HTTP 503 LLM_UNAVAILABLE

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
```

---

## ✅ Ellenőrzési Checklist

- [ ] Code módosított
- [ ] Unit tesztek futnak (pytest)
- [ ] Regressziós teszt 1 (message-only): HTTP 200
- [ ] Regressziós teszt 2 (input_data snake_case): HTTP 200
- [ ] Regressziós teszt 3 (input_data camelCase): HTTP 200
- [ ] Manual API test: correlation ID rögzítve
- [ ] Logs/requests.jsonl: 200 status, nem 503
- [ ] Git commit + push

---

## 📌 Addiционális P1-P3 Javaslatok

### P1: Kategóriamapping Validálása
- Classifier szabályok sorrendje: BANK_TRANSFER_FEES előbb kell az általános BANKING-nél
- Regex: `r"átutal.*díj|vontak\s+le"` validálása

### P2: RAG Index Ellenőrzése
- ChromaDB indexeltsége: `python -m app.services.index_documents --bank OTP`
- Min similarity küszöb: 0.7 vagy 0.5 (tuning)

### P3: Error Handling Javítása
- ValueError/ModelGenerationError nem kell 503-as HTTP hibát adni
- Specifikus hiba kódok: PROMPT_INPUT_INVALID, GENERATION_VALIDATION_FAILED

---

**Ajánlott Prioritás:** Azonnal implementálni ezt az P0 fixet, mert KRITIKUS hiba
