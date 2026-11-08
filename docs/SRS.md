# Szoftverkövetelmény-specifikáció (SRS)

**Projekt:** Comply Flow AI Service  
**A dokumentum célja:** a repositoryban megvalósított szolgáltatás és üzleti logika dokumentálása
**Állapot:** implementációalapú leírás; nem jövőbeli célállapot vagy jogi megfelelőségi ígéret

## 1. Bevezetés

### 1.1 Cél és hatókör

Ez a dokumentum a Comply Flow AI Service jelenlegi HTTP API-ját, feldolgozási folyamatait, modell- és adattároló-integrációit, valamint működési korlátait írja le. A szolgáltatás FastAPI-alapú backend, amely ügyfélüzeneteket tisztít, anonimizál, tényeket nyer ki és kategorizál, továbbá prompt és nyelvi modell segítségével válaszokat generál. Opcionálisan előre indexelt referencia-dokumentumokból releváns részleteket keres.

A repositoryban nincs felhasználói felület, hitelesítési vagy jogosultságkezelési réteg, adatbázis az ügyfélügyeknek, illetve önálló szabálymotor. A `ruleDecision`, `ruleReason` és `matchedRules` mezők a generálás számára átadott kontextusok; a szolgáltatás ezeket nem állítja elő szabálymotorral.

### 1.2 Fogalmak

- **Előfeldolgozás:** Unicode-normalizálás, szövegtisztítás, mintázatalapú anonimizálás, entitás- és ténykinyerés, ügytípus-besorolás.
- **Prompt:** a `prompts` könyvtár `.txt` fájljából betöltött sablon; a `{{kulcs}}` alakú változók bemeneti mezőkből töltődnek ki.
- **LLM:** a válaszgeneráláshoz használt, Hugging Face Transformers-alapú nyelvi modell.
- **RAG / retrieval:** beágyazásalapú keresés az előre indexelt referencia-dokumentumok szövegrészletei között.
- **Részleges előfeldolgozási siker:** a tisztítás és anonimizálás elkészült, de a ténykinyerés vagy a besorolás valamelyike nem állt rendelkezésre.

## 2. Rendszeráttekintés és folyamatok

### 2.1 Külső végpontok

Az alkalmazás alapértelmezés szerint az alábbi, URL-prefix nélküli végpontokat teszi elérhetővé:

- `GET /health`
- `POST /preprocessing`
- `GET /models`
- `POST /generate`

A FastAPI alapértelmezett `/openapi.json`, `/docs` és `/redoc` felületei aktívak.

### 2.2 Előfeldolgozási folyamat

1. A tisztító NFC Unicode-normalizálást végez, egységesíti a sortöréseket, soronként összevonja a whitespace-t, és legfeljebb egy üres sort hagy egymás után.
2. Az anonimizáló ismert mintákat helyettesítő címkékre cserél.
3. Üres vagy whitespace-szövegnél a folyamat üres `message` és `facts`, `null` `category` értékkel tér vissza; a modelleket nem kell meghívnia.
4. Nem üres szövegnél a ténykinyerés és kategorizálás egymástól függetlenül fut.
5. Ha valamelyik modellkomponens hibázik, az adott kimeneti mező üres marad, és a válasz figyelmeztetéssel, `partial_success: true` értékkel tér vissza. A másik komponens ettől még sikerülhet.

### 2.3 Generálási folyamat

1. A kliens vagy közvetlen `input_data` objektumot küld, vagy `message` mezőt ad meg.
2. `message` esetén a szolgáltatás előfeldolgozza az üzenetet. A generálás csak nem üres üzenet, nem részleges előfeldolgozás és nem `null` kategória esetén folytatódik.
3. Ha a retrieval be van kapcsolva és elérhető, a szolgáltatás a tisztított üzenet, kategória, kinyert tények és kapott szabálykontextus alapján keres dokumentumrészleteket.
4. A prompt szolgáltatás kiválasztja a promptot, majd behelyettesíti a bemeneti adatokat.
5. Az LLM szolgáltatás kiválasztja a modellt, validálja a paramétereket, és meghívja a modellt.
6. A generálási szolgáltatás elutasítja az üres, belső promptformátumú vagy a rendelkezésre álló szabálydöntéssel nem alátámasztott válaszokat.
7. Az API visszaadja a választ és a modell/prompt metaadatait, valamint – ha kértek – a referencia-válasszal számított token-F1 pontszámot.

## 3. Üzleti logika

### 3.1 Tisztítás és anonimizálás

Az anonimizáló a következő ismert mintázatokat helyettesíti: e-mail-cím (`[EMAIL]`), magyar telefonszám (`[PHONE]`), IBAN (`[IBAN]`), 13–19 számjegyű bankkártyaszám (`[CARD]`), címkézett TAJ-/adóazonosító-/adószám (`[ID]`), valamint bizonyos címkézett vagy „… vagyok” alakú teljes személynevek (`[NAME]`). A névfelismerés és az egyéb reguláris kifejezések nem általános személyesadat-felismerők.

Az anonimizálás csak az előfeldolgozás eredményére és az abból épített `message` alapú generálási bemenetre vonatkozik. A közvetlenül küldött `input_data`-t a szolgáltatás nem dolgozza át automatikusan. Az anonimizálás nem teljes körű adatvédelmi garancia.

### 3.2 Tény- és entitáskinyerés

A `facts` sztringkulcsokhoz rendelt sztringlisták objektuma. A célzott reguláris kifejezések:

- `PRODUCT`: bizonyos „termék …:” kezdetű, címkézett termékleírásokból;
- `PRICE`: `Ft`, `HUF` vagy `forint` utótagú összegekből.

A NER-modell entitáscímkéi közül a `PER`/`PERSON`, `LOC`/`LOCATION` és `ORG`/`ORGANIZATION` rendre `PERSON`, `LOCATION` és `ORGANIZATION` kulcsra normalizálódik. Az ismétlődő, azonos értékeket a kinyerő egy kulcson belül nem duplikálja. A modell további címkéi is megjelenhetnek saját címkeként. A termék- és árkinyerés, valamint a NER eredménye nem tekinthető teljes vagy validált adatkinyerésnek.

### 3.3 Kategorizálás

A besorolás először magyar, reguláris kifejezéseken alapuló szabályokat vizsgál, az alábbi prioritási sorrendben; az első illeszkedő szabály eredménye érvényesül:

1. `TRANSACTION_TAX`
2. `SAVINGS_ACCOUNT_INTEREST`
3. `SAVINGS_ACCOUNT_TERMS`
4. `BANK_TRANSFER_FEES`
5. `SAVINGS_ACCOUNT_FEES`
6. `BANKING`
7. `REFUND`
8. `CANCELLATION`
9. `WARRANTY`
10. `DELIVERY`
11. `DEFECTIVE_PRODUCT`
12. `GENERAL_COMPLAINT`
13. `OTHER`

A szabályok többek között tranzakciós illetékre, megtakarításokra és kamatra, banki díjakra, visszatérítésre, lemondásra, garanciára, szállításra, hibás termékre, általános panaszra és néhány információkérésre keresnek kifejezéseket. Szabálytalálat hiányában a szolgáltatás többnyelvű zero-shot osztályozót hív meg, amely ugyanezeket a kategóriákat használja. A `0,7` alatti modellpontszám `OTHER` eredményt ad. A besorolási minőség magyar panaszokon nincs reprezentatív tesztkészlettel igazolva.

### 3.4 `message` és `input_data` összeállítása

- Legalább `message` vagy `input_data` megadása szükséges. Ha mindkettő szerepel, a `message` alapú bemenet az irányadó.
- `message` esetén a `cleanedText` és `customerMessage` az anonimizált, tisztított üzenet; `caseType` a kérésben megadott `category`, ha van, egyébként az előfeldolgozás kategóriája.
- `structuredCaseState` a kérésbeli `facts`, ha megadták, egyébként az előfeldolgozás tényei.
- `ruleDecision` alapértéke `not_evaluated`; `ruleReason` alapértéke `No rule-engine decision is available.`; `matchedRules` alapértéke üres lista. Ezeket a bemeneti mezőket a kliens biztosíthatja; a backend nem futtat szabálymotort.
- A `retrievedContext` a találatok szövegéből készül, találat hiányában `Nincs kellően releváns dokumentumrészlet.`.
- Közvetlen `input_data` esetén a mezőket a kliens biztosítja; a szolgáltatás csak a prompt behelyettesítésekor használt kulcsokat ellenőrzi.
- A `reference_answer` kizárólag az értékelési pontszámhoz használatos, nem kerül a modell promptjába.

### 3.5 Prompt és válaszgenerálás

A prompt alapértelmezése `v1.1`, illetve a `DEFAULT_PROMPT_NAME` környezeti változóval módosítható. A fájl kiterjesztés nélküli neve a prompt azonosítója. A `{{ kulcs }}` változóhoz hiányzó bemeneti kulcs érvénytelen generálási bemenet; a `null` érték `Not provided` szöveggé alakul. A lista-, tuple- és szótárértékek olvasható JSON-ként kerülnek a promptba, más értékek szöveges formára alakítódnak.

A válasz nem lehet üres. A szolgáltatás elutasítja a tipikus belső döntési/választerv-fejléceket, JSON-választerveket, táblázatokat, egyes ténykinyerési mezőcímkéket és ismétlődő kimeneteket. Ha a `ruleDecision` nincs kiértékelve, a generált szöveg nem állíthatja, hogy a panasz, díj vagy terhelés jogos/jogtalan. Ezek a kimeneti ellenőrzések heurisztikusak, nem garantálják a modell minden hibájának felismerését.

### 3.6 Dokumentum-visszakeresés

A retrieval opcionális, és alapértelmezés szerint ki van kapcsolva (`RETRIEVAL_ENABLED=false`). Bekapcsolt retrieval esetén a szolgáltatás a keresési lekérdezést a panaszszövegből, kategóriából, tényekből és a rendelkezésre álló szabálymezőkből építi. Üres vektortár esetén nincs találat és beágyazás sem készül.

A `/generate` kérés `bank` mezője választja ki a szabálykészletet: `OTP` vagy `Raiffeisen`; elhagyásakor `OTP` az alapérték. A keresés kizárólag a kiválasztott bank vektorindexét kérdezi le, a két bank tartalma elkülönített tárolóútvonalon van, így a másik bank dokumentumai nem keverednek a találatok közé. A keresés legfeljebb `top_k` találatot kér; a távolságból `similarity = 1 - distance` képlettel számol hasonlóságot, majd az értéket 0 és 1 közé korlátozza. Csak az alapértelmezett vagy kérésbeli küszöböt elérő találatok maradnak meg. Alapértelmezés: `top_k=5`, minimális hasonlóság `0,7`. A generálási prompt a releváns szövegrészleteket kapja meg, az API-válasz pedig a forrás dokumentumnevét, részletindexét, szövegét, karakterpozícióit és hasonlóságát is visszaadja.

A szolgáltatás normál induláskor nem olvassa be a szabályfájlokat; azokat külön kell indexelni. A `/generate` bankválasztása a már előre indexelt, bankonként elkülönített vektortárat választja ki; ha a retrieval be van kapcsolva, csak az adott bank indexe kerül lekérdezésre. A dokumentumok indexelése külön parancs: `python -m app.services.index_documents --bank OTP` vagy `python -m app.services.index_documents --bank Raiffeisen`. Egy parancsfutás csak a kiválasztott bank dokumentumait olvassa be és indexeli.

## 4. HTTP API-szerződés

Minden végpont ugyanazt a korrelációs azonosítót használja a válasz `X-Correlation-ID` fejlécében. A kliens által küldött, legfeljebb 128 karakteres, engedélyezett karaktereket tartalmazó `X-Correlation-ID` elfogadható; egyébként a szolgáltatás UUID-t hoz létre.

### 4.1 `GET /health`

- **Bemenet:** nincs.
- **Siker:** `200 {"status":"ok"}`.
- **Jelentés:** az alkalmazás HTTP-rétege válaszol; az endpoint nem ellenőrzi az LLM, az előfeldolgozó modellek, a promptok vagy a vektortár állapotát.

### 4.2 `POST /preprocessing`

- **Kérés:** `{"message": string}`.
- **Sikeres válasz:** `PreprocessingResponse`:

```json
{
  "success": true,
  "message": "Az anonimizált és tisztított üzenet",
  "facts": {"PRODUCT": ["ASUS TUF A16 laptop"], "PRICE": ["419000 Ft"]},
  "category": "DEFECTIVE_PRODUCT",
  "partial_success": false,
  "warnings": [],
  "warning_details": {}
}
```

- `category` a 3.3. szakasz kategóriáinak egyike vagy `null`.
- Részleges eredménynél a `warnings` a `FACT_EXTRACTION_UNAVAILABLE` és/vagy `CLASSIFICATION_UNAVAILABLE` kódot tartalmazza; a `warning_details` ugyanazokhoz a kódokhoz a kivétel típusát rendeli. Ilyenkor `partial_success: true`, és a megfelelő `facts` üres, illetve `category` null.
- Üres vagy whitespace bemenetnél az anonimizált `message` és a `facts` üres, a `category` null; a válasz nem részleges siker.
- **Hibák:** `422 VALIDATION_ERROR` hibás kérésnél; `503 PREPROCESSING_UNAVAILABLE`, ha a pipeline vagy modell nem érhető el; `504 PREPROCESSING_TIMEOUT` tényleges `TimeoutError` esetén; `500 INTERNAL_SERVER_ERROR` más feldolgozási hibánál.

### 4.3 `GET /models`

- **Bemenet:** nincs.
- **Siker:** `{"success":true,"models":[{"name":"qwen2.5-0.5b-instruct","is_default":true}, ...]}`.
- A lista az alkalmazásban konfigurált modellkulcsokat adja vissza; az `is_default` jelzi az alapértelmezett modellt.
- **Hiba:** `503 LLM_UNAVAILABLE`, ha az LLM szolgáltatás nincs konfigurálva.

### 4.4 `POST /generate`

**Kérésmezők:**

| Mező | Típus / korlátozás | Funkció |
|---|---|---|
| `input_data` | tetszőleges JSON-objektum, opcionális | Közvetlen promptbemenet. |
| `message` | sztring, opcionális | Előfeldolgozott ügyfélüzenet. |
| `bank` | `OTP` vagy `Raiffeisen`; alapértelmezés: `OTP` | A retrievalhez használt, bankonként elkülönített szabálykészlet kiválasztása. |
| `facts` | sztringlistákat tartalmazó objektum, opcionális | `message` esetén felülírja a kinyert tényeket. |
| `category` | megengedett kategória, opcionális | `message` esetén felülírja a besorolást. |
| `rule_decision` | sztring, opcionális | `ruleDecision` promptmező; a szolgáltatás nem értékeli ki. |
| `rule_reason` | sztring, opcionális | `ruleReason` promptmező. |
| `matched_rules` | sztringlista, alapértelmezés: `[]` | `matchedRules` promptmező. |
| `retrieval_top_k` | pozitív egész vagy `null` | Kérésenkénti találatszám-felülírás. |
| `retrieval_min_similarity` | 0 és 1 közötti szám vagy `null` | Kérésenkénti hasonlóságküszöb. |
| `prompt_name` | nem üres sztring vagy `null` | Promptazonosító; alapértelmezés a konfigurált prompt. |
| `model_name` | nem üres sztring vagy `null` | Modellkulcs; alapértelmezés a konfigurált modell. |
| `model_parameters` | objektum vagy `null` | Generálási paraméter-felülírások: `max_new_tokens`, `do_sample`, `temperature`, `top_p`. |
| `reference_answer` | nem üres sztring vagy `null` | Opcionális token-F1 összehasonlításhoz; nem kerül a promptba. |

Legalább a `message` vagy az `input_data` szükséges. A kérés ismeretlen legfelső szintű mezője validációs hibát ad. A `model_parameters` objektum ismeretlen paramétere a szolgáltatási validáció során `400 MODEL_PARAMETERS_INVALID` hibát ad.

**Sikeres válaszmezők:**

- `success: true`, `response`: nem üres, ügyfélnek szánt szöveg;
- `prompt_version`, `model_name`, `model_id` és a tényleges `model_parameters`;
- `reference_similarity` és `reference_similarity_metric`, amelyek referencia nélkül `null` értékűek, referencia esetén a metrika `token_f1`;
- `sources`: a retrieval forrástalálatai; retrieval nélkül üres lista.

A token-F1 a kis-/nagybetűtől független, NFKC-normalizált szavak multihalmazának átfedését méri 0–1 között. Lexikai hasonlóság, nem ténybeli vagy jogi helyességi mérőszám.

**Jelentősebb hibák:**

| HTTP | Hibakód | Feltétel |
|---|---|---|
| `400` | `GENERATION_INPUT_INVALID` | Üres/whitespace üzenet vagy hibás promptbemenet. |
| `400` | `MODEL_NOT_SUPPORTED` | Ismeretlen modellkulcs. |
| `400` | `MODEL_PARAMETERS_INVALID` | Nem támogatott vagy érvénytelen generálási paraméter. |
| `404` | `PROMPT_VERSION_NOT_FOUND` | A kért prompt nincs betöltve. |
| `422` | `VALIDATION_ERROR` | Kérésmező- vagy típushiba. |
| `503` | `GENERATION_UNAVAILABLE` | Generálási szolgáltatás nincs konfigurálva. |
| `503` | `PREPROCESSING_UNAVAILABLE` / `PREPROCESSING_INCOMPLETE` | `message` alapú generálásnál nincs pipeline, vagy az előfeldolgozás nem teljes. |
| `503` | `RETRIEVAL_UNAVAILABLE` | A retrieval hibás vagy nem érhető el. |
| `503` | `LLM_UNAVAILABLE` | A modell nem tudta kiszolgálni a kérést, vagy az eredménye nem fogadható el ügyfélválaszként. |
| `504` | `PREPROCESSING_TIMEOUT`, `RETRIEVAL_TIMEOUT` vagy `GENERATION_TIMEOUT` | Az adott művelet `TimeoutError`-ral állt le. |
| `500` | `INTERNAL_SERVER_ERROR` | Nem várt belső hiba. |

Az összes API-hibaválasz alakja: `{"success":false,"error":{"code":string,"message":string,"correlation_id":string,"details"?: [...]}}`. A `422` válasz `details` mezője a mezőútvonalat és a validációs kódot tartalmazza, nem a beküldött mezőértéket.

## 5. Modell- és külső szolgáltatásintegrációk

### 5.1 Előfeldolgozó modellek

- **NER:** `NYTK/named-entity-recognition-nerkor-hubert-hungarian`, Hugging Face Transformers `ner` pipeline, `aggregation_strategy="simple"`. A modell/tokenizáló első entitáskinyeréskor, lustán töltődik be.
- **Zero-shot osztályozó:** `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, Transformers `zero-shot-classification` pipeline. Csak akkor töltődik be, ha a magyar szabályok egyike sem illeszkedik. A modellpontszám alsó küszöbe `0.7`.
- A modellek eléréséhez szükséges csomagokat, modellletöltést, cache-t és futtatási erőforrásokat a környezetnek kell biztosítania. A modellhibák az önálló preprocessing API-ban részleges eredményt okozhatnak.

### 5.2 Generatív modellek

Az LLM-modellek közös `HuggingFaceChatModel` adaptert használnak. A Hugging Face `AutoTokenizer` és `AutoModelForCausalLM` betöltése az első generáláskor történik; a tokenizer chat template-jével készül az egyfelhasználós üzenet, a modell `generate` hívása `torch.inference_mode()` alatt fut. A válasz a bemeneti tokenek utáni kimenetből dekódolódik.

| API-modellkulcs | Alapértelmezett Hugging Face modellazonosító |
|---|---|
| `qwen2.5-0.5b-instruct` | `Qwen/Qwen2.5-0.5B-Instruct` |
| `qwen2-0.5b-instruct` | `Qwen/Qwen2-0.5B-Instruct` |
| `llama-tiny-1.1b-chat` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `llama-3.2-1b-instruct` | `meta-llama/Llama-3.2-1B-Instruct` |

A modellkulcsot kell a `model_name` mezőben megadni, nem a Hugging Face azonosítót. Alapértelmezett modell a `qwen2.5-0.5b-instruct`. A beállítások induláskor ellenőrződnek. Alapértékük `max_new_tokens=512`, `do_sample=false`, `temperature=1.0`, `top_p=1.0`. A temperature és a top-p csak bekapcsolt mintavételezéskor kerül a modellnek átadott paraméterek közé.

### 5.3 Beágyazás és vektortár

- **Beágyazó modell:** alapértelmezés szerint `intfloat/multilingual-e5-small`. A dokumentumok `passage: `, a keresési lekérdezések `query: ` prefixet kapnak. A modell tokenizált kimenetéből attention-maskolt mean pooling, majd L2-normalizálás készül.
- **Vektortár:** tartós ChromaDB `PersistentClient`, alapértelmezett gyűjteménynév `reference_documents`, cosine távolságmetrikával. Az alapértelmezett könyvtár `~/.comply-flow/vector_store`.
- A beágyazó modell szintén lustán töltődik be. A vektortár életciklusának lezárásakor az alkalmazás bezárja a klienst.
- **Dokumentumforrások:** BOM alapján UTF-16/UTF-8, ennek hiányában UTF-8, majd CP1250 kódolással olvashatók.
- **Bankonkénti fájlkészletek:** OTP esetén `OTP-Uzletszabalyzat.txt`, `2012. évi CXVI. törvény a pénzügyi.txt`, `OTP-HIRDETMÉNY.txt`; Raiffeisen esetén `2012. évi CXVI. törvény a pénzügyi.txt`, `HIRDETENY-Raiffeisen.txt`, `Lakossagi-uzletszabalyzat-Raiffeisen.txt`. Alapértelmezett könyvtáruk `~/OTP` és `~/Raiffeisen`; az `OTP_RULES_DIR` és `RAIFFEISEN_RULES_DIR` változókkal külön-külön felülírhatók. A régi `REFERENCE_DOCUMENTS_DIR` változó továbbra is az OTP könyvtár felülírásaként használható.
- **Darabolás:** az alapértelmezett részletméret 2400 karakter, az átfedés 300 karakter. A daraboló előnyben részesíti a bekezdés-, mondat- és szóhatárokat, a találatok forrás-karakterpozíciót és dokumentumon belüli indexet kapnak.
- **Indexelés:** a `DocumentIndexer` a dokumentumokat feldarabolja, beágyazza, majd dokumentumonként lecseréli a vektortárban szereplő részleteket; az elavult részleteket törli. A CLI az eredményül indexelt dokumentum- és darabszámot kiírja.

## 6. Adatok és konfiguráció

### 6.1 Kategóriaértékek

Az előfeldolgozási kategória literáljai:

`TRANSACTION_TAX`, `SAVINGS_ACCOUNT_INTEREST`, `SAVINGS_ACCOUNT_TERMS`, `BANK_TRANSFER_FEES`, `SAVINGS_ACCOUNT_FEES`, `BANKING`, `REFUND`, `DEFECTIVE_PRODUCT`, `DELIVERY`, `WARRANTY`, `CANCELLATION`, `GENERAL_COMPLAINT`, `OTHER`.

### 6.2 Környezeti változók

| Változó | Alapérték / funkció |
|---|---|
| `PROMPT_DIR` | Repository `prompts` könyvtára; a prompt `.txt` fájlok helye. |
| `DEFAULT_PROMPT_NAME` | `v1.1`; alapértelmezett promptazonosító. |
| `DEFAULT_LLM_MODEL` | `qwen2.5-0.5b-instruct`; alapértelmezett API-modellkulcs. |
| `QWEN2_5_MODEL_NAME` | `Qwen/Qwen2.5-0.5B-Instruct`; ha nincs megadva, `QWEN_MODEL_NAME` is használható. |
| `QWEN2_MODEL_NAME` | `Qwen/Qwen2-0.5B-Instruct`. |
| `LLAMA_MODEL_NAME` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. |
| `LLAMA3_2_MODEL_NAME` | `meta-llama/Llama-3.2-1B-Instruct`. |
| `LLM_MAX_NEW_TOKENS`, `LLM_DO_SAMPLE`, `LLM_TEMPERATURE`, `LLM_TOP_P` | Közös generálási alapértékek; rendre `512`, `false`, `1.0`, `1.0`. |
| `QWEN2_5_*`, `QWEN2_*`, `LLAMA_TINY_*`, `LLAMA3_2_*` | Modellcsaládonként felülírható generálási változók: `_MAX_NEW_TOKENS`, `_DO_SAMPLE`, `_TEMPERATURE`, `_TOP_P`. |
| `ZERO_SHOT_CLASSIFIER_MODEL` | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`. |
| `RETRIEVAL_ENABLED` | `false`; retrieval bekapcsolása. |
| `REFERENCE_DOCUMENTS_DIR` | Régi, OTP-re vonatkozó könyvtárbeállítás; elsődlegesen az `OTP_RULES_DIR` használandó. |
| `OTP_RULES_DIR` | OTP szabályfájlok könyvtára; alapértelmezés `~/OTP`. |
| `RAIFFEISEN_RULES_DIR` | Raiffeisen szabályfájlok könyvtára; alapértelmezés `~/Raiffeisen`. |
| `EMBEDDING_MODEL_NAME` | `intfloat/multilingual-e5-small`. |
| `RETRIEVAL_TOP_K` | `5`; alapértelmezett keresési találatszám. |
| `RETRIEVAL_MIN_SIMILARITY` | `0.7`; alapértelmezett hasonlósági küszöb. |
| `VECTOR_STORE_DIR` | ChromaDB gyökérkönyvtára; OTP index közvetlenül itt, Raiffeisen index a `raiffeisen` alkönyvtárban van. Alapértelmezés `~/.comply-flow/vector_store`. |
| `REQUEST_LOG_FILE` | `logs/requests.jsonl`; relatív érték a repository gyökérkönyvtárához viszonyul. |

A promptkönyvtárnak léteznie kell, legalább egy nem üres `.txt` fájlt kell tartalmaznia, és a beállított alapértelmezett promptnak is léteznie kell. Ha a prompt-konfiguráció betöltése hibázik, az alkalmazás ettől még elindulhat, de a generálási szolgáltatás nem konfigurált. Hiányzó vagy hibás szabályfájl esetén a külön indexelő parancs hibával leáll; az API nem tölti be automatikusan a forrásfájlokat. A retrieval bekapcsolása külön vektortárat és modellfüggőségeket igényel.

### 6.3 Adattárolás és naplózás

Az alkalmazás nem tárol ügyfélügyeket vagy generálási eredményeket adatbázisban. A ChromaDB tartósan referencia-dokumentumrészleteket és beágyazásokat tárol.

A korrelációs naplózás a kérés kezdetét/befejezését és a hibakódokat rögzíti. Emellett a `RequestFileLoggingMiddleware` alapértelmezés szerint JSONL-fájlba írja a HTTP-kérések törzsét is (`request_body`), a metódussal, útvonallal, query stringgel, státusszal és korrelációs azonosítóval együtt; választestet nem ír. Ezért a nyers, még anonimizálatlan ügyfélüzenet is bekerülhet a kérésnaplóba. A fájl elérési útja `REQUEST_LOG_FILE` változóval állítható, és a napló megőrzését, hozzáférését, törlését, illetve adatvédelmi kezelését az üzemeltetőnek kell meghatároznia.

## 7. Nem funkcionális korlátok és nyitott kérdések

- A hitelesítés, jogosultságkezelés, rate limiting, SLA, válaszidő- és terhelési célérték nincs implementálva vagy meghatározva.
- A `/health` nem függőség-ellenőrző readiness/liveness vizsgálat.
- A személyes adatokat felismerő reguláris kifejezések nem teljes körűek; a request-fájl napló a nyers kérést is tárolja.
- A modellminőség, magyar kategorizálási pontosság, válaszok tényszerűsége és jogi megfelelősége nincs elfogadási adatkészlettel igazolva.
- A generáláshoz, NER-hez, osztályozáshoz és beágyazáshoz nincs általános, konfigurált inference-timeout; a `504` csak észlelt `TimeoutError` esetén keletkezik.
- A modellletöltés, tárhely-, CPU/GPU-igény és válaszidő a futtatási környezettől függ. A betöltés és következtetés helyi Hugging Face/Transformers-függőségre épül.
- A retrieval találata nem bizonyítja, hogy a generált válasz helyes; a token-F1 pedig csak szöveges egyezést jelez.
- A besorolás szabályai heurisztikusak és sorrendérzékenyek; a zero-shot modell küszöbe, címkéi és pontossága üzleti validációt igényel.

## 8. Elfogadási alap

A jelenlegi implementáció a specifikációnak megfelelő, ha:

- a négy dokumentált HTTP-végpont a leírt szerződés és hibakezelés szerint működik;
- az előfeldolgozás a tisztítási, anonimizálási, ténykinyerési, kategorizálási és részleges sikerre vonatkozó szabályokat alkalmazza;
- `message` alapú generálás a teljes előfeldolgozási eredményt, a promptot, a kiválasztott LLM-et és – ha be van kapcsolva – a retrieval eredményeit használja;
- a promptbemenet, modellparaméterek, válaszkimenet és opcionális token-F1 a fent leírt módon validálódik;
- a konfigurálatlan vagy elérhetetlen függőségek a megfelelő hibával vagy előfeldolgozási figyelmeztetéssel jelennek meg;
- a dokumentum nem állít nem létező szabálymotort, teljes körű anonimizálást, biztonsági funkciót, válaszminőséget vagy automatikus dokumentumindexelést.
