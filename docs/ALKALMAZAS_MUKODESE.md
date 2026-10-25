# Az alkalmazás működése

## Áttekintés

A Comply Flow AI Service egy FastAPI-alapú backend ügyfélüzenetek előfeldolgozására és nyelvi modellel történő válaszgenerálására. A kliensek HTTP API-n keresztül érik el; az alkalmazás nem tartalmaz felhasználói felületet vagy adatbázist.

Az alkalmazás három fő funkciót tesz elérhetővé:

- `GET /health`: egyszerű állapotjelzés.
- `POST /preprocessing`: egy üzenet tisztítása, anonimizálása, entitásainak kinyerése és kategorizálása, ha az előfeldolgozó pipeline be van állítva.
- `GET /models` és `POST /generate`: a konfigurált modellek listázása, illetve válasz generálása.

## A kérés útja

### Előfeldolgozás

A `POST /preprocessing` végpont egy `message` szöveget vár. A beállított pipeline a következő lépéseket végzi:

1. Normalizálja és megtisztítja a szöveget.
2. Regexalapú mintákkal anonimizál bizonyos személyes adatokat.
3. Entitásokat nyer ki az üzenetből, és ténylistát készít.
4. Kategóriát rendel az üzenethez.

A válaszban az átalakított `message`, a `facts` és a `category` szerepel. Ha valamelyik modellkomponens nem elérhető, a válasz `partial_success: true` értékkel és gépi `warnings` kódokkal jelzi a részleges eredményt. Üresre tisztított üzenet esetén a ténylista üres, a kategória pedig `null`.

Az alapértelmezett `app = create_app()` hívás létrehozza a `StateBuilder`-t, ezért a preprocessing route elérhető. Egyedi `create_app()` használatakor a `StateBuilder` injektálható; ha az explicit `None`, a végpont `503 PREPROCESSING_UNAVAILABLE` választ ad.

Az előfeldolgozó komponensek szerepe:

| Komponens | Mit csinál? | Használja a generálás automatikusan? |
|---|---|---|
| `MessageCleaner` | NFC Unicode-normalizálás, sortörések egységesítése, soron belüli whitespace összevonása és szélső whitespace eltávolítása. | Nem. A `StateBuilder` hívja az előfeldolgozási kérésben. |
| `MessageAnonymizer` | Regexekkel ismert e-mail-, magyar telefonszám-, IBAN-, kártyaszám-, egyes adó-/TAJ-azonosító- és névmintákat helyettesít, például `[EMAIL]` értékkel. A névminta felismeri az „a nevem Kovács Anna” és „Kovács Anna vagyok” formákat is. | Nem. Csak a preprocessing pipeline hívja. |
| `FactExtractor` | A NER-modell találataiból címkézett tényeket állít össze; többek között a személy-, hely- és szervezetcímkéket egységesíti. | Nem. A preprocessing válasz része, a kliensnek kell továbbadnia. |
| `NERModel` | A `NYTK/named-entity-recognition-nerkor-hubert-hungarian` Hugging Face modellt használja entitásfelismerésre. | Nem. A `FactExtractor` használja. |
| `MessageClassifier` | A `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` többnyelvű NLI-modellel zero-shot kategóriabesorolást végez hét magyar nyelvű kategórialeírás alapján. | Igen, az alapértelmezett `StateBuilder` részeként; a modell lustán, az első osztályozásnál töltődik be. |
| `StateBuilder` | A cleaner → anonymizer → extractor → classifier sorrendben futtatja a pipeline-t. A kinyerés és kategorizálás már az anonimizált szöveget kapja; modellhiba esetén részleges eredményt és figyelmeztetést ad. | Igen. A `/preprocessing` végpont használja; alapértelmezés szerint az alkalmazás létrehozza. |

Az alapértelmezett `create_app()` létrehozza és beköti a `StateBuilder`-t. A NER és a zero-shot classifier modellek az első preprocessing kéréskor töltődnek be. Ha valamelyik modell nem elérhető, a végpont a többi sikeresen végrehajtott lépés eredményét `partial_success` jelöléssel adja vissza.

A zero-shot modell többnyelvű NLI-adatokon tanult, de a magyar panaszokra adott kategóriaminősége nincs a projektben validálva. Éles használat előtt magyar, címkézett panaszokon kell mérni; a modell nem helyettesíti a saját adatokon végzett minőségellenőrzést.

### Válaszgenerálás

A `POST /generate` kétféle bemenetet fogad:

- `message`: egyszerű üzenet; ebből az API alapértelmezett esetadatokat állít elő. A szöveg ügyfélüzenetként kerül átadásra, a kategória alapértelmezés szerint `OTHER`, a ténylista üres.
- `input_data`: tetszőleges kulcs-érték objektum, amely közvetlenül a prompt változóit tölti ki.

Ha mindkettő meg van adva, a `message` alapján felépített adatok kerülnek felhasználásra. A generálás önmagában nem hívja meg az előfeldolgozást: ha annak eredményeit is használni kell, a kliensnek kell azokat a generálási bemenetbe átadnia.

Ha a kérés a `message` mezőt használja, a request modell a következő promptadatokat állítja elő: `customerMessage` és `cleanedText` értéke maga az eredeti üzenet; `caseType` a megadott kategória vagy `OTHER`; `structuredCaseState` a megadott `facts` vagy üres objektum; a `ruleDecision`, `ruleReason`, `matchedRules` és `responsePlan` pedig rögzített alapértéket kap. Ebben az útvonalban sem történik tisztítás, anonimizálás, NER vagy automatikus kategorizálás.

Ha a kliens a preprocessing eredményét is szeretné felhasználni, külön meghívja a `/preprocessing` végpontot, majd az eredményt megfelelő kulcsokkal az `input_data` objektumban adja át a `/generate` kérésnek. A prompt csak azokat a mezőket használja, amelyekre a betöltött `.txt` sablonban `{{kulcs}}` helyőrző hivatkozik.

A feldolgozás menete:

1. A rendszer kiválasztja a kért promptot, vagy a `v1.1` alapértelmezett promptot.
2. A prompt `{{kulcs}}` alakú változóit behelyettesíti a bemeneti adatokból. Hiányzó kulcs esetén hiba keletkezik; a lista- és objektumértékek JSON-ként kerülnek a promptba.
3. Kiválasztja a kért modellt, vagy az alapértelmezettet.
4. Meghívja a Hugging Face-alapú modellt, majd a nem üres választ a használt prompt és modell metaadataival együtt visszaadja.
5. Ha a kérés `reference_answer` mezőt is tartalmaz, a rendszer kiszámítja a generált válasz token-szintű F1-egyezését a referencia-válasszal.

A `reference_answer` opcionális, nem kerül be a modell promptjába. Megadásakor a válasz `reference_similarity` mezője 0 és 1 közötti token-F1 pontszámot tartalmaz (1 = teljes tokenegyezés), a `reference_similarity_metric` pedig `token_f1`. A mérés kis-/nagybetűtől független, Unicode-normalizált szótokeneket és ismétlődő szavakat is figyelembe vesz. Ez lexikai egyezés, nem szemantikai vagy tényszerű pontosság: parafrázisnál alacsony lehet, azonos szavak mellett pedig nem igazolja a tények helyességét. Referencia nélkül mindkét mező `null`.

A modellt és a tokenizálót az alkalmazás lusta módon, az első generálás során tölti be. Ezért az első kérés hosszabb ideig tarthat, és a modell elérhetősége, letöltése, valamint a futtatáshoz szükséges erőforrások a környezettől függenek.

### API-válaszok és hibák

A sikeres `/preprocessing` válasz tartalmazza a `success: true`, `partial_success`, `warnings`, `warning_details`, `message`, `facts` és `category` mezőket. A `partial_success: true` azt jelzi, hogy a tisztítás/anonimizálás sikerült, de egy modellkomponens nem adott eredményt. A `warning_details` kizárólag kivételtípusokat tartalmaz, nem kivételüzenetet vagy fájlútvonalat. A `/models` válasz ugyanígy sikerjelzőt tartalmaz. A `/generate` válasz mezői:

```json
{
  "success": true,
  "response": "A modell által előállított szöveg",
  "prompt_version": "v1.1",
  "model_name": "qwen2.5-0.5b-instruct",
  "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
  "model_parameters": {
    "max_new_tokens": 512,
    "do_sample": false
  },
  "reference_similarity": 0.8,
  "reference_similarity_metric": "token_f1"
}
```

Példa referencia-választ tartalmazó kérésre: `{"message":"Hol van a rendelésem?","reference_answer":"A rendelését hamarosan kézbesítjük."}`.

Hiba esetén nincs HTTP 200-as sikernek álcázott válasz. A hibaséma:

```json
{
  "success": false,
  "error": {
    "code": "PROMPT_VERSION_NOT_FOUND",
    "message": "The requested prompt version was not found.",
    "correlation_id": "..."
  }
}
```

A kérés `X-Correlation-ID` fejlécét a szerver csak biztonságos karakterekkel és legfeljebb 128 karakterrel fogadja el; más esetben új azonosítót generál. Az azonosítót válaszfejlécben is visszaadja. A naplózás útvonalat, metódust, státuszt, korrelációs azonosítót és szükség esetén kivételtípust rögzít, de nem naplózza a kérés vagy válasz törzsét.

| Eset | HTTP-státusz | Hibakód |
|---|---:|---|
| Hibás JSON vagy mezővalidáció | `422` | `VALIDATION_ERROR` |
| Hiányzó generálási bemenet vagy érvénytelen promptadat | `400` | `GENERATION_INPUT_INVALID` |
| Nem támogatott modell vagy modellparaméter | `400` | `MODEL_NOT_SUPPORTED` / `MODEL_PARAMETERS_INVALID` |
| Nem található promptverzió | `404` | `PROMPT_VERSION_NOT_FOUND` |
| Modell, prompt vagy preprocessing függőség nem érhető el | `503` | végponttól függő `*_UNAVAILABLE` |
| A modell vagy pipeline ténylegesen `TimeoutError` kivételt jelez | `504` | `GENERATION_TIMEOUT` / `PREPROCESSING_TIMEOUT` |
| Nem várt belső hiba | `500` | `INTERNAL_SERVER_ERROR` |

Az alkalmazás alapértelmezés szerint a repository `prompts` könyvtárából tölti be a promptokat, így külön `PROMPT_DIR` beállítás nélkül is elindul a generálási szolgáltatás. Egyedi könyvtár a `PROMPT_DIR` változóval adható meg. Ha a könyvtár vagy az alapértelmezett prompt nem érhető el, az alkalmazás naplózza a konfigurációs hibát, elindul, a `/generate` pedig strukturált `503` választ ad. A `/health` továbbra is egyszerű folyamat-állapotjelzés.

## API-példák

Egyszerű generálási kérés:

```json
{
  "message": "Szeretném megtudni, hogyan kezelik a személyes adataimat.",
  "facts": {
    "language": ["hu"]
  },
  "category": "GENERAL_COMPLAINT",
  "prompt_name": "v1.1",
  "model_name": "qwen2.5-0.5b-instruct"
}
```

Az opcionális `prompt_name` és `model_name` elhagyható; ilyenkor az alapértelmezett értékek érvényesülnek. A válasz például:

```json
{
  "success": true,
  "response": "A modell által előállított szöveg",
  "prompt_version": "v1.1",
  "model_name": "qwen2.5-0.5b-instruct",
  "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
  "model_parameters": {
    "max_new_tokens": 512,
    "do_sample": false
  }
}
```

Elérhető modellek lekérése:

```http
GET /models
```

Egészségjelzés lekérése:

```http
GET /health
```

A `/health` csak azt jelzi, hogy az alkalmazás válaszol; nem ellenőrzi a modell vagy az előfeldolgozó működőképességét. Az API pontos sémái és a hibaválaszok a FastAPI OpenAPI dokumentációjában is megtekinthetők.

## Indítás és konfiguráció

A függőségeket és az alkalmazást ugyanazzal a projekt virtuális környezettel használd. Ha a `.venv` még nem létezik, egyszer hozd létre, majd telepítsd bele a csomagokat:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

A fejlesztői szerver indítása PowerShellből:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Ne indítsd az alkalmazást egy másik, globális `python` interpreterrel: annak külön csomagkörnyezete van, és hiányozhat belőle például a `transformers`, ami a NER és a zero-shot kategorizáló betöltéséhez szükséges.

A promptkönyvtár alapértelmezett helye a repository `prompts` könyvtára, és tartalmazza a `v1.1.txt` fájlt. Egyedi promptkészlet használatához állítsd be a `PROMPT_DIR` változót, például `$env:PROMPT_DIR = "C:\utvonal\a\promptokhoz"`. A könyvtárnak léteznie kell, és legalább egy nem üres `.txt` fájlt kell tartalmaznia. A fájlnév kiterjesztés nélküli része a prompt azonosítója; például a `v1.1.txt` azonosítója `v1.1`. Az alkalmazás induláskor beolvassa a promptokat, és ellenőrzi az alapértelmezett prompt meglétét. A hiányzó vagy nem olvasható konfiguráció naplóbejegyzést és a generálási végponton `503` választ eredményez; a `/health` ettől még válaszol.

A generálás alapértelmezett promptazonosítója a `DEFAULT_PROMPT_NAME` változóval állítható be; alapértéke `v1.1`. Az API-kérés `prompt_name` mezője ezt felülírhatja. Az induláskor a szolgáltatás ellenőrzi, hogy az alapértelmezett prompt megtalálható-e a betöltött promptok között.

Az alapértelmezett modellazonosító `qwen2.5-0.5b-instruct`, amit a `DEFAULT_LLM_MODEL` változóval lehet felülírni. A kliens a kérés `model_name` mezőjével választhat a regisztrált modellek közül. A Hugging Face modellazonosítók a `QWEN2_5_MODEL_NAME`, `QWEN_MODEL_NAME`, `QWEN2_MODEL_NAME`, `LLAMA_MODEL_NAME` és `LLAMA3_2_MODEL_NAME` változókkal konfigurálhatók.

A generálási beállítások alapértékei: `LLM_MAX_NEW_TOKENS` (`512`), `LLM_DO_SAMPLE` (`false`), `LLM_TEMPERATURE` (`1.0`) és `LLM_TOP_P` (`1.0`). Ezek modellenként felülírhatók a `QWEN2_5_*`, `QWEN2_*`, `LLAMA_TINY_*` és `LLAMA3_2_*` változókkal; a változónév végződése `MAX_NEW_TOKENS`, `DO_SAMPLE`, `TEMPERATURE` vagy `TOP_P`. A zero-shot classifier modellazonosítója a `ZERO_SHOT_CLASSIFIER_MODEL` változóval cserélhető, alapértéke `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`. A kérés opcionális `model_parameters` objektuma egy-egy generálásnál felülírhatja az LLM beállításait. A pozitív `max_new_tokens`, pozitív és véges `temperature`, valamint a `(0, 1]` tartományú `top_p` kötelező. A `temperature` és `top_p` csak akkor kerül ténylegesen a modellhívásba, ha a mintavételezés engedélyezett; a sikeres válasz csak a ténylegesen alkalmazott paramétereket tartalmazza.

A Transformers modellbetöltése és inferenciája helyi, szinkron művelet. A kód nem szab rá mesterséges falióra-időkorlátot, és nem tudja megszakítani a CPU/GPU-n futó inference-t; `504` kizárólag akkor keletkezik, ha a függőség tényleges `TimeoutError`-t dob. A modellletöltéshez és futtatáshoz megfelelő hálózati hozzáférés és számítási erőforrás szükséges lehet.

**Szabálykészlet:** jelenleg nincs implementált Rule Engine vagy verziózott szabálykészlet-betöltés. A `message` alapú kérés fix `provide_information` értéket, indoklást és üres `matchedRules` listát állít elő; ez nem szabályfuttatás. Valódi ruleset-verzió alkalmazásához szükség van a szabálykészlet formátumára, a döntési szabályokra és a kívánt verziók meghatározására.

## Korlátok

- Az anonimizálás meghatározott mintákra épül; nem ismer fel minden személyes adatot, és önmagában nem jelent adatvédelmi garanciát.
- A NER és zero-shot modellek első használatakor Hugging Face letöltést vagy cache-hozzáférést igényelnek; ezek kiesése részleges eredményt okoz.
- A zero-shot kategóriabesorolás magyar panaszokon elért pontossága nincs mérve vagy elfogadási küszöbhöz kötve.
- Nincs hitelesítés, jogosultságkezelés, sebességkorlátozás vagy adatbázis.
- A válasz minősége nem garantált; a modell determinisztikus beállítással legfeljebb 512 új tokent generál.
- A `504` nem szerveroldali inference-megszakítás: a jelenlegi lokális modellfuttatásnak nincs külön beállított időkerete.

Részletes követelményekért, API-hibákért és a konfiguráció teljes modell-listájáért lásd a [Szoftverkövetelmény-specifikációt](SRS.md).