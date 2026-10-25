# Szoftverkövetelmény-specifikáció (SRS)

**Projekt:** Comply Flow AI Service  
**A dokumentum célja:** a repositoryban található implementáció jelenlegi állapotának rögzítése  
**Állapot:** implementációalapú, nem célállapotot leíró specifikáció

## 1. Bevezetés

### 1.1 Cél

Ez a dokumentum a Comply Flow AI Service jelenleg megvalósított funkcióit, API-ját, működési feltételeit és ismert korlátait írja le. A specifikáció a forráskód alapján készült; nem tekintendő olyan képesség ígéretének, amelyhez a repositoryban nincs implementáció vagy konfiguráció.

### 1.2 Hatókör

A szolgáltatás egy FastAPI-alapú backend, amely fogyasztói vagy ügyfélszolgálati üzenetek előfeldolgozására, valamint prompt és nyelvi modell felhasználásával történő válaszgenerálásra ad API-t. A repository nem tartalmaz felhasználói felületet, adatbázist vagy önálló szabálymotort.

### 1.3 Fogalmak

- **Előfeldolgozás:** az üzenet tisztítása, mintázatalapú anonimizálása, entitásainak kinyerése és kategorizálása.
- **Prompt:** külső `.txt` fájlból betöltött szövegsablon; a `{{kulcs}}` formájú változók a bemeneti adatokból helyettesítődnek.
- **LLM:** a generáláshoz használt nyelvi modell.
- **SRS:** szoftverkövetelmény-specifikáció.

## 2. Rendszeráttekintés

A szolgáltatás három fő útvonalat biztosít:

1. `GET /health` – egyszerű állapotjelzés.
2. `POST /preprocessing` – üzenet előfeldolgozása, amennyiben a pipeline be van konfigurálva.
3. `GET /models` és `POST /generate` – elérhető generatív modellek listázása és válasz előállítása.

A generálás folyamata: a kérés bemenetének normalizálása, prompt kiválasztása és változóinak behelyettesítése, a kiválasztott LLM meghívása, majd a generált szöveg visszaadása. A Hugging Face modellek és tokenizálók betöltése lusta módon, az első generálási kérésnél történik.

## 3. Felhasználói szerepkörök és külső rendszerek

- **API-fogyasztó:** HTTP-kéréseket küld a health, preprocessing és generation végpontokra.
- **Üzemeltető/fejlesztő:** megadja a promptkönyvtárat és szükség esetén felülírja a modellneveket környezeti változókkal.
- **Külső modellforrás:** a Hugging Face modell- és tokenizáló-azonosítók alapján szolgáltatott artefaktumok. A szükséges modell elérhetősége és letöltése futtatási környezetfüggő.

A jelenlegi kódban nincs hitelesítési vagy jogosultságkezelési szerepkör.

## 4. Funkcionális követelmények

Az alábbi követelmények a jelenlegi implementáció viselkedését rögzítik.

| Azonosító | Követelmény | Jelenlegi állapot |
|---|---|---|
| FR-01 | A rendszernek a `GET /health` kérésre `{"status":"ok"}` választ kell adnia. | Megvalósítva; ez nem ellenőrzi az LLM vagy az előfeldolgozó rendelkezésre állását. |
| FR-02 | A `POST /preprocessing` kérésnek egy `message` szöveget kell fogadnia. | Megvalósítva Pydantic-kérésmodellel. |
| FR-03 | Az előfeldolgozó pipeline-nek normalizálnia kell a Unicode-szöveget, egységesítenie kell a sortöréseket, össze kell vonnia a soron belüli whitespace-t, és korlátoznia kell az egymást követő üres sorokat. | Megvalósítva. |
| FR-04 | Az előfeldolgozó pipeline-nek ismert személyesadat-mintázatokat helyettesítő címkékkel kell anonimizálnia. | Megvalósítva regexekkel e-mail-címekre, magyar telefonszámokra, IBAN-ra, bankkártyaszámokra, egyes magyar adó-/TAJ-azonosítókra, címkézett nevekre, valamint az „a nevem Kovács Anna” és „Kovács Anna vagyok” formájú nevekre. Nem teljes körű adatvédelmi garancia. |
| FR-05 | Az előfeldolgozó pipeline-nek entitásokat kell kinyernie, majd kategóriát kell rendelnie az üzenethez. | A default `StateBuilder` a magyar NER-Kor/HuBERT entitáskinyerőt használja; a `PER`, `LOC`, `ORG` címkéket `PERSON`, `LOCATION`, `ORGANIZATION` kulcsokra képezi le. Egy helyi próba megmutatta, hogy az entitáshatárok (például a névragok) hibásak lehetnek. Kategorizáláshoz többnyelvű NLI-modellre épülő zero-shot osztályozót használ. Modellhiba esetén részleges eredményt és figyelmeztetést ad. |
| FR-06 | Üres vagy whitespace-re redukálódó üzenet esetén a pipeline üres üzenetet, üres ténylistát és `null` kategóriát adjon vissza. | A `StateBuilder` szintjén megvalósítva. |
| FR-07 | A `POST /generate` kérés `input_data` objektumot vagy `message` mezőt fogadhat. | Megvalósítva. Ha mindkettő szerepel, a `message` alapján létrehozott bemenet az irányadó. |
| FR-08 | `message` használatakor a rendszer a szöveget ügyfélüzenetként adja át, alapértelmezett `OTHER` kategóriával, üres tényekkel, `provide_information` döntéssel, valamint professzionális magyar választervvel. | Megvalósítva a kérésmodellben. |
| FR-09 | A generálásnak a megadott promptot, vagy ennek hiányában a `v1.1` promptot kell használnia. | Megvalósítva; a promptnak betöltött `.txt` fájlként léteznie kell. |
| FR-10 | A prompt `{{kulcs}}` változóit a bemeneti adatok megfelelő értékeivel kell helyettesíteni; hiányzó kulcs esetén hibát kell jelezni. | Megvalósítva. Lista-, tuple- és szótárértékek JSON-ként kerülnek beillesztésre. |
| FR-11 | A kliens megadhatja a használni kívánt modellt, vagy elhagyhatja azt az alapértelmezett modell használatához. | Megvalósítva. |
| FR-12 | A `GET /models` végpontnak fel kell sorolnia a konfigurált modelleket és jeleznie kell az alapértelmezettet. | Megvalósítva. |
| FR-13 | A generálásnak nem üres szöveges választ kell visszaadnia. | Üres modellválasz esetén a rendszer hibát jelez. |
| FR-14 | A `POST /generate` opcionális referencia-válasszal számítson token-szintű egyezési pontszámot. | Megvalósítva kis-/nagybetűtől független token-F1-ként; ez lexikai hasonlóság, nem tényszerű pontosság. Referencia nélkül a pontszám és a metrika neve `null`. |

## 5. Külső interfészek

### 5.1 HTTP API

| Metódus és útvonal | Bemenet | Sikeres válasz | Jelentősebb hibák |
|---|---|---|---|
| `GET /health` | Nincs | `{"status":"ok"}` | A kódban nincs külön függőségi állapotvizsgálat. |
| `POST /preprocessing` | `{"message": string}` | `success`, `partial_success`, `warnings`, `warning_details`, `message`, `facts`, `category` | `422` sémahiba; `503` hiányzó pipeline/modell; `504` tényleges `TimeoutError`; `500` feldolgozási hiba. |
| `GET /models` | Nincs | `{"success": true, "models":[{"name": string, "is_default": boolean}]}` | `503`, ha az LLM szolgáltatás nincs konfigurálva. |
| `POST /generate` | `input_data` és/vagy `message`; opcionálisan `facts`, `category`, `prompt_name`, `model_name`, `model_parameters`, `reference_answer` | `success`, `response`, prompt- és modellmetaadatok, valamint `reference_similarity` és `reference_similarity_metric` | `400` érvénytelen bemenet/modell/paraméter; `404` hiányzó prompt; `422` request-validáció; `503` modellhiba; `504` tényleges timeout; `500` nem várt hiba. |

A FastAPI alapértelmezett OpenAPI és Swagger UI felülete nincs kikapcsolva.

Hibaválasz minden végponton egységes formátumú: `{"success": false, "error": {"code": string, "message": string, "correlation_id": string}}`. A request validációs hiba `details` mezője csak mezőútvonalat és validációs kódot tartalmaz, a felhasználói értéket nem. Az `X-Correlation-ID` kérésfejlécet a szolgáltatás validálja, majd a válaszfejlécben is visszaadja. A naplózás nem tartalmaz kérés- vagy választestet.

### 5.2 Generálási kérés részletei

- Az `input_data` általános JSON-objektumként adható át.
- A `message` alternatív, egyszerű bemenet. Használatakor a rendszer a `cleanedText`, `customerMessage`, `caseType`, `structuredCaseState`, `ruleDecision`, `ruleReason`, `matchedRules` és `responsePlan` kulcsokat állítja elő.
- A `facts`, `category`, `prompt_name`, `model_name` és opcionális `model_parameters` közvetlenül a `message` alapú bemenet létrehozásakor használható.
- A `reference_answer` opcionális elvárt válasz; csak az értékeléshez használjuk, a modell promptjába nem kerül be. A válasz token-F1 pontszáma 0 és 1 közé esik, és lexikai egyezést mér, nem tényszerű helyességet.
- Az `input_data` és `message` egyikének hiánya futás közben validációs hibához vezet.

## 6. Adatkövetelmények

### 6.1 Előfeldolgozási kategóriák

A válasz kategóriája az alábbi értékek egyike vagy `null` lehet:

- `REFUND`
- `DEFECTIVE_PRODUCT`
- `DELIVERY`
- `WARRANTY`
- `CANCELLATION`
- `GENERAL_COMPLAINT`
- `OTHER`

### 6.2 Tények és entitások

A `facts` mező sztringkulcsokhoz rendelt sztringlistákat tartalmaz. A személy-, hely- és szervezet-entitások címkéi rendre `PERSON`, `LOCATION` és `ORGANIZATION` értékre normalizálódnak; más modellcímkék is megjelenhetnek.

### 6.3 Tárolás

A repositoryban nincs adatbázis vagy tartós üzletiadat-tárolás implementálva. A generálási és előfeldolgozási kérések feldolgozását a kód memóriabeli műveletekkel végzi; a külső naplózásról és megőrzésről nincs specifikáció.

## 7. Nem funkcionális követelmények és korlátok

- **Adatvédelem:** az anonimizálás regexalapú és csak meghatározott mintázatokra terjed ki. Nem garantálja valamennyi személyes adat felismerését, és nem helyettesít adatvédelmi felülvizsgálatot.
- **Biztonság:** hitelesítés, jogosultságkezelés, sebességkorlátozás és titokkezelési konfiguráció nincs megvalósítva a vizsgált kódban. Éles telepítésnél ezekről külön kell gondoskodni.
- **Elérhetőség:** a `/health` válasza nem igazolja az előfeldolgozó, a promptkönyvtár vagy az LLM működőképességét.
- **Teljesítmény:** válaszidőre, áteresztőképességre, párhuzamosságra és erőforráskorlátokra nincs rögzített célérték. A modell helyi betöltése és inferenciája futtatási környezet- és hardverfüggő; a jelenlegi kód nem állít be inference-időtúllépést. A `504` csak ténylegesen észlelt `TimeoutError` esetén használatos.
- **Generálási viselkedés:** a modellhívás alapértelmezett maximuma 512 új token; a mintavételezés ki van kapcsolva (`do_sample=False`).
- **Konfigurálhatóság:** a promptkönyvtár alapértelmezés szerint a repository `prompts` könyvtára; egyedi hely a `PROMPT_DIR` változóval adható meg. A modellnevek és generálási beállítások környezeti változókkal felülírhatók, a modellkonfigurációk induláskor validálódnak.
- **Naplózás:** a kérések kezdetét és befejezését korrelációs azonosítóval naplózza; a kérés- és választestet nem írja naplóba. A kliensválaszok nem tartalmaznak belső kivételrészletet.
- **Tesztelhetőség:** az alkalmazásgyár lehetővé teszi a szolgáltatások injektálását; az API workflow tesztjei mockolt modellekkel ellenőrzik a HTTP-szerződést.

## 8. Telepítési és konfigurációs előfeltételek

- A függőségek a gyökérkönyvtár `requirements.txt` fájljában vannak rögzítve.
- A `prompts` könyvtár tartalmazza a működéshez szükséges `v1.1.txt` alapértelmezett promptot. Egyedi promptkönyvtár a `PROMPT_DIR` változóval adható meg; a könyvtárnak legalább egy nem üres `.txt` fájlt kell tartalmaznia. A fájlnév kiterjesztés nélküli része a prompt azonosítója.
- Az alkalmazás létrehozásakor a promptkönyvtár beolvasásra kerül, és a konfigurált alapértelmezett promptot ellenőrzi. Hiányzó prompt-konfiguráció esetén az alkalmazás elindul, de a generálási szolgáltatás nem elérhető, és a `/generate` `503` hibát ad.
- A nyelvi modellek konfigurációja:

| Környezeti változó | Alapértelmezett modellazonosító |
|---|---|
| `DEFAULT_LLM_MODEL` | `qwen2.5-0.5b-instruct` |
| `QWEN2_5_MODEL_NAME` vagy másodlagosan `QWEN_MODEL_NAME` | `Qwen/Qwen2.5-0.5B-Instruct` |
| `QWEN2_MODEL_NAME` | `Qwen/Qwen2-0.5B-Instruct` |
| `LLAMA_MODEL_NAME` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `LLAMA3_2_MODEL_NAME` | `meta-llama/Llama-3.2-1B-Instruct` |

- A NER és zero-shot modellek első használatkor töltenek be; hálózat/cache vagy modellhiba esetén a preprocessing válasza `partial_success: true` és `warnings` mezőkkel jelzi a kiesett lépést.
- A zero-shot `MessageClassifier` alapértelmezett modellje `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, a `ZERO_SHOT_CLASSIFIER_MODEL` változóval felülírható. A magyar panaszokon elért pontosság nincs validálva.

## 9. Ismert hiányosságok és nyitott kérdések

1. A zero-shot classifier magyar panaszokon mért pontossága, elfogadási küszöbe és kategórialeírásainak minősége nincs meghatározva.
2. Az anonimizáló mintázatok pontossági és teljességi követelményei, valamint az anonimizálás utáni adatkezelés szabályai nincsenek rögzítve.
3. Nincs meghatározva a válaszminőség mérési módja, elfogadási küszöbe vagy jogi megfelelőségi ellenőrzése.
4. Nincsenek rögzítve telepítési, megfigyelhetőségi, naplózási, adatmegőrzési, SLA- és terhelési célok.
5. A lokális modellbetöltés és inferencia megszakítására nincs külön timeout-mechanizmus.

## 10. Követelmény-elfogadási alap

A jelenlegi állapot akkor tekinthető a dokumentumnak megfelelőnek, ha:

- a `/health` a megadott állapotobjektumot adja;
- a `/models` visszaadja a konfigurált modellazonosítókat és az alapértelmezett modellt;
- egy érvényes, létező promptot és elérhető modellt használó `/generate` kérés nem üres szöveget ad vissza;
- az előfeldolgozás eredménye a leírt válaszsémát követi, ha az összes szükséges komponenst konfigurálták;
- a fent dokumentált hiányzó konfigurációk és nem implementált biztonsági, minőségi, teljesítmény- és üzemeltetési célok nem kerülnek kész képességként feltüntetésre.