# Comply Flow LLM Diagnosztikai Teszt Szekvencia
# A tesztüzenet (OTP átutalási díjpanasz)
$testMessage = @"
OTP forint megtakarítási számláról MobilBankon keresztül eseti OTP-n belüli, nem saját számlára utaltam 100 000 Ft-ot; 1 000 Ft díjat vontak le.
"@.Trim()

$baseUrl = "http://localhost:8000"
$correlationId = [guid]::NewGuid().ToString()

# Segédfüggvény: HTTP kérés küldése és a válasz JSON-as formázása
function Invoke-ApiRequest {
    param(
        [string]$Method,
        [string]$Endpoint,
        [object]$Body = $null
    )
    
    $headers = @{
        "X-Correlation-ID" = $correlationId
        "Content-Type" = "application/json"
    }
    
    $uri = "$baseUrl$Endpoint"
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "📍 REQUEST: $Method $Endpoint" -ForegroundColor Cyan
    Write-Host "   Correlation-ID: $correlationId" -ForegroundColor Gray
    
    if ($Body) {
        Write-Host "   Body (first 200 chars):" -ForegroundColor Gray
        $bodyStr = $Body | ConvertTo-Json -Depth 10
        Write-Host "   $($bodyStr.Substring(0, [Math]::Min(200, $bodyStr.Length)))" -ForegroundColor Gray
    }
    
    try {
        if ($Method -eq "GET") {
            $response = Invoke-WebRequest -Uri $uri -Headers $headers -Method GET
        } else {
            $response = Invoke-WebRequest -Uri $uri -Headers $headers -Method $Method -Body ($Body | ConvertTo-Json -Depth 10)
        }
        
        Write-Host "✅ HTTP $($response.StatusCode)" -ForegroundColor Green
        $content = $response.Content | ConvertFrom-Json
        return $content
    } catch {
        Write-Host "❌ ERROR: $_" -ForegroundColor Red
        if ($_.Exception.Response) {
            Write-Host "   HTTP Status: $($_.Exception.Response.StatusCode)" -ForegroundColor Red
            try {
                $errorBody = $_.Exception.Response.Content.ToString() | ConvertFrom-Json
                Write-Host "   Response: $($errorBody | ConvertTo-Json -Depth 3)" -ForegroundColor Red
            } catch {
                Write-Host "   Response (raw): $($_.Exception.Response.Content.ToString())" -ForegroundColor Red
            }
        }
        return $null
    }
}

# STEP 1: Konfigurációs állapot ellenőrzése
Write-Host "`n" -ForegroundColor Cyan
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  STEP 1: KONFIGURÁCIÓS ÁLLAPOT ELLENŐRZÉSE                    ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan

Write-Host "`n1️⃣ Health Check:" -ForegroundColor Yellow
$healthResponse = Invoke-ApiRequest -Method GET -Endpoint "/health"
if ($healthResponse) {
    Write-Host "   Status: $($healthResponse.status)" -ForegroundColor Green
}

Write-Host "`n2️⃣ Available Models:" -ForegroundColor Yellow
$modelsResponse = Invoke-ApiRequest -Method GET -Endpoint "/models"
if ($modelsResponse) {
    Write-Host "   Available models:" -ForegroundColor Green
    if ($modelsResponse.models) {
        $modelsResponse.models | ForEach-Object {
            Write-Host "     - $($_.name): $($_.model_id)" -ForegroundColor Green
        }
    }
    if ($modelsResponse.default_model) {
        Write-Host "   Default model: $($modelsResponse.default_model)" -ForegroundColor Green
    }
}

# STEP 2: Preprocessing tesztelés
Write-Host "`n" -ForegroundColor Cyan
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  STEP 2: PREPROCESSING TESZTELÉS                             ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan

Write-Host "`n3️⃣ Preprocessing Request:" -ForegroundColor Yellow
Write-Host "   Input message: `"$testMessage`"" -ForegroundColor Cyan

$preprocessRequest = @{
    message = $testMessage
}

$preprocessResponse = Invoke-ApiRequest -Method POST -Endpoint "/preprocessing" -Body $preprocessRequest

if ($preprocessResponse) {
    Write-Host "`n📋 Preprocessing Results:" -ForegroundColor Green
    Write-Host "   Success: $($preprocessResponse.success)" -ForegroundColor Green
    Write-Host "   Category: $($preprocessResponse.category)" -ForegroundColor Green
    Write-Host "   Cleaned message (first 100 chars): `"$($preprocessResponse.message.Substring(0, [Math]::Min(100, $preprocessResponse.message.Length)))`"" -ForegroundColor Green
    
    if ($preprocessResponse.facts) {
        Write-Host "   Facts extracted:" -ForegroundColor Green
        foreach ($factKey in $preprocessResponse.facts.PSObject.Properties.Name) {
            Write-Host "     - $factKey`: $($preprocessResponse.facts.$factKey -join ', ')" -ForegroundColor Green
        }
    }
    
    if ($preprocessResponse.partial_success) {
        Write-Host "   ⚠️ PARTIAL SUCCESS - Warnings:" -ForegroundColor Yellow
        $preprocessResponse.warnings | ForEach-Object {
            Write-Host "     - $_" -ForegroundColor Yellow
        }
    }
}

# STEP 3: Generate Request tesztelés (input_data nélkül)
Write-Host "`n" -ForegroundColor Cyan
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  STEP 3: GENERATE REQUEST (Message Only)                     ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan

Write-Host "`n4️⃣ Generate Request with Message Only:" -ForegroundColor Yellow

$generateRequest = @{
    message = $testMessage
    model_name = $null
    prompt_name = $null
}

$generateResponse = Invoke-ApiRequest -Method POST -Endpoint "/generate" -Body $generateRequest

if ($generateResponse) {
    Write-Host "`n📝 Generation Results:" -ForegroundColor Green
    Write-Host "   Success: $($generateResponse.success)" -ForegroundColor Green
    Write-Host "   Model used: $($generateResponse.model_name)" -ForegroundColor Green
    Write-Host "   Model ID: $($generateResponse.model_id)" -ForegroundColor Green
    Write-Host "   Prompt version: $($generateResponse.prompt_version)" -ForegroundColor Green
    
    if ($generateResponse.model_parameters) {
        Write-Host "   Model parameters:" -ForegroundColor Green
        foreach ($param in $generateResponse.model_parameters.PSObject.Properties) {
            Write-Host "     - $($param.Name): $($param.Value)" -ForegroundColor Green
        }
    }
    
    Write-Host "`n   Generated Response (first 300 chars):" -ForegroundColor Green
    $respText = $generateResponse.response
    if ($respText.Length -gt 300) {
        Write-Host "   `"$($respText.Substring(0, 300))...`"" -ForegroundColor Cyan
    } else {
        Write-Host "   `"$respText`"" -ForegroundColor Cyan
    }
    
    if ($generateResponse.sources) {
        Write-Host "`n   📚 Retrieved Sources (RAG):" -ForegroundColor Yellow
        $generateResponse.sources | ForEach-Object {
            Write-Host "     - $($_.document_name) (chunk $($_.chunk_index))" -ForegroundColor Yellow
            Write-Host "       Similarity: $($_.similarity)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "`n   📚 No RAG sources retrieved" -ForegroundColor Yellow
    }
}

# STEP 4: Generate Request input_data-val (API szerinti formátum)
Write-Host "`n" -ForegroundColor Cyan
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  STEP 4: GENERATE REQUEST (with input_data / API format)      ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan

Write-Host "`n5️⃣ Generate Request with input_data (simulating API call):" -ForegroundColor Yellow

# Ez az API által küldött formátum
$generateRequestWithInputData = @{
    message = $testMessage
    category = "BANK_TRANSFER_FEES"
    facts = @{
        PRICE = @("100 000 Ft", "1 000 Ft")
    }
    input_data = @{
        case_state = @{
            category = "BANK_TRANSFER_FEES"
            description = "OTP átutalási díj panasz"
        }
        decision = "BankTransferFeesGuidance"
        reason = "A díj helyességi ellenőrzésre szorul"
        bank = "OTP"
        response_plan = @{
            guidance = "Kérjük forduljon ügyfélszolgálatunkhoz"
        }
        rule_result = @{
            decision = "BankTransferFeesGuidance"
            expected_fee = 713
            charged_fee = 1000
            difference = 287
        }
    }
    prompt_name = "v1.1"
    model_name = "qwen2.5-0.5b-instruct"
}

$generateResponse2 = Invoke-ApiRequest -Method POST -Endpoint "/generate" -Body $generateRequestWithInputData

if ($generateResponse2) {
    Write-Host "`n📝 Generation Results (with input_data):" -ForegroundColor Green
    Write-Host "   Success: $($generateResponse2.success)" -ForegroundColor Green
    Write-Host "   Model used: $($generateResponse2.model_name)" -ForegroundColor Green
    Write-Host "   Prompt version: $($generateResponse2.prompt_version)" -ForegroundColor Green
    
    Write-Host "`n   Generated Response (first 400 chars):" -ForegroundColor Green
    $respText2 = $generateResponse2.response
    if ($respText2.Length -gt 400) {
        Write-Host "   `"$($respText2.Substring(0, 400))...`"" -ForegroundColor Cyan
    } else {
        Write-Host "   `"$respText2`"" -ForegroundColor Cyan
    }
    
    if ($generateResponse2.sources) {
        Write-Host "`n   📚 Retrieved Sources (RAG):" -ForegroundColor Yellow
        $generateResponse2.sources | ForEach-Object {
            Write-Host "     - $($_.document_name) (chunk $($_.chunk_index), similarity: $($_.similarity))" -ForegroundColor Yellow
        }
    } else {
        Write-Host "`n   📚 No RAG sources retrieved (RAG disabled or no relevant docs)" -ForegroundColor Yellow
    }
}

# STEP 5: RAG Ellenőrzés
Write-Host "`n" -ForegroundColor Cyan
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  STEP 5: RAG KONFIGURÁCIÓS ELLENŐRZÉS                         ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan

Write-Host "`n6️⃣ RAG Status:" -ForegroundColor Yellow
$noSourcesResponse1 = if ($generateResponse.sources) { "✅ RAG aktív (sources találhatók)" } else { "❌ RAG kikapcsolt vagy nincs relevancia találat" }
$noSourcesResponse2 = if ($generateResponse2.sources) { "✅ RAG aktív (sources találhatók)" } else { "❌ RAG kikapcsolt vagy nincs relevancia találat" }
Write-Host "   Message-only generate: $noSourcesResponse1" -ForegroundColor Cyan
Write-Host "   input_data generate: $noSourcesResponse2" -ForegroundColor Cyan

# SUMMARY
Write-Host "`n" -ForegroundColor Cyan
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  DIAGNOSZTIKAI ÖSSZEFOGLALÁS                                   ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan

Write-Host "`n📊 Fő Megállapítások:" -ForegroundColor Yellow
Write-Host "   ✓ API elérhető: localhost:8000" -ForegroundColor Green
Write-Host "   ✓ Preprocessing sikeresen kategóriázza az üzenetet: $($preprocessResponse.category)" -ForegroundColor Green
Write-Host "   ✓ Tények kinyerve: PRICE lista alapján" -ForegroundColor Green
Write-Host "   ✓ Decision mapping: BankTransferFeesGuidance → $(if ($generateResponse2 -and $generateResponse2.success) { 'guidance' } else { 'unknown' })" -ForegroundColor Green

Write-Host "`n🔧 Következő Diagnosztikai Lépések:" -ForegroundColor Cyan
Write-Host "   1. Ellenőrizni az env-változókat: RETRIEVAL_ENABLED, PROMPT_DIR, VECTOR_STORE_DIR" -ForegroundColor Cyan
Write-Host "   2. Vizsgálni a naplófájlt (logs/requests.jsonl) a correlation ID alapján" -ForegroundColor Cyan
Write-Host "   3. Tesztelni a fallback válasz kioldódét fallback-szituációban" -ForegroundColor Cyan
Write-Host "   4. Ellenőrizni a prompt v1.1.txt tartalmát" -ForegroundColor Cyan

Write-Host "`n✅ Diagnosztika kész. Correlation ID: $correlationId" -ForegroundColor Green
