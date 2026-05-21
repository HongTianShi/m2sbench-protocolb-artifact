$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

Write-Host "M2S-Bench optional Bloomberg adapter"
Write-Host "Make sure Bloomberg Terminal is running and logged in."

$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logPath = Join-Path $logDir ("adapter_console_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
Set-Content -LiteralPath $logPath -Value "M2S-Bench optional Bloomberg adapter" -Encoding UTF8

function Invoke-Python {
    param([object]$Python, [string[]]$Arguments)
    $allArgs = @()
    $allArgs += $Python.Args
    $allArgs += $Arguments
    & $Python.Exe @allArgs
}

function Invoke-PythonLogged {
    param([object]$Python, [string[]]$Arguments, [string]$Label)
    Write-Host $Label
    Add-Content -LiteralPath $logPath -Value ""
    Add-Content -LiteralPath $logPath -Value ("## " + $Label)
    $allArgs = @()
    $allArgs += $Python.Args
    $allArgs += $Arguments
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & $Python.Exe @allArgs 2>&1
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldPreference
    }
    foreach ($line in $output) {
        $text = [string]$line
        Write-Host $text
        Add-Content -LiteralPath $logPath -Value $text
    }
    if ($null -eq $code) { $code = 0 }
    return [pscustomobject]@{ ExitCode = $code; Output = $output }
}

function Test-Python {
    param([object]$Python)
    try {
        Invoke-Python $Python @("--version") | Out-Host
        return ($LASTEXITCODE -eq 0 -or $null -eq $LASTEXITCODE)
    } catch {
        return $false
    }
}

function Find-Python {
    $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        $candidate = [pscustomobject]@{ Exe = $venvPython; Args = @(); Source = "package .venv" }
        if (Test-Python $candidate) { return $candidate }
    }
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        foreach ($versionArg in @("-3.11", "-3")) {
            $candidate = [pscustomobject]@{ Exe = $py.Source; Args = @($versionArg); Source = "Python launcher" }
            if (Test-Python $candidate) { return $candidate }
        }
    }
    foreach ($name in @("python", "python3")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) {
            $candidate = [pscustomobject]@{ Exe = $cmd.Source; Args = @(); Source = $name }
            if (Test-Python $candidate) { return $candidate }
        }
    }
    return $null
}

function Ensure-PythonEnvironment {
    $python = Find-Python
    if ($null -eq $python) {
        Write-Host "No Python executable found. Running SETUP_PYTHON_ENV.ps1..."
        powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "SETUP_PYTHON_ENV.ps1")
        $python = Find-Python
    }
    if ($null -eq $python) {
        Write-Error "No Python executable found after setup. See $logPath"
    }

    Write-Host ("Using Python executable: {0} {1}" -f $python.Exe, ($python.Args -join " "))
    $check = Invoke-PythonLogged $python @("-c", "import blpapi; print('blpapi OK')") "Checking Bloomberg blpapi module..."
    if ($check.ExitCode -eq 0) {
        return $python
    }

    Write-Host ""
    Write-Host "blpapi is not available in the selected Python. Running SETUP_PYTHON_ENV.ps1 once..."
    powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "SETUP_PYTHON_ENV.ps1") -SkipWinget
    $python = Find-Python
    if ($null -eq $python) {
        Write-Error "Python disappeared after setup. See $logPath"
    }
    Write-Host ("Using Python executable after setup: {0} {1}" -f $python.Exe, ($python.Args -join " "))
    $check = Invoke-PythonLogged $python @("-c", "import sys, blpapi; print(sys.executable); print('blpapi OK')") "Rechecking Bloomberg blpapi module..."
    if ($check.ExitCode -ne 0) {
        Write-Host ""
        Write-Host "blpapi still failed to import. Copy this log back for diagnosis:"
        Write-Host $logPath
        Write-Error "Bloomberg blpapi is not importable."
    }
    return $python
}

$pythonCommand = Ensure-PythonEnvironment

Write-Host "Checking Python executable..."
$pyCheck = Invoke-PythonLogged $pythonCommand @("-c", "import sys; print(sys.executable); print(sys.version)") "Checking Python executable..."
if ($pyCheck.ExitCode -ne 0) {
    Write-Error "Python command exists but did not run correctly. See $logPath"
}

Write-Host "Running adapter..."
$run = Invoke-PythonLogged $pythonCommand @("run_bloomberg_adapter.py", "--spec", "request_spec.json", "--out-dir", "exports") "Running adapter..."
if ($run.ExitCode -ne 0) {
    Write-Error "Adapter script failed. See $logPath"
}

$latestZip = Get-ChildItem -Path (Join-Path $PSScriptRoot "exports") -Filter "bloomberg_optional_adapter_*.zip" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($null -eq $latestZip) {
    Write-Error "No export zip was created. See $logPath"
}

Write-Host ""
Write-Host "Done. Copy this file back to the M2S-Bench machine:"
Write-Host $latestZip.FullName
Write-Host "Console log:"
Write-Host $logPath
