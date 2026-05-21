param(
    [string]$PythonWingetId = "Python.Python.3.11",
    [string]$BloombergPipIndex = "https://blpapi.bloomberg.com/repository/releases/python/simple/",
    [switch]$SkipWinget
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Invoke-NativeChecked {
    param(
        [scriptblock]$Command,
        [string]$Label,
        [string]$LogPath
    )
    Write-Host $Label
    Add-Content -LiteralPath $LogPath -Value ""
    Add-Content -LiteralPath $LogPath -Value ("## " + $Label)
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & $Command 2>&1
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldPreference
    }
    foreach ($line in $output) {
        $text = [string]$line
        Write-Host $text
        Add-Content -LiteralPath $LogPath -Value $text
    }
    if ($null -eq $code) { $code = 0 }
    if ($code -ne 0) {
        throw "$Label failed with exit code $code. See $LogPath"
    }
}

function Invoke-Python {
    param([object]$Python, [string[]]$Arguments)
    $allArgs = @()
    $allArgs += $Python.Args
    $allArgs += $Arguments
    & $Python.Exe @allArgs
}

function Test-Python {
    param([object]$Python)
    try {
        Invoke-Python $Python @("-c", "import sys; print(sys.executable); print(sys.version)")
        return ($LASTEXITCODE -eq 0 -or $null -eq $LASTEXITCODE)
    } catch {
        return $false
    }
}

function Find-Python {
    $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        $candidate = [pscustomobject]@{ Exe = $venvPython; Args = @() }
        if (Test-Python $candidate) { return $candidate }
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        foreach ($versionArg in @("-3.11", "-3")) {
            $candidate = [pscustomobject]@{ Exe = $py.Source; Args = @($versionArg) }
            if (Test-Python $candidate) { return $candidate }
        }
    }

    foreach ($name in @("python", "python3")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) {
            $candidate = [pscustomobject]@{ Exe = $cmd.Source; Args = @() }
            if (Test-Python $candidate) { return $candidate }
        }
    }
    return $null
}

Write-Host "M2S-Bench Bloomberg adapter Python setup"
Write-Host "Working directory: $PSScriptRoot"
$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logPath = Join-Path $logDir ("python_setup_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
Set-Content -LiteralPath $logPath -Value "M2S-Bench Bloomberg adapter Python setup" -Encoding UTF8

$python = Find-Python
if ($null -eq $python -and -not $SkipWinget) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        Write-Host ""
        Write-Host "Python was not found. Installing Python 3.11 with winget..."
        & $winget.Source install --id $PythonWingetId --exact --source winget --accept-package-agreements --accept-source-agreements --scope user
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "User-scope winget install failed; retrying without --scope user."
            & $winget.Source install --id $PythonWingetId --exact --source winget --accept-package-agreements --accept-source-agreements
        }
        $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        $env:Path = "$machinePath;$userPath"
        $python = Find-Python
    }
}

if ($null -eq $python) {
    Write-Host ""
    Write-Host "No usable Python executable was found."
    Write-Host "Fast manual route: install 64-bit Python 3.11 for Windows, enable 'Add python.exe to PATH', then rerun this script."
    Write-Error "Python setup cannot continue without Python."
}

Write-Host ""
Write-Host "Creating package-local virtual environment..."
$venvDir = Join-Path $PSScriptRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    Invoke-Python $python @("-m", "venv", $venvDir)
}
if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Error "Virtual environment was not created: $venvPython"
}

Write-Host ""
Write-Host "Installing Bloomberg Python package..."
Invoke-NativeChecked -Label "Upgrade pip tooling" -LogPath $logPath -Command {
    & $venvPython -m pip install --upgrade pip setuptools wheel
}

$installedBlpapi = $false
$bundledWheelDir = Join-Path $PSScriptRoot "offline_wheels"
if (Test-Path -LiteralPath $bundledWheelDir) {
    $bundledWheels = Get-ChildItem -LiteralPath $bundledWheelDir -Filter "blpapi*.whl" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending
    foreach ($wheel in $bundledWheels) {
        try {
            Invoke-NativeChecked -Label "Install bundled offline wheel: $($wheel.Name)" -LogPath $logPath -Command {
                & $venvPython -m pip install --force-reinstall $wheel.FullName
            }
            $installedBlpapi = $true
            break
        } catch {
            Write-Warning $_.Exception.Message
        }
    }
}

try {
    if (-not $installedBlpapi) {
        Invoke-NativeChecked -Label "Install blpapi from Bloomberg Python package index" -LogPath $logPath -Command {
            & $venvPython -m pip install --index-url=$BloombergPipIndex blpapi
        }
        $installedBlpapi = $true
    }
} catch {
    Write-Warning $_.Exception.Message
}

if (-not $installedBlpapi) {
    try {
        Invoke-NativeChecked -Label "Install blpapi from default PyPI index" -LogPath $logPath -Command {
            & $venvPython -m pip install blpapi
        }
        $installedBlpapi = $true
    } catch {
        Write-Warning $_.Exception.Message
    }
}

if (-not $installedBlpapi) {
    $localSdkCandidates = @(
        "C:\blp\API\APIv3\Python",
        "C:\blp\api\APIv3\Python",
        "C:\Bloomberg\API\APIv3\Python"
    )
    foreach ($candidate in $localSdkCandidates) {
        if (Test-Path -LiteralPath $candidate) {
            try {
                Invoke-NativeChecked -Label "Install blpapi from local Bloomberg SDK: $candidate" -LogPath $logPath -Command {
                    & $venvPython -m pip install $candidate
                }
                $installedBlpapi = $true
                break
            } catch {
                Write-Warning $_.Exception.Message
            }
        }
    }
}

if (-not $installedBlpapi -and (Test-Path -LiteralPath "C:\blp")) {
    $setupPy = Get-ChildItem -LiteralPath "C:\blp" -Recurse -Filter "setup.py" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match "\\Python\\" -or $_.DirectoryName -match "blpapi" } |
        Select-Object -First 1
    if ($setupPy) {
        $setupDir = $setupPy.DirectoryName
        try {
            Invoke-NativeChecked -Label "Install blpapi from discovered setup.py: $setupDir" -LogPath $logPath -Command {
                & $venvPython -m pip install $setupDir
            }
            $installedBlpapi = $true
        } catch {
            Write-Warning $_.Exception.Message
        }
    }
}

Write-Host ""
Write-Host "Verifying blpapi import..."
Invoke-NativeChecked -Label "Verify blpapi import" -LogPath $logPath -Command {
    & $venvPython -c "import sys, blpapi; print(sys.executable); print('blpapi OK'); print(getattr(blpapi, '__version__', 'version unknown'))"
}

Write-Host ""
Write-Host "Done. Next step:"
Write-Host "  Double-click RUN_BLOOMBERG_ADAPTER.bat"
Write-Host "Setup log:"
Write-Host $logPath
