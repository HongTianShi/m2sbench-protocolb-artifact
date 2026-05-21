param(
    [string]$SdkPath = ""
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logPath = Join-Path $logDir ("local_sdk_install_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
Set-Content -LiteralPath $logPath -Value "M2S-Bench local Bloomberg SDK installer" -Encoding UTF8

function Write-Log {
    param([string]$Text)
    Write-Host $Text
    Add-Content -LiteralPath $logPath -Value $Text
}

function Invoke-NativeLogged {
    param([scriptblock]$Command, [string]$Label)
    Write-Log ""
    Write-Log ("## " + $Label)
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & $Command 2>&1
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldPreference
    }
    foreach ($line in $output) { Write-Log ([string]$line) }
    if ($null -eq $code) { $code = 0 }
    return [pscustomobject]@{ ExitCode = $code; Output = $output }
}

function Get-PythonCommand {
    $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) { return $venvPython }
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        & $py.Source -3.11 -m venv (Join-Path $PSScriptRoot ".venv")
    } else {
        $python = Get-Command python -ErrorAction SilentlyContinue
        if (-not $python) { Write-Error "No Python found. Install Python 3.11 first." }
        & $python.Source -m venv (Join-Path $PSScriptRoot ".venv")
    }
    if (-not (Test-Path -LiteralPath $venvPython)) {
        Write-Error "Could not create .venv at $venvPython"
    }
    return $venvPython
}

function Expand-SdkArchive {
    param([string]$ArchivePath)
    $safeName = [IO.Path]::GetFileNameWithoutExtension($ArchivePath) -replace '[^A-Za-z0-9_.-]', '_'
    if ($ArchivePath.EndsWith(".tar.gz", [StringComparison]::OrdinalIgnoreCase)) {
        $safeName = ([IO.Path]::GetFileName($ArchivePath) -replace '\.tar\.gz$', '') -replace '[^A-Za-z0-9_.-]', '_'
    }
    $dest = Join-Path $PSScriptRoot (Join-Path ".sdk_extract" $safeName)
    if (Test-Path -LiteralPath $dest) { Remove-Item -LiteralPath $dest -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Write-Log "Extracting SDK archive to $dest"
    if ($ArchivePath.EndsWith(".zip", [StringComparison]::OrdinalIgnoreCase)) {
        Expand-Archive -LiteralPath $ArchivePath -DestinationPath $dest -Force
    } else {
        tar -xf $ArchivePath -C $dest
    }
    return $dest
}

function Find-InstallCandidates {
    param([string[]]$Roots)
    $items = New-Object System.Collections.Generic.List[string]
    foreach ($root in $Roots) {
        if ([string]::IsNullOrWhiteSpace($root) -or -not (Test-Path -LiteralPath $root)) { continue }
        $resolved = (Resolve-Path -LiteralPath $root).Path
        if (Test-Path -LiteralPath $resolved -PathType Leaf) {
            if ($resolved -match 'blpapi.*\.(whl|zip)$' -or $resolved -match 'blpapi.*\.tar\.gz$') {
                $items.Add($resolved) | Out-Null
            }
            continue
        }
        Get-ChildItem -LiteralPath $resolved -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^blpapi.*\.(whl|zip)$' -or $_.Name -match '^blpapi.*\.tar\.gz$' } |
            ForEach-Object { $items.Add($_.FullName) | Out-Null }
        Get-ChildItem -LiteralPath $resolved -Recurse -File -Filter "setup.py" -ErrorAction SilentlyContinue |
            Where-Object { $_.DirectoryName -match 'blpapi|Python|python|APIv3' } |
            ForEach-Object { $items.Add($_.DirectoryName) | Out-Null }
        Get-ChildItem -LiteralPath $resolved -Recurse -File -Filter "pyproject.toml" -ErrorAction SilentlyContinue |
            Where-Object { $_.DirectoryName -match 'blpapi|Python|python|APIv3' } |
            ForEach-Object { $items.Add($_.DirectoryName) | Out-Null }
    }
    return $items.ToArray() | Sort-Object -Unique
}

function Test-Blpapi {
    param([string]$PythonExe)
    $result = Invoke-NativeLogged -Label "Verify blpapi import" -Command {
        & $PythonExe -c "import sys, blpapi; print(sys.executable); print('blpapi OK'); print(getattr(blpapi, '__version__', 'version unknown'))"
    }
    return ($result.ExitCode -eq 0)
}

$pythonExe = Get-PythonCommand
Invoke-NativeLogged -Label "Upgrade pip tooling" -Command {
    & $pythonExe -m pip install --upgrade pip setuptools wheel
} | Out-Null

$roots = New-Object System.Collections.Generic.List[string]
if (-not [string]::IsNullOrWhiteSpace($SdkPath)) { $roots.Add($SdkPath) | Out-Null }
$roots.Add($PSScriptRoot) | Out-Null
$roots.Add((Join-Path $env:USERPROFILE "Downloads")) | Out-Null
$roots.Add((Join-Path $env:USERPROFILE "Desktop")) | Out-Null
$roots.Add("C:\blp\API\APIv3\Python") | Out-Null
$roots.Add("C:\blp\api\APIv3\Python") | Out-Null

$candidates = Find-InstallCandidates -Roots $roots.ToArray()
Write-Log ""
Write-Log "Install candidates:"
if ($candidates.Count -eq 0) {
    Write-Log "  none found"
} else {
    foreach ($candidate in $candidates) { Write-Log ("  " + $candidate) }
}

$expandedRoots = New-Object System.Collections.Generic.List[string]
foreach ($candidate in $candidates) {
    if ($candidate -match '\.(zip|tar\.gz)$') {
        try {
            $expandedRoots.Add((Expand-SdkArchive $candidate)) | Out-Null
        } catch {
            Write-Log ("Archive extract failed: " + $_.Exception.Message)
        }
    }
}
if ($expandedRoots.Count -gt 0) {
    $more = Find-InstallCandidates -Roots $expandedRoots.ToArray()
    foreach ($item in $more) { $candidates += $item }
    $candidates = $candidates | Sort-Object -Unique
}

$installed = $false
foreach ($candidate in $candidates) {
    Write-Log ""
    Write-Log "Trying candidate: $candidate"
    $result = Invoke-NativeLogged -Label "pip install $candidate" -Command {
        & $pythonExe -m pip install $candidate
    }
    if ($result.ExitCode -eq 0 -and (Test-Blpapi $pythonExe)) {
        $installed = $true
        break
    }
}

if (-not $installed) {
    Write-Log ""
    Write-Log "Local SDK install did not produce an importable blpapi."
    Write-Log "If BDEV downloaded a zip to a different path, rerun:"
    Write-Log "  powershell -NoProfile -ExecutionPolicy Bypass -File .\INSTALL_LOCAL_BLPAPI_SDK.ps1 -SdkPath C:\path\to\downloaded_sdk.zip"
    Write-Error "blpapi is still not importable. See $logPath"
}

Write-Log ""
Write-Log "Done. Next step:"
Write-Log "  .\.venv\Scripts\python.exe run_bloomberg_adapter.py --spec request_spec.json --out-dir exports"
Write-Log "Log:"
Write-Log $logPath
