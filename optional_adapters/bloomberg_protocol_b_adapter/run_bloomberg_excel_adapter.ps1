param(
    [string]$Spec = "request_spec.json",
    [string]$OutDir = "exports",
    [int]$WaitSeconds = 180,
    [switch]$DryRun,
    [switch]$UseBloombergRibbonRefresh,
    [switch]$NoPrompt
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Write-JsonFile {
    param([string]$Path, [object]$Object)
    $parent = Split-Path -Parent $Path
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    $Object | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Get-Sha256 {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) {
        return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    }
    return ""
}

function Get-Ymd {
    param([string]$DateText)
    if ([string]::IsNullOrWhiteSpace($DateText)) {
        return (Get-Date).ToString("yyyyMMdd")
    }
    return ([datetime]::Parse($DateText)).ToString("yyyyMMdd")
}

function Get-SafeSheetName {
    param([string]$Security, [string]$Field, [int]$Index)
    $name = ("{0}_{1}_{2}" -f $Index, $Security, $Field)
    $name = $name -replace '[\\\/\?\*\[\]\:]', '_'
    if ($name.Length -gt 31) {
        $name = $name.Substring(0, 31)
    }
    return $name
}

function Convert-ExcelDate {
    param([object]$Value)
    if ($null -eq $Value) { return $null }
    if ($Value -is [double] -or $Value -is [int]) {
        try { return ([datetime]::FromOADate([double]$Value)).ToString("yyyy-MM-dd") } catch { return $null }
    }
    if ($Value -is [datetime]) {
        return $Value.ToString("yyyy-MM-dd")
    }
    $text = [string]$Value
    $parsed = [datetime]::MinValue
    if ([datetime]::TryParse($text, [ref]$parsed)) {
        return $parsed.ToString("yyyy-MM-dd")
    }
    return $null
}

function Convert-Number {
    param([object]$Value)
    if ($null -eq $Value) { return $null }
    $text = ([string]$Value).Replace(",", "").Trim()
    if ([string]::IsNullOrWhiteSpace($text)) { return $null }
    $out = 0.0
    if ([double]::TryParse($text, [ref]$out)) {
        if ([double]::IsNaN($out) -or [double]::IsInfinity($out)) { return $null }
        return $out
    }
    return $null
}

function Invoke-BloombergRefresh {
    param([object]$Excel, [object]$Workbook, [int]$WaitSeconds, [bool]$UseRibbonRefresh)
    try { $Excel.CutCopyMode = $false } catch {}
    try { $Excel.Interactive = $true } catch {}
    try { $Excel.EnableEvents = $true } catch {}
    try { $Workbook.Worksheets.Item(1).Activate() | Out-Null } catch {}
    try { $Workbook.Worksheets.Item(1).Range("A1").Select() | Out-Null } catch {}

    if ($UseRibbonRefresh) {
        try { $Workbook.RefreshAll() | Out-Null } catch {}
        $macros = @(
            "RefreshEntireWorkbook",
            "RefreshAllStaticData",
            "BloombergUI.xla!RefreshEntireWorkbook",
            "BloombergUI.xla!RefreshAllStaticData"
        )
        foreach ($macro in $macros) {
            try { $Excel.Run($macro) | Out-Null } catch {}
        }
    } else {
        Write-Host "Passive Excel mode: not calling Bloomberg Ribbon refresh macros."
    }

    try { $Excel.CalculateFullRebuild() | Out-Null } catch {}
    $end = (Get-Date).AddSeconds($WaitSeconds)
    while ((Get-Date) -lt $end) {
        Start-Sleep -Seconds 5
        try { $Excel.Calculate() | Out-Null } catch {}
        try { $Excel.CalculateUntilAsyncQueriesDone() | Out-Null } catch {}
    }
}

function Read-SheetRows {
    param([object]$Sheet, [string]$Security, [string]$Field)
    $rows = New-Object System.Collections.Generic.List[object]
    $range = $Sheet.UsedRange
    $values = $range.Value2
    if ($null -eq $values) { return $rows }
    if ($values -isnot [System.Array]) {
        return $rows
    }
    $r0 = $values.GetLowerBound(0)
    $r1 = $values.GetUpperBound(0)
    $c0 = $values.GetLowerBound(1)
    $c1 = $values.GetUpperBound(1)
    if (($c1 - $c0) -lt 1) { return $rows }
    for ($r = $r0; $r -le $r1; $r++) {
        $date = Convert-ExcelDate $values.GetValue($r, $c0)
        $number = Convert-Number $values.GetValue($r, $c0 + 1)
        if ($null -ne $date -and $null -ne $number) {
            $rows.Add([pscustomobject]@{
                security = $Security
                date = $date
                field = $Field
                value = $number
            }) | Out-Null
        }
    }
    return $rows
}

function Get-SheetStatus {
    param([object]$Sheet, [string]$Security, [string]$Field, [int]$ParsedRows)
    $usedRows = 0
    $usedCols = 0
    $errorCount = 0
    $samples = New-Object System.Collections.Generic.List[string]
    try {
        $range = $Sheet.UsedRange
        $usedRows = [int]$range.Rows.Count
        $usedCols = [int]$range.Columns.Count
        $maxRows = [Math]::Min($usedRows, 2500)
        $maxCols = [Math]::Min($usedCols, 8)
        for ($r = 1; $r -le $maxRows; $r++) {
            for ($c = 1; $c -le $maxCols; $c++) {
                $text = [string]$Sheet.Cells.Item($r, $c).Text
                if ($text -like "#*") {
                    $errorCount += 1
                    if ($samples.Count -lt 5) {
                        $samples.Add(("R{0}C{1}:{2}" -f $r, $c, $text)) | Out-Null
                    }
                }
            }
        }
    } catch {
        if ($samples.Count -lt 5) {
            $samples.Add(("status_error:{0}" -f $_.Exception.Message)) | Out-Null
        }
    }
    return [pscustomobject]@{
        sheet = $Sheet.Name
        security = $Security
        field = $Field
        parsed_rows = $ParsedRows
        used_rows = $usedRows
        used_columns = $usedCols
        visible_error_cells = $errorCount
        error_samples = ($samples.ToArray() -join " | ")
    }
}

function Write-SheetStatusCsv {
    param([object[]]$Rows, [string]$Path)
    if ($Rows.Count -eq 0) {
        "sheet,security,field,parsed_rows,used_rows,used_columns,visible_error_cells,error_samples" |
            Set-Content -LiteralPath $Path -Encoding UTF8
        return
    }
    $Rows | Export-Csv -LiteralPath $Path -NoTypeInformation -Encoding UTF8
}

function Write-WideCsv {
    param([object[]]$Rows, [string]$Path)
    if ($Rows.Count -eq 0) {
        "date" | Set-Content -LiteralPath $Path -Encoding UTF8
        return
    }
    $columns = $Rows | ForEach-Object { "$($_.security)|$($_.field)" } | Sort-Object -Unique
    $byDate = @{}
    foreach ($row in $Rows) {
        if (-not $byDate.ContainsKey($row.date)) { $byDate[$row.date] = @{} }
        $byDate[$row.date]["$($row.security)|$($row.field)"] = $row.value
    }
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add(("date," + (($columns | ForEach-Object { '"' + ($_ -replace '"', '""') + '"' }) -join ","))) | Out-Null
    foreach ($date in ($byDate.Keys | Sort-Object)) {
        $vals = foreach ($column in $columns) {
            if ($byDate[$date].ContainsKey($column)) { [string]$byDate[$date][$column] } else { "" }
        }
        $lines.Add(($date + "," + ($vals -join ","))) | Out-Null
    }
    $lines | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Read-WorkbookIntoBuffers {
    param([object]$Workbook, [object]$AllRows, [object]$SheetStatuses)
    $AllRows.Clear()
    $SheetStatuses.Clear()
    foreach ($sheet in $Workbook.Worksheets) {
        if ($sheet.Name -eq "README") { continue }
        $security = [string]$sheet.Range("E1").Value2
        $field = [string]$sheet.Range("E2").Value2
        $sheet.Columns.Item(1).NumberFormat = "yyyy-mm-dd"
        $sheet.Columns.Item(2).NumberFormat = "0.########"
        $rows = Read-SheetRows $sheet $security $field
        $SheetStatuses.Add((Get-SheetStatus $sheet $security $field $rows.Count)) | Out-Null
        foreach ($row in $rows) { $AllRows.Add($row) | Out-Null }
    }
}

$specPath = Resolve-Path -LiteralPath $Spec
$specObj = Get-Content -LiteralPath $specPath.Path -Raw -Encoding UTF8 | ConvertFrom-Json
$runId = "bloomberg_optional_adapter_excel_" + (Get-Date -Format "yyyyMMdd_HHmmss")
$exportDir = Join-Path $PSScriptRoot (Join-Path $OutDir $runId)
$rawDir = Join-Path $exportDir "raw"
$logsDir = Join-Path $exportDir "logs"
New-Item -ItemType Directory -Force -Path $rawDir, $logsDir | Out-Null
Copy-Item -LiteralPath $specPath.Path -Destination (Join-Path $exportDir "request_spec_used.json") -Force

$audit = [ordered]@{
    run_id = $runId
    mode = "excel_bdh_fallback"
    status = "started"
    timestamp_local = (Get-Date).ToString("s")
    wait_seconds = $WaitSeconds
    dry_run = [bool]$DryRun
    use_bloomberg_ribbon_refresh = [bool]$UseBloombergRibbonRefresh
    no_prompt = [bool]$NoPrompt
    errors = @()
}

$excel = $null
$workbook = $null
$allRows = New-Object System.Collections.Generic.List[object]
$sheetStatuses = New-Object System.Collections.Generic.List[object]
$workbookPath = Join-Path $exportDir "bloomberg_bdh_workbook.xlsx"

try {
    if ($DryRun) {
        $audit.status = "dry_run"
    } else {
        $excel = New-Object -ComObject Excel.Application
        $excel.Visible = $true
        $excel.DisplayAlerts = $true
        $workbook = $excel.Workbooks.Add()
        while ($workbook.Worksheets.Count -gt 1) {
            $workbook.Worksheets.Item($workbook.Worksheets.Count).Delete()
        }
        $control = $workbook.Worksheets.Item(1)
        $control.Name = "README"
        $control.Range("A1").Value2 = "M2S-Bench optional Bloomberg adapter"
        $control.Range("A2").Value2 = "Wait for Bloomberg formulas to populate, then the script exports CSV and zip files."
        $control.Range("A3").Value2 = "If cells stay at #N/A Requesting Data, press Esc in Excel, click a blank cell, wait or refresh from Bloomberg, then return to the console and press Enter."
        $startYmd = Get-Ymd $specObj.start_date
        $endYmd = Get-Ymd $specObj.end_date
        $sheetIndex = 0
        foreach ($security in $specObj.securities) {
            foreach ($field in $specObj.fields) {
                $sheetIndex += 1
                $sheet = $workbook.Worksheets.Add()
                $sheet.Name = Get-SafeSheetName $security $field $sheetIndex
                $sheet.Range("A1").Formula = ('=BDH("{0}","{1}","{2}","{3}","Dir=V","Dts=S","Sort=A")' -f $security, $field, $startYmd, $endYmd)
                $sheet.Columns.Item(1).NumberFormat = "yyyy-mm-dd"
                $sheet.Columns.Item(2).NumberFormat = "0.########"
                $sheet.Range("D1").Value2 = "security"
                $sheet.Range("E1").Value2 = $security
                $sheet.Range("D2").Value2 = "field"
                $sheet.Range("E2").Value2 = $field
            }
        }
        $control.Activate() | Out-Null
        $control.Range("A1").Select() | Out-Null
        $workbook.SaveAs($workbookPath)
        Invoke-BloombergRefresh $excel $workbook $WaitSeconds ([bool]$UseBloombergRibbonRefresh)
        Read-WorkbookIntoBuffers $workbook $allRows $sheetStatuses
        if ($allRows.Count -eq 0 -and -not $NoPrompt) {
            Write-Host ""
            Write-Host "No parsed Bloomberg rows yet."
            Write-Host "In the open Excel workbook: press Esc, click a blank cell, and wait until BDH cells show dates/prices."
            Write-Host "If needed, use Bloomberg's own refresh controls after leaving edit mode."
            Write-Host "Then return to this console and press Enter to export what is currently loaded."
            [void](Read-Host)
            try { $Excel.Calculate() | Out-Null } catch {}
            Read-WorkbookIntoBuffers $workbook $allRows $sheetStatuses
        }
        $longCsv = Join-Path $rawDir "bdh_daily_long.csv"
        $wideCsv = Join-Path $rawDir "bdh_daily_wide.csv"
        $statusCsv = Join-Path $logsDir "bdh_sheet_status.csv"
        $allRows | Export-Csv -LiteralPath $longCsv -NoTypeInformation -Encoding UTF8
        Write-WideCsv -Rows $allRows.ToArray() -Path $wideCsv
        Write-SheetStatusCsv -Rows $sheetStatuses.ToArray() -Path $statusCsv
        $workbook.Save()
        $audit.status = if ($allRows.Count -gt 0) { "ok" } else { "no_rows" }
    }
} catch {
    $audit.status = "excel_error"
    $audit.errors += [string]$_.Exception.Message
    $audit.traceback = [string]$_
} finally {
    if ($null -ne $workbook) {
        try { $workbook.Close($true) | Out-Null } catch {}
    }
    if ($null -ne $excel) {
        try { $excel.Quit() | Out-Null } catch {}
    }
}

$audit.n_long_rows = $allRows.Count
$audit.sheet_status = $sheetStatuses.ToArray()
Write-JsonFile -Path (Join-Path $exportDir "adapter_audit.json") -Object $audit
$manifest = [ordered]@{
    run_id = $runId
    adapter_name = "m2sbench_optional_bloomberg_excel_adapter"
    status = $audit.status
    mode = "excel_bdh_fallback"
    licensing_note = "Returned Bloomberg data remains subject to the license and entitlements of the account that ran this adapter."
    request = @{
        start_date = $specObj.start_date
        end_date = if ([string]::IsNullOrWhiteSpace($specObj.end_date)) { (Get-Date).ToString("yyyy-MM-dd") } else { $specObj.end_date }
        periodicity = $specObj.periodicity
        n_securities = $specObj.securities.Count
        fields = $specObj.fields
    }
    files = @{
        "raw/bdh_daily_long.csv" = Get-Sha256 (Join-Path $rawDir "bdh_daily_long.csv")
        "raw/bdh_daily_wide.csv" = Get-Sha256 (Join-Path $rawDir "bdh_daily_wide.csv")
        "logs/bdh_sheet_status.csv" = Get-Sha256 (Join-Path $logsDir "bdh_sheet_status.csv")
        "bloomberg_bdh_workbook.xlsx" = Get-Sha256 $workbookPath
        "adapter_audit.json" = ""
    }
}
Write-JsonFile -Path (Join-Path $exportDir "adapter_manifest.json") -Object $manifest
"Copy this zip back to M2S-Bench and run python scripts/import_bloomberg_optional_export.py <zip>." |
    Set-Content -LiteralPath (Join-Path $exportDir "RETURN_THIS_EXPORT.txt") -Encoding UTF8

$zipPath = Join-Path (Join-Path $PSScriptRoot $OutDir) ($runId + ".zip")
if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
Compress-Archive -LiteralPath $exportDir -DestinationPath $zipPath -Force

Write-Host "Wrote export: $zipPath"
Write-Host ("Status: {0}; rows: {1}" -f $audit.status, $allRows.Count)
if ($audit.status -ne "ok" -and $audit.status -ne "dry_run") {
    Write-Host "Open adapter_audit.json inside the zip for diagnostics."
}
