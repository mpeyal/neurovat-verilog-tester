# Windows fallback when LibreOffice is unavailable: export only the supplied
# report using an owned, hidden Word instance. Leave user documents untouched.
param([Parameter(Mandatory=$true)][string]$DocumentPath,
      [Parameter(Mandatory=$true)][string]$PdfPath)
$ErrorActionPreference = 'Stop'
$reportInput = (Resolve-Path -LiteralPath $DocumentPath).Path
$reportOutput = [System.IO.Path]::GetFullPath($PdfPath)
$null = New-Item -ItemType Directory -Force -Path (Split-Path -Parent $reportOutput)
$wordInstance = $null
$reportDocument = $null
try {
    $wordInstance = New-Object -ComObject Word.Application
    $wordInstance.Visible = $false
    $wordInstance.DisplayAlerts = 0
    $reportDocument = $wordInstance.Documents.Open($reportInput, $false, $true, $false)
    $reportDocument.ExportAsFixedFormat($reportOutput, 17)
    Write-Output $reportOutput
} finally {
    if ($null -ne $reportDocument) {
        $reportDocument.Close(0)
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($reportDocument)
    }
    if ($null -ne $wordInstance) {
        $wordInstance.Quit()
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordInstance)
    }
}
