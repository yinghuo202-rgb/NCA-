param(
  [string]$Manifest = "precollection\runs\20260131_234759\outputs\final_manifest.csv",
  [string]$OutFile = "precollection\runs\20260131_234759\logs\progress.txt",
  [int]$IntervalSec = 30
)

$stale = 0
$lastLine = ""

while ($true) {
  if (Test-Path $Manifest) {
    $rows = Import-Csv $Manifest
    $total = $rows.Count
    $ok = ($rows | Where-Object { $_.subtitle_status -eq 'OK' }).Count
    $err = ($rows | Where-Object { $_.subtitle_status -eq 'ERR' }).Count
    $pending = $total - $ok - $err
    $pct = if ($total -gt 0) { [math]::Round(($ok / $total) * 100, 2) } else { 0 }

    $barLen = 30
    $filled = if ($total -gt 0) { [math]::Floor(($pct / 100) * $barLen) } else { 0 }
    $bar = ('#' * $filled).PadRight($barLen, '-')
    $line = "$(Get-Date -Format s) [$bar] $pct% OK=$ok ERR=$err PENDING=$pending TOTAL=$total"
    Set-Content -Path $OutFile -Value $line

    if ($line -eq $lastLine) {
      $stale++
    } else {
      $stale = 0
      $lastLine = $line
    }
  }

  $running = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -match '02_download_subtitles.py' }).Count -gt 0
  if (-not $running -and $stale -ge 3) {
    break
  }
  Start-Sleep -Seconds $IntervalSec
}
