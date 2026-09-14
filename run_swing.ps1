$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONPATH = $PSScriptRoot
Get-Content .env | Where-Object { $_ -match "^[A-Z_]+=" -and $_ -notmatch "^\s*#" } |
  ForEach-Object { $p = $_ -split "=",2; [Environment]::SetEnvironmentVariable($p[0].Trim(), $p[1].Trim(), "Process") }
& "$PSScriptRoot\.venv\Scripts\python.exe" -m app.swing_supervisor
