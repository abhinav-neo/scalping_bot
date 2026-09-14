$root = "E:\MyDevelopment\GitHub\scalping_bot"
Set-Location $root
$env:PYTHONPATH = $root
Get-Content .env | Where-Object { $_ -match "^[A-Z_]+=" -and $_ -notmatch "^\s*#" } |
  ForEach-Object { $p = $_ -split "=",2; [Environment]::SetEnvironmentVariable($p[0].Trim(), $p[1].Trim(), "Process") }
& "$root\.venv\Scripts\python.exe" -m app.monitor
