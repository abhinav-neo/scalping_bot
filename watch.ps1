# Live monitor -- tails the bot log with a status header.
$root = "E:\MyDevelopment\GitHub\scalping_bot"
Set-Location $root
$host.UI.RawUI.WindowTitle = "SwingBot Monitor"
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host " SWING BOT - LIVE MONITOR" -ForegroundColor Cyan
Write-Host " decision window 15:40 ET each session" -ForegroundColor DarkGray
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
$proc = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*swing_supervisor*" }
if ($proc) { Write-Host " STATUS: RUNNING (pid $($proc.ProcessId))" -ForegroundColor Green }
else       { Write-Host " STATUS: NOT RUNNING" -ForegroundColor Red }
Write-Host " log: logs\supervisor_out.log"
Write-Host " (leave this window open; Ctrl+C to stop watching - the bot keeps running)" -ForegroundColor DarkGray
Write-Host ""
Get-Content "$root\logs\supervisor_out.log" -Tail 15 -Wait
