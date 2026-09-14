# Switch config: .\switch.ps1 h6   |   .\switch.ps1 h12   |   .\switch.ps1 h3
param([Parameter(Mandatory=$true)][ValidateSet("h3","h6","h12")][string]$cfg)
$map = @{
  h3  = @{ HORIZON_BARS=3;  MAX_HOLD_MINUTES=16; PT_MULT=0.7; SL_MULT=1.0; META_THRESHOLD=0.58; dir="models" }
  h6  = @{ HORIZON_BARS=6;  MAX_HOLD_MINUTES=32; PT_MULT=1.0; SL_MULT=1.0; META_THRESHOLD=0.55; dir="models_h6" }
  h12 = @{ HORIZON_BARS=12; MAX_HOLD_MINUTES=62; PT_MULT=1.5; SL_MULT=1.0; META_THRESHOLD=0.55; dir="models_h12" }
}
$s = $map[$cfg]
$c = Get-Content .env
foreach ($k in @("HORIZON_BARS","MAX_HOLD_MINUTES","PT_MULT","SL_MULT","META_THRESHOLD")) {
  $c = $c -replace "^$k=.*", "$k=$($s[$k])"
}
$c | Set-Content .env
Copy-Item "$($s.dir)\*" models\ -Force
Write-Host "switched to $cfg : horizon=$($s.HORIZON_BARS) hold=$($s.MAX_HOLD_MINUTES)m pt/sl=$($s.PT_MULT)/$($s.SL_MULT) thr=$($s.META_THRESHOLD)"
Write-Host "models copied from $($s.dir). Now: docker compose up -d --force-recreate bot"
