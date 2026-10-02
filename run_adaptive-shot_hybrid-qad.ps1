# run_adaptive-shot_hybrid-qad.ps1
# Paper-specific entry point. Executes the complete reproducibility pipeline.

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "`n[adaptive-shot-hybrid-qad] Starting full pipeline..." -ForegroundColor Cyan
& "$ScriptDir\run_pipeline.ps1"
