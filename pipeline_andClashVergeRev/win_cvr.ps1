# Compatibility wrapper. The maintained Windows script is win_cvr.cmd.

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$cmd = Join-Path $scriptDir "win_cvr.cmd"

Write-Host "win_cvr.ps1 is a wrapper. Prefer running win_cvr.cmd directly."
& $cmd @args
exit $LASTEXITCODE
