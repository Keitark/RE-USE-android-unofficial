param([switch]$ExactExp)
$ErrorActionPreference = 'Stop'
$reuseRoot = Split-Path -Parent $PSScriptRoot
$reuseBuild = Join-Path $reuseRoot 'native\build-windows'
if ($ExactExp) { $reuseBuild = Join-Path $reuseRoot 'native\build-windows-exact' }
$reuseExact = if ($ExactExp) { 'ON' } else { 'OFF' }
& cmake -S (Join-Path $reuseRoot 'native') -B $reuseBuild -G 'Visual Studio 17 2022' -A x64 "-DREUSE_EXACT_EXP=$reuseExact"
if ($LASTEXITCODE -ne 0) { throw 'Native CMake configure failed' }
& cmake --build $reuseBuild --config Release --parallel 4
if ($LASTEXITCODE -ne 0) { throw 'Native CMake build failed' }
Write-Output (Join-Path $reuseBuild 'Release\reuse_ort_ops.dll')
