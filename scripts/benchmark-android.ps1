param(
    [Parameter(Mandatory=$true)][string]$Serial,
    [Parameter(Mandatory=$true)][string]$ModelDir,
    [string]$FixtureDir,
    [int]$Threads = 2,
    [int]$Warmup = 10,
    [string]$AdbPath,
    [string]$OutputDir = 'dist/android-benchmark'
)
$ErrorActionPreference = 'Stop'
$reuseRoot = Split-Path -Parent $PSScriptRoot
$reuseModel = (Resolve-Path -LiteralPath $ModelDir).Path
$reuseManifest = Get-Content -Raw -LiteralPath (Join-Path $reuseModel 'manifest.json') | ConvertFrom-Json
if (-not $AdbPath) {
    if ($env:ANDROID_HOME -and (Test-Path -LiteralPath (Join-Path $env:ANDROID_HOME 'platform-tools/adb.exe'))) {
        $AdbPath = Join-Path $env:ANDROID_HOME 'platform-tools/adb.exe'
    } elseif (Get-Command adb -ErrorAction SilentlyContinue) {
        $AdbPath = (Get-Command adb).Source
    } else { throw 'Set ANDROID_HOME or pass -AdbPath to an Android SDK adb.exe.' }
}
function Invoke-ReuseAdb([string[]]$Arguments) {
    & $AdbPath -s $Serial @Arguments
    if ($LASTEXITCODE -ne 0) { throw "adb failed: $($Arguments -join ' ')" }
}
$reuseApp = 'io.github.reuse.benchmark'
$reuseRemote = "/data/user/0/$reuseApp/files/model"
Invoke-ReuseAdb @('install','-r',(Join-Path $reuseRoot 'android/benchmark/build/outputs/apk/debug/benchmark-debug.apk'))
Invoke-ReuseAdb @('install','-r',(Join-Path $reuseRoot 'android/benchmark/build/outputs/apk/androidTest/debug/benchmark-debug-androidTest.apk'))
Invoke-ReuseAdb @('shell','run-as',$reuseApp,'mkdir','-p','files/model')
function Copy-ReuseInput([string]$LocalFile, [string]$Name) {
    # Scoped storage gives shell-created external directories the shell UID.
    # Stage a world-readable file, then let the debug app create its own copy.
    $reuseTemporary = "/data/local/tmp/reuse-benchmark-$Name"
    Invoke-ReuseAdb @('push',$LocalFile,$reuseTemporary)
    Invoke-ReuseAdb @('shell','chmod','644',$reuseTemporary)
    Invoke-ReuseAdb @('shell','run-as',$reuseApp,'cp',$reuseTemporary,"files/model/$Name")
    Invoke-ReuseAdb @('shell','rm','-f',$reuseTemporary)
}
foreach ($reuseFile in @('model.onnx','manifest.json')) {
    Copy-ReuseInput (Join-Path $reuseModel $reuseFile) $reuseFile
}
# Remove only this benchmark's stale inputs/report, never other apps' data.
Invoke-ReuseAdb @('shell','run-as',$reuseApp,'rm','-f','files/model/input.f32','files/model/expected.f32','files/model/report.json','files/model/enhanced.f32')
if ($FixtureDir) {
    $reuseFixture = (Resolve-Path -LiteralPath $FixtureDir).Path
    foreach ($reuseFile in @('input.f32','expected.f32')) {
        Copy-ReuseInput (Join-Path $reuseFixture $reuseFile) $reuseFile
    }
}
$reuseResult = Invoke-ReuseAdb @('shell','am','instrument','-w','-e','modelDir',$reuseRemote,
    '-e','sampleRate',([string]$reuseManifest.sample_rate),'-e','exitLayers',([string]$reuseManifest.exit_layers),
    '-e','threads',([string]$Threads),'-e','warmup',([string]$Warmup),
    "$reuseApp.test/io.github.reuse.benchmark.ReuseBenchmarkInstrumentation")
$reuseResult | Write-Output
$reuseOut = [IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputDir))
New-Item -ItemType Directory -Force -Path $reuseOut | Out-Null
$reuseResult | Set-Content -Encoding utf8 -LiteralPath (Join-Path $reuseOut 'instrumentation.txt')
foreach ($reuseFile in @('report.json','enhanced.f32')) {
    # Keep binary float PCM intact on every supported PowerShell version.
    $reuseProcess = [Diagnostics.Process]::new()
    $reuseProcess.StartInfo.FileName = $AdbPath
    $reuseProcess.StartInfo.UseShellExecute = $false
    $reuseProcess.StartInfo.CreateNoWindow = $true
    $reuseProcess.StartInfo.RedirectStandardOutput = $true
    $reuseProcess.StartInfo.RedirectStandardError = $true
    foreach ($reuseArgument in @('-s',$Serial,'exec-out','run-as',$reuseApp,'cat',"files/model/$reuseFile")) {
        $reuseProcess.StartInfo.ArgumentList.Add($reuseArgument)
    }
    $reuseStream = [IO.File]::Create((Join-Path $reuseOut $reuseFile))
    try {
        $null = $reuseProcess.Start()
        $reuseProcess.StandardOutput.BaseStream.CopyTo($reuseStream)
        $reuseError = $reuseProcess.StandardError.ReadToEnd()
        $reuseProcess.WaitForExit()
        if ($reuseProcess.ExitCode -ne 0) { throw "Cannot receive $reuseFile : $reuseError" }
    } finally { $reuseStream.Dispose(); $reuseProcess.Dispose() }
}
Copy-Item -LiteralPath (Join-Path $reuseModel 'manifest.json') -Destination (Join-Path $reuseOut 'manifest.json')
$reuseReport = Get-Content -Raw -LiteralPath (Join-Path $reuseOut 'report.json') | ConvertFrom-Json
if (($reuseResult -join "`n") -match 'ERROR:|INSTRUMENTATION_FAILED' -or
    $reuseReport.lifecycle_pass -ne $true -or
    ($FixtureDir -and $reuseReport.parity_pass -ne $true)) {
    throw "Android correctness check failed; see $reuseOut"
}
Write-Output "Android correctness passed. RTF=$($reuseReport.real_time_factor), p95=$($reuseReport.hop_ms_p95) ms. Physical-phone performance requires a physical-phone run."
