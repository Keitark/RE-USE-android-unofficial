# SPDX-License-Identifier: MIT
[CmdletBinding()]
param(
    [string]$Sdk,
    [string]$NdkVersion = '26.1.10909125',
    [string]$JavaHome,
    [string]$StockAar,
    [int]$Parallel = 10,
    [ValidateSet('Release', 'MinSizeRel')]
    [string]$Configuration = 'Release',
    [switch]$PackageOnly
)
$ErrorActionPreference = 'Stop'
if (!$Sdk) {
    if ($env:ANDROID_HOME) { $Sdk = $env:ANDROID_HOME }
    elseif ($env:ANDROID_SDK_ROOT) { $Sdk = $env:ANDROID_SDK_ROOT }
    elseif ($env:LOCALAPPDATA) { $Sdk = Join-Path $env:LOCALAPPDATA 'Android\Sdk' }
    else { throw 'Pass -Sdk or set ANDROID_HOME to your Android SDK directory.' }
}
if (!$JavaHome) {
    if ($env:JAVA_HOME) { $JavaHome = $env:JAVA_HOME }
    else {
        $javacCommand = Get-Command javac.exe -ErrorAction SilentlyContinue
        if ($javacCommand) { $JavaHome = Split-Path -Parent (Split-Path -Parent $javacCommand.Source) }
        else { throw 'Pass -JavaHome or set JAVA_HOME to your JDK 17 directory.' }
    }
}
$taskRoot = Split-Path -Parent $PSScriptRoot
$sourceDir = Join-Path $taskRoot '.research\ort-gpu-build-source'
$buildDir = Join-Path $taskRoot '.research\ort-gpu-build-arm64'
$toolDir = Join-Path $taskRoot '.research\ort-gpu-build-tools'
$outputDir = Join-Path $taskRoot '.research\ort-gpu-build'
$python = Join-Path $taskRoot '.venv\Scripts\python.exe'
$cmake = Join-Path $toolDir 'cmake\data\bin\cmake.exe'
$ninjaDir = Join-Path $Sdk 'cmake\3.22.1\bin'
$ndk = Join-Path $Sdk ('ndk\' + $NdkVersion)
$patch = Join-Path $taskRoot 'native\ort-gpu\gpu-only-device-filter.patch'
$commit = 'a83fc4d58cb48eb68890dd689f94f28288cf2278'

function Assert-Exit([string]$Operation) {
    if ($LASTEXITCODE -ne 0) { throw "$Operation failed with exit code $LASTEXITCODE" }
}

if (!(Test-Path -LiteralPath $python)) { throw 'Prepare the project Python environment before building.' }
if (!(Test-Path -LiteralPath $ndk)) { throw "NDK not found: $ndk" }
if (!(Test-Path -LiteralPath (Join-Path $JavaHome 'bin\java.exe'))) { throw "JDK not found: $JavaHome" }
if (!(Test-Path -LiteralPath $sourceDir)) {
    & git -c core.longpaths=true clone --depth 1 --branch v1.23.2 --single-branch https://github.com/microsoft/onnxruntime.git $sourceDir
    Assert-Exit 'ORT source clone'
}
$actualCommit = (& git -C $sourceDir rev-parse HEAD).Trim()
Assert-Exit 'ORT source revision query'
if ($actualCommit -ne $commit) { throw "ORT revision differs from the pinned source: $actualCommit" }

# Reapply only to the exact original source, or accept an already-applied full patch.
& git -C $sourceDir apply --reverse --check $patch 2>$null
if ($LASTEXITCODE -ne 0) {
    & git -C $sourceDir apply --check $patch
    Assert-Exit 'GPU-only patch preflight'
    & git -C $sourceDir apply $patch
    Assert-Exit 'GPU-only patch application'
}

if (!$PackageOnly) {
    & git -C $sourceDir -c core.longpaths=true submodule update --init --depth 1 cmake/external/onnx
    Assert-Exit 'Pinned ONNX submodule setup'
    if (!(Test-Path -LiteralPath $cmake)) {
        & $python -m pip install --target $toolDir cmake==3.30.9
        Assert-Exit 'Isolated CMake installation'
    }
    $env:JAVA_HOME = $JavaHome
    $env:PATH = (Split-Path -Parent $cmake) + ';' + $ninjaDir + ';' + $env:PATH
    $arguments = @(
        (Join-Path $sourceDir 'tools\ci_build\build.py'),
        '--build_dir', $buildDir, '--config', $Configuration, '--update', '--build',
        '--parallel', "$Parallel", '--android', '--android_sdk_path', $Sdk,
        '--android_ndk_path', $ndk, '--android_abi', 'arm64-v8a', '--android_api', '29',
        '--use_nnapi', '--build_shared_lib', '--skip_tests', '--skip_submodule_sync',
        '--cmake_generator', 'Ninja', '--cmake_path', $cmake,
        '--disable_ml_ops', '--compile_no_warning_as_error',
        '--cmake_extra_defines', 'onnxruntime_BUILD_UNIT_TESTS=OFF',
        'CMAKE_ANDROID_PAGE_SIZE=16384', 'CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=16384'
    )
    & $python @arguments
    Assert-Exit 'Android ORT native build'
}

if (!$StockAar) {
    $cache = Join-Path $env:USERPROFILE '.gradle\caches\modules-2\files-2.1\com.microsoft.onnxruntime\onnxruntime-android\1.23.2'
    $found = @(Get-ChildItem -LiteralPath $cache -Recurse -File -Filter 'onnxruntime-android-1.23.2.aar')
    if ($found.Count -ne 1) { throw 'Pass -StockAar with the pinned Maven Central ORT 1.23.2 AAR.' }
    $StockAar = $found[0].FullName
}
& $python (Join-Path $taskRoot 'native\ort-gpu\package_runtime.py') `
    --source $sourceDir --native-library (Join-Path $buildDir ($Configuration + '\libonnxruntime.so')) `
    --stock-aar $StockAar --patch $patch --output $outputDir --build-config $Configuration
Assert-Exit 'GPU-only runtime packaging'
Write-Output ('Local Maven repository: ' + (Join-Path $outputDir 'repository'))
Write-Output ('Runtime provenance: ' + (Join-Path $outputDir 'runtime-provenance.json'))
