#Requires -Version 7.0
# SPDX-License-Identifier: MIT
[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidatePattern('^[A-Za-z0-9_.:\[\]-]+$')][string]$Serial,
    [ValidatePattern('^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+$')][string]$PackageId = 'io.github.reuse.phone.gpu',
    [string]$AdbPath,
    [string]$OutputDirectory,
    [ValidateRange(1, 1440)][int]$MaxMinutes = 60,
    [ValidateRange(1, 1024)][int]$MaxLogMiB = 50
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (!$AdbPath) {
    $command = Get-Command adb.exe -ErrorAction SilentlyContinue
    if ($command) { $AdbPath = $command.Source }
    else {
        $sdk = if ($env:ANDROID_HOME) { $env:ANDROID_HOME } elseif ($env:ANDROID_SDK_ROOT) { $env:ANDROID_SDK_ROOT } else { Join-Path $env:LOCALAPPDATA 'Android\Sdk' }
        $AdbPath = Join-Path $sdk 'platform-tools\adb.exe'
    }
}
if (!(Test-Path -LiteralPath $AdbPath -PathType Leaf)) { throw 'Pass -AdbPath or put Android platform-tools on PATH.' }
$AdbPath = [IO.Path]::GetFullPath($AdbPath)
if (!$OutputDirectory) {
    $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
    $OutputDirectory = Join-Path (Split-Path -Parent $PSScriptRoot) ('.research\phone-live-logs\' + $stamp)
}
$captureRoot = [IO.Path]::GetFullPath($OutputDirectory)
$rootPrefix = $captureRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
$budget = [long]$MaxLogMiB * 1024 * 1024
$utf8 = [Text.UTF8Encoding]::new($false, $true)
$fingerprints = @{}
$remoteStamps = @{}
$live = $null; $logStream = $null; $errorStream = $null
$copyTask = $null; $errorTask = $null
$state = [ordered]@{
    state = 'preparing'; started_utc = [DateTime]::UtcNow.ToString('o'); package = $PackageId
    output_directory = $captureRoot
    serial = $Serial; app_uid = $null; recorder_pid = $PID; adb_logcat_pid = $null
    metadata_files_copied = 0; metadata_errors = 0; last_error = ''; stop_reason = ''
    max_minutes = $MaxMinutes; max_capture_bytes = $budget
    scope = 'App-UID logcat and whitelisted JSON diagnostics only; no microphone or audio-file access'
}

function Get-CapturePath([string]$Relative) {
    $path = [IO.Path]::GetFullPath((Join-Path $captureRoot $Relative))
    if (!$path.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Capture path escaped output directory.' }
    return $path
}

function Write-Status {
    $state.updated_utc = [DateTime]::UtcNow.ToString('o')
    $target = Get-CapturePath 'recorder-status.json'
    $temp = Get-CapturePath 'recorder-status.json.tmp'
    [IO.File]::WriteAllText($temp, ($state | ConvertTo-Json -Depth 5) + "`n", $utf8)
    [IO.File]::Move($temp, $target, $true)
}

function New-AdbProcess([string[]]$Arguments) {
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $AdbPath; $info.UseShellExecute = $false; $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true; $info.RedirectStandardError = $true
    foreach ($argument in @('-s', $Serial) + $Arguments) { $info.ArgumentList.Add($argument) }
    $process = [Diagnostics.Process]::new(); $process.StartInfo = $info
    if (!$process.Start()) { throw 'Could not start the app-scoped ADB client.' }
    return $process
}

function Invoke-Adb([string[]]$Arguments, [int]$TimeoutSeconds = 8) {
    $process = New-AdbProcess $Arguments
    $memory = [IO.MemoryStream]::new()
    try {
        $read = $process.StandardOutput.BaseStream.CopyToAsync($memory)
        $errors = $process.StandardError.ReadToEndAsync()
        if (!$process.WaitForExit($TimeoutSeconds * 1000)) {
            $process.Kill(); [void]$process.WaitForExit(2000)
            throw 'ADB command timed out; check the explicit wireless connection.'
        }
        [void]$read.GetAwaiter().GetResult()
        return [pscustomobject]@{ ExitCode = $process.ExitCode; Bytes = $memory.ToArray(); ErrorText = $errors.GetAwaiter().GetResult() }
    } finally { $memory.Dispose(); $process.Dispose() }
}

function Get-CaptureBytes {
    return [long](Get-ChildItem -LiteralPath $captureRoot -Recurse -File | Measure-Object -Property Length -Sum).Sum
}

function Save-Metadata {
    $listing = Invoke-Adb @('exec-out', 'run-as', $PackageId, 'find', 'files/captures', '-maxdepth', '2', '-type', 'f')
    if ($listing.ExitCode -ne 0) { throw "App-private metadata unavailable: $($listing.ErrorText.Trim())" }
    foreach ($remote in ($utf8.GetString($listing.Bytes) -split '\r?\n')) {
        if ([DateTime]::UtcNow -ge $deadline -or $live.HasExited -or
                (Test-Path -LiteralPath (Get-CapturePath 'stop.requested'))) { break }
        # This allowlist intentionally cannot match WAV/PCM, arbitrary paths, or nested directories.
        if ($remote -notmatch '^files/captures/(?<session>[0-9]{8}-[0-9]{6}-[0-9a-f]{8})/(?<name>stats\.json|phone-capture\.json|nnapi-warmup[A-Za-z0-9_.-]*\.json)$') { continue }
        $session = $Matches.session; $name = $Matches.name
        # Do not repeatedly transfer a large unchanged warm-up profile over a mobile VPN.
        $stat = Invoke-Adb @('exec-out', 'run-as', $PackageId, 'stat', '-c', '%s:%y', '--', $remote)
        if ($stat.ExitCode -ne 0) { continue }
        $stamp = $utf8.GetString($stat.Bytes).Trim()
        if ($stamp -notmatch '^(?<size>[0-9]+):') { continue }
        if ([long]$Matches.size -gt 32 * 1024 * 1024) { throw 'A JSON diagnostic exceeds the 32 MiB per-file limit.' }
        if ($remoteStamps[$remote] -eq $stamp) { continue }
        $result = Invoke-Adb @('exec-out', 'run-as', $PackageId, 'cat', '--', $remote)
        if ($result.ExitCode -ne 0) { continue }
        if ($result.Bytes.Length -gt 32 * 1024 * 1024) { throw 'A JSON diagnostic exceeds the 32 MiB per-file limit.' }
        # Reject partially written/invalid JSON and preserve an earlier complete local copy.
        try { $null = $utf8.GetString($result.Bytes) | ConvertFrom-Json -ErrorAction Stop }
        catch { $state.metadata_errors++; $state.last_error = 'A diagnostic was incomplete; it will be retried.'; continue }
        $hasher = [Security.Cryptography.SHA256]::Create()
        try { $hash = [Convert]::ToBase64String($hasher.ComputeHash($result.Bytes)) } finally { $hasher.Dispose() }
        if ($fingerprints[$remote] -eq $hash) { $remoteStamps[$remote] = $stamp; continue }
        $target = Get-CapturePath (Join-Path $session $name)
        $previousSize = if (Test-Path -LiteralPath $target) { (Get-Item -LiteralPath $target).Length } else { 0 }
        if ((Get-CaptureBytes) - $previousSize + $result.Bytes.Length -ge $budget) {
            $state.last_error = 'Capture budget prevents copying another diagnostic.'; continue
        }
        [void][IO.Directory]::CreateDirectory((Split-Path -Parent $target))
        $temp = Get-CapturePath (Join-Path $session ($name + '.tmp'))
        [IO.File]::WriteAllBytes($temp, $result.Bytes)
        [IO.File]::Move($temp, $target, $true)
        $fingerprints[$remote] = $hash; $remoteStamps[$remote] = $stamp; $state.metadata_files_copied++
    }
}

if (Test-Path -LiteralPath (Get-CapturePath 'recorder-status.json')) { throw 'Choose a new output directory; an existing capture will not be overwritten.' }
[void][IO.Directory]::CreateDirectory($captureRoot)
try {
    $packages = Invoke-Adb @('shell', 'cmd', 'package', 'list', 'packages', '-U', '--user', '0', $PackageId)
    if ($packages.ExitCode -ne 0) { throw "Could not query the installed app: $($packages.ErrorText.Trim())" }
    $escaped = [regex]::Escape($PackageId)
    $uids = @($utf8.GetString($packages.Bytes) -split '\r?\n' | ForEach-Object {
        if ($_ -match "^package:$escaped\s+uid:(?<uid>[0-9]+)\s*$") { $Matches.uid }
    })
    if ($uids.Count -ne 1) { throw 'The exact demo package must already be installed on the connected device.' }
    $state.app_uid = $uids[0]
    $initial = Invoke-Adb @('logcat', '-d', "--uid=$($state.app_uid)", '-v', 'threadtime')
    if ($initial.ExitCode -ne 0) { throw "App-UID logcat failed: $($initial.ErrorText.Trim())" }
    if ($initial.Bytes.Length -ge $budget) { throw 'Initial app log buffer exceeds the capture budget.' }
    [IO.File]::WriteAllBytes((Get-CapturePath 'initial-app-logcat.txt'), $initial.Bytes)
    $live = New-AdbProcess @('logcat', "--uid=$($state.app_uid)", '-v', 'threadtime', '-T', '1')
    $state.adb_logcat_pid = $live.Id
    $logStream = [IO.FileStream]::new((Get-CapturePath 'live-app-logcat.txt'), [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read, 1, [IO.FileOptions]::Asynchronous)
    $errorStream = [IO.FileStream]::new((Get-CapturePath 'adb-logcat-stderr.txt'), [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read, 1, [IO.FileOptions]::Asynchronous)
    $copyTask = $live.StandardOutput.BaseStream.CopyToAsync($logStream)
    $errorTask = $live.StandardError.BaseStream.CopyToAsync($errorStream)
    $deadline = [DateTime]::UtcNow.AddMinutes($MaxMinutes)
    $state.state = 'recording'; $state.stop_reason = 'interrupted'
    Write-Status
    Write-Output ($state | ConvertTo-Json -Compress)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Test-Path -LiteralPath (Get-CapturePath 'stop.requested')) { $state.stop_reason = 'stop_requested'; break }
        if ($live.HasExited) { $state.stop_reason = 'adb_logcat_exited'; break }
        if ((Get-CaptureBytes) -ge $budget) { $state.stop_reason = 'size_limit'; break }
        try { Save-Metadata } catch { $state.metadata_errors++; $state.last_error = $_.Exception.Message }
        Write-Status
        Start-Sleep -Seconds 5
    }
    if ([DateTime]::UtcNow -ge $deadline) { $state.stop_reason = 'duration_limit' }
} catch {
    $state.state = 'failed'; $state.stop_reason = 'error'; $state.last_error = $_.Exception.Message
    throw
} finally {
    if ($live) {
        if (!$live.HasExited) { $live.Kill(); [void]$live.WaitForExit(2000) }
        foreach ($task in @($copyTask, $errorTask)) { if ($task) { try { [void]$task.GetAwaiter().GetResult() } catch { } } }
        $live.Dispose()
    }
    if ($logStream) { $logStream.Dispose() }; if ($errorStream) { $errorStream.Dispose() }
    if ($state.state -ne 'failed') { $state.state = 'stopped' }
    Write-Status
}
