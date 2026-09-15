# Runs the Baballonia test suite WITHOUT the hardware-dependent tests.
#
# Why a filter exists: the stock suite mixes real unit tests with integration tests that probe
# serial COM ports, ESP32 boards, wifi and physical cameras. On a normal dev machine those hang
# for minutes and then crash the test host, which makes the suite useless as a regression gate.
# The excluded classes are exactly the hardware-touching ones; everything else must stay green.
#
# Usage:  pwsh -File scripts\run-tests.ps1  [-Filter "<extra vstest filter>"]

param(
    [string]$Filter = "",
    [string]$Configuration = "Debug",
    [switch]$NoRestore
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot 'resolve-dotnet.ps1')
$dotnet = Resolve-BaballoniaDotNet

$repo = Split-Path -Parent $PSScriptRoot
$proj = Join-Path $repo "src\Baballonia.Tests\Baballonia.Tests.csproj"

# Tests install synthetic adapters and write recordings. Always isolate them from home profiles.
if ([string]::IsNullOrWhiteSpace($env:BABALLONIA_PROFILE)) { $env:BABALLONIA_PROFILE = 'tests' }
if ([string]::IsNullOrWhiteSpace($env:BABALLONIA_DATA_ROOT)) {
    $env:BABALLONIA_DATA_ROOT = Join-Path $repo 'artifacts\test-profiles'
}

# Hardware-dependent classes: serial/firmware boards, wifi provisioning, real webcams,
# the external trainer executable, and the SteamVR overlay.
$excluded = @(
    "Baballonia.Tests.FirmwareTests.FirmwareIntegrationTest",
    "Baballonia.Tests.FirmwareTests.FirmwareServiceTest",
    "Baballonia.Tests.Models.FirmwareSessionFactoryTest",
    "Baballonia.Tests.Models.FirmwareSessionV2Test",
    "Baballonia.Tests.OpenCvCaptureTest",
    "Baballonia.Tests.Services.Inference.SingleCameraSourceTest",
    "Baballonia.Tests.Trainer.TrainerServiceTest",
    "Baballonia.Tests.Calibration.OverlayTrainerServiceTest"
)

$clauses = $excluded | ForEach-Object { "FullyQualifiedName!~$_" }
$expr = $clauses -join "&"
if ($Filter -ne "") { $expr = "($expr)&($Filter)" }

$extraArguments = @()
if ($NoRestore) { $extraArguments += '--no-restore' }
& $dotnet test $proj -c $Configuration -m:1 -nr:false -p:UseSharedCompilation=false --filter $expr --logger "console;verbosity=normal" @extraArguments
exit $LASTEXITCODE
