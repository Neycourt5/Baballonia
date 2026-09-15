# Build a fresh self-contained Windows directory and ZIP. Existing builds are retained.
[CmdletBinding()]
param(
    [string]$Version = '0.0.0-c2-preview.20260915',
    [switch]$RequireCleanSource
)
$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT') { throw 'Run this Windows package script on Windows x64 with the .NET 10 SDK.' }
$repo = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
. (Join-Path $PSScriptRoot 'resolve-dotnet.ps1')
$dotnet = Resolve-BaballoniaDotNet
$revision = (& git -C $repo rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot determine the source revision.' }
$dirty = @(& git -C $repo status --porcelain --untracked-files=normal).Count -gt 0
if ($LASTEXITCODE -ne 0) { throw 'Cannot determine source cleanliness.' }
if ($RequireCleanSource -and $dirty) { throw 'Commit the reviewed source changes before creating a release package.' }
$submoduleRevision = (& git -C (Join-Path $repo 'src/HyperText.Avalonia') rev-parse HEAD).Trim()
$submoduleEntry = & git -C $repo ls-files -s src/HyperText.Avalonia
if ($LASTEXITCODE -ne 0 -or $submoduleEntry -notmatch ('^160000 ' + $submoduleRevision + ' ')) {
    throw 'Initialize the pinned HyperText submodule: git submodule update --init src/HyperText.Avalonia'
}
& (Join-Path $repo 'download_dependencies.ps1')

$buildId = [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
$releaseRoot = Join-Path $repo ('artifacts/release/' + $buildId)
$publishDirectory = Join-Path $releaseRoot 'Baballonia-C2-Windows-x64'
$buildArtifacts = Join-Path $releaseRoot 'build'
[IO.Directory]::CreateDirectory($releaseRoot) | Out-Null
$project = Join-Path $repo 'src/Baballonia.Desktop/Baballonia.Desktop.csproj'
& $dotnet publish $project -r win-x64 -c Release --self-contained true -f net10.0 `
    --artifacts-path $buildArtifacts -o $publishDirectory -m:1 -nr:false -p:UseSharedCompilation=false `
    -p:PublishSingleFile=false -p:PublishTrimmed=false -p:PublishReadyToRun=false `
    -p:DebugType=None -p:DebugSymbols=false `
    "-p:Version=$Version" "-p:SourceRevisionId=$revision" "-p:InformationalVersion=$Version+$revision" `
    "-p:PathMap=$repo=/_/src"
if ($LASTEXITCODE -ne 0) { throw "Windows publish failed. Output retained at $releaseRoot" }

$requiredFiles = @(
    'Baballonia.Desktop.exe', 'Baballonia.Desktop.dll', 'Baballonia.dll',
    'Baballonia.Desktop.runtimeconfig.json', 'Baballonia.Desktop.deps.json',
    'coreclr.dll', 'hostfxr.dll', 'hostpolicy.dll',
    'OpenCvSharp.dll', 'OpenCvSharpExtern.dll', 'opencv_videoio_ffmpeg4130_64.dll',
    'onnxruntime.dll', 'onnxruntime_providers_shared.dll', 'libSkiaSharp.dll', 'libHarfBuzzSharp.dll',
    'faceModel.onnx', 'eyeModel.onnx', 'LocalSettings.json',
    'Modules/Baballonia.SerialCameraCapture.dll', 'Modules/Baballonia.OpenCVCapture.dll',
    'Modules/Baballonia.IPCameraCapture.dll', 'Modules/Baballonia.VFTCapture.dll',
    'training/babble_personal/c2.py', 'training/requirements.txt',
    'Calibration/Windows/Trainer/BabbleTrainer.exe',
    'Calibration/Windows/Overlay/BabbleCalibration.x86_64.exe',
    'Calibration/Windows/Overlay/BabbleCalibration.x86_64.pck',
    'Calibration/Windows/Overlay/libgodot_openvr_release.dll',
    'Calibration/Windows/Overlay/data_BabbleCalibration_windows_x86_64/BabbleCalibration.dll',
    'Firmware/Windows/espflash.exe'
)
foreach ($relative in $requiredFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $publishDirectory $relative) -PathType Leaf)) {
        throw "Required runtime file missing: $relative. Unfinished package retained at $releaseRoot"
    }
}
# Fail closed if cached compiler output still embeds local source/profile paths.
$localPaths = @($repo, [Environment]::GetFolderPath('UserProfile')) |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
    ForEach-Object { $_.Replace('\', '/') }
$applicationAssemblies = Get-ChildItem -LiteralPath $publishDirectory -Recurse -File |
    Where-Object { $_.Name -like 'Baballonia*.dll' -or $_.Name -eq 'HyperText.Avalonia.dll' }
foreach ($assembly in $applicationAssemblies) {
    $bytes = [IO.File]::ReadAllBytes($assembly.FullName)
    # UTF-16 strings may begin at either byte alignment inside a binary image.
    foreach ($format in @('utf8', 'utf16-even', 'utf16-odd')) {
        $decoded = switch ($format) {
            'utf8' { [Text.Encoding]::UTF8.GetString($bytes) }
            'utf16-even' { [Text.Encoding]::Unicode.GetString($bytes) }
            'utf16-odd' { [Text.Encoding]::Unicode.GetString($bytes, 1, $bytes.Length - 1) }
        }
        $decoded = $decoded.Replace('\', '/')
        foreach ($localPath in $localPaths) {
            if ($decoded.IndexOf($localPath, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
                throw 'A local source/profile path remains in application binaries. Package creation stopped; rebuild from a clean checkout.'
            }
        }
    }
}

$unexpected = @(Get-ChildItem -LiteralPath $publishDirectory -Recurse -File | Where-Object {
    $_.Extension -in '.npz', '.pt', '.pth', '.log', '.dmp' -or $_.Name -like 'personalFaceModel*.onnx'
})
if ($unexpected.Count -gt 0) { throw 'Unexpected training data, personal model, or diagnostic file in publish output. Review the retained output before packaging.' }

# Only remove disposable leaf files from this newly created distribution, never source/build folders.
$publishPrefix = [IO.Path]::GetFullPath($publishDirectory) + [IO.Path]::DirectorySeparatorChar
foreach ($file in (Get-ChildItem -LiteralPath $publishDirectory -Recurse -File | Where-Object { $_.Extension -in '.pdb', '.lib' })) {
    if (-not $file.FullName.StartsWith($publishPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Package path validation failed.' }
    Remove-Item -LiteralPath $file.FullName
}
$licenseDirectory = Join-Path $publishDirectory 'licenses'
[IO.Directory]::CreateDirectory($licenseDirectory) | Out-Null
Copy-Item -LiteralPath (Join-Path $repo 'LICENSE') -Destination (Join-Path $publishDirectory 'LICENSE')
Copy-Item -LiteralPath (Join-Path $repo 'CREDITS.md') -Destination (Join-Path $publishDirectory 'CREDITS.md')
Copy-Item -LiteralPath (Join-Path $repo 'src/HyperText.Avalonia/LICENSE') -Destination (Join-Path $licenseDirectory 'HyperText.Avalonia-LICENSE.txt')
Get-ChildItem -LiteralPath (Join-Path $repo 'artifacts/windows-dependencies/licenses') -File | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $licenseDirectory
}
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'windows-dependencies.json') -Destination (Join-Path $publishDirectory 'WINDOWS-DEPENDENCIES.json')
$readme = @(
    'Run Baballonia.Desktop.exe to start Baballonia.',
    '',
    'Extract the entire ZIP to a folder first. Keep the files and folders together.',
    'Windows x64; the .NET runtime is included.',
    '',
    'This is an unofficial experimental fork of Project-Babble/Baballonia.',
    'A new profile uses public stock models. No contributor personal C/C2 model,',
    'training recordings, or calibration profile is included. Select your cameras',
    'and calibrate for your own hardware. C2 requires your own trained Model C first.',
    '',
    'Personalization / C2 training is optional and requires Python 3.13 and the',
    'training tools installed from the app Personalization setup. Tracking itself',
    'does not require Python. Headset calibration requires SteamVR and compatible',
    'tracking hardware. Keep your own profile and training data private.',
    '',
    'Source, setup and limitations: https://github.com/Neycourt5/Baballonia/tree/publish/c2-preview',
    'Upstream: https://github.com/Project-Babble/Baballonia'
) -join [Environment]::NewLine
[IO.File]::WriteAllText((Join-Path $publishDirectory 'README.txt'), $readme + [Environment]::NewLine)
$metadata = [ordered]@{
    version = $Version
    sourceRevision = $revision
    uncommittedSourceChanges = $dirty
    hyperTextRevision = $submoduleRevision
    sdk = (& $dotnet --version).Trim()
    configuration = 'Release'
    runtimeIdentifier = 'win-x64'
    selfContained = $true
    singleFile = $false
    trimmed = $false
    builtAtUtc = [DateTime]::UtcNow.ToString('o')
}
[IO.File]::WriteAllText((Join-Path $publishDirectory 'BUILD-INFO.json'), ($metadata | ConvertTo-Json) + [Environment]::NewLine)
$zipPath = Join-Path $releaseRoot 'Baballonia-C2-Windows-x64.zip'
# Windows tar supports long runtime dependency paths that PowerShell 5's ZipFile rejects.
$tar = Get-Command tar.exe -CommandType Application -ErrorAction Stop
& $tar.Source -a -c -f $zipPath -C $releaseRoot 'Baballonia-C2-Windows-x64'
if ($LASTEXITCODE -ne 0) { throw "ZIP creation failed. Output retained at $releaseRoot" }
$hash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText($zipPath + '.sha256', $hash + '  Baballonia-C2-Windows-x64.zip' + [Environment]::NewLine)
Write-Host "Windows ZIP: $zipPath"
Write-Host "SHA-256: $hash"
Write-Output $zipPath
