# Download public, pinned Windows dependencies without replacing differing local files.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$repo = [IO.Path]::GetFullPath($PSScriptRoot)
$manifest = Get-Content -LiteralPath (Join-Path $repo 'scripts/windows-dependencies.json') -Raw | ConvertFrom-Json
$cacheRoot = Join-Path $repo 'artifacts/windows-dependencies'
[IO.Directory]::CreateDirectory($cacheRoot) | Out-Null
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Get-VerifiedDownload($item) {
    $filename = [IO.Path]::GetFileName(([Uri]$item.url).AbsolutePath)
    $cached = Join-Path $cacheRoot ($item.sha256 + '-' + $filename)
    if (-not (Test-Path -LiteralPath $cached)) {
        # Keep temporary paths short enough for PowerShell 5's filesystem provider.
        # Appending the full digest, archive name and GUID can exceed Windows MAX_PATH.
        $temporary = Join-Path $cacheRoot ('.download-' + [Guid]::NewGuid().ToString('N'))
        Write-Host ('Downloading ' + $item.name)
        $curl = Get-Command curl.exe -CommandType Application -ErrorAction SilentlyContinue
        if ($curl) {
            & $curl.Source --fail --location --silent --show-error --retry 3 --output $temporary $item.url
            if ($LASTEXITCODE -ne 0) { throw "Download failed; partial file retained at $temporary" }
        } else {
            Invoke-WebRequest -UseBasicParsing -Uri $item.url -OutFile $temporary
        }
        if ((Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash -ne $item.sha256) {
            throw "Checksum mismatch for $($item.name); untrusted download retained at $temporary and not installed."
        }
        Move-Item -LiteralPath $temporary -Destination $cached
    }
    if ((Get-FileHash -LiteralPath $cached -Algorithm SHA256).Hash -ne $item.sha256) {
        throw "Cached checksum mismatch for $($item.name): $cached"
    }
    return $cached
}

function Copy-DependencyFile([string]$source, [string]$destination) {
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($destination)) | Out-Null
    if (Test-Path -LiteralPath $destination) {
        if ((Get-FileHash -LiteralPath $source).Hash -ne (Get-FileHash -LiteralPath $destination).Hash) {
            throw "Existing dependency differs; preserved without changes: $destination. Use a fresh checkout to obtain the pinned release."
        }
        return
    }
    Copy-Item -LiteralPath $source -Destination $destination
}

function Expand-DependencyArchive([string]$archivePath, [string]$destination) {
    $prefix = [IO.Path]::GetFullPath($destination).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    $archive = [IO.Compression.ZipFile]::OpenRead($archivePath)
    try {
        foreach ($entry in $archive.Entries) {
            $target = [IO.Path]::GetFullPath((Join-Path $destination $entry.FullName))
            if (-not $target.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Archive entry escapes its dependency directory: $($entry.FullName)"
            }
            if ([string]::IsNullOrEmpty($entry.Name)) {
                [IO.Directory]::CreateDirectory($target) | Out-Null
                continue
            }
            $inputStream = $entry.Open()
            try {
                if (Test-Path -LiteralPath $target) {
                    $sha = [Security.Cryptography.SHA256]::Create()
                    try { $expected = [BitConverter]::ToString($sha.ComputeHash($inputStream)).Replace('-', '') }
                    finally { $sha.Dispose() }
                    if ((Get-FileHash -LiteralPath $target).Hash -ne $expected) {
                        throw "Existing dependency differs; preserved without changes: $target. Use a fresh checkout."
                    }
                    continue
                }
                [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target)) | Out-Null
                $outputStream = [IO.File]::Open($target, [IO.FileMode]::CreateNew)
                try { $inputStream.CopyTo($outputStream) } finally { $outputStream.Dispose() }
            } finally { $inputStream.Dispose() }
        }
    } finally { $archive.Dispose() }
}

foreach ($dependency in $manifest.dependencies) {
    $cached = Get-VerifiedDownload $dependency
    $destination = [IO.Path]::GetFullPath((Join-Path $repo $dependency.destination))
    if (-not $destination.StartsWith($repo + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Dependency destination must stay inside this checkout.'
    }
    if ($dependency.kind -eq 'zip') { Expand-DependencyArchive $cached $destination }
    elseif ($dependency.kind -eq 'file') { Copy-DependencyFile $cached $destination }
    else { throw "Unknown dependency kind: $($dependency.kind)" }
    Write-Host ('Verified ' + $dependency.name + ' ' + $dependency.version)
}
foreach ($license in $manifest.licenses) {
    Copy-DependencyFile (Get-VerifiedDownload $license) (Join-Path $cacheRoot ('licenses/' + $license.name))
}
Write-Host 'Public Windows dependencies and licenses are ready.'
