# Returns a .NET executable with the required SDK, without changing machine settings.
function Resolve-BaballoniaDotNet {
    $candidates = @()
    if ($env:DOTNET_HOST_PATH) { $candidates += $env:DOTNET_HOST_PATH }
    $onPath = Get-Command dotnet -CommandType Application -ErrorAction SilentlyContinue
    if ($onPath) { $candidates += $onPath.Source }
    $executable = if ($env:OS -eq 'Windows_NT') { 'dotnet.exe' } else { 'dotnet' }
    if ($env:DOTNET_ROOT) { $candidates += Join-Path $env:DOTNET_ROOT $executable }
    $profileDirectory = [Environment]::GetFolderPath('UserProfile')
    if ($profileDirectory) { $candidates += Join-Path $profileDirectory ".dotnet/$executable" }
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
        $sdks = @(& $candidate --list-sdks 2>$null)
        if ($LASTEXITCODE -eq 0 -and ($sdks | Where-Object { $_ -match '^10\.\d+\.\d+\s' })) {
            return $candidate
        }
    }
    throw 'Install the .NET 10 SDK, or set DOTNET_HOST_PATH to its dotnet executable. A runtime-only installation cannot build Baballonia.'
}
