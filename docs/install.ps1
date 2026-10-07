#requires -version 5
<#
  Install ipsupport-code on Windows from GitHub Releases.

  Newest nightly (tracks main):
    iex (irm https://ipsupport-llc.github.io/ipsupport-code/install.ps1)

  A specific channel/tag (iex can't pass args, so use the scriptblock form):
    & ([scriptblock]::Create((irm https://ipsupport-llc.github.io/ipsupport-code/install.ps1))) latest
    & ([scriptblock]::Create((irm https://ipsupport-llc.github.io/ipsupport-code/install.ps1))) v0.22.0

  Installs to %LOCALAPPDATA%\Programs\ipsupport-code and adds it to your user PATH.
#>
[CmdletBinding()]
param(
  [string]$Tag = 'nightly',   # 'nightly' | 'latest' | a tag like v0.22.0
  [string]$Dest               # optional full path to the .exe
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # the progress bar cripples download speed on PS 5.1
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$repo = 'ipsupport-llc/ipsupport-code'
# The machine's architecture, from WMI rather than $env:PROCESSOR_ARCHITECTURE,
# which an x64 PowerShell under emulation on ARM64 reports as AMD64. 12 = ARM64.
$arch = 'amd64'
try { if ((Get-CimInstance Win32_Processor).Architecture -contains 12) { $arch = 'arm64' } } catch { }
$ua = @{ 'User-Agent' = 'ipsupport-code-install' }

if (-not $Dest) {
  $dir = Join-Path $env:LOCALAPPDATA 'Programs\ipsupport-code'
  $Dest = Join-Path $dir 'ipsupport-code.exe'
} else {
  $dir = Split-Path -Parent $Dest
}
New-Item -ItemType Directory -Force -Path $dir | Out-Null

$api = if ($Tag -eq 'latest') {
  "https://api.github.com/repos/$repo/releases/latest"
} else {
  "https://api.github.com/repos/$repo/releases/tags/$Tag"
}

Write-Host "-> resolving $Tag release for windows-$arch ..."
$rel = Invoke-RestMethod -Headers $ua -Uri $api
$zip = $rel.assets | Where-Object { $_.name -like "*_windows-$arch.zip" } | Select-Object -First 1
if (-not $zip -and $arch -eq 'arm64') {
  # Releases before v0.62.5 carry no ARM64 build; the x64 one runs under
  # emulation on Windows 11 (not on Windows 10 on ARM).
  Write-Host "-> no windows-arm64 build in the '$Tag' release; installing windows-amd64 (runs under emulation on Windows 11)"
  $arch = 'amd64'
  $zip = $rel.assets | Where-Object { $_.name -like "*_windows-$arch.zip" } | Select-Object -First 1
}
$sum = $rel.assets | Where-Object { $_.name -eq 'checksums.txt' } | Select-Object -First 1
if (-not $zip) { throw "no windows-$arch asset in the '$Tag' release" }
# A release can carry more than one build for a platform (the rolling nightly
# does while a run swaps its assets). checksums.txt names the current build, so
# install that one rather than whichever the API happens to list first.
if ($sum) {
  $sumsText = (Invoke-WebRequest -UseBasicParsing -Headers $ua -Uri $sum.browser_download_url).Content
  if ($sumsText -is [byte[]]) { $sumsText = [Text.Encoding]::UTF8.GetString($sumsText) }
  $want = ($sumsText -split "`n" | ForEach-Object { ($_.Trim() -split '\s+')[-1] } |
           Where-Object { $_ -like "*_windows-$arch.zip" } | Select-Object -First 1)
  if (-not $want) { throw "checksums.txt in the '$Tag' release lists no windows-$arch build" }
  $zip = $rel.assets | Where-Object { $_.name -eq $want } | Select-Object -First 1
  if (-not $zip) { throw "the '$Tag' release has no asset $want" }
}

$tmp = Join-Path ([IO.Path]::GetTempPath()) ('ipscode-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
try {
  $zipPath = Join-Path $tmp $zip.name
  Write-Host "-> downloading $($zip.name)"
  Invoke-WebRequest -Headers $ua -Uri $zip.browser_download_url -OutFile $zipPath

  if ($sum) {
    $sumsPath = Join-Path $tmp 'checksums.txt'
    Invoke-WebRequest -Headers $ua -Uri $sum.browser_download_url -OutFile $sumsPath
    $line = Get-Content $sumsPath | Where-Object { $_ -match ([regex]::Escape($zip.name) + '$') } | Select-Object -First 1
    $expected = (($line -split '\s+')[0]).ToLower()
    $actual = (Get-FileHash -Algorithm SHA256 -Path $zipPath).Hash.ToLower()
    if ($expected -and $expected -eq $actual) { Write-Host "-> checksum OK" }
    else { throw "checksum mismatch for $($zip.name)" }
  }

  Expand-Archive -Path $zipPath -DestinationPath $tmp -Force
  $exe = Join-Path $tmp 'ipsupport-code.exe'
  if (-not (Test-Path $exe)) { throw 'ipsupport-code.exe not found in the archive' }
  Copy-Item -Force $exe $Dest
} finally {
  Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}

Write-Host "-> installed: $Dest"
& $Dest -version

# The short name: ipco. A one-line ipco.cmd beside the .exe, not a link — a
# symlink needs admin or developer mode, and a hardlink or copy would keep
# running the old build after the first self-update renames the .exe.
# Never over an ipco.cmd that is not ours.
$shim = Join-Path $dir 'ipco.cmd'
$leaf = Split-Path -Leaf $Dest
if (-not (Test-Path $shim) -or (Get-Content -Raw $shim) -match 'ipsupport-code') {
  Set-Content -Path $shim -Value "@`"%~dp0$leaf`" %*" -Encoding Ascii
  Write-Host "-> also as:   ipco"
}

# Put the install dir on the user PATH (idempotent).
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if (-not $userPath) { $userPath = '' }
Write-Host ''
if (($userPath -split ';') -notcontains $dir) {
  [Environment]::SetEnvironmentVariable('Path', ($userPath.TrimEnd(';') + ';' + $dir), 'User')
  Write-Host "OK - added $dir to your user PATH. Open a NEW terminal, then run:  ipsupport-code"
} else {
  Write-Host "OK - $dir is on your PATH. Run it with:  ipsupport-code"
}
