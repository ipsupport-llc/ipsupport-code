#requires -version 5
<#
  Install ipsupport-code on Windows from GitHub Releases.

  Newest nightly (tracks main):
    iex (irm https://ipsupport-llc.github.io/ipsupport-code/install.ps1)

  A specific channel/tag (iex can't pass args, so use the scriptblock form):
    & ([scriptblock]::Create((irm https://ipsupport-llc.github.io/ipsupport-code/install.ps1))) latest
    & ([scriptblock]::Create((irm https://ipsupport-llc.github.io/ipsupport-code/install.ps1))) v0.15.0

  Installs to %LOCALAPPDATA%\Programs\ipsupport-code and adds it to your user PATH.
#>
[CmdletBinding()]
param(
  [string]$Tag = 'nightly',   # 'nightly' | 'latest' | a tag like v0.15.0
  [string]$Dest               # optional full path to the .exe
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # the progress bar cripples download speed on PS 5.1
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$repo = 'ipsupport-llc/ipsupport-code'
$arch = 'amd64'   # only a windows-amd64 build is published (runs under emulation on ARM64)
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
$sum = $rel.assets | Where-Object { $_.name -eq 'checksums.txt' } | Select-Object -First 1
if (-not $zip) { throw "no windows-$arch asset in the '$Tag' release" }

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
