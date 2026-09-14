# Renders docs/demo.gif from docs/demo.tape.
#
#   powershell docs/make_demo_gif.ps1
#
# Needs vhs, ttyd and ffmpeg (winget install charmbracelet.vhs tsl0922.ttyd Gyan.FFmpeg)
# and the project venv built (.venv). vhs captures the frames; ffmpeg is run here
# because vhs 0.12 on Windows silently skips it.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$work = Join-Path $env:TEMP "tabaudit-demo-gif"
$frames = Join-Path $work "demo_frames"

# Put winget-installed tools on PATH for this session if they are not already.
$pk = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages"
foreach ($dir in (Get-ChildItem $pk -Directory -ErrorAction SilentlyContinue)) {
    Get-ChildItem $dir.FullName -Recurse -Include vhs.exe, ttyd.exe, ffmpeg.exe -ErrorAction SilentlyContinue |
        ForEach-Object { $env:PATH = $_.DirectoryName + ";" + $env:PATH }
}
foreach ($tool in "vhs", "ttyd", "ffmpeg") {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool not found on PATH" }
}

if (Test-Path $work) { Remove-Item -Recurse -Force $work }
New-Item -ItemType Directory -Force $work | Out-Null
Set-Location $work
vhs (Join-Path $repo "docs\demo.tape")
if (-not (Test-Path $frames)) { throw "vhs produced no frames" }

# vhs records at 50 fps: frame-text-* is the terminal, frame-cursor-* only a cursor overlay.
# 20 fps is plenty for a terminal and keeps the GIF small; the palette pass keeps text crisp.
$out = Join-Path $repo "docs\demo.gif"
ffmpeg -y -loglevel error -framerate 50 -i (Join-Path $frames "frame-text-%05d.png") `
    -vf "fps=20,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=none" $out
Write-Host "wrote $out ($([math]::Round((Get-Item $out).Length / 1KB)) KB)"
