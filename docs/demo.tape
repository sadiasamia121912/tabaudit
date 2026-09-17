# vhs script for docs/demo.gif (https://github.com/charmbracelet/vhs).
#
# Build it with:  powershell docs/make_demo_gif.ps1
#
# Two things worth knowing if you edit this:
#  * `tabaudit demo` prints ~100 lines; only the first 45 (header, score, the two CRITICAL
#    findings and one MEDIUM) fit in the frame, so the hidden set-up wraps the command to
#    truncate the output and print a "... 5 more findings" line. Rich drops colours when
#    piped on Windows, so the wrapper forces ANSI output.
#  * vhs 0.12 on Windows captures frames fine but silently fails to call ffmpeg, so this
#    tape emits PNG frames and make_demo_gif.ps1 runs ffmpeg itself.

Output demo_frames/

Set Shell "powershell"
Set FontSize 13
Set Width 880
Set Height 800
Set Padding 16
Set TypingSpeed 70ms

Hide
Type 'cd C:\Users\User\dev\tabaudit' Enter
Type '$env:PATH = "$PWD\.venv\Scripts;" + $env:PATH; $env:PYTHONIOENCODING = "utf-8"; [Console]::OutputEncoding = [Text.Encoding]::UTF8' Enter
Type 'function tabaudit { python -c "import rich.console as rc; _C = rc.Console; rc.Console = lambda *a, **k: _C(*a, force_terminal=True, force_interactive=False, legacy_windows=False, width=100, **k); from tabaudit.cli import app; app()" @args | Select-Object -First 46; Write-Host "  ... 7 more findings (MEDIUM x4, LOW x2, INFO x1) - full output in the HTML report" -ForegroundColor DarkGray }' Enter
Type 'function prompt { "$ " }' Enter
Type 'cls' Enter
Sleep 500ms
Show

Sleep 1s
Type "tabaudit demo"
Sleep 700ms
Enter
# The run takes ~35 s on this machine (the impact check fits two extra models) and the
# pipe holds Python's output until it exits, so skip the dead time instead of recording it.
Sleep 1s
Hide
Sleep 45s
Show
Sleep 6s
