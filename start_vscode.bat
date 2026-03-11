@echo off

call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set PATH=C:\apps\vscode\bin;%PATH%
code --new-window "%~dp0.vscode\thalia.code-workspace"
