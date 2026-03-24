@echo off

if "%COMPUTERNAME%"=="NXSRV42" (
    set MS_VS_HOME=C:\Program Files\Microsoft Visual Studio\2022\Community
    set VSCODE_HOME=C:\tools\vscode
)
if "%COMPUTERNAME%"=="NXPKHMED" (
    set MS_VS_HOME=C:\Program Files\Microsoft Visual Studio\18\Community
    set VSCODE_HOME=C:\apps\vscode
)

call "%MS_VS_HOME%\VC\Auxiliary\Build\vcvarsall.bat" x64
set PATH=%VSCODE_HOME%\bin;%PATH%
code --new-window "%~dp0.vscode\thalia.code-workspace"
