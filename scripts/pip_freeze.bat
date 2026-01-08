@echo off
setlocal EnableDelayedExpansion

for %%i in ("%~dp0..") do set WORKSPACE_ROOT=%%~fi
set VENV_DIR=%WORKSPACE_ROOT%\.venv
set REQUIREMENTS_FILE=%WORKSPACE_ROOT%\requirements.txt

pushd "%WORKSPACE_ROOT%"
call "%VENV_DIR%\Scripts\activate.bat"
pip freeze > "%REQUIREMENTS_FILE%"
call "%VENV_DIR%\Scripts\deactivate.bat"
popd

exit /b 0
