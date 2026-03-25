@echo off

pushd "%~dp0"

@REM NOTE: Clean up pytest cache
echo Deleting .pytest_cache directory...
if exist ".pytest_cache" rd /s /q ".pytest_cache"
echo Done! .pytest_cache directory has been removed.

@REM NOTE: Clean up all __pycache__ directories in scripts\, src\, and tests\
echo Deleting all __pycache__ directories in src\...
for /d /r src %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
echo Deleting all __pycache__ directories in scripts\...
for /d /r scripts %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
echo Deleting all __pycache__ directories in tests\...
for /d /r tests %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
echo Done! All __pycache__ directories have been removed.

@REM NOTE: Clean up all _build directories in src\ (C++ kernel build artifacts)
echo Deleting all _build directories (C++ kernel build artifacts) in src\...
for /d /r src %%d in (_build) do @if exist "%%d" rd /s /q "%%d"
echo Done! All _build directories have been removed.

popd

pause
