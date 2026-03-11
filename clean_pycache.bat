@echo off

echo Deleting all __pycache__ directories...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
echo Done! All __pycache__ directories have been removed.

echo Deleting all _philox_build directories...
for /d /r . %%d in (_philox_build) do @if exist "%%d" rd /s /q "%%d"
echo Done! All _philox_build directories have been removed.

pause
