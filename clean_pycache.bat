@echo off
echo Deleting all __pycache__ directories...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
echo Done! All __pycache__ directories have been removed.
pause
