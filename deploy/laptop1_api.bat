@echo off
REM PathSnap - Laptop 1 (model + API). Starts the FastAPI backend for the evaluator UI.
REM
REM   deploy\laptop1_api.bat                  demo: serves the evaluator_test trips
REM   deploy\laptop1_api.bat student_test     practice run on student_test
REM
REM Works from Command Prompt, PowerShell or a double-click. Uses .venv directly, so the
REM venv does not need to be activated. Data: "Student folder" next to the repo folder,
REM or set PATHSNAP_DATA first. Stop the API with Ctrl+C.
setlocal
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
  echo .venv not found in %CD%
  echo Set it up once:  py -m venv .venv   then   .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)

set "PART=%~1"
if "%PART%"=="" set "PART=evaluator_test"

echo Starting PathSnap API on partition %PART% ...
".venv\Scripts\python.exe" -m pathsnap.api --partition %PART%
if errorlevel 1 pause
endlocal
