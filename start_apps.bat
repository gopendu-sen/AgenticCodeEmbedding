@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "PYTHON_EXE="
set "PYTHON_ARGS="
set "VENV_PY=%~dp0.venv\Scripts\python.exe"

if exist "%VENV_PY%" (
  set "PYTHON_EXE=%VENV_PY%"
) else (
  where py >nul 2>nul
  if %errorlevel%==0 (
    set "PYTHON_EXE=py"
    set "PYTHON_ARGS=-3"
  ) else (
    where python >nul 2>nul
    if errorlevel 1 (
      echo Python 3 is required but was not found in PATH.
      echo Create a local virtual environment with:
      echo   py -3 -m venv .venv
      exit /b 1
    )
    set "PYTHON_EXE=python"
  )
)

where npm >nul 2>nul
if errorlevel 1 (
  echo npm is required but was not found in PATH.
  exit /b 1
)

"%PYTHON_EXE%" %PYTHON_ARGS% -c "import fastapi, pydantic, requests, uvicorn, chromadb" >nul 2>nul
if errorlevel 1 (
  echo Required Python dependencies are missing for [%PYTHON_EXE% %PYTHON_ARGS%].
  echo Install them in this environment with:
  echo   "%PYTHON_EXE%" %PYTHON_ARGS% -m pip install -r requirements.txt
  exit /b 1
)

start "Vyom Chat API" cmd /k "cd /d \"%~dp0\" && \"%PYTHON_EXE%\" %PYTHON_ARGS% -m chat_module.api --config config.chat.yml"
start "Vyom Embedding API" cmd /k "cd /d \"%~dp0\" && \"%PYTHON_EXE%\" %PYTHON_ARGS% -m ops_module.api --config config.embedding.yml"
start "Vyom Web UI" cmd /k "cd /d \"%~dp0web_ui\" && npm run dev"

echo Started chat API, embedding API, and Web UI in separate terminals.
echo Port values are read from config.chat.yml and config.embedding.yml.
