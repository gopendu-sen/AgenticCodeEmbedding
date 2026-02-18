@echo off
setlocal EnableExtensions

cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  set "PYTHON_CMD=py -3"
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo Python 3 is required but was not found in PATH.
    exit /b 1
  )
  set "PYTHON_CMD=python"
)

where npm >nul 2>nul
if errorlevel 1 (
  echo npm is required but was not found in PATH.
  exit /b 1
)

start "Vyom Chat API" cmd /k "cd /d \"%~dp0\" && %PYTHON_CMD% -m chat_module.api --config config.chat.yml"
start "Vyom Embedding API" cmd /k "cd /d \"%~dp0\" && %PYTHON_CMD% -m ops_module.api --config config.embedding.yml"
start "Vyom Web UI" cmd /k "cd /d \"%~dp0web_ui\" && npm run dev"

echo Started chat API, embedding API, and Web UI in separate terminals.
echo Port values are read from config.chat.yml and config.embedding.yml.
