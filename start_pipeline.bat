@echo off
setlocal

set "ROOT=%~dp0"
set "RUST_RL_SHARED_DATA=%ROOT%shared-data"
set "RUST_RL_BOT_COUNT=1"
set "RUST_RL_NUM_ENVS=1"
set "RUST_RL_INVULNERABLE=true"
set "RUST_RL_OBSERVATION_TIMEOUT=10"

if not defined RUST_DEDICATED_EXE set "RUST_DEDICATED_EXE=%ROOT%server\server\steamapps\common\rust_dedicated\RustDedicated.exe"
set "PYTHON_EXE=%ROOT%venv\Scripts\python.exe"

if not exist "%RUST_DEDICATED_EXE%" (
    echo [ERROR] RustDedicated.exe not found:
    echo         %RUST_DEDICATED_EXE%
    echo Set RUST_DEDICATED_EXE to your local server executable.
    exit /b 1
)

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python virtual environment not found:
    echo         %PYTHON_EXE%
    echo Create it and install requirements.txt first.
    exit /b 1
)

if not exist "%RUST_RL_SHARED_DATA%" mkdir "%RUST_RL_SHARED_DATA%"

echo [ACTION] Starting private Rust server...
start "Rust RL Server" /min "%RUST_DEDICATED_EXE%" ^
    -batchmode -nographics ^
    +server.ip 127.0.0.1 ^
    +server.port 28015 ^
    +server.tickrate 10 ^
    +server.hostname "Rust RL Private MVP" ^
    +server.identity "rust-rl-agent"

echo [ACTION] Waiting for Carbon telemetry...
set /a WAITED=0
:WAIT_FOR_TELEMETRY
if exist "%RUST_RL_SHARED_DATA%\vision_0.json" goto START_TRAINING
if %WAITED% GEQ 180 (
    echo [ERROR] No telemetry file after 180 seconds.
    echo Confirm Carbon loaded BotController.cs and RUST_RL_SHARED_DATA is correct.
    exit /b 1
)
timeout /t 1 /nobreak >nul
set /a WAITED+=1
goto WAIT_FOR_TELEMETRY

:START_TRAINING
echo [ACTION] Starting one-bot PPO training...
start "Rust RL Training" cmd /k ""%PYTHON_EXE%" "%ROOT%ai-agent\train_resnet_v2.py" --num-envs 1 --total-timesteps 1000000 --device auto"

echo [SUCCESS] Server and training processes started.
echo Use Ctrl+C in the training window to stop and save a checkpoint.
endlocal
