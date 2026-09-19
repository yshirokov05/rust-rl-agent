@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
pushd "%ROOT%" >nul || (
    echo [ERROR] Could not enter repository directory: %ROOT%
    exit /b 1
)

rem Preserve caller overrides and supply safe one-bot smoke defaults.
if not defined RUST_RL_SHARED_DATA set "RUST_RL_SHARED_DATA=%ROOT%shared-data"
if not defined RUST_RL_BOT_COUNT set "RUST_RL_BOT_COUNT=1"
if not defined RUST_RL_NUM_ENVS set "RUST_RL_NUM_ENVS=1"
if not defined RUST_RL_INVULNERABLE set "RUST_RL_INVULNERABLE=true"
if not defined RUST_RL_OBSERVATION_TIMEOUT set "RUST_RL_OBSERVATION_TIMEOUT=10"
if not defined RUST_RL_TOTAL_TIMESTEPS set "RUST_RL_TOTAL_TIMESTEPS=10000"
if not defined RUST_RL_DEVICE set "RUST_RL_DEVICE=cpu"
if not defined RUST_RL_START_TENSORBOARD set "RUST_RL_START_TENSORBOARD=true"
if not defined RUST_RL_TENSORBOARD_PORT set "RUST_RL_TENSORBOARD_PORT=6006"
if not defined RUST_RL_BRIDGE_SMOKE_TIMEOUT set "RUST_RL_BRIDGE_SMOKE_TIMEOUT=180"
if not defined RUST_RL_GATHER_SMOKE_TIMEOUT set "RUST_RL_GATHER_SMOKE_TIMEOUT=180"
if not defined RUST_RL_SERVER_SEED set "RUST_RL_SERVER_SEED=11111"
if not defined RUST_RL_SERVER_WORLDSIZE set "RUST_RL_SERVER_WORLDSIZE=3000"

if not defined RUST_DEDICATED_EXE set "RUST_DEDICATED_EXE=%ROOT%server\server\steamapps\common\rust_dedicated\RustDedicated.exe"
if not defined RUST_RL_PYTHON_EXE set "RUST_RL_PYTHON_EXE=%ROOT%venv\Scripts\python.exe"

for %%I in ("%RUST_DEDICATED_EXE%") do set "RUST_SERVER_DIR=%%~dpI"
if not defined RUST_RL_BUNDLE_PATH if exist "%RUST_SERVER_DIR%Bundles" set "RUST_RL_BUNDLE_PATH=%RUST_SERVER_DIR%Bundles"
if not defined RUST_RL_BUNDLE_PATH set "RUST_RL_BUNDLE_PATH=%RUST_SERVER_DIR%RustDedicated_Data\Bundles"

if not "%RUST_RL_BOT_COUNT%"=="%RUST_RL_NUM_ENVS%" (
    echo [ERROR] RUST_RL_BOT_COUNT and RUST_RL_NUM_ENVS must match.
    goto FAIL
)

if not exist "%RUST_DEDICATED_EXE%" (
    echo [ERROR] RustDedicated.exe not found:
    echo         %RUST_DEDICATED_EXE%
    echo Set RUST_DEDICATED_EXE before running this launcher.
    goto FAIL
)

if not exist "%RUST_RL_PYTHON_EXE%" (
    echo [ERROR] Python virtual environment not found:
    echo         %RUST_RL_PYTHON_EXE%
    echo Create venv and install requirements.txt first.
    goto FAIL
)

if not exist "%ROOT%scripts\bridge_smoke.py" (
    echo [ERROR] scripts\bridge_smoke.py is missing.
    goto FAIL
)
if not exist "%ROOT%scripts\gather_smoke.py" (
    echo [ERROR] scripts\gather_smoke.py is missing.
    goto FAIL
)

if not exist "%RUST_RL_SHARED_DATA%" (
    mkdir "%RUST_RL_SHARED_DATA%"
    if errorlevel 1 (
        echo [ERROR] Could not create shared-data directory: %RUST_RL_SHARED_DATA%
        goto FAIL
    )
)

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "RUN_STAMP=%%I"
if not defined RUN_STAMP set "RUN_STAMP=latest"
set "RUN_DIR=%ROOT%artifacts\runs\%RUN_STAMP%"
if not exist "%RUN_DIR%" mkdir "%RUN_DIR%" >nul 2>&1
if not exist "%RUN_DIR%" (
    echo [ERROR] Could not create run directory: %RUN_DIR%
    goto FAIL
)
set "SERVER_LOG=%RUN_DIR%\rust_server.log"
set "BRIDGE_LOG=%RUN_DIR%\bridge_smoke.log"
set "GATHER_LOG=%RUN_DIR%\gather_smoke.log"
set "TRAIN_LOG=%RUN_DIR%\training.log"

tasklist /FI "IMAGENAME eq RustDedicated.exe" 2>nul | find /I "RustDedicated.exe" >nul
if not errorlevel 1 (
    echo [ERROR] RustDedicated.exe is already running.
    echo Stop it first, or run scripts\bridge_smoke.py manually against that server.
    goto FAIL
)

echo [ACTION] Starting private Rust server...
echo [INFO]   Server log: %SERVER_LOG%
start "Rust RL Server" /min /D "%RUST_SERVER_DIR%" "%RUST_DEDICATED_EXE%" ^
    -batchmode -nographics ^
    +server.ip 127.0.0.1 ^
    +server.port 28015 ^
    +server.tickrate 10 ^
    +server.hostname "Rust RL Private MVP" ^
    +server.identity "rust-rl-agent" ^
    +server.seed "%RUST_RL_SERVER_SEED%" ^
    +server.worldsize "%RUST_RL_SERVER_WORLDSIZE%" ^
    +bundle.path "%RUST_RL_BUNDLE_PATH%" ^
    -logfile "%SERVER_LOG%"
set "SERVER_STARTED=true"

echo [ACTION] Waiting for exact reset, forward, and stop acknowledgements...
"%RUST_RL_PYTHON_EXE%" "%ROOT%scripts\bridge_smoke.py" ^
    --bot-id 0 ^
    --shared-data-dir "%RUST_RL_SHARED_DATA%" ^
    --timeout "%RUST_RL_BRIDGE_SMOKE_TIMEOUT%" ^
    --evidence-dir "%RUN_DIR%" >"%BRIDGE_LOG%" 2>&1
set "SMOKE_EXIT=%ERRORLEVEL%"
type "%BRIDGE_LOG%"
if not "%SMOKE_EXIT%"=="0" (
    echo [ERROR] Bridge smoke failed. Training was not started.
    echo [INFO]  Inspect %BRIDGE_LOG% and %SERVER_LOG%
    goto FAIL
)

echo [ACTION] Requiring a scripted approach, attack, and real wood gain...
"%RUST_RL_PYTHON_EXE%" "%ROOT%scripts\gather_smoke.py" ^
    --bot-id 0 ^
    --shared-data-dir "%RUST_RL_SHARED_DATA%" ^
    --timeout "%RUST_RL_GATHER_SMOKE_TIMEOUT%" ^
    --evidence-dir "%RUN_DIR%" >"%GATHER_LOG%" 2>&1
set "GATHER_EXIT=%ERRORLEVEL%"
type "%GATHER_LOG%"
if not "%GATHER_EXIT%"=="0" (
    echo [ERROR] Wood-gather smoke failed. Training was not started.
    echo [INFO]  Inspect %GATHER_LOG% and %SERVER_LOG%
    goto FAIL
)

if /I "%RUST_RL_START_TENSORBOARD%"=="true" (
    echo [ACTION] Starting TensorBoard at http://127.0.0.1:%RUST_RL_TENSORBOARD_PORT%/
    if not exist "%ROOT%runs\mvp" mkdir "%ROOT%runs\mvp"
    start "Rust RL TensorBoard" /min "%RUST_RL_PYTHON_EXE%" -m tensorboard.main ^
        --logdir "%ROOT%runs\mvp" ^
        --port "%RUST_RL_TENSORBOARD_PORT%"
)

echo [ACTION] Starting one-bot PPO smoke training...
echo [INFO]   Device: %RUST_RL_DEVICE%
echo [INFO]   Timesteps: %RUST_RL_TOTAL_TIMESTEPS%
echo [INFO]   Training log: %TRAIN_LOG%
set "TRAIN_COMMAND=& '%RUST_RL_PYTHON_EXE%' '%ROOT%ai-agent\train_resnet_v2.py' --num-envs %RUST_RL_NUM_ENVS% --total-timesteps %RUST_RL_TOTAL_TIMESTEPS% --device '%RUST_RL_DEVICE%' 2>&1 | Tee-Object -FilePath '%TRAIN_LOG%'"
start "Rust RL Training" powershell -NoProfile -NoExit -Command "%TRAIN_COMMAND%"

echo [SUCCESS] Bridge and wood-gather gates passed; the 10k CPU run was started.
echo [INFO]    Run evidence: %RUN_DIR%
echo [INFO]    Use Ctrl+C in the training window to stop and save a checkpoint.
popd >nul
endlocal
exit /b 0

:FAIL
if defined SERVER_STARTED echo [INFO] The server process was left running so its log can be inspected.
popd >nul
endlocal
exit /b 1
