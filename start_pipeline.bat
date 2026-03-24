@echo off
title Vanguard AI: Controller
color 0A

:: Kill any existing ghosts and python zombies first to free UDP ports
echo [ACTION] Cleaning up previous processes...
taskkill /f /im RustDedicated.exe >nul 2>&1
taskkill /f /im python.exe /t >nul 2>&1

echo ===================================================
echo   Vanguard AI: Controller (Separation Mode)
echo ===================================================
echo.

:: 1. Force a detached server launch via PowerShell Bridge
echo [ACTION] Spawning Rust Engine via PowerShell...
powershell -Command "Start-Process 'C:\Projects\rust-rl-agent\server\server\steamapps\common\rust_dedicated\RustDedicated.exe' -ArgumentList '-batchmode -nographics +server.ip 0.0.0.0 +server.port 28015 +server.tickrate 10 +server.hostname Vanguard-RL +server.seed 12345 +server.worldsize 3000 +bundle.path C:\Projects\rust-rl-agent\server\server\steamapps\common\rust_dedicated\RustDedicated_Data\Bundles -logfile rust_server.log' -WorkingDirectory 'C:\Projects\rust-rl-agent\server\server\steamapps\common\rust_dedicated'"

echo [SUCCESS] Server process detached.
echo [TIMER] 180 second countdown for Server Initialization and Bot Auto-Spawning...
echo.

:: 2. Wait for Server & Plugin OnServerInitialized
timeout /t 180 /nobreak

echo [READY] Server warmup complete. Launching Training and Monitoring...
echo.

:: 3. Launch PyTorch Training in a NEW visible window
echo [ACTION] Launching 6-Worker PyTorch Pipeline...
start "Vanguard AI Training" cmd /k "cd /d C:\Projects\rust-rl-agent\ai-agent && call C:\Projects\rust-rl-agent\venv\Scripts\activate.bat && python train_resnet.py"

:: 4. Launch TensorBoard in a NEW visible window
echo [ACTION] Launching TensorBoard Server...
start "Vanguard TensorBoard" cmd /k "cd /d C:\Projects\rust-rl-agent && call C:\Projects\rust-rl-agent\venv\Scripts\activate.bat && tensorboard --logdir=C:\Projects\ml_logs\tensorboard_logs --port=6006"

echo [SUCCESS] All systems operational. You can close this controller window now.
pause