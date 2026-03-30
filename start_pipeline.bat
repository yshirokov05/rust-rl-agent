@echo off
title Vanguard AI: Controller
color 0A

:: Kill any existing ghosts and python zombies first to free UDP ports
echo [ACTION] Cleaning up previous processes...
taskkill /f /im "RustDedicated.exe" >nul 2>&1
taskkill /f /im "python.exe" /t >nul 2>&1

echo ===================================================
echo   Vanguard AI: Controller (No-Fail Hardcoded)
echo ===================================================
echo.

:: 1. Force a detached server launch (Hardcoded Paths)
echo [ACTION] Spawning Rust Engine...
cd /d "C:\Projects\rust-rl-agent"
start "Rust Server" /min "C:\Projects\rust-rl-agent\server\server\steamapps\common\rust_dedicated\RustDedicated.exe" -batchmode -nographics +server.ip 0.0.0.0 +server.port 28015 +server.tickrate 10 +server.hostname "Vanguard-RL" +server.seed 11111 +server.worldsize 3000 +bundle.path "C:\Projects\rust-rl-agent\server\server\steamapps\common\rust_dedicated\RustDedicated_Data\Bundles" -logfile "C:\Projects\rust-rl-agent\server\server\steamapps\common\rust_dedicated\rust_server.log"

echo [SUCCESS] Server process detached.
echo [TIMER] 180 second countdown for Server Initialization...
timeout /t 180 /nobreak

:: 2. Launch PyTorch Training (Verified Hardware: i5-8600k / 5700 XT)
echo [ACTION] Launching 6-Worker Optimized PyTorch Pipeline...
if not exist "C:\Projects\rust-rl-agent\venv\Scripts\python.exe" (
    echo [ERROR] Virtual Environment not found at C:\Projects\rust-rl-agent\venv
    pause
    exit /b
)
start "Vanguard Training" cmd /k ""C:\Projects\rust-rl-agent\venv\Scripts\python.exe" "C:\Projects\rust-rl-agent\ai-agent\train_resnet_v2.py""

:: 3. Launch TensorBoard (Hardcoded Paths)
echo [ACTION] Launching TensorBoard Server...
start "Vanguard TensorBoard" "C:\Projects\rust-rl-agent\venv\Scripts\python.exe" -m tensorboard.main --logdir="C:\Projects\ml_logs\tensorboard_logs_v2" --port=6006

echo [SUCCESS] All systems operational.
pause