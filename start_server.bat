@echo off
setlocal

:: Paths
set SERVER_ROOT=C:\Projects\rust-rl-agent\server\steamapps\common\rust_dedicated
set IDENTITY=rust_server
set SEED=12345
set WORLDSIZE=1500
set PORT=28015
set RCON_PORT=28016
set RCON_PASSWORD=antigravity
set MAXPLAYERS=10
set HOSTNAME="Rust RL Development"

echo [STARTING] Rust Dedicated Server with Carbon...
cd /d "%SERVER_ROOT%"

RustDedicated.exe -batchmode ^
 +server.port %PORT% ^
 +server.level "Procedural Map" ^
 +server.seed %SEED% ^
 +server.worldsize %WORLDSIZE% ^
 +server.maxplayers %MAXPLAYERS% ^
 +server.hostname %HOSTNAME% ^
 +server.identity "%IDENTITY%" ^
 +rcon.port %RCON_PORT% ^
 +rcon.password "%RCON_PASSWORD%" ^
 +rcon.web 1 ^
 +rcon.ip 0.0.0.0 ^
 -logfile "rust_server.log"

pause
