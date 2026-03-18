@echo off
cd server\steamapps\common\rust_dedicated
RustDedicated.exe ^
 -batchmode ^
 +server.port 28015 ^
 +server.level "Procedural Map" ^
 +server.seed 12345 ^
 +server.worldsize 1500 ^
 +server.maxplayers 10 ^
 +server.hostname "RL_Training_Server" ^
 +server.identity "rust_rl" ^
 +rcon.port 28016 ^
 +rcon.password "password" ^
 +rcon.web 1 ^
 -logfile "rust_server.log"
