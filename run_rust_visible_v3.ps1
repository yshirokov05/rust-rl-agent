$bundlesPath = "C:\Projects\rust-rl-agent\server\server\steamapps\common\rust_dedicated\Bundles"
$proc = Start-Process "C:\Projects\rust-rl-agent\server\server\steamapps\common\rust_dedicated\RustDedicated.exe" -ArgumentList "+server.identity `" `"rust-rl +server.port 28015 +server.hostname AntigravityBot +rcon.port 28016 +rcon.password secret +server.seed 12345 +server.worldsize 3000 +bundle.path `"$bundlesPath`"" -PassThru
if ($proc) {
    Write-Output "Successfully launched RustDedicated (PID: $($proc.Id))"
} else {
    Write-Output "Failed to launch RustDedicated"
}
