$root = "C:\RustServer_Full"
$identity = "rust-rl"

# Server-only flags to bypass client-side shader rendering and fix NullRef
$params = @(
    "-batchmode",
    "-nographics",
    "-silent-crashes",
    "+server.ip 0.0.0.0",
    "+server.port 28015",
    "+server.tickrate 10",
    "+server.level `"Procedural Map`"",
    "+server.worldsize 3000",
    "+server.hostname 'Rust RL Agent (3D Vision - Server Only)'",
    "+server.identity `"$identity`"",
    "+server.seed 12345",
    "+server.maxplayers 10",
    "+bundle.path $root\Bundles",
    "-logfile $root\rust_server.log",
    "+carbon.reload AgentEyes"
)

Write-Output "Launching RustDedicated with Server-Only Bypass from $root..."
Start-Process -FilePath "$root\RustDedicated.exe" -ArgumentList $params -WorkingDirectory $root -WindowStyle Normal

$proc = Get-Process RustDedicated -ErrorAction SilentlyContinue
if ($proc) {
    Write-Output "Successfully launched RustDedicated (PID: $($proc.Id))"
} else {
    Write-Output "Failed to launch RustDedicated."
}
