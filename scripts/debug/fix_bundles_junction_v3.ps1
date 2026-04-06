$root = "C:\Projects\rust-rl-agent\server\steamapps\common\rust_dedicated"
$realBundles = "C:\Projects\rust-rl-agent\server\server\steamapps\common\rust_dedicated\Bundles"

Write-Output "Purging invalid junctions in C:\Projects\rust-rl-agent\server\..."
Get-ChildItem -Path "C:\Projects\rust-rl-agent\server\" -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.Attributes -match "ReparsePoint" } | ForEach-Object {
    Write-Output "Removing Junction: $($_.FullName)"
    Remove-Item $_.FullName -Force
}

if (Test-Path "$root\Bundles") {
    Write-Output "Removing existing Bundles folder/link at root..."
    Remove-Item "$root\Bundles" -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Output "Creating Junction link: $root\Bundles -> $realBundles"
New-Item -ItemType Junction -Path "$root\Bundles" -Target "$realBundles"

$check = Get-Item "$root\Bundles"
Write-Output "Junction Created: $($check.FullName) (Target: $($check.Target))"
