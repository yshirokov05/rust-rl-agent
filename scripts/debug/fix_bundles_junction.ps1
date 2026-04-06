$root = "C:\Projects\rust-rl-agent\server\server\steamapps\common\rust_dedicated"
$realBundles = "C:\Projects\rust-rl-agent\server\server\steamapps\common\rust_dedicated\Bundles"

if (Test-Path "$root\Bundles") {
    Write-Output "Removing existing Bundles folder/link at root..."
    Remove-Item "$root\Bundles" -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Output "Creating Junction link: $root\Bundles -> $realBundles"
New-Item -ItemType Junction -Path "$root\Bundles" -Target "$realBundles"

$check = Get-Item "$root\Bundles"
Write-Output "Junction Created: $($check.FullName) (Target: $($check.Target))"
