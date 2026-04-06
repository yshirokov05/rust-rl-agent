$target = "C:\Projects\rust-rl-agent\server\steamapps\common\rust_dedicated\Bundles"
$source = "C:\Projects\rust-rl-agent\server\server\steamapps\common\rust_dedicated\Bundles"

if (Test-Path $target) {
    Write-Output "Force removing $target..."
    # Use cmd /c rd to be absolutely sure we remove the junction/folder
    cmd /c "rd /s /q `"$target`""
}

Write-Output "Creating Junction: $target -> $source"
New-Item -ItemType Junction -Path $target -Target $source

$check = Get-Item $target
Write-Output "Result: $($check.FullName) -> $($check.Target)"
