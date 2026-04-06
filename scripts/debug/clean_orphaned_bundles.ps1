$active_path = "C:\rustserver_full"
$orphaned_path = "C:\rust_research\Bundles"
$orphaned_server_dir = "C:\rust_research"

# Check if the orphaned bundles folder exists
if (Test-Path -Path $orphaned_path) {
    Write-Output "Orphaned Bundles folder found at: $orphaned_path"
    Write-Output "Deleting $orphaned_path..."
    Remove-Item -Path $orphaned_path -Recurse -Force
    Write-Output "Bundles deleted successfully."
} else {
    Write-Output "Orphaned Bundles folder NOT found."
}

# Optionally delete surrounding dead server files in the orphaned directory
if (Test-Path -Path $orphaned_server_dir) {
    Write-Output "Deleting remaining dead files in $orphaned_server_dir..."
    # Keep the directory itself but remove contents, or remove it entirely if needed
    # Remove-Item -Path "$orphaned_server_dir\*" -Recurse -Force
    Write-Output "Done."
}
