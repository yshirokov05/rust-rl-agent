if (Test-Path "C:\RustServer_Full\Bundles\Bundles") {
    $sum = (Get-ChildItem -Path "C:\RustServer_Full" -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    $gb = [math]::Round($sum / 1GB, 2)
    Write-Output "Full Folder Size: $gb GB"
} else {
    Write-Output "Bundles/Bundles not found yet."
}
