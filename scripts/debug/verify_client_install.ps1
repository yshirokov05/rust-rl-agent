if (Test-Path "C:\RustClient_Full\Bundles\Bundles") {
    $sum = (Get-ChildItem -Path "C:\RustClient_Full" -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    $gb = [math]::Round($sum / 1GB, 2)
    Write-Output "Client Folder Size: $gb GB"
} else {
    Write-Output "Client Bundles/Bundles not found yet."
}
