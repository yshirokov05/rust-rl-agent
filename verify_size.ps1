$sum = (Get-ChildItem -Path "C:\Projects\rust-rl-agent\server\steamapps\common\rust_dedicated" -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
$gb = [math]::Round($sum / 1GB, 2)
Write-Output "Final Folder Size: $gb GB"
