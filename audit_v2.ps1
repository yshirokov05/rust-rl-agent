Get-ChildItem -Path C:\Projects\rust-rl-agent -Directory | ForEach-Object {
    $sum = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    $gb = [math]::Round($sum / 1GB, 2)
    Write-Output ("{0} : {1:N2} GB" -f $_.FullName, $gb)
}
