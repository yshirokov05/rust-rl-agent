Get-ChildItem -Path C:\ -Filter "Bundles" -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.PSIsContainer } | ForEach-Object {
    $size = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB
    Write-Output ("{0} : {1:N2} GB" -f $_.FullName, $size)
}
