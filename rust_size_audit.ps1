Get-ChildItem -Path C:\ -Filter "rust_dedicated" -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.PSIsContainer } | ForEach-Object {
    try {
        $sum = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $gb = [math]::Round($sum / 1GB, 2)
        Write-Output ("{0} : {1:N2} GB" -f $_.FullName, $gb)
    } catch {
        Write-Output "$($_.FullName) : Access Denied or Error"
    }
}
