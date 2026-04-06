Get-ChildItem -Path C:\ -Filter "*rust*" -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.PSIsContainer } | ForEach-Object {
    try {
        $sum = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        if ($sum -gt 1GB) {
            $gb = [math]::Round($sum / 1GB, 2)
            Write-Output ("{0} : {1:N2} GB" -f $_.FullName, $gb)
        }
    } catch {
        # Skip access denied
    }
}
