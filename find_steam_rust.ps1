$paths = @("C:\Program Files (x86)\Steam", "C:\SteamLibrary", "C:\")
foreach ($path in $paths) {
    if (Test-Path $path) {
        Write-Output "Scanning $path..."
        Get-ChildItem -Path $path -Filter "*rust*" -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.PSIsContainer } | ForEach-Object {
            $size = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB
            if ($size -gt 1) {
                Write-Output ("{0} : {1:N2} GB" -f $_.FullName, $size)
            }
        }
    }
}
