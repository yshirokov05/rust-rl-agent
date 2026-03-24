Get-PSDrive -PSProvider FileSystem | ForEach-Object {
    $driveName = $_.Name
    $rootPath = "$($driveName):\"
    Write-Output "Scanning drive $driveName for 'Bundles'..."
    Get-ChildItem -Path $rootPath -Filter "Bundles" -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.PSIsContainer } | ForEach-Object {
        $size = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB
        if ($size -gt 1) {
            Write-Output ("{0} : {1:N2} GB" -f $_.FullName, $size)
        }
    }
}
