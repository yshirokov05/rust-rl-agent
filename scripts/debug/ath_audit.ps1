$maxR = -1.0
$maxStep = 0
$maxPath = ""

Get-ChildItem -Path C:\Projects\rust-rl-agent -Recurse -Filter "*.monitor.csv" | ForEach-Object {
    $content = Get-Content $_.FullName
    $header = $content[1] # Skip the comment/first line
    $rows = $content | Select-Object -Skip 2
    foreach ($row in $rows) {
        $parts = $row -split ","
        if ($parts.Count -ge 3) {
            $r = [float]$parts[0]
            $l = [int]$parts[1]
            if ($r -gt $maxR) {
                $maxR = $r
                $maxStep = $l
                $maxPath = $_.FullName
            }
        }
    }
}

Write-Output "All-Time High Reward: $maxR | Achieved at Step: $maxStep | File: $maxPath"
