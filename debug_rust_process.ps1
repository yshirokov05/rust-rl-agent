$process = Get-WmiObject Win32_Process -Filter "Name = 'RustDedicated.exe'"
if ($process) {
    "CommandLine: " + $process.CommandLine
} else {
    "Process RustDedicated.exe not found."
}
