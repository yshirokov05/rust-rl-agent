Get-ChildItem -Path C:\Projects\rust-rl-agent\server -Recurse -File | Select-String -Pattern "vision_0.json" | Select-Object -Unique Path
