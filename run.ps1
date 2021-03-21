Write-Host "Start"
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
$null = New-Item -ItemType Directory -Force -Path ".\outputs\default"
$timestamp = Get-Date -UFormat "%Y%m%d_%H%M%S"
Write-Host "Train:         " -NoNewline
python train.py --production > $(".\outputs\default\train_" + $timestamp + ".log")
Write-Host "Done"
Write-Host "Inference:     " -NoNewline
python inference.py > (".\outputs\default\inference_" + $timestamp + ".log")
Write-Host "Done"
Rename-Item -Path ".\outputs\default" -NewName ("output_" + $timestamp)
Write-Host "Complete"
