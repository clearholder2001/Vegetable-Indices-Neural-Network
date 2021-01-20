$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
$time = Get-Date -UFormat "%Y%m%d_%H%M%S"
python train_tf_v2.py --production > $(".\log\train_" + $time + ".log")
python inference.py > (".\log\inference_" + $time + ".log")
Rename-Item -Path ".\weights\trained_model.h5" -NewName ("trained_model_" + $time + ".h5")
