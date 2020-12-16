$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
python train.py > .\log\train_log.txt
python inference.py > .\log\inference_log.txt