



# train ml model 
train_ml.py --json result/part1/3_CVD_prediction/json//Arrhythmia.json --train result/part1/train_imputed.pkl --test result/part1/test_imputed.pkl --out result/part1/3_CVD_prediction/Arrhythmia/
train_ml.py --json result/part1/3_CVD_prediction/json//Stroke.json --train result/part1/train_imputed.pkl --test result/part1/test_imputed.pkl --out result/part1/3_CVD_prediction/Stroke/
train_ml.py --json result/part1/3_CVD_prediction/json//Peripheral_vascular_disease.json --train result/part1/train_imputed.pkl --test result/part1/test_imputed.pkl --out result/part1/3_CVD_prediction/Peripheral_vascular_disease/
train_ml.py --json result/part1/3_CVD_prediction/json//Hypertension.json --train result/part1/train_imputed.pkl --test result/part1/test_imputed.pkl --out result/part1/3_CVD_prediction/Hypertension/
train_ml.py --json result/part1/3_CVD_prediction/json//incident_cad.json --train result/part1/train_imputed.pkl --test result/part1/test_imputed.pkl --out result/part1/3_CVD_prediction/incident_cad/