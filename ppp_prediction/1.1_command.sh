
# run logit association
cal_corr.py --json result/part1/1_CVD_association_result/disease.json --whole_file  result/part1/data.pkl --out result/part1/1_CVD_association_result/
# vocano

vocano.R -i result/part1/1_CVD_association_result/Arrhythmia.csv -o result/part1/1_CVD_association_result/Arrhythmia -x coef -y pvalue --runfdr --title Arrhythmia
vocano.R -i result/part1/1_CVD_association_result/Stroke.csv -o result/part1/1_CVD_association_result/Stroke -x coef -y pvalue --runfdr --title Stroke
vocano.R -i result/part1/1_CVD_association_result/Peripheral_vascular_disease.csv -o result/part1/1_CVD_association_result/Peripheral_vascular_disease -x coef -y pvalue --runfdr --title Peripheral_vascular_disease
vocano.R -i result/part1/1_CVD_association_result/Hypertension.csv -o result/part1/1_CVD_association_result/Hypertension -x coef -y pvalue --runfdr --title Hypertension
vocano.R -i result/part1/1_CVD_association_result/incident_cad.csv -o result/part1/1_CVD_association_result/incident_cad -x coef -y pvalue --runfdr --title incident_cad


# train ml model 
train_ml.py --json result/part1/3_CVD_prediction/json//Arrhythmia.json --train result/part1/train_imputed.pkl --test result/part1/test_imputed.pkl --out result/part1/3_CVD_prediction/Arrhythmia/
train_ml.py --json result/part1/3_CVD_prediction/json//Stroke.json --train result/part1/train_imputed.pkl --test result/part1/test_imputed.pkl --out result/part1/3_CVD_prediction/Stroke/
train_ml.py --json result/part1/3_CVD_prediction/json//Peripheral_vascular_disease.json --train result/part1/train_imputed.pkl --test result/part1/test_imputed.pkl --out result/part1/3_CVD_prediction/Peripheral_vascular_disease/
train_ml.py --json result/part1/3_CVD_prediction/json//Hypertension.json --train result/part1/train_imputed.pkl --test result/part1/test_imputed.pkl --out result/part1/3_CVD_prediction/Hypertension/
train_ml.py --json result/part1/3_CVD_prediction/json//incident_cad.json --train result/part1/train_imputed.pkl --test result/part1/test_imputed.pkl --out result/part1/3_CVD_prediction/incident_cad/