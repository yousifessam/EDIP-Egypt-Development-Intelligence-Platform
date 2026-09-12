import pandas as pd, numpy as np, json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv('/mnt/user-data/uploads/1787955593753_Merged_ML_Cleaned.csv')

features = [
    'Population_Density','Urban_Percentage','Birth_Rate','Migration_Rate',
    'Vaccination_Rate','Health_Insurance_Coverage','Doctors','Beds',
    'Literacy_Rate','Exam_Score','Graduation_Rate','Dropout_Rate','Student_Teacher_Ratio',
    'GDP_per_Capita','Employment_Rate','Poverty_Rate','Average_Income',
    'Electricity_Coverage','Water_Coverage','Internet_Coverage','Waste_Collection','Water_Quality_Index'
]
target = 'DevIndexBase'

train = df[df.ML_Split=='Train']
val = df[df.ML_Split=='Validation']
test = df[df.ML_Split=='Test']

Xtr, ytr = train[features], train[target]
Xval, yval = val[features], val[target]
Xte, yte = test[features], test[target]

gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42, subsample=0.9)
gb.fit(Xtr, ytr)

pred_te = gb.predict(Xte)
r2 = r2_score(yte, pred_te)
mae = mean_absolute_error(yte, pred_te)
rmse = np.sqrt(mean_squared_error(yte, pred_te))
pred_val = gb.predict(Xval)
r2v = r2_score(yval, pred_val)
print('TEST  R2=%.5f MAE=%.5f RMSE=%.5f'%(r2,mae,rmse))
print('VAL   R2=%.5f'%r2v)

# feature importance
imp = sorted(zip(features, gb.feature_importances_), key=lambda x:-x[1])
for f,i in imp: print(f, round(i,4))

# stats for input ranges (sliders)
stats = df[features].agg(['min','max','mean','median']).round(2).to_dict()
with open('feature_stats.json','w') as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

# Export trees to JSON structure
init_pred = float(gb.init_.constant_[0][0])  # base prediction (mean-based)
lr = gb.learning_rate

trees_json = []
n_nodes_total = 0
for est in gb.estimators_[:,0]:
    t = est.tree_
    n_nodes_total += t.node_count
    trees_json.append({
        'feature': t.feature.tolist(),
        'threshold': [round(x,6) for x in t.threshold.tolist()],
        'left': t.children_left.tolist(),
        'right': t.children_right.tolist(),
        'value': [round(v[0][0],6) for v in t.value]
    })

model_export = {
    'features': features,
    'init_pred': round(init_pred,6),
    'learning_rate': lr,
    'trees': trees_json,
    'metrics': {'r2_test': round(float(r2),4), 'mae_test': round(float(mae),4), 'rmse_test': round(float(rmse),4), 'r2_val': round(float(r2v),4)}
}

with open('model_export.json','w') as f:
    json.dump(model_export, f)

print('total nodes:', n_nodes_total)
import os
print('file size KB:', os.path.getsize('model_export.json')/1024)
