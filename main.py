import pandas as pd
from ExpertRule import judge_with_rules, get_prevals, get_postvals

PRE_COLS = [27, 29, 31, 33, 35, 37, 39, 41, 43, 45]
POST_COLS = [47, 49, 51, 53, 55, 57, 59, 61, 63, 65]
COL_ALARM = 15
COL_THRES = 16
COL_LABEL = 5
TP = FP = TN = FN = 0

df = pd.read_csv('data.csv')
print(f"总记录数: {len(df)}")
print(f"列数: {len(df.columns)}")

for idx, row in df.iterrows():
    label = str(row.iloc[COL_LABEL])
    if label not in ('误报', '非误报'):
        continue

    alarmval = float(row.iloc[COL_ALARM])
    threshold = float(row.iloc[COL_THRES])
    prevals = get_prevals(row, PRE_COLS)
    postvals = get_postvals(row, POST_COLS)

    results = judge_with_rules(prevals, postvals, alarmval, threshold)
    ok1 = any(k for k , _ in results.values())
    ok2 = (label == '误报')

    if ok1 and ok2:TP+=1
    elif ok1 and not ok2:FP+=1
    elif not ok1 and not ok2:TN+=1 
    else:FN+=1

prec = TP/(TP+FP) if TP+FP>0 else 0
recall = TP/(TP+FN) if TP+FN > 0 else 0
f1 = 2*prec*recall/(prec+recall) if prec+recall > 0 else 0

print("专家系统方法：")
print(f"TP={TP}, FP={FP}, TN={TN}, FN={FN}")
print(f"精确率={prec:.2%}")
print(f"召回率={recall:.2%}")
print(f"F1={f1:.2%}")