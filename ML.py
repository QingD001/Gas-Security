import numpy as np
import pandas as pd
from typing import List, Optional
from main import PRE_COLS, POST_COLS, COL_ALARM, COL_THRES, COL_LABEL

from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

def tofloat(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None

#获取报警前后的序列
def get_prevals(row, cols: List[int]) -> List[float]:
    vals = []
    for c in cols:
        x = tofloat(row.iloc[c])
        if x is not None:
            vals.append(x)
    return vals

def get_postvals(row, cols: List[int]) -> List[float]:
    vals = []
    for c in cols:
        x = tofloat(row.iloc[c])
        if x is not None:
            vals.append(x)
    return vals

def get_features(row ) -> pd.DataFrame:
    res = {}
    alarmval = tofloat(row.iloc[COL_ALARM])
    threshold = tofloat(row.iloc[COL_THRES])

    res['alarmval'] = alarmval
    res['threshold'] = threshold
    res['alarm_exc'] = max(0.0, alarmval - threshold) if alarmval is not None and threshold is not None else 0

    prevals = get_prevals(row, PRE_COLS)

    res['pre_avg'] = np.mean(prevals) if prevals else 0
    res['pre_max'] = np.max(prevals) if prevals else 0
    res['pre_min'] = np.min(prevals) if prevals else 0
    res['pre_std'] = np.std(prevals) if prevals else 0
    res['pre_zero_ratio'] = sum(1 for v in prevals if v==0) / len(prevals) if prevals else 0

    if len(prevals) >= 2:
        diff1 = np.diff(prevals)
        res['pre_diff_mean'] = np.mean(diff1)
        res['pre_rising'] = np.mean(diff1 > 0)
        res['pre_maxchange'] = np.max(np.abs(diff1))
    else:
        res['pre_diff_mean'] = 0
        res['pre_rising'] = 0
        res['pre_maxchange'] = 0

    res['pre_last3_mean'] = np.mean(prevals[-3:]) if len(prevals) >= 3 else (np.mean(prevals) if prevals else 0)

    postvals = get_postvals(row, POST_COLS)

    res['post_avg'] = np.mean(postvals) if postvals else 0
    res['post_max'] = np.max(postvals) if postvals else 0
    res['post_min'] = np.min(postvals) if postvals else 0
    res['post_std'] = np.std(postvals) if postvals else 0
    res['post_zero_ratio'] = sum(1 for v in postvals if v==0) / len(postvals) if postvals else 0
    
    post5 = postvals[:5] if len(postvals)>=5 else postvals
    res['post_first5zero_ratio'] = sum(1 for v in post5 if v==0) / len(post5) if post5 else 0
    if len(postvals) >= 2:
        diff2 = np.diff(postvals)
        res['post_diff_mean'] = np.mean(diff2)
        res['post_fall_ratio'] = np.mean(diff2 < 0)
        res['post_maxchange'] = np.max(np.abs(diff2))
    else:
        res['post_diff_mean'] = 0
        res['post_fall_ratio'] = 0
        res['post_maxchange'] = 0

    vals = prevals + [alarmval] + postvals if alarmval is not None else prevals + postvals
    res['alarm_is_max'] = 1 if alarmval is not None and vals and alarmval >= max(vals) else 0
    res['alarm_vs_premax'] = alarmval / res['pre_max'] if res['pre_max'] > 0 and alarmval is not None else 0
    res['postpre_avg_diff'] = res['post_avg'] - res['pre_avg']

    full = np.array(vals)
    if len(full) >= 3:
        res['full_mean'] = np.mean(full)
        res['full_std'] = np.std(full)
        res['full_zero_ratio'] = np.mean(full == 0)
        res['full_skew'] = tofloat(pd.Series(full).skew()) if np.std(full) > 0 else 0
        res['full_kurtosis'] = tofloat(pd.Series(full).kurtosis()) if np.std(full) > 0 else 0
        res['alarm_percentile'] = np.mean(full <= alarmval) if alarmval is not None else 0
    else:
        for k in ['full_mean', 'full_std', 'full_zero_ratio', 'full_skew', 'full_kurtosis', 'alarm_percentile']:
            res[k] = 0

    res['pre_last_diff'] = float(np.diff(prevals)[-1]) if len(prevals) >= 2 else 0
    cnt = 0
    for v in reversed(prevals):
        if v==0:
            cnt+=1
        else:
            break
    res['pre_consecutive_zero'] = cnt

    res['post_first_vs_alarm'] = postvals[0] / alarmval if postvals and alarmval and alarmval > 0 else 0
    res['post_decay_rate'] = (postvals[-1] - postvals[0]) / len(postvals) if len(postvals) >= 3 else 0
    res['post_monotone_decrease'] = 1 if len(postvals) >= 3 and all(postvals[i] >= postvals[i + 1] for i in range(len(postvals) - 1)) else 0

    res['post_above_thr_ratio'] = sum(1 for v in postvals if v > threshold) / len(postvals) if threshold and threshold > 0 and postvals else 0

    return pd.DataFrame([res])

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    frames = [get_features(i) for _,i in df.iterrows()]
    return pd.concat(frames, ignore_index=True)

if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    labels = df[df.iloc[:,COL_LABEL].isin(['误报', '非误报'])]
    y = (labels.iloc[:,COL_LABEL] == '误报').astype(int)
    X = feature_engineering(labels) 
    X = X.fillna(0)

    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # 验证
    model = XGBClassifier(
        random_state=42,
        max_depth=7,
        learning_rate=0.2,
        eval_metric='logloss'
    )
    y_pred = cross_val_predict(model,X,y,cv=cv)
    y_prob = cross_val_predict(model,X,y,cv=cv,method='predict_proba')[:,1]

    print("XGBoost方法：")  
    print(f"精确率={precision_score(y, y_pred):.2%}")
    print(f"召回率={recall_score(y, y_pred):.2%}")
    print(f"F1={f1_score(y, y_pred):.2%}")
    print(f"AUC={roc_auc_score(y, y_prob):.4f}")