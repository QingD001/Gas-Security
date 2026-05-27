"""阈值优化分析：寻找最优阈值使得漏判率可接受"""
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

pred = pd.read_csv('output/mid_predictions.csv')
y_true = pred['label'].astype(int)
y_prob = pred['true_alarm_prob']

# 原始标签类型
label_types = pred['label_type'].values

print('=' * 70)
print('不同阈值下的模型表现')
print('=' * 70)
print(f"{'阈值':>8s}  {'TP':>5s}  {'FP':>5s}  {'TN':>5s}  {'FN':>5s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>8s}  {'漏判FN':>8s}")

thresholds = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
results = []
for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)

    # 还会关注FP中的false_alarm和no_alarm分布
    fp_mask = (y_true == 0) & (y_pred == 1)
    fp_false_alarm = (label_types[fp_mask] == 'false_alarm').sum()
    fp_no_alarm = (label_types[fp_mask] == 'no_alarm').sum()

    print(f'{t:>8.2f}  {tp:>5d}  {fp:>5d}  {tn:>5d}  {fn:>5d}  {p:>10.1%}  {r:>10.1%}  {f:>8.3f}  {fn:>8d}')

    results.append({
        'threshold': t,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': p, 'recall': r, 'f1': f,
        'fp_false_alarm': fp_false_alarm,
        'fp_no_alarm': fp_no_alarm,
    })

# ====== FN 深度特征分析 ======
print('\n' + '=' * 70)
print('FN样本深度特征分析')
print('=' * 70)

alarm = pd.read_csv('output/mid_alarm_predictions.csv')
true_alarm = alarm[alarm['label_type'] == 'true_alarm'].copy()
fn = true_alarm[true_alarm['is_wrong'] == True].copy()
tp = true_alarm[true_alarm['is_wrong'] == False].copy()

vcols = [f'v_{i}' for i in range(1, 21)]
vcols = [c for c in vcols if c in fn.columns]

def row_features(df):
    vals = df[vcols].values
    return pd.DataFrame({
        'nonzero_mean': np.array([
            vals[i][vals[i] != 0].mean() if (vals[i] != 0).any() else 0
            for i in range(len(vals))
        ]),
        'zero_rate': (vals == 0).mean(axis=1),
        'std': vals.std(axis=1),
        'range': vals.max(axis=1) - vals.min(axis=1),
        'last_minus_first': vals[:, -1] - vals[:, 0],
        'n_unique': [len(np.unique(v)) for v in vals],
    }, index=df.index)

fn_f = row_features(fn)
tp_f = row_features(tp)

print('\n--- FN vs TP 特征均值对比 ---')
for col in fn_f.columns:
    print(f'{col:20s}: FN={fn_f[col].mean():.4f}, TP={tp_f[col].mean():.4f}, diff={tp_f[col].mean() - fn_f[col].mean():.4f}')

# 按jczb_code细分
print('\n--- 按jczb_code的漏判率 ---')
for code in ['RQ0101', 'PS0101']:
    subset = true_alarm[true_alarm['jczb_code'] == code]
    fn_sub = fn[fn['jczb_code'] == code]
    tp_sub = tp[tp['jczb_code'] == code]
    print(f'\n{code}:')
    print(f'  总计: {len(subset)}, TP: {len(tp_sub)}, FN: {len(fn_sub)}, 漏判率: {len(fn_sub)/len(subset):.1%}')

    if len(fn_sub) > 0:
        fn_sub_f = row_features(fn_sub)
        tp_sub_f = row_features(tp_sub)
        for col in fn_sub_f.columns:
            print(f'  {col:20s}: FN={fn_sub_f[col].mean():.4f}, TP={tp_sub_f[col].mean():.4f}')

# ====== FP (误报被判为真实报警) 深度分析 ======
print('\n' + '=' * 70)
print('FP样本(误报被判为真实报警)特征分析')
print('=' * 70)
false_alarm = alarm[alarm['label_type'] == 'false_alarm']
fp = false_alarm[false_alarm['is_wrong'] == True]
tn_fa = false_alarm[false_alarm['is_wrong'] == False]

print(f'FP数量: {len(fp)} (false_alarm中)')
print(f'\n--- FP jczb_code分布 ---')
print(fp['jczb_code'].value_counts().to_string())

fp_f = row_features(fp)
tn_fa_f = row_features(tn_fa)
print('\n--- FP vs TN(false_alarm) 特征均值对比 ---')
for col in fp_f.columns:
    print(f'{col:20s}: FP={fp_f[col].mean():.4f}, TN={tn_fa_f[col].mean():.4f}, diff={fp_f[col].mean() - tn_fa_f[col].mean():.4f}')

# ====== 针对RQ0101的FN做更细致的模式分析 ======
print('\n' + '=' * 70)
print('RQ0101 FN样本模式分析')
print('=' * 70)
rq_fn = fn[fn['jczb_code'] == 'RQ0101']
rq_tp = tp[tp['jczb_code'] == 'RQ0101']

# 分类：全零样本 vs 部分零样本 vs 无零样本
def classify_pattern(vals):
    zeros = (np.array(vals) == 0).sum()
    if zeros == 20:
        return '全零'
    elif zeros >= 15:
        return '高零值(>=15)'
    elif zeros >= 5:
        return '中零值(5-14)'
    elif zeros > 0:
        return '低零值(1-4)'
    else:
        return '无零值'

rq_fn_vals = rq_fn[vcols].values
rq_tp_vals = rq_tp[vcols].values

fn_patterns = [classify_pattern(v) for v in rq_fn_vals]
tp_patterns = [classify_pattern(v) for v in rq_tp_vals]

print('\nRQ0101 - FN 零值模式分布:')
for p in ['全零', '高零值(>=15)', '中零值(5-14)', '低零值(1-4)', '无零值']:
    fn_c = fn_patterns.count(p)
    tp_c = tp_patterns.count(p)
    fn_pct = fn_c / len(rq_fn) if len(rq_fn) > 0 else 0
    tp_pct = tp_c / len(rq_tp) if len(rq_tp) > 0 else 0
    print(f'  {p:20s}: FN={fn_c:>3d} ({fn_pct:.0%}), TP={tp_c:>3d} ({tp_pct:.0%})')

# 判断FN中是否存在"稳定值"模式（如全部是同一个非零值）
print('\nRQ0101 - FN 中"稳定非零值"样本 (std<0.01 且 zero_rate=0):')
stable_fn = rq_fn[(rq_fn[vcols].std(axis=1) < 0.01) & ((rq_fn[vcols] == 0).sum(axis=1) == 0)]
print(f'  数量: {len(stable_fn)}')
if len(stable_fn) > 0:
    for _, row in stable_fn.iterrows():
        val = row.get('v_1', 'N/A')
        print(f'  sbbsm={row["sbbsm"]}, score={row["true_alarm_prob"]:.4f}, 常值={val}')

print('\nRQ0101 - FN 中"有波动非零值"样本 (std>=0.01 且 zero_rate=0):')
varying_fn = rq_fn[(rq_fn[vcols].std(axis=1) >= 0.01) & ((rq_fn[vcols] == 0).sum(axis=1) == 0)]
print(f'  数量: {len(varying_fn)}')

print('\n========== 总结与建议 ==========')
print(f'''
1. 阈值调整:
   - 当前0.5阈值下FN=44 (漏判率37.6%)
   - 若降至0.25: FN预计大幅减少，但FP会增加

2. FN特征规律:
   - FN平均零值率21.2% vs TP平均零值率52.1%
   - FN平均有效值15.8个 vs TP平均有效值9.6个
   - 即: 数据越"完整"，模型越倾向于判为非报警
   - 很多FN是"稳定常值"或"小幅波动"形态

3. PS0101漏判率异常高(61.5% vs RQ0101的34.6%)
   - PS0101样本量少(仅13个)，需更多数据

4. FP特征:
   - FP多是RQ0101的"单点异常脉冲"模式
   - 大量0值 + 中间一个极大值
''')
