"""分析漏判样本（FN: true_alarm被判为non_true_alarm）的特征"""
import pandas as pd
import numpy as np

# 读取报警样本预测
alarm = pd.read_csv('output/mid_alarm_predictions.csv')

# 分离 true_alarm 样本
true_alarm = alarm[alarm['label_type'] == 'true_alarm'].copy()

# FN: 漏判 (真实报警预测为非报警)
fn = true_alarm[true_alarm['is_wrong'] == True].copy()
# TP: 正确判为真实报警
tp = true_alarm[true_alarm['is_wrong'] == False].copy()

print('=' * 70)
print('漏判样本(FN) 总体概览')
print('=' * 70)
print(f'FN数量: {len(fn)}')
print(f'TP数量: {len(tp)}')
print(f'漏判率: {len(fn) / (len(fn) + len(tp)):.1%}')

print('\n--- FN 分数分布 ---')
print(fn['true_alarm_prob'].describe().to_string())

print('\n--- TP 分数分布 (对比) ---')
print(tp['true_alarm_prob'].describe().to_string())

# 分数区间分布
print('\n--- FN 分数区间分布 ---')
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5]
labels = ['0~0.1', '0.1~0.2', '0.2~0.3', '0.3~0.4', '0.4~0.45', '0.45~0.5']
fn['score_bin'] = pd.cut(fn['true_alarm_prob'], bins=bins, labels=labels, right=False)
print(fn['score_bin'].value_counts().sort_index().to_string())

print('\n--- 紧贴阈值 (0.45~0.5) 的 FN 样本 ---')
near = fn[fn['true_alarm_prob'] >= 0.45]
print(f'数量: {len(near)} / {len(fn)} ({len(near)/len(fn):.0%})')
if len(near) > 0:
    print(near[['sbbsm', 'jczb_code', 'true_alarm_prob']].to_string())

# 监测指标类型分布
print('\n--- FN 按监测指标类型(jczb_code)分布 ---')
print(fn['jczb_code'].value_counts().to_string())
print('\n--- TP 按监测指标类型(jczb_code)分布 (对比) ---')
print(tp['jczb_code'].value_counts().to_string())

# 每个 jczb_code 的漏判率
print('\n--- 各监测指标的漏判率 ---')
for code in true_alarm['jczb_code'].unique():
    total = len(true_alarm[true_alarm['jczb_code'] == code])
    missed = len(fn[fn['jczb_code'] == code])
    print(f'{code}: 漏判 {missed}/{total} ({missed/total:.1%})')

# 20次流水值特征分析
print('\n' + '=' * 70)
print('20次流水值特征对比 (FN vs TP)')
print('=' * 70)

vcols = [f'v_{i}' for i in range(1, 21)]
vcols = [c for c in vcols if c in fn.columns]

# 计算每个样本的统计量
def calc_row_stats(df):
    vals = df[vcols].values
    stats = pd.DataFrame({
        'mean': vals.mean(axis=1),
        'std': vals.std(axis=1),
        'max': vals.max(axis=1),
        'min': vals.min(axis=1),
        'zero_rate': (vals == 0).mean(axis=1),
        'valid_cnt': (vals != 0).sum(axis=1),
        'last_value': vals[:, -1],
        'first_value': vals[:, 0],
        'range': vals.max(axis=1) - vals.min(axis=1),
        'nonzero_mean': np.where((vals != 0).sum(axis=1) > 0,
                                 (vals * (vals != 0)).sum(axis=1) / (vals != 0).sum(axis=1),
                                 0),
    })
    return stats

fn_stats = calc_row_stats(fn)
tp_stats = calc_row_stats(tp)

print('\n--- FN 流水统计量 ---')
print(fn_stats.describe().to_string())

print('\n--- TP 流水统计量 ---')
print(tp_stats.describe().to_string())

print('\n--- 关键差异 (TP均值 - FN均值) ---')
diff = tp_stats.mean() - fn_stats.mean()
print(diff.to_string())

# 零值形态分析
print('\n--- 零值率分布 ---')
print(f"FN 平均零值率: {fn_stats['zero_rate'].mean():.1%}")
print(f"TP 平均零值率: {tp_stats['zero_rate'].mean():.1%}")
print(f"FN 零值率=1.0 (全零) 的样本数: {(fn_stats['zero_rate'] == 1.0).sum()}")
print(f"TP 零值率=1.0 (全零) 的样本数: {(tp_stats['zero_rate'] == 1.0).sum()}")
print(f"FN 有效值数均值: {fn_stats['valid_cnt'].mean():.1f}")
print(f"TP 有效值数均值: {tp_stats['valid_cnt'].mean():.1f}")

# 列出 FN 中特别值得关注的样本
print('\n' + '=' * 70)
print('FN样本明细 (按分数从高到低)')
print('=' * 70)
fn_detail = fn.sort_values('true_alarm_prob', ascending=False)
for i, (_, row) in enumerate(fn_detail.iterrows()):
    v_vals = [row.get(f'v_{j}', np.nan) for j in range(1, 21)]
    nonzero = [v for v in v_vals if v != 0 and not np.isnan(v)]
    print(f"\n[{i+1}] sbbsm={row['sbbsm']}, jczb={row['jczb_code']}")
    print(f"    score={row['true_alarm_prob']:.4f}, alarm_time={row.get('alarm_time', 'N/A')}")
    print(f"    非零值数={len(nonzero)}, 零值率={(20-len(nonzero))/20:.1%}")
    if nonzero:
        print(f"    非零值: {[f'{v:.2f}' for v in nonzero[:10]]}")

print('\n' + '=' * 70)
print('误报样本(FP: false_alarm被判为true_alarm) 简析')
print('=' * 70)
false_alarm = alarm[alarm['label_type'] == 'false_alarm']
fp = false_alarm[false_alarm['is_wrong'] == True]
print(f'FP数量: {len(fp)}')
print(f'FP jczb_code分布:')
print(fp['jczb_code'].value_counts().to_string())
print(f'\nFP 分数分布:')
print(fp['true_alarm_prob'].describe().to_string())
