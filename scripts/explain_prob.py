"""展示 XGBoost 概率分数的完整计算公式和计算过程"""
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# ============================================================
# 1. 重新训练模型（与 ML2.py 完全一致）
# ============================================================
df = pd.read_csv('mid_dataset.csv', dtype={'sbbsm': str})
df = df[df['label_type'].isin(['true_alarm', 'false_alarm', 'no_alarm'])].copy()
y = df['label'].astype(int)

vcols = [c for c in df.columns if c.startswith('v_') and c[2:].isdigit()]
fcols = [
    'valid_cnt', 'win_mean', 'win_max', 'win_min', 'win_std',
    'win_zero_rate', 'win_range', 'win_first_last_diff', 'win_max_jump',
    'win_diff_mean', 'win_rising_rate', 'win_fall_rate',
    'win_skew', 'win_kurtosis', 'win_max_pos', 'win_max_pos_rate',
    'win_max_near_end', 'head_zero_cnt', 'tail_zero_cnt',
    'max_consecutive_zero',
    'early_mean', 'early_max', 'early_std', 'early_zero_rate',
    'mid_mean', 'mid_max', 'mid_std', 'mid_zero_rate',
    'late_mean', 'late_max', 'late_std', 'late_zero_rate',
    'late_early_mean_diff', 'late_early_max_diff',
]
cols = vcols + [c for c in fcols if c in df.columns]
X = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
code = pd.get_dummies(df['jczb_code'], prefix='jczb')
X = pd.concat([X, code], axis=1)

train_idx, test_idx = train_test_split(
    df.index, test_size=0.3, random_state=42, stratify=df['label_type'],
)
X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_train, y_test = y.loc[train_idx], y.loc[test_idx]

model = XGBClassifier(
    random_state=42, max_depth=4, learning_rate=0.05,
    n_estimators=200, subsample=0.9, colsample_bytree=0.9,
    eval_metric='logloss',
)
model.fit(X_train, y_train)

# ============================================================
# 2. 公式
# ============================================================
booster = model.get_booster()
ntrees = booster.num_boosted_rounds()
feat_names = X_test.columns.tolist()

print('=' * 68)
print('概率分数的数学公式')
print('=' * 68)
print(f'''
模型 = {ntrees} 棵决策树的集成

核心公式（非常简洁）:

    log_odds  =  Σ [ 每棵树输出一个叶子的分数 ]  +  base_score

    true_alarm_prob  =  sigmoid( log_odds )
                      =       1
                        ──────────────
                        1 + e^(-log_odds)

    e = 2.71828...（自然常数）

只需两步:
  Step 1: 把样本扔进 {ntrees} 棵树，每棵树根据特征判定一条路径，
          落到某个叶子节点上，取出该叶子的分数。
          全部叶子分数加在一起 = log_odds

  Step 2: 对 log_odds 做 sigmoid，映射到 (0, 1)

直观理解:
  log_odds > 0  →  prob > 0.5  (被判为真报警)
  log_odds < 0  →  prob < 0.5  (被判为非报警)
  log_odds = 0  →  prob = 0.5  (刚好在阈值上)
''')

# ============================================================
# 3. 用具体样本演示计算过程
# ============================================================
test = df.loc[test_idx].copy()
prob_full = model.predict_proba(X_test)[:, 1]
test['score'] = prob_full

# 取 FN 和 TP 各一个
fn_sample = test[(test['label_type'] == 'true_alarm') & (test['score'] < 0.5)].iloc[0]
tp_sample = test[(test['label_type'] == 'true_alarm') & (test['score'] >= 0.5)].iloc[0]

for label, sample in [('FN (真报警漏判)', fn_sample), ('TP (真报警正确判)', tp_sample)]:
    print('=' * 68)
    print(f'样本: {label}')
    print(f'sbbsm: {sample["sbbsm"]}, 指标: {sample["jczb_code"]}')
    print()

    row_idx = X_test.index.get_loc(sample.name)
    x_row = X_test.iloc[[row_idx]]

    # pred_contribs 返回每个特征的贡献 + bias（最后一项）
    contribs = booster.predict(xgb.DMatrix(x_row), pred_contribs=True)[0]

    bias = contribs[-1]           # base_score
    feat_contribs = contribs[:-1] # 每个特征的log-odds贡献

    log_odds = bias + feat_contribs.sum()

    print('【Step 1: log_odds 计算】')
    print(f'  base_score = {bias:.6f}')
    print(f'  所有特征的贡献总和 = {feat_contribs.sum():.6f}')
    print(f'  log_odds = {bias:.6f} + {feat_contribs.sum():.6f} = {log_odds:.6f}')

    print()
    print('【Step 2: sigmoid 变换】')
    print(f'  公式: 1 / (1 + e^(-log_odds))')
    print(f'       = 1 / (1 + e^(-({log_odds:.6f})))')
    print(f'       = 1 / (1 + {np.exp(-log_odds):.6f})')
    manual_prob = 1 / (1 + np.exp(-log_odds))
    print(f'       = {manual_prob:.6f}')

    print()
    print(f'  sklearn 输出的 predict_proba = {sample["score"]:.6f}')
    print(f'  手算结果一致: {abs(manual_prob - sample["score"]) < 1e-10}')

    # 贡献最大的特征
    print()
    print('【每棵树贡献的 log-odds 拆解 (每棵树 = 一轮 boosting)】')
    # dump 每棵树的信息
    dump = booster.get_dump(with_stats=True)
    leaf_indices = booster.predict(xgb.DMatrix(x_row), pred_leaf=True)[0]

    # 展示前3棵树的结构
    for t in range(min(3, len(dump))):
        tree_text = dump[t]
        leaf_idx = int(leaf_indices[t])
        # 提取该树的叶子分数
        import re
        leaf_lines = [l.strip() for l in tree_text.split('\n') if 'leaf=' in l]
        # XGBoost dump 叶子是按遍历顺序列出的
        leaf_values = [float(re.search(r'leaf=([-\d.e+]+)', l).group(1)) for l in leaf_lines]
        leaf_val = leaf_values[leaf_idx] if leaf_idx < len(leaf_values) else 0
        print(f'  Tree {t}: 路由到叶子#{leaf_idx}, 输出分数 = {leaf_val:+.6f}')

    print(f'  ... (共 {ntrees} 棵树)')

    # 前10大特征贡献
    print()
    print('【按特征的前10大 log-odds 贡献】')
    pairs = sorted(zip(feat_names, feat_contribs), key=lambda x: abs(x[1]), reverse=True)
    for feat, c in pairs[:10]:
        d = '+' if c >= 0 else '-'
        print(f'  {feat:28s}: {c:+10.6f}  ({d} 推高/压低)')

print()
print('=' * 68)
print('一句话总结')
print('=' * 68)
print(f'''
log_odds = 每棵树给一个叶子分数，{ntrees}棵树的分全加起来 + base_score
prob = sigmoid(log_odds)

这就是"概率"的完整定义。没有更"简单"的公式了，
因为每棵树内部就是几十个 if-else 嵌套。
如果要一个可读的简化公式，可以用逻辑回归近似，
但会损失精度（当前AUC=0.9615）。
''')
