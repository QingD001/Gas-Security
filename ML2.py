import argparse

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:
    XGBClassifier = None

from sklearn.ensemble import RandomForestClassifier


DATA = 'mid_dataset.csv'
ROC_OUT = 'output/mid_dataset_roc.svg'
PRED_OUT = 'output/mid_predictions.csv'
ALARM_PRED_OUT = 'output/mid_alarm_predictions.csv'
IMP_OUT = 'output/mid_feature_importance.csv'


def feature_engineering(df):
    vcols = [c for c in df.columns if c.startswith('v_') and c[2:].isdigit()]
    fcols = [
        'valid_cnt',
        'win_mean',
        'win_max',
        'win_min',
        'win_std',
        'win_zero_rate',
        'win_range',
        'win_first_last_diff',
        'win_max_jump',
        'win_diff_mean',
        'win_rising_rate',
        'win_fall_rate',
        'win_skew',
        'win_kurtosis',
        'win_max_pos',
        'win_max_pos_rate',
        'win_max_near_end',
        'head_zero_cnt',
        'tail_zero_cnt',
        'max_consecutive_zero',
        'early_mean',
        'early_max',
        'early_std',
        'early_zero_rate',
        'mid_mean',
        'mid_max',
        'mid_std',
        'mid_zero_rate',
        'late_mean',
        'late_max',
        'late_std',
        'late_zero_rate',
        'late_early_mean_diff',
        'late_early_max_diff',
    ]
    cols = vcols + [c for c in fcols if c in df.columns]
    X = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    code = pd.get_dummies(df['jczb_code'], prefix='jczb')
    return pd.concat([X, code], axis=1)


def get_features(df):
    return feature_engineering(df)


def get_model():
    if XGBClassifier is not None:
        return 'XGBoost', XGBClassifier(
            random_state=42,
            max_depth=4,
            learning_rate=0.05,
            n_estimators=200,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric='logloss',
        )

    return 'RandomForest', RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )


def feature_group(name):
    if name.startswith('v_'):
        return '20次原始流水值'
    if name.startswith('jczb_'):
        return '监测指标类型'
    return '窗口统计特征'


def feature_importance(model, cols):
    if hasattr(model, 'get_booster'):
        score = model.get_booster().get_score(importance_type='gain')
        data = [{'feature': c, 'importance': float(score.get(c, 0))} for c in cols]
    else:
        data = [
            {'feature': c, 'importance': float(v)}
            for c, v in zip(cols, model.feature_importances_)
        ]

    imp = pd.DataFrame(data)
    total = imp['importance'].sum()
    if total == 0:
        imp['contribution'] = 0
    else:
        imp['contribution'] = imp['importance'] / total
    imp['group'] = imp['feature'].map(feature_group)
    return imp.sort_values('contribution', ascending=False)


def save_roc_curve(y_test, prob, auc, out):
    from pathlib import Path

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, prob)
    width = 720
    height = 560
    left = 80
    right = 30
    top = 40
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    def xy(x, y):
        return left + x * plot_w, top + (1 - y) * plot_h

    points = ' '.join(f'{x:.2f},{y:.2f}' for x, y in (xy(x, y) for x, y in zip(fpr, tpr)))
    diag_start = xy(0, 0)
    diag_end = xy(1, 1)
    ticks = []
    for i in range(6):
        v = i / 5
        x, y0 = xy(v, 0)
        _, y1 = xy(v, 1)
        x0, y = xy(0, v)
        x1, _ = xy(1, v)
        ticks.append(f'<line x1="{x:.2f}" y1="{y0:.2f}" x2="{x:.2f}" y2="{y0 + 6:.2f}" stroke="#333"/>')
        ticks.append(f'<text x="{x:.2f}" y="{y0 + 24:.2f}" text-anchor="middle" font-size="13">{v:.1f}</text>')
        ticks.append(f'<line x1="{x0 - 6:.2f}" y1="{y:.2f}" x2="{x0:.2f}" y2="{y:.2f}" stroke="#333"/>')
        ticks.append(f'<text x="{x0 - 12:.2f}" y="{y + 5:.2f}" text-anchor="end" font-size="13">{v:.1f}</text>')
        if i not in (0, 5):
            ticks.append(f'<line x1="{x:.2f}" y1="{y0:.2f}" x2="{x:.2f}" y2="{y1:.2f}" stroke="#e6e6e6"/>')
            ticks.append(f'<line x1="{x0:.2f}" y1="{y:.2f}" x2="{x1:.2f}" y2="{y:.2f}" stroke="#e6e6e6"/>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="24" text-anchor="middle" font-size="20" font-family="Arial">ROC Curve</text>
{''.join(ticks)}
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1.5"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1.5"/>
<line x1="{diag_start[0]:.2f}" y1="{diag_start[1]:.2f}" x2="{diag_end[0]:.2f}" y2="{diag_end[1]:.2f}" stroke="#999" stroke-dasharray="6 6"/>
<polyline points="{points}" fill="none" stroke="#1f77b4" stroke-width="3"/>
<text x="{left + plot_w - 10}" y="{top + plot_h - 20}" text-anchor="end" font-size="15" font-family="Arial">XGBoost AUC={auc:.4f}</text>
<text x="{left + plot_w / 2}" y="{height - 22}" text-anchor="middle" font-size="15" font-family="Arial">False Positive Rate</text>
<text x="22" y="{top + plot_h / 2}" text-anchor="middle" font-size="15" font-family="Arial" transform="rotate(-90 22 {top + plot_h / 2})">True Positive Rate</text>
</svg>
'''
    with open(out, 'w', encoding='utf-8') as f:
        f.write(svg)


def save_predictions(test, pred, out, alarm_out):
    from pathlib import Path

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(alarm_out).parent.mkdir(parents=True, exist_ok=True)
    detail = test.copy()
    detail['true_alarm_prob'] = detail['score']
    detail['non_true_alarm_prob'] = 1 - detail['score']
    detail['pred_label'] = pred
    detail['pred_type'] = detail['pred_label'].map({1: 'true_alarm', 0: 'non_true_alarm'})
    detail['is_wrong'] = detail['label'] != detail['pred_label']

    base_cols = [
        'source',
        'sbbsm',
        'jczb_code',
        'alarm_time',
        'label_type',
        'label',
        'pred_type',
        'pred_label',
        'true_alarm_prob',
        'non_true_alarm_prob',
        'is_wrong',
    ]
    value_cols = [f'v_{i}' for i in range(1, 21) if f'v_{i}' in detail.columns]
    keep_cols = [c for c in base_cols + value_cols if c in detail.columns]
    detail = detail[keep_cols].sort_values(['is_wrong', 'true_alarm_prob'], ascending=[False, False])
    detail.to_csv(out, index=False, encoding='utf-8-sig')

    alarm_detail = detail[detail['source'] == 'event'].copy()
    alarm_detail.to_csv(alarm_out, index=False, encoding='utf-8-sig')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=DATA)
    parser.add_argument('--roc-out', default=ROC_OUT)
    parser.add_argument('--pred-out', default=PRED_OUT)
    parser.add_argument('--alarm-pred-out', default=ALARM_PRED_OUT)
    parser.add_argument('--imp-out', default=IMP_OUT)
    args = parser.parse_args()

    df = pd.read_csv(args.data, dtype={'sbbsm': str})
    df = df[df['label_type'].isin(['true_alarm', 'false_alarm', 'no_alarm'])].copy()
    y = df['label'].astype(int)
    X = get_features(df)

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.3,
        random_state=42,
        stratify=df['label_type'],
    )
    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    name, model = get_model()
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    test = df.loc[test_idx].copy()
    test['score'] = prob
    save_predictions(test, pred, args.pred_out, args.alarm_pred_out)

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    auc = roc_auc_score(y_test, prob)
    print(
        f'数据: {args.data}\n'
        f'模型: {name}\n'
        f'训练集: {len(train_idx):,}\n'
        f'测试集: {len(test_idx):,}\n'
        f'TP={tp}, FP={fp}, TN={tn}, FN={fn}\n'
        f'准确率={precision_score(y_test, pred, zero_division=0):.2%}\n'
        f'召回率={recall_score(y_test, pred, zero_division=0):.2%}\n'
        f'F1={f1_score(y_test, pred, zero_division=0):.2%}\n'
        f'AUC={auc:.4f}'
    )
    save_roc_curve(y_test, prob, auc, args.roc_out)
    print(f'ROC曲线: {args.roc_out}')
    print(f'测试集预测明细: {args.pred_out}')
    print(f'报警样本预测明细: {args.alarm_pred_out}')

    print('\n各类窗口分数:')
    print(test.groupby('label_type')['score'].agg(['count', 'mean', 'median', 'min', 'max']).to_string())

    print('\n样本分布:')
    print(df['label_type'].value_counts().to_string())

    imp = feature_importance(model, X.columns)
    from pathlib import Path
    Path(args.imp_out).parent.mkdir(parents=True, exist_ok=True)
    imp.to_csv(args.imp_out, index=False, encoding='utf-8-sig')
    print(f'特征贡献表: {args.imp_out}')

    print('\n特征贡献率Top 20:')
    top = imp.head(20).copy()
    top['contribution'] = top['contribution'].map(lambda x: f'{x:.2%}')
    print(top[['feature', 'group', 'contribution']].to_string(index=False))

    print('\n特征贡献率分组:')
    group = imp.groupby('group')['contribution'].sum().sort_values(ascending=False)
    print(group.map(lambda x: f'{x:.2%}').to_string())


if __name__ == '__main__':
    main()
