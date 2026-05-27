import argparse
import glob
import os

import pandas as pd


EVENT_FILE = 'data.csv'
RULE_FILE = 'sbyxwh_jcxyz_202605131521.csv'
PART_DIR = 'data'
OUT_FILE = 'mid_dataset.csv'

WINDOW = 20
NO_ALARM_PER_EVENT = 3

JCLX_MAP = {
    '可燃气体探测器_可燃气体浓度': 'RQ0101',
    '燃气压力设备_压力': 'PS0101',
    '液位传感器_液位': 'RQ0401',
    '温度传感器_温度': 'RQ0501',
}


def to_num(s):
    return pd.to_numeric(s.replace({'/': None, '[NULL]': None, '': None}), errors='coerce')


def clean_values(values):
    s = pd.Series(values)
    s = to_num(s)
    return s


def value_cols():
    pre = [f'报警前{i}次推送值' for i in range(10, 0, -1)]
    post = [f'报警后{i}次推送值' for i in range(1, 10)]
    return pre + ['初次报警值'] + post


def event_windows():
    df = pd.read_csv(EVENT_FILE, dtype=str)
    df = df[df['综合研判-误报'].isin(['误报', '非误报'])].copy()

    rows = []
    cols = value_cols()
    for _, r in df.iterrows():
        vals = clean_values([r.get(c) for c in cols]).tolist()

        threshold = pd.to_numeric(r.get('报警阈值'), errors='coerce')
        label_type = 'true_alarm' if r['综合研判-误报'] == '非误报' else 'false_alarm'

        row = {
            'source': 'event',
            'label_type': label_type,
            'label': 1 if label_type == 'true_alarm' else 0,
            'sbbsm': r.get('设备标识码'),
            'jczb_code': r.get('监测指标'),
            'alarm_time': r.get('初次报警时间'),
            'threshold': threshold,
        }
        for i, v in enumerate(vals, 1):
            row[f'v_{i}'] = v
        rows.append(row)

    return pd.DataFrame(rows)


def read_rules():
    df = pd.read_csv(RULE_FILE, dtype={'sbbsm': str, 'jczb': str})
    df['bjyz'] = pd.to_numeric(df['bjyz'], errors='coerce')
    df = df[(df['bjyz'].notna()) & (df['bjyz'] > 0)]
    df = df.sort_values(['sbbsm', 'jczb', 'bjyz'])
    df = df.drop_duplicates(['sbbsm', 'jczb'], keep='first')
    df = df.rename(columns={'jczb': 'jczb_code', 'bjyz': 'threshold'})
    return df[['sbbsm', 'jczb_code', 'threshold']]


def no_alarm_windows(n):
    rules = read_rules()
    parts = sorted(glob.glob(os.path.join(PART_DIR, 'rq_jc_ssjc_part_*.csv')))
    rows = []

    for path in parts:
        if len(rows) >= n:
            break

        df = pd.read_csv(
            path,
            usecols=['sbbsm', 'jczb', 'jcz', 'jccjsj'],
            dtype={'sbbsm': str},
            low_memory=False,
        )
        df['jczb_code'] = df['jczb'].map(JCLX_MAP)
        df['value'] = clean_values(df['jcz'])
        df['time'] = pd.to_datetime(df['jccjsj'], errors='coerce')
        df = df.dropna(subset=['jczb_code', 'value', 'time'])
        df = df.merge(rules, on=['sbbsm', 'jczb_code'], how='inner')

        for (sbbsm, code), g in df.groupby(['sbbsm', 'jczb_code']):
            if len(rows) >= n:
                break
            g = g.sort_values('time').drop_duplicates(['time', 'value'])
            if len(g) < WINDOW:
                continue

            step = max(WINDOW, len(g) // 10)
            for start in range(0, len(g) - WINDOW + 1, step):
                win = g.iloc[start:start + WINDOW]
                threshold = float(win['threshold'].iloc[0])
                if (win['value'] >= threshold).any():
                    continue

                row = {
                    'source': 'raw',
                    'label_type': 'no_alarm',
                    'label': 0,
                    'sbbsm': sbbsm,
                    'jczb_code': code,
                    'alarm_time': '',
                    'threshold': threshold,
                }
                for i, v in enumerate(win['value'].tolist(), 1):
                    row[f'v_{i}'] = v
                rows.append(row)
                if len(rows) >= n:
                    break

    return pd.DataFrame(rows)


def zero_head_count(row):
    cnt = 0
    for v in row:
        if v == 0:
            cnt += 1
        else:
            break
    return cnt


def zero_tail_count(row):
    cnt = 0
    for v in row[::-1]:
        if v == 0:
            cnt += 1
        else:
            break
    return cnt


def zero_max_count(row):
    best = 0
    cur = 0
    for v in row:
        if v == 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def add_part_features(df, vals):
    parts = {
        'early': vals.iloc[:, :7],
        'mid': vals.iloc[:, 7:13],
        'late': vals.iloc[:, 13:],
    }
    for name, part in parts.items():
        df[f'{name}_mean'] = part.mean(axis=1)
        df[f'{name}_max'] = part.max(axis=1)
        df[f'{name}_std'] = part.std(axis=1).fillna(0)
        df[f'{name}_zero_rate'] = (part == 0).mean(axis=1)

    df['late_early_mean_diff'] = df['late_mean'] - df['early_mean']
    df['late_early_max_diff'] = df['late_max'] - df['early_max']
    return df


def add_features(df):
    vcols = [f'v_{i}' for i in range(1, WINDOW + 1)]
    vals = df[vcols].apply(pd.to_numeric, errors='coerce')
    df[vcols] = vals
    df['valid_cnt'] = vals.notna().sum(axis=1)
    df[vcols] = vals.ffill(axis=1).bfill(axis=1).fillna(0)

    vals = df[vcols]
    threshold = pd.to_numeric(df['threshold'], errors='coerce').replace(0, pd.NA)
    ratio = vals.div(threshold, axis=0)

    df['win_mean'] = vals.mean(axis=1)
    df['win_max'] = vals.max(axis=1)
    df['win_min'] = vals.min(axis=1)
    df['win_std'] = vals.std(axis=1).fillna(0)
    df['win_zero_rate'] = (vals == 0).mean(axis=1)
    df['win_above_thr_cnt'] = ratio.ge(1).sum(axis=1)
    df['win_above_80_rate'] = ratio.ge(0.8).mean(axis=1)
    df['win_above_50_rate'] = ratio.ge(0.5).mean(axis=1)
    df['win_max_ratio'] = ratio.max(axis=1)
    df['win_last_ratio'] = vals[f'v_{WINDOW}'] / threshold
    df['win_range'] = df['win_max'] - df['win_min']
    df['win_first_last_diff'] = vals[f'v_{WINDOW}'] - vals['v_1']
    diff = vals.diff(axis=1)
    df['win_max_jump'] = diff.abs().max(axis=1).fillna(0)
    df['win_diff_mean'] = diff.mean(axis=1).fillna(0)
    df['win_rising_rate'] = diff.gt(0).sum(axis=1) / (WINDOW - 1)
    df['win_fall_rate'] = diff.lt(0).sum(axis=1) / (WINDOW - 1)
    df['win_skew'] = vals.skew(axis=1).fillna(0)
    df['win_kurtosis'] = vals.kurtosis(axis=1).fillna(0)
    df['win_max_pos'] = vals.values.argmax(axis=1) + 1
    df['win_max_pos_rate'] = df['win_max_pos'] / WINDOW
    df['win_max_near_end'] = (df['win_max_pos'] >= WINDOW - 4).astype(int)

    zero_rows = vals.to_numpy()
    df['head_zero_cnt'] = [zero_head_count(row) for row in zero_rows]
    df['tail_zero_cnt'] = [zero_tail_count(row) for row in zero_rows]
    df['max_consecutive_zero'] = [zero_max_count(row) for row in zero_rows]
    df = add_part_features(df, vals)
    df['threshold'] = pd.to_numeric(df['threshold'], errors='coerce').fillna(0)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=OUT_FILE)
    parser.add_argument('--no-alarm-ratio', type=int, default=NO_ALARM_PER_EVENT)
    args = parser.parse_args()

    events = event_windows()
    no_alarm_n = len(events) * args.no_alarm_ratio
    normals = no_alarm_windows(no_alarm_n)
    df = pd.concat([events, normals], ignore_index=True)
    df = add_features(df)
    df.to_csv(args.out, index=False, encoding='utf-8-sig')

    print(
        f'保存: {args.out}\n'
        f'总样本: {len(df):,}\n'
        f'{df["label_type"].value_counts().to_string()}'
    )


if __name__ == '__main__':
    main()
