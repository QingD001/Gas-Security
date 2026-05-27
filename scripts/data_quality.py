import os
import pandas as pd

FILE = 'rq_jc_ssjc_202605131842.csv'
SAMPLE = 100000

def getfile(name):
    if os.path.exists(name):
        return name
    root = os.path.join(os.path.dirname(__file__), '..', name)
    if os.path.exists(root):
        return root
    raise FileNotFoundError(name)

def pct(n, total):
    return n / total if total else 0

def show_top(s, name, n=10):
    print(f'\n{name}:')
    print(s.value_counts(dropna=False).head(n).to_string())

file = getfile(FILE)

line = '=' * 60
print(f'{line}\n数据质量检查\n文件: {FILE}\n抽样: {SAMPLE:,} 行\n{line}')

df = pd.read_csv(file, nrows=SAMPLE)
rows = len(df)

print(f'\n1. 基本信息\n样本行数: {rows:,}\n列数: {len(df.columns)}\n列名: {df.columns.tolist()}')

if rows == 0:
    print('\n没读到数据，先停。')
    raise SystemExit

print('\n2. 缺失值')
null_count = df.isnull().sum()
null_rate = df.isnull().mean()

for col in df.columns:
    n = int(null_count[col])
    if n == 0:
        continue
    flag = '  建议删' if n == rows else ''
    print(f'  {col}: {n:,} ({null_rate[col]:.2%}){flag}')

print('\n3. 常数列')
for col in df.columns:
    uniq = df[col].nunique(dropna=True)
    if uniq <= 1:
        vals = df[col].dropna().unique()
        val = vals[0] if len(vals) else None
        print(f'  {col}: 只有一个值 {val}，基本没信息')

print('\n4. 重复值')
dup = int(df.duplicated().sum())
msg = f'  完全重复行: {dup:,} ({pct(dup, rows):.2%})'

if 'bsm' in df.columns:
    dup_bsm = int(df['bsm'].duplicated().sum())
    msg += f'\n  bsm重复: {dup_bsm:,} ({pct(dup_bsm, rows):.2%})'

print(msg)

print('\n5. 时间列')
for col in ['jccjsj', 'jcsbsj', 'tbsj']:
    if col not in df.columns:
        print(f'  {col}: 没有这列')
        continue

    d = pd.to_datetime(df[col], errors='coerce')
    bad = int(d.isnull().sum())
    print(f'  {col}: 有效 {rows - bad:,}，无效 {bad:,}，范围 [{d.min()}, {d.max()}]')

if 'jcz' in df.columns:
    print('\n6. jcz总体')
    jcz = pd.to_numeric(df['jcz'], errors='coerce')
    ok_jcz = jcz.dropna()
    not_zero = ok_jcz[ok_jcz != 0]

    print(f'  空值: {jcz.isnull().sum():,}')
    print(
        f'  0值: {(jcz == 0).sum():,} ({(jcz == 0).mean():.2%})\n'
        f'  负值: {(jcz < 0).sum():,} ({(jcz < 0).mean():.2%})\n'
        f'  正值: {(jcz > 0).sum():,}'
    )

    if len(ok_jcz):
        print(f'  min={ok_jcz.min():.4e}, max={ok_jcz.max():.4e}')

    if len(not_zero):
        print(f'  均值(去0)={not_zero.mean():.2f}')
        print(f'  标准差(去0)={not_zero.std():.2f}')

if 'jcz' in df.columns and 'jczb' in df.columns:
    print('\n7. jcz按监测指标分组')
    for name, grp in df.groupby('jczb', dropna=False):
        j = pd.to_numeric(grp['jcz'], errors='coerce')
        good = j.dropna()
        print(
            f'  {name}:\n'
            f'    样本={len(j):,}, 空值={j.isnull().sum():,}, '
            f'0值={(j == 0).sum():,}({(j == 0).mean():.2%})'
        )
        if not len(good):
            continue
        q1 = good.quantile(0.25)
        q3 = good.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        out = ((good < lower) | (good > upper)).sum()
        nz = good[good != 0]

        avg = f'    均值(去0)={nz.mean():.2f}' if len(nz) else '    均值(去0)=无'
        print(
            f'{avg}\n'
            f'    中位数={good.median():.2f}\n'
            f'    IQR范围=[{lower:.2f}, {upper:.2f}]\n'
            f'    异常值={out:,} ({pct(out, len(good)):.2%})\n'
            f'    min={good.min():.4e}, max={good.max():.4e}'
        )

print('\n8. 分类列')
if 'jczb' in df.columns:
    print(f'  jczb种类: {df["jczb"].nunique(dropna=False)}')
    show_top(df['jczb'], '  jczb分布')

if 'sjtbzt' in df.columns:
    show_top(df['sjtbzt'], '  sjtbzt分布')

if 'sbzt' in df.columns:
    show_top(df['sbzt'], '  sbzt分布')

if 'bz' in df.columns:
    print('\n9. bz非空样例')
    bz = df[df['bz'].notna()]['bz']
    print(f'  非空: {len(bz):,}')
    if len(bz):
        print(f'  样例: {bz.head(5).tolist()}')

print(f'\n{line}\n检查完\n{line}')
