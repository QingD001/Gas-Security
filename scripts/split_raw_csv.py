import argparse
import csv
import os


RAW_FILE = 'rq_jc_ssjc_202605131842.csv'
OUT_DIR = 'data'
ROWS_PER_FILE = 500000


def getfile(name):
    if os.path.exists(name):
        return name
    root = os.path.join(os.path.dirname(__file__), '..', name)
    if os.path.exists(root):
        return root
    raise FileNotFoundError(name)


def open_new(out_dir, idx, header):
    name = os.path.join(out_dir, f'rq_jc_ssjc_part_{idx:03d}.csv')
    if os.path.exists(name):
        raise FileExistsError(f'{name} already exists')

    f = open(name, 'w', newline='', encoding='utf-8-sig')
    w = csv.writer(f)
    w.writerow(header)
    return name, f, w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=ROWS_PER_FILE)
    parser.add_argument('--out-dir', default=OUT_DIR)
    args = parser.parse_args()

    src = getfile(RAW_FILE)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(src, 'r', newline='', encoding='utf-8-sig') as f:
        r = csv.reader(f)
        header = next(r)

        part = 1
        row_in_part = 0
        total = 0
        name, out, w = open_new(args.out_dir, part, header)
        print(f'写入 {name}')

        try:
            for row in r:
                if row_in_part >= args.rows:
                    out.close()
                    part += 1
                    row_in_part = 0
                    name, out, w = open_new(args.out_dir, part, header)
                    print(f'写入 {name}，已处理 {total:,} 行')

                w.writerow(row)
                row_in_part += 1
                total += 1
        finally:
            out.close()

    print(f'完成，总行数 {total:,}，文件数 {part}')


if __name__ == '__main__':
    main()
