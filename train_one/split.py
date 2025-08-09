import csv
import os
from pathlib import Path

CSV_FILE = 'initial_generated_data.csv'
OUTPUT_DIR = Path('./input')
N_SPLITS = 100        # 固定 100 份

# 1. 计算总行数（不含表头）
with open(CSV_FILE, 'r', newline='') as f:
    total_rows = sum(1 for _ in f) - 1          # 减去表头

rows_per_split = total_rows // N_SPLITS
remainder = total_rows % N_SPLITS               # 余数，前 remainder 份多分 1 行

# 2. 创建输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 3. 按块写出文件
with open(CSV_FILE, 'r', newline='') as infile:
    reader = csv.reader(infile)
    header = next(reader)                       # 读取表头

    start_row = 0
    for idx in range(N_SPLITS):
        # 计算当前文件应有的行数
        extra = 1 if idx < remainder else 0
        end_row = start_row + rows_per_split + extra

        # 构造输出文件名
        out_path = OUTPUT_DIR / f'{idx}.csv'

        # 写出
        with open(out_path, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)
            for _ in range(end_row - start_row):
                try:
                    writer.writerow(next(reader))
                except StopIteration:
                    break

        start_row = end_row

print(f'Done. {total_rows} rows split into {N_SPLITS} files in {OUTPUT_DIR}')