# fix_header.py
import csv
import os

src = 'three_solve_generated_data.csv'
bak = 'data.csv.bak'

# 1. 先备份
os.rename(src, bak)

# 2. 读取首行，插入缺失列名
with open(bak, newline='') as f_in, open(src, 'w', newline='') as f_out:
    r = csv.reader(f_in)
    w = csv.writer(f_out)

    # 取表头
    header = next(r)
    # 在 y(k) 后面插入 u(k-4)、u(k-5)
    new_header = header[:1] + ['u(k-4)', 'u(k-5)'] + header[1:]
    w.writerow(new_header)

    # 其余原样写回
    for row in r:
        w.writerow(row)

print('表头已修复，原文件备份为', bak)