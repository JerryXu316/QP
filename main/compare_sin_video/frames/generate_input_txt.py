import os
import re

# 获取当前目录下所有png文件
files = [f for f in os.listdir('.') if f.endswith('.png')]

# 使用正则提取 k 和 l 的数值，并进行排序
pattern = re.compile(r'^(\d{3})_(\d{3})\.png$')
parsed_files = []

for f in files:
    match = pattern.match(f)
    if match:
        k = int(match.group(1))
        l = int(match.group(2))
        parsed_files.append((k, l, f))

# 按 k, l 排序
parsed_files.sort()

# 写入 input.txt（ffmpeg格式）
with open('input.txt', 'w') as f:
    for _, _, filename in parsed_files:
        f.write(f"file '{filename}'\n")

print("✅ 已生成 input.txt，可直接用 ffmpeg 拼接视频。")
