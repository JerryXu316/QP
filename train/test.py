import os
import pandas as pd

def check_csv_file(file_path):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在！")
        return

    # 获取文件大小
    file_size = os.path.getsize(file_path)
    print(f"文件大小: {file_size} 字节")

    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 检查行数和列数
        num_rows, num_cols = df.shape
        print(f"行数: {num_rows}")
        print(f"列数: {num_cols}")

        # 检查是否有缺失值
        missing_values = df.isnull().sum()
        print("缺失值统计：")
        print(missing_values)

        # 检查某个关键字段是否连续（假设字段名为 'id'）
        if 'id' in df.columns:
            print("检查 'id' 字段是否连续：")
            if df['id'].is_monotonic_increasing:
                print("字段 'id' 是连续的。")
            else:
                print("字段 'id' 不是连续的。")
        else:
            print("字段 'id' 不存在。")

        # 检查文件的前几行和后几行
        print("\n文件的前5行：")
        print(df.head())
        print("\n文件的后5行：")
        print(df.tail())

    except pd.errors.EmptyDataError:
        print("CSV文件为空！")
    except pd.errors.ParserError:
        print("CSV文件格式错误，无法解析！")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

# 调用函数检查文件
file_path = 'initial_generated_data.csv'
check_csv_file(file_path)