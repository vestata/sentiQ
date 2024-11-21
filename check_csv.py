import pandas as pd

def check_csv_format(file_name="exp_output.csv"):
    try:
        # 讀取 CSV 檔案
        data = pd.read_csv(file_name, index_col=0)
        print(data)
    except FileNotFoundError:
        print(f"{file_name} not found. Make sure it has been created.")
    except Exception as e:
        print(f"Error reading {file_name}: {e}")

# 測試檢查
check_csv_format()