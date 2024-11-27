import os
import pandas as pd
from datetime import datetime


def save_output_to_csv(question, model, output, file):

    file_name = f"{file}_output.csv"
    # 抽取需要儲存的資料
    try:
        plain_generate = output["plain_generate"][1]
    except:
        plain_generate = ""
    try:
        rag_generate = output["rag_generate"][1]
    except:
        rag_generate = ""
    
    try: 
        plain_dangerous = output["plain_generate"][0]
    except:
        plain_dangerous = ""
    try:
        rag_dangerous = output["rag_generate"][0]
    except:
        rag_dangerous = ""
        
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 將資料存成 DataFrame 格式
    new_data = pd.DataFrame(
        {
            "question": [question],
            "model": [model],
            "plain_dangerous": [plain_dangerous],
            "plain": [plain_generate],
            "rag_dangerous": [rag_dangerous],
            "rag + web": [rag_generate],
        },
        index=[timestamp],
    )

    # 檢查檔案是否已存在
    if os.path.exists(file_name):
        # 若檔案存在則附加資料
        existing_data = pd.read_csv(file_name, index_col=0)
        updated_data = pd.concat([existing_data, new_data])
    else:
        # 若檔案不存在則建立新檔案
        updated_data = new_data

    # 儲存至 CSV
    updated_data.to_csv(file_name)
    print(f"Results saved to {file_name}.")
