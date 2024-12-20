import os
import pandas as pd
from datetime import datetime


def save_output_to_csv(question, output, file):
    file_name = f"{file}_output.csv"
    try:
        plain_generate = output["plain_generate"]
    except:
        plain_generate = ""
    try:
        rag_generate = output["rag_generate"]
    except:
        rag_generate = ""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_data = pd.DataFrame(
        {
            "question": [question],
            "plain": [plain_generate],
            "rag + web": [rag_generate],
        },
        index=[timestamp],
    )

    if os.path.exists(file_name):
        existing_data = pd.read_csv(file_name, index_col=0)
        updated_data = pd.concat([existing_data, new_data])
    else:
        updated_data = new_data

    updated_data.to_csv(file_name)
    print(f"Results saved to {file_name}.")
