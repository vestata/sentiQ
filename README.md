# sentiQ
SentIQ is an intelligent security system that combines RAG, LLM, and sensor data to detect threats, monitor behaviors, and generate alerts for enhanced safety .

## Quick Usage
將開發文件中的 openai API key 複製到 `config.py` 中。
使用 `conda` 建立虛擬環境，並使用以下命令安裝環境：
```
pip install -r requirements.txt
```
使用以下命令進行簡單的測試：
```
python3 run.py
```

為了省去本地大型語言模型的建置時間，目前測試是使用 openai API 以加速功能的開發，openai API key 不能上傳 github，請去開發文件中查閱。

## githook
執行命令 `bash setup-hooks.sh` 來啟動 git hook，在每次 commit 時執行以下內容：
- 自動排版 git commit message
- 自動排版 python 程式
- 自動清除 API key
