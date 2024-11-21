from langchain_core.pydantic_v1 import BaseModel, Field

# 定義兩個工具的 DataModel
class web_search(BaseModel):
    """
    網路搜尋工具。若問題與如何識別可疑人物/物品"無關"，請使用web_search工具搜尋解答。
    """
    query: str = Field(description="使用網路搜尋時輸入的問題")
    
class vectorstore(BaseModel):
    """
    紀錄關於"如何識別可疑人物/物品"的向量資料庫工具。若問題與"如何識別可疑人物/物品"有關，請使用此工具搜尋解答。
    """
    query: str = Field(description="搜尋向量資料庫時輸入的問題")