from langchain_openai.chat_models import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

"""### Retrieval Grader"""

class GradeDocuments(BaseModel):
    """
    確認提取文章與問題是否有關(yes/no)
    """

    binary_score: str = Field(description="請問文章與問題是否相關。('yes' or 'no')")

"""### Hallucination Grader"""

class GradeHallucinations(BaseModel):
    """
    確認答案是否為虛構(yes/no)
    """

    binary_score: str = Field(description="答案是否由為虛構。('yes' or 'no')")

"""### Answer Grader"""

class GradeAnswer(BaseModel):
    """
    確認答案是否可回應問題
    """

    binary_score: str = Field(description="答案是否回應問題。('yes' or 'no')")