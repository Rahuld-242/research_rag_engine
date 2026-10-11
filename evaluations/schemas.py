from typing import Literal
from pydantic import BaseModel

class RelevanceJudgement(BaseModel):
    grade: Literal[0,1,2]
    reason: str
    