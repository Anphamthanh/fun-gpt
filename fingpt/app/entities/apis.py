from pydantic import BaseModel
from enum import Enum


class LlmName(str, Enum):
    GPT_4o = "gpt-4o"


class TickerReqDto(BaseModel):
    symbol: str
    sec_api_key: str
    llm_api_key: str
    llm_name: LlmName = LlmName.GPT_4o
