from pydantic import BaseModel
from enum import Enum


class LlmName(str, Enum):
    Mistral = "mistral:7b-instruct-v0.3-fp16"
    GPT_4o = "gpt-4o"
    Llama = "llama3:8b-instruct-fp16"


class TickerReqDto(BaseModel):
    symbol: str
    sec_api_key: str
    llm_api_key: str
    llm_name: LlmName
