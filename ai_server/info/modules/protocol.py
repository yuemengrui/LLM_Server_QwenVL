# *_*coding:utf-8 *_*
# @Author : YueMengRui
from pydantic import BaseModel, Field
from typing import Dict, List


class ErrorResponse(BaseModel):
    object: str = 'error'
    errcode: int
    errmsg: str


class ChatRequest(BaseModel):
    prompt: List
    history: List = Field(default=[], description="历史记录")
    generation_configs: Dict = Field(default={})
    stream: bool = Field(default=True, description="是否流式输出")


class TokenCountRequest(BaseModel):
    prompt: str


class TokenCountResponse(BaseModel):
    object: str = 'token_count'
    model_name: str
    prompt: str
    prompt_tokens: int
    max_tokens: int
    status: str
