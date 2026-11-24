from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any

@dataclass
class TextDelta:
    content:str

    def __str__(self):
        return self.content


class StreamEventType(str,Enum):
    TEXT_DELTA = "text_delta"
    MESSAGE_COMPLETE="message completed"
    ERROR= "error"

    TOOL_CALL_START="tool_call_start"
    TOOL_CALL_DELTA="tool_call_delta"
    TOOL_CALL_COMPLETE="tool_call_complete"

@dataclass
class TokenUsage:
    prompt_token: int= 0
    completion_token: int  =0
    total_token: int =0
    cache_token:int=0

    def __add__(self,other:TokenUsage):
        return TokenUsage(
            prompt_token=self.prompt_token + other.prompt_token,
            completion_token=self.completion_token + other.completion_token,
            total_token=self.total_token + other.total_token,
            cache_token=self.cache_token + other.cache_token,
        )

@dataclass
class ToolCallDelta:
    call_id:str
    name:str|None = None
    arguments_delta : str = ""

@dataclass
class ToolCall:
    call_id:str = ""
    name:str|None = None
    arguments: str = ""



@dataclass
class StreamEvent:
    type:EventType
    text_delta:TextDelta | None = None
    error:str|None=None
    finish_reason:str|None=None
    usage:TokenUsage |None=None

    tool_call_delta: ToolCallDelta|None = None
    tool_call: ToolCall|None = None


@dataclass
class ToolResultMessage:
    tool_call_id:str
    content: str
    is_error:bool = False

    @property
    def to_openai_message(self) -> dict[str,Any]:
        return {
            'role':'tool',
            'tool_call_id':self.tool_call_id,
            'content': self.content
        }

import json
def parse_tool_call_arguments(arguments_str:str) -> dict[str,Any]:
    if not arguments_str:
        return{}
    try:
        return json.loads(arguments_str)
    except json.JSONDecodeError:
        return {'raw_arguments':arguments_str}