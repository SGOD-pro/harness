from __future__ import annotations
import re
from typing import AsyncGenerator
from .event import AgentEvent,AgentEventType
from client.llm_client import LLMClient
from client.response import StreamEventType,ToolCall,ToolResultMessage
from context.manager import ContextManager
from tools.registry import create_default_registry
from pathlib import Path
_THINK_RE = re.compile(r"^(?:<think>)?.*?</think>|<think>.*?</think>", re.DOTALL)

def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models."""
    return _THINK_RE.sub("", text).lstrip("\n")

class Agent:
    def __init__(self):
        self.client=LLMClient()
        self.context_manager = ContextManager()
        self.tool_registry = create_default_registry()

    async def run(self, message: str):
        yield AgentEvent.agent_start(message)
        self.context_manager.add_user_message(message)


        final_res: str | None = None
        async for event in self._agentic_loop(message):
            yield event

            if event.type == AgentEventType.TEXT_COMPLETE:
                final_res = event.data.get("content")

        yield AgentEvent.agent_end(final_res, None)

    async def _agentic_loop(self, message: str):
        messages = self.context_manager.get_messages()

        raw_response = ""

        tool_schema = self.tool_registry.get_schemas()
        tool_calls: list[ToolCall] = []
        async for event in self.client.chat_complition(
            messages, 
            tools= tool_schema if tool_schema else None, 
            stream=True
        ):
        
            if event.type == StreamEventType.TEXT_DELTA:
                if event.text_delta:
                    raw_response += event.text_delta.content
            elif event.type == StreamEventType.TOOL_CALL_COMPLETE:
                if event.tool_call:
                    tool_calls.append(event.tool_call)
            elif event.type == StreamEventType.ERROR:
                yield AgentEvent.agent_error(
                    event.error or "Unknown error occurred",
                    None
                )

        

        
        if raw_response:
            response = _strip_think(raw_response)
            for char in response:
                yield AgentEvent.text_delta(char)
            self.context_manager.add_assistant_message(response)
            yield AgentEvent.text_complete(response)
        self.context_manager.add_assistant_message(None)

        tool_call_results: list[ToolResultMessage]=[]

        for tool_call in tool_calls:
            yield AgentEvent.tool_call_start(
                tool_call.call_id,
                tool_call.name,
                tool_call.arguments
            )

            result= await self.tool_registry.invoke(
                tool_call.name,
                tool_call.arguments,
                Path.cwd()
            )

            yield AgentEvent.tool_call_complete(
                tool_call.call_id,
                tool_call.name,
                result
            )
            
            tool_call_results.append(
                ToolResultMessage(
                    tool_call.call_id,
                    content = result.to_model_ouput(),
                    is_error = not result.sucess
                )
            )
        
        for tool_result in tool_call_results:
            self.context_manager.add_tool_result(
                tool_result.tool_call_id,
                tool_result.content,
            )

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(self,exc_type, exc_tb,exc_val) -> Agent:
        if self.client:
            await self.client.close()
            self.client=None
        