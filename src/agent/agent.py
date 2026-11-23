from __future__ import annotations
import re
from typing import AsyncGenerator
from .event import AgentEvent,AgentEventType
from client.llm_client import LLMClient
from client.response import StreamEventType
from context.manager import ContextManager
from tools.registry import create_default_registry
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

        tool_schema=self.tool_registry.get_schemas()

        async for event in self.client.chat_complition(
            messages, 
            tools= tool_schema if tool_schema else None, 
            stream=True
        ):
            print(event)
            if event.type == StreamEventType.TEXT_DELTA:
                if event.text_delta:
                    raw_response += event.text_delta.content

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
            

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(self,exc_type, exc_tb,exc_val) -> Agent:
        if self.client:
            await self.client.close()
            self.client=None
        