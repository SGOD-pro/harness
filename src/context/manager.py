from prompts.system import get_system_prompt
from config.config import Config
from dataclasses import dataclass
from utils.text import count_tokens
from typing import Any

@dataclass
class MessageItem:
    role: str
    content: str
    token_count: int | None

    def to_dict(self) -> int | None:
        result: dict[str, Any] = {'role': self.role}

        if self.content:
            result['content']=self.content

        return result

class ContextManager:
    def __init__(self):
        self._system_prompt = get_system_prompt(Config())
        self._model_name = "qwen.qwen3-32b"
        self._messages: list[MessageItem] =[]

    def add_user_message(self,content) -> None:
        item=MessageItem(
            role="user",
            content=content,
            token_count=count_tokens(text=content,model=self._model_name)
        )

        self._messages.append(item)

    def add_assistant_message(self,content) -> None:
        content = content or ""
        item=MessageItem(
            role="assistant",
            content=content or "",
            token_count=count_tokens(text=content,model=self._model_name)
        )

        self._messages.append(item)

    def get_messages(self) -> list[dict[str,Any]]:
        messages=[]

        if self._system_prompt:
            messages.append({
                'role':"system",
                'content':self._system_prompt
            })

        for item in self._messages:
            messages.append(item.to_dict())

        return messages
