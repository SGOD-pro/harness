from typing import Any
from client.llm_client import LLMClient
import asyncio
import click

class CLI:
    def __init__(self):
        pass

    def run_single(self):
        pass


async def run(messages: list[dict[str, Any]]):
    client = LLMClient()
    async for event in client.chat_complition(messages, True):
        print(event)


@click.command()
@click.argument("prompt", nargs=-1)

# uv run main.py "hey whats up buddy"
def main(prompt: tuple[str, ...]):
    text = " ".join(prompt).strip() if prompt else "Hello!"
    messages = [
        {
            "role": "user",
            "content": text
        }
    ]
    asyncio.run(run(messages))
    


# asyncio.run(main())
main()
    