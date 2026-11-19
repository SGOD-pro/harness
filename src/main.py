from typing import Any
from client.llm_client import LLMClient
import asyncio
import click
import sys
from agent.agent import Agent
from agent.event import AgentEventType
from ui.tui import TUI,get_console


console=get_console()

class CLI:
    def __init__(self):
        self.agent:Agent | None =None
        self.tui=TUI(console)


    async def run_single(self,message:str) -> str | None:
        async with Agent() as agent:
            self.agent = agent
            return await self._process_message(message)

    async def _process_message(self,message: str) -> str |None:
        if not self.agent:
            return None

        assistant_streaming = False
        final_res: str | None = None

        async for event in self.agent.run(message):
            if event.type == AgentEventType.TEXT_DELTA:
               content = event.data.get("content","")
               if not assistant_streaming:
                   self.tui.begin_assist()
                   assistant_streaming = True

               self.tui.stream_assistant_delta(content)
            elif event.type == AgentEventType.TEXT_COMPLETE:
                final_res = event.data.get("content")
                if assistant_streaming:
                    assistant_streaming = False
            
            elif event.type == AgentEventType.AGENT_ERROR:
                error = event.data.get("error","Unknown error")
                # if assistant_streaming:
                console.print(f"\n[error] Error: {error} [/error]")


        if assistant_streaming:
            self.tui.end_assist()

        return final_res


# async def run(messages: list[dict[str, Any]]):
#     client = LLMClient()
#     async for event in client.chat_complition(messages, True):
#         print(event)


@click.command()
@click.argument("prompt", nargs=-1)

# uv run main.py "hey whats up buddy"
def main(prompt: tuple[str, ...]):
    text = " ".join(prompt).strip() if prompt else "Hello!"
    messages = [{"role": "user","content": text}]

    cli= CLI()
    if prompt:
        res=asyncio.run(cli.run_single(text))
        if res is None:
            sys.exit(1)

    # asyncio.run(run(messages))
    


# asyncio.run(main())
main()
    