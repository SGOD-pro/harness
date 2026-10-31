from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError
from typing import Any, AsyncGenerator
from .response import TextDelta,TokenUsage,StreamEvent,EventType
import os
from dotenv import load_dotenv
load_dotenv()
import asyncio

class LLMClient:
    def __init__(self):
        self._client: AsyncOpenAI  |None =None
        self._max_retries:int =3

    def get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client=AsyncOpenAI(
                    api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"],
                    base_url="https://bedrock-mantle.ap-south-1.api.aws/v1",
                )
        
        return self._client



    async def close(self):
        if self._client:
            await self._client.close()
            self._client=None

    async def chat_complition(
            self,
            messages:list[dict[str,Any]],
            stream: bool
        ) -> AsyncGenerator[StreamEvent,None]:

        client=self.get_client()

        kwargs={
                "model":"qwen.qwen3-32b",
                "messages":messages,
                "stream":stream,
            }

        for attempt in range(self._max_retries + 1):
            try:

                if stream:
                    async for event in self._stream_response(client,kwargs):
                        yield event

                else:
                    event = await self._non_stream_response(client,kwargs)
                    yield event

                return
                
            except RateLimitError as e:
                if attempt < self._max_retries:
                    wait = 2**attempt

                    await asyncio.sleep(wait)
                else:
                    yield StreamEvent(
                        type=EventType.ERROR,
                        error=f"RateLimit Excced {e}"
                    )

                return
            except APIConnectionError as e:
                if attempt < self._max_retries:
                    wait = 2**attempt

                    await asyncio.sleep(wait)
                else:
                    yield StreamEvent(
                        type=EventType.ERROR,
                        error=f"Connection error: {e}"
                    )

                return
            except APIError as e:
                yield StreamEvent(
                        type=EventType.ERROR,
                        error=f"API error: {e}"
                    )

                return

                

    
    async def _stream_response(
            self,
            client: AsyncOpenAI,
            kwargs:dict[str,Any]
        ) -> AsyncGenerator[StreamEvent,None]:
        
        res = await client.chat.completions.create(**kwargs)

        finish_reason:str|None=None
        usage: TokenUsage | None=None
        async for chunk in res:
            if hasattr(chunk,"usage") and chunk.usage:
                usage=TokenUsage(
                    prompt_token=chunk.usage.prompt_tokens,
                    completion_token=chunk.usage.completion_tokens,
                    total_token=chunk.usage.total_tokens,
                    cache_token=chunk.usage.prompt_tokens_details.cached_tokens if chunk.usage.prompt_tokens_details else None
                )
            if not chunk.choices:
                continue

            choise=chunk.choices[0]
            delta=choise.delta
            if choise.finish_reason:
                finish_reason=choise.finish_reason

            if delta.content:
                yield StreamEvent(
                    type=EventType.TEXT_DELTA,
                    text_delta=TextDelta(delta.content)

                )
    
        yield StreamEvent(
            type=EventType.MESSAGE_COMPLETE,
            finish_reason=finish_reason,
            usage=usage
        )



    async def _non_stream_response(
            self,
            client: AsyncOpenAI,
            kwargs:dict[str,Any]
        ):
        res=await client.chat.completions.create(**kwargs)
        choice=res.choices[0]
        message=choice.message

        text_delta=None

        if message.content:
            text_delta=TextDelta(message.content)


        usage=None

        if res.usage:
            usage=TokenUsage(
                    prompt_token=res.usage.prompt_tokens,
                    completion_token=res.usage.completion_tokens,
                    total_token=res.usage.total_tokens,
                    cache_token=res.usage.prompt_tokens_details.cached_tokens if res.usage.prompt_tokens_details else None
                )
        return StreamEvent(
            type=EventType.MESSAGE_COMPLETE,
            text_delta=text_delta,
            finish_reason=choice.finish_reason,
            usage=usage,
        )
