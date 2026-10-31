# import os
# from dotenv import load_dotenv
# from openai import OpenAI

# load_dotenv()

# MODEL = "openai.gpt-oss-20b"

# client = OpenAI(
#     api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"],
#     base_url="https://bedrock-mantle.ap-south-1.api.aws/v1",
# )


# def call_llm(messages: list, tools: list = None) -> tuple:
#     """Call the LLM and return (message, usage)."""
#     kwargs = {
#         "model": MODEL,
#         "messages": messages,
#     }
#     if tools:
#         kwargs["tools"] = tools
#         kwargs["tool_choice"] = "auto"

#     response = client.chat.completions.create(**kwargs)

#     message = response.choices[0].message
#     usage = response.usage
#     return message, usage


from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError
from typing import Any, AsyncGenerator
import os
from dotenv import load_dotenv
load_dotenv()

class LLmClient:
    def __init__(self):
        self._client: AsyncOpenAI  |None =None

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
            stream: bool=False
        ):

        client=self.get_client()

        kwargs={
            "model":"qwen.qwen3-32b",
            "messages":messages,
            "stream":stream,
        }

        if stream:
            self._stream_response()

        else:
            self._non_stream_response(client,kwargs)

    
    async def _stream_response(self):
        pass
    
    async def _non_stream_response(
            self,
            client: AsyncOpenAI,
            kwargs:dict[str,Any]
        ):
        res=await client.chat.completions.create(**kwargs)
        print(res)





