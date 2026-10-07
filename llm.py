import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "openai.gpt-oss-20b"

client = OpenAI(
    api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"],
    base_url="https://bedrock-mantle.ap-south-1.api.aws/v1",
)


def call_llm(messages: list, tools: list = None) -> tuple:
    """Call the LLM and return (message, usage)."""
    kwargs = {
        "model": MODEL,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    response = client.chat.completions.create(**kwargs)

    message = response.choices[0].message
    usage = response.usage
    return message, usage
