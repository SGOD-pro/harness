"""
List all models available on the AWS Bedrock account, sorted by model ID.

Uses the same OpenAI-compatible endpoint as the harness agent.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"],
    base_url="https://bedrock-mantle.ap-south-1.api.aws/v1",
)


def list_models() -> list:
    """Fetch and return all available models, sorted by ID."""
    response = client.models.list()
    models   = list(response)
    models.sort(key=lambda m: m.id.lower())
    return models


if __name__ == "__main__":
    models = list_models()
    print(f"{'MODEL ID':<55}  {'OWNED BY'}")
    print("-" * 75)
    for m in models:
        owned_by = getattr(m, "owned_by", "—") or "—"
        print(f"{m.id:<55}  {owned_by}")
    print(f"\n{len(models)} model(s) available.")
