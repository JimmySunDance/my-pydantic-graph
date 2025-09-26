from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider


OLLAMA_MODEL = OpenAIChatModel(
    model_name="qwen2.5:32b",
    provider=OllamaProvider(base_url="http://localhost:11434/v1")
)