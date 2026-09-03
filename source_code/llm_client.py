"""
llm_client.py
-------------
Thin wrapper around the Google Gemini API.
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not found. Create a .env file in the Project/ "
                "folder with a line: GEMINI_API_KEY=your_key_here"
            )
        _client = genai.Client(api_key=api_key)
    return _client


def call_llm(prompt: str, system: str = None, max_tokens: int = 600, model: str = "gemini-flash-latest",
             json_mode: bool = False) -> str:
    client = get_client()
    config_kwargs = dict(
        max_output_tokens=max_tokens,
        system_instruction=system if system else None,
    )
    if json_mode:
        # Forces the API itself to only ever return syntactically valid JSON -
        # far more reliable than asking nicely in the prompt and hoping.
        config_kwargs["response_mime_type"] = "application/json"
    config = types.GenerateContentConfig(**config_kwargs)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return response.text