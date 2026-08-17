import os

ALLOWED_USER_IDS = {
    os.getenv("ADMIN_USER_ID", ""): "Osama"
}

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OLLAMA_HOST = os.getenv("OLLAMA_HOST")

DEFAULT_LLM_MODEL = os.getenv(
    "DEFAULT_LLM_MODEL",
    "gpt-5-mini"
)

YT_DL_DIR = os.getenv("YT_DL_DIR")
YT_DL_URL = os.getenv("YT_DL_URL")

SCHEDULES = None

USER_PERMISSIONS = {
    os.getenv("ADMIN_USER_ID", ""): {
        "is_admin": True,
        "can_use_tools": True,
        "can_use_ollama_llm_models": False,
        "can_use_replicate_models": True,
        "can_use_memory_tool": True,
        "exclude_replicate_models": []
    },
    "default": {
        "is_admin": False,
        "can_use_tools": True,
        "can_use_ollama_llm_models": False,
        "can_use_replicate_models": False,
        "can_use_memory_tool": False,
        "exclude_replicate_models": []
    }
}

MCP_FETCH_URL = os.getenv("MCP_FETCH_URL")

WEBAPP_PORT = int(os.getenv("PORT", "8180"))

GREEK_LEARNING_WEBAPP_URL = os.getenv(
    "GREEK_LEARNING_WEBAPP_URL",
    f"http://localhost:{WEBAPP_PORT}/greek/"
)

WEBAPP_BASE_URL = os.getenv(
    "WEBAPP_BASE_URL",
    f"http://localhost:{WEBAPP_PORT}"
)