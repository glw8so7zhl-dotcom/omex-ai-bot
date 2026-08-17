import os


# ============================================================
# Telegram
# ============================================================

ALLOWED_USER_IDS = {
    os.getenv("ADMIN_USER_ID", ""): "Osama"
}

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")


# ============================================================
# AI API Keys
# ============================================================

# OpenAI - optional
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Anthropic - optional
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Replicate - optional
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")

# OpenRouter - used for the free AI model
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# ============================================================
# Ollama
# ============================================================

OLLAMA_HOST = os.getenv("OLLAMA_HOST")


# ============================================================
# Default LLM Model
# ============================================================

# The project already defines this model as an OpenRouter model:
#
# deepseek/deepseek-r1-0528:free
#
# Therefore we use it as the default instead of gpt-5-mini,
# which requires OpenAI API quota.
#
# This can still be overridden through Railway Variables
# using DEFAULT_LLM_MODEL.

DEFAULT_LLM_MODEL = os.getenv(
    "DEFAULT_LLM_MODEL",
    "deepseek/deepseek-r1-0528:free"
)


# ============================================================
# YouTube Downloader
# ============================================================

YT_DL_DIR = os.getenv("YT_DL_DIR")
YT_DL_URL = os.getenv("YT_DL_URL")


# ============================================================
# Schedules
# ============================================================

SCHEDULES = None


# ============================================================
# User Permissions
# ============================================================

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


# ============================================================
# MCP
# ============================================================

MCP_FETCH_URL = os.getenv("MCP_FETCH_URL")


# ============================================================
# WebApp
# ============================================================

WEBAPP_PORT = int(os.getenv("PORT", "8180"))

GREEK_LEARNING_WEBAPP_URL = os.getenv(
    "GREEK_LEARNING_WEBAPP_URL",
    f"http://localhost:{WEBAPP_PORT}/greek/"
)

WEBAPP_BASE_URL = os.getenv(
    "WEBAPP_BASE_URL",
    f"http://localhost:{WEBAPP_PORT}"
)