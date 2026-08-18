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

# Optional Facebook Page publishing
FACEBOOK_PAGE_ID = os.getenv("FACEBOOK_PAGE_ID", "")
FACEBOOK_PAGE_ACCESS_TOKEN = os.getenv("FACEBOOK_PAGE_ACCESS_TOKEN", "")
FACEBOOK_GRAPH_VERSION = os.getenv("FACEBOOK_GRAPH_VERSION", "v23.0")

# Vision model for product-image analysis. Override in Railway if needed.
VISION_MODEL = os.getenv(
    "VISION_MODEL",
    "nvidia/nemotron-nano-12b-v2-vl:free"
)


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
    "openai/gpt-oss-20b:free"
)

# ============================================================
# Performance / Speed
# ============================================================

AI_TIMEOUT_SECONDS = float(os.getenv("AI_TIMEOUT_SECONDS", "45"))
AI_MAX_RETRIES = int(os.getenv("AI_MAX_RETRIES", "1"))
TEXT_MAX_TOKENS = int(os.getenv("TEXT_MAX_TOKENS", "900"))
VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "1200"))

# Keep this False for maximum speed. If a vision model omits search metadata,
# the bot uses a local fallback instead of making a second AI request.
GENERATE_SEARCH_QUERIES_FALLBACK = (
    os.getenv("GENERATE_SEARCH_QUERIES_FALLBACK", "false").lower() == "true"
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