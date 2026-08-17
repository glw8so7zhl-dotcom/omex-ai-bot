import asyncio
import os
import time

from telebot import ExceptionHandler
from telebot.types import Message, WebAppInfo
from telebot.asyncio_storage import StateMemoryStorage
from telebot import types

from telebot_nav import TeleBotNav

from config import TELEGRAM_TOKEN
from config import ALLOWED_USER_IDS

from lib.permissions import is_replicate_available
from lib.user_helpers import get_user_display_name

import config
import openai_module
import llm_module
import replicate_module
import youtube_dl_module
import scheduler_module
import tools_module
import greek_learning_module
import webapp_server
import webapp_apps_module

from logger import logger


class ExceptionH(ExceptionHandler):
    def handle(self, exception: Exception):
        logger.exception(exception)


async def _open_shared_webapp(
    botnav: TeleBotNav,
    message: Message,
    app_id: str
) -> None:
    """Respond to a deep-link that points at a user-generated web app."""

    app_path = os.path.join(
        os.path.dirname(__file__),
        "webapp",
        "apps",
        app_id,
        "index.html"
    )

    if not os.path.exists(app_path):
        await botnav.bot.send_message(
            message.chat.id,
            "Web app not found."
        )
        return

    app_url = (
        f"{config.WEBAPP_BASE_URL.rstrip('/')}"
        f"/apps/{app_id}/index.html"
    )

    markup = types.InlineKeyboardMarkup()

    markup.add(
        types.InlineKeyboardButton(
            text="🌐 Open Web App",
            web_app=WebAppInfo(url=app_url)
        )
    )

    await botnav.bot.send_message(
        message.chat.id,
        "Click the button to open the shared web app:",
        reply_markup=markup
    )


async def start(
    botnav: TeleBotNav,
    message: Message
) -> None:

    botnav.clean_default_handler(message)
    botnav.clean_next_handler(message)

    user = botnav.get_user(message)

    user_id = user.id
    user_id_str = str(user_id)

    display_name = get_user_display_name(user_id)

    logger.info(
        f"{display_name} (ID: {user_id}) {message.chat.id}"
    )

    # ---------------------------------------------------------
    # Shared Web App deep-link
    # ---------------------------------------------------------

    text = message.text or ""
    args = text.split()

    if len(args) > 1 and args[1].startswith("app_"):
        app_id = args[1][4:]

        await _open_shared_webapp(
            botnav,
            message,
            app_id
        )

        return

    # ---------------------------------------------------------
    # User access control
    # ---------------------------------------------------------

    if ALLOWED_USER_IDS and user_id_str not in ALLOWED_USER_IDS:

        logger.info(
            f"{display_name} (ID: {user_id}) not allowed"
        )

        await botnav.bot.send_message(
            message.chat.id,
            "Build your own bot, here is a source code:\n"
            "https://github.com/Sets88/sets88_telegram_bot"
        )

        return

    # ---------------------------------------------------------
    # Main menu
    # ---------------------------------------------------------

    buttons = {}

    # OpenAI legacy menu
    if config.OPENAI_API_KEY:
        buttons["🧠 OpenAI"] = openai_module.start_openai

    # LLM Router
    #
    # IMPORTANT:
    # Only expose the LLM menu if LLMRouter actually exists.
    #
    if (
        hasattr(llm_module, "LLMRouter")
        and (
            config.OPENAI_API_KEY
            or config.ANTHROPIC_API_KEY
            or config.OPENROUTER_API_KEY
            or config.OLLAMA_HOST
        )
    ):
        buttons["🧠 LLM"] = llm_module.LLMRouter.run

    # Greek Learning
    if config.GREEK_LEARNING_WEBAPP_URL:
        buttons["🇬🇷 Greek Learning"] = (
            greek_learning_module.start_greek
        )

    # Replicate
    if is_replicate_available(user_id):
        buttons["💻 Replicate"] = (
            replicate_module.start_replicate
        )

    # My Web Apps
    if config.WEBAPP_BASE_URL:
        buttons["📱 My Web Apps"] = (
            webapp_apps_module.start_my_apps
        )

    # Youtube
    buttons["📼 Youtube-DL"] = (
        youtube_dl_module.start_youtube_dl
    )

    # Tools
    buttons["Tools"] = tools_module.start_tools

    # Scheduled scripts
    if config.SCHEDULES:
        buttons["Scheduled scripts"] = (
            scheduler_module.start_schedules
        )

    await botnav.print_buttons(
        message.chat.id,
        buttons,
        "Choose",
        row_width=2
    )

    # ---------------------------------------------------------
    # Telegram commands
    # ---------------------------------------------------------

    botnav.wipe_commands(message)

    botnav.add_command(
        message,
        "start",
        "🏁 Start the bot",
        start
    )

    await botnav.send_commands(message)


async def main() -> None:

    # ---------------------------------------------------------
    # Scheduler
    # ---------------------------------------------------------

    if config.SCHEDULES:
        await scheduler_module.manager.run(botnav)

    # ---------------------------------------------------------
    # Web App server
    # ---------------------------------------------------------

    if config.GREEK_LEARNING_WEBAPP_URL:

        asyncio.create_task(
            webapp_server.start_server(botnav)
        )

    # ---------------------------------------------------------
    # Telegram bot
    # ---------------------------------------------------------

    await botnav.send_init_commands(
        {
            "start": "🏁 Start the bot"
        }
    )

    await botnav.set_global_default_handler(start)

    await botnav.bot.polling(
        non_stop=True
    )


# -------------------------------------------------------------
# Telegram Bot
# -------------------------------------------------------------

botnav = TeleBotNav(
    TELEGRAM_TOKEN,
    state_storage=StateMemoryStorage(),
    exception_handler=ExceptionH()
)


# -------------------------------------------------------------
# Application entry point
# -------------------------------------------------------------

if __name__ == "__main__":

    try:

        asyncio.run(main())

    except Exception as exc:

        logger.exception(exc)

        time.sleep(10)

        raise exc