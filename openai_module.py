import io
import asyncio
from io import BytesIO
import functools
from typing import BinaryIO

from openai import AsyncOpenAI
from telebot.types import Message
from pydub import AudioSegment

import config
from telebot_nav import TeleBotNav
from logger import logger


# ============================================================
# OpenAI Adapter
# ============================================================

class OpenAiAdapter:

    def __init__(self) -> None:

        self.conversations = {}

        self.client = AsyncOpenAI(
            api_key=config.OPENAI_API_KEY
        )

    # --------------------------------------------------------
    # DALL-E
    # --------------------------------------------------------

    async def dalle_generate_image(
        self,
        prompt: str
    ) -> str:

        response = await self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )

        return response.data[0].url

    # --------------------------------------------------------
    # Whisper
    # --------------------------------------------------------

    async def whisper_transcribe(
        self,
        audio: BinaryIO
    ) -> str:

        response = await self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audio,
        )

        return response.text

    # --------------------------------------------------------
    # TTS
    # --------------------------------------------------------

    async def tts_generate_audio(
        self,
        text: str,
        voice: str
    ) -> BinaryIO:

        response = await self.client.audio.speech.create(
            model="tts-1",
            input=text,
            voice=voice
        )

        return response


# ============================================================
# Whisper Router
# ============================================================

class WhisperRouter:

    @classmethod
    def get_mp3_from_ogg(
        cls,
        file_content: BinaryIO
    ) -> BytesIO:

        file = BytesIO(file_content)

        file.seek(0)

        ogg = AudioSegment.from_ogg(file)

        mp3 = BytesIO()

        ogg.export(
            mp3,
            format="mp3"
        )

        mp3.seek(0)

        return mp3

    # --------------------------------------------------------
    # Extract voice text
    # --------------------------------------------------------

    @classmethod
    async def extract_text_from_voice(
        cls,
        botnav: TeleBotNav,
        message: Message
    ) -> str:

        if not message.voice:
            return ""

        file_info = await botnav.bot.get_file(
            message.voice.file_id
        )

        file_content = await botnav.bot.download_file(
            file_info.file_path
        )

        file = await asyncio.to_thread(
            cls.get_mp3_from_ogg,
            file_content
        )

        file.name = "voice.mp3"

        text = await openai_instance.whisper_transcribe(
            file
        )

        return text

    # --------------------------------------------------------
    # Whisper message handler
    # --------------------------------------------------------

    @classmethod
    async def whisper_message_handler(
        cls,
        botnav: TeleBotNav,
        message: Message
    ) -> None:

        if message.content_type != "voice":
            return

        try:

            text = await botnav.await_coro_sending_action(
                message.chat.id,

                cls.extract_text_from_voice(
                    botnav,
                    message
                ),

                "typing"
            )

            if text:

                await botnav.bot.send_message(
                    message.chat.id,
                    text
                )

        except Exception as exc:

            await botnav.bot.send_message(
                message.chat.id,
                "Something went wrong, try again later"
            )

            logger.exception(exc)

    # --------------------------------------------------------
    # Start Whisper
    # --------------------------------------------------------

    @classmethod
    async def run(
        cls,
        botnav: TeleBotNav,
        message: Message
    ) -> None:

        botnav.wipe_commands(
            message,
            preserve=[
                "start",
                "openai"
            ]
        )

        await botnav.bot.send_message(
            message.chat.id,
            "Welcome to Whisper, send me voice message to transcribe!"
        )

        botnav.set_default_handler(
            message,
            cls.whisper_message_handler
        )

        botnav.clean_next_handler(
            message
        )

        await botnav.send_commands(
            message
        )


# ============================================================
# DALL-E Router
# ============================================================

class DallERouter:

    # --------------------------------------------------------
    # DALL-E message handler
    # --------------------------------------------------------

    @classmethod
    async def dalle_message_handler(
        cls,
        botnav: TeleBotNav,
        message: Message
    ) -> None:

        if message.content_type != "text":
            return

        try:

            url = await botnav.await_coro_sending_action(
                message.chat.id,

                openai_instance.dalle_generate_image(
                    message.text
                ),

                "upload_photo"
            )

            await botnav.bot.send_photo(
                message.chat.id,
                url
            )

        except Exception as exc:

            await botnav.bot.send_message(
                message.chat.id,
                "Something went wrong, try again later"
            )

            logger.exception(exc)

    # --------------------------------------------------------
    # Start DALL-E
    # --------------------------------------------------------

    @classmethod
    async def run(
        cls,
        botnav: TeleBotNav,
        message: Message
    ):

        botnav.wipe_commands(
            message,
            preserve=[
                "start",
                "openai"
            ]
        )

        await botnav.bot.send_message(
            message.chat.id,
            "Welcome to DALL-E, ask me to draw something!"
        )

        botnav.set_default_handler(
            message,
            cls.dalle_message_handler
        )

        botnav.clean_next_handler(
            message
        )

        await botnav.send_commands(
            message
        )


# ============================================================
# TTS Router
# ============================================================

class TTSRouter:

    # --------------------------------------------------------
    # TTS message handler
    # --------------------------------------------------------

    @classmethod
    async def tts_message_handler(
        cls,
        botnav: TeleBotNav,
        message: Message
    ) -> None:

        if message.content_type != "text":
            return

        if "openai_params" not in message.state_data:

            message.state_data[
                "openai_params"
            ] = {}

        voice = message.state_data[
            "openai_params"
        ].get(
            "tts_voice",
            "alloy"
        )

        try:

            response = await botnav.await_coro_sending_action(
                message.chat.id,

                openai_instance.tts_generate_audio(
                    message.text,
                    voice
                ),

                "upload_audio"
            )

            await botnav.bot.send_voice(
                message.chat.id,
                io.BytesIO(
                    response.content
                )
            )

        except Exception as exc:

            await botnav.bot.send_message(
                message.chat.id,
                "Something went wrong, try again later"
            )

            logger.exception(exc)

    # --------------------------------------------------------
    # Start TTS
    # --------------------------------------------------------

    @classmethod
    async def run(
        cls,
        botnav: TeleBotNav,
        message: Message
    ):

        await botnav.print_buttons(
            message.chat.id,

            {
                "Alloy": functools.partial(
                    set_openai_param,
                    "tts_voice",
                    "alloy"
                ),

                "Echo": functools.partial(
                    set_openai_param,
                    "tts_voice",
                    "echo"
                ),

                "Fable": functools.partial(
                    set_openai_param,
                    "tts_voice",
                    "fable"
                ),

                "Onyx": functools.partial(
                    set_openai_param,
                    "tts_voice",
                    "onyx"
                ),

                "Nova": functools.partial(
                    set_openai_param,
                    "tts_voice",
                    "nova"
                ),

                "Shimmer": functools.partial(
                    set_openai_param,
                    "tts_voice",
                    "shimmer"
                ),
            },

            "Available voices:",

            row_width=3
        )

        botnav.wipe_commands(
            message,
            preserve=[
                "start",
                "openai"
            ]
        )

        await botnav.bot.send_message(
            message.chat.id,
            "Welcome to TTS, send me text to speech!"
        )

        botnav.set_default_handler(
            message,
            cls.tts_message_handler
        )

        botnav.clean_next_handler(
            message
        )

        await botnav.send_commands(
            message
        )


# ============================================================
# OpenAI Parameters
# ============================================================

async def set_openai_param(
    param: str,
    value: str,
    botnav: TeleBotNav,
    message: Message
) -> None:

    if "openai_params" not in message.state_data:

        message.state_data[
            "openai_params"
        ] = {}

    message.state_data[
        "openai_params"
    ][param] = value

    await botnav.bot.send_message(
        message.chat.id,
        f"OpenAI param {param} was set to {value}"
    )


# ============================================================
# CHAT GPT
# ============================================================

async def start_chatgpt(
    botnav: TeleBotNav,
    message: Message
) -> None:

    # استيراد LLMRouter هنا لمنع Circular Import
    from llm_module import LLMRouter

    # إزالة أي handler سابق
    botnav.clean_next_handler(
        message
    )

    # الاحتفاظ بأوامر start و openai
    botnav.wipe_commands(
        message,
        preserve=[
            "start",
            "openai"
        ]
    )

    # ========================================================
    # IMPORTANT
    #
    # لا نشغل LLMRouter.run الآن.
    #
    # نجعل الرسالة القادمة من المستخدم هي التي تذهب إليه.
    # ========================================================

    botnav.set_default_handler(
        message,
        LLMRouter.run
    )

    await botnav.bot.send_message(
        message.chat.id,

        "ChatGPT جاهز.\n\n"
        "اكتب رسالتك الآن وسأرسلها إلى الذكاء الاصطناعي.\n\n"
        "للعودة إلى القائمة الرئيسية استخدم /start"
    )

    await botnav.send_commands(
        message
    )


# ============================================================
# DALL-E Button
# ============================================================

async def start_dalle(
    botnav: TeleBotNav,
    message: Message
) -> None:

    await DallERouter.run(
        botnav,
        message
    )


# ============================================================
# Whisper Button
# ============================================================

async def start_whisper(
    botnav: TeleBotNav,
    message: Message
) -> None:

    await WhisperRouter.run(
        botnav,
        message
    )


# ============================================================
# TTS Button
# ============================================================

async def start_tts(
    botnav: TeleBotNav,
    message: Message
) -> None:

    await TTSRouter.run(
        botnav,
        message
    )


# ============================================================
# OPENAI MENU
# ============================================================

async def start_openai(
    botnav: TeleBotNav,
    message: Message
) -> None:

    # لا نستورد LLMRouter في أعلى الملف
    # لمنع Circular Import.
    from llm_module import LLMRouter

    await botnav.print_buttons(
        message.chat.id,

        {
            # =================================================
            # IMPORTANT FIX
            #
            # لا تستخدم:
            #
            # "🤖 Chat GPT": LLMRouter.run
            #
            # لأن هذا يشغل ChatGPT فور الضغط على الزر.
            #
            # نستخدم start_chatgpt حتى ننتظر رسالة المستخدم.
            # =================================================

            "🤖 Chat GPT": start_chatgpt,

            "🖌️ Dall-E": start_dalle,

            "🗣️ Whisper": start_whisper,

            "💬 TTS": start_tts
        },

        "Choose",

        row_width=2
    )

    # ========================================================
    # Commands
    # ========================================================

    botnav.wipe_commands(
        message,
        preserve=[
            "start"
        ]
    )

    botnav.add_command(
        message,
        "openai",
        "🧠 OpenAI models",
        start_openai
    )

    await botnav.send_commands(
        message
    )


# ============================================================
# OpenAI Instance
# ============================================================

openai_instance = OpenAiAdapter()