import logging
from typing import Dict, List

from openai import AsyncOpenAI

import config


logger = logging.getLogger(__name__)


class LLMRouter:
    """
    LLM router for the Telegram bot.

    Priority:
    1. OpenRouter
    2. OpenAI
    3. Anthropic is reserved for future integration
    """

    conversations: Dict[int, List[dict]] = {}

    @classmethod
    def _get_client(cls):
        """
        Create the appropriate async AI client.
        """

        # Prefer OpenRouter because the configured default model
        # is an OpenRouter model.
        if config.OPENROUTER_API_KEY:
            return AsyncOpenAI(
                api_key=config.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
            )

        # Fallback to OpenAI.
        if config.OPENAI_API_KEY:
            return AsyncOpenAI(
                api_key=config.OPENAI_API_KEY
            )

        return None

    @classmethod
    def _get_model(cls) -> str:
        """
        Return the configured default model.
        """

        return getattr(
            config,
            "DEFAULT_LLM_MODEL",
            "deepseek/deepseek-r1-0528:free",
        )

    @classmethod
    def _get_system_prompt(cls) -> str:
        return (
            "You are OMEX AI Assistant. "
            "Answer clearly and accurately. "
            "The user may communicate in Arabic or English. "
            "If the user writes Arabic, respond in Arabic. "
            "Do not invent information."
        )

    @classmethod
    async def ask(
        cls,
        user_id: int,
        text: str,
    ) -> str:
        """
        Send a message to the configured LLM.
        """

        client = cls._get_client()

        if client is None:
            return (
                "لم يتم إعداد مفتاح API للذكاء الاصطناعي.\n\n"
                "أضف OPENROUTER_API_KEY أو OPENAI_API_KEY "
                "في Railway Variables."
            )

        if not text or not text.strip():
            return "أرسل رسالة نصية أولًا."

        text = text.strip()

        # Create conversation history for this Telegram user.
        if user_id not in cls.conversations:
            cls.conversations[user_id] = [
                {
                    "role": "system",
                    "content": cls._get_system_prompt(),
                }
            ]

        history = cls.conversations[user_id]

        history.append(
            {
                "role": "user",
                "content": text,
            }
        )

        # Prevent unlimited memory growth.
        # Keep system message + latest 20 messages.
        if len(history) > 21:
            cls.conversations[user_id] = [
                history[0]
            ] + history[-20:]

        history = cls.conversations[user_id]

        try:
            response = await client.chat.completions.create(
                model=cls._get_model(),
                messages=history,
                temperature=0.7,
            )

            answer = response.choices[0].message.content

            if not answer:
                answer = "لم يتم استلام رد من نموذج الذكاء الاصطناعي."

            answer = answer.strip()

            history.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            return answer

        except Exception as exc:
            logger.exception(
                "LLM request failed: %s",
                exc,
            )

            # Remove the failed user message so the conversation
            # does not become corrupted.
            if (
                history
                and history[-1].get("role") == "user"
                and history[-1].get("content") == text
            ):
                history.pop()

            raise

        finally:
            try:
                await client.close()
            except Exception:
                pass

    @classmethod
    async def run(cls, botnav, message):
        """
        Telegram handler.
        """

        if message.content_type != "text":
            return

        user_id = message.from_user.id
        text = message.text or ""

        # Commands
        if text == "/clear":
            cls.conversations.pop(user_id, None)

            await botnav.bot.send_message(
                message.chat.id,
                "تم مسح المحادثة والذاكرة الحالية."
            )
            return

        try:
            # Show typing status while waiting for the model.
            try:
                answer = await botnav.await_coro_sending_action(
                    message.chat.id,
                    cls.ask(user_id, text),
                    "typing",
                )
            except AttributeError:
                # Fallback if this TeleBotNav version does not
                # provide await_coro_sending_action.
                answer = await cls.ask(user_id, text)

            # Telegram has a message length limit.
            max_length = 4096

            if len(answer) <= max_length:
                await botnav.bot.send_message(
                    message.chat.id,
                    answer,
                )
                return

            # Split long responses.
            for i in range(0, len(answer), max_length):
                chunk = answer[i:i + max_length]

                await botnav.bot.send_message(
                    message.chat.id,
                    chunk,
                )

        except Exception as exc:
            logger.exception(
                "LLM Telegram handler failed: %s",
                exc,
            )

            await botnav.bot.send_message(
                message.chat.id,
                "حدث خطأ أثناء الاتصال بنموذج الذكاء الاصطناعي.\n"
                "راجع Railway Logs لمعرفة الخطأ الحقيقي."
            )