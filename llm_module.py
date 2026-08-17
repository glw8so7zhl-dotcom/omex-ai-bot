import base64
import logging
from typing import Dict, List, Optional

from openai import AsyncOpenAI

import config


logger = logging.getLogger(__name__)


class LLMRouter:
    """
    OMEX AI Assistant LLM Router.

    Supports:
    - Normal text conversations
    - Product image analysis
    - Arabic marketing descriptions
    - OpenRouter
    - OpenAI fallback for text
    """

    conversations: Dict[int, List[dict]] = {}

    # =========================================================
    # CLIENT
    # =========================================================

    @classmethod
    def _get_client(cls, vision: bool = False):
        """
        Create the appropriate async AI client.

        OpenRouter is preferred.

        Vision requests use OpenRouter because the configured
        free router can automatically select a model that supports
        image understanding.
        """

        # -----------------------------------------------------
        # OpenRouter
        # -----------------------------------------------------

        if config.OPENROUTER_API_KEY:

            return AsyncOpenAI(
                api_key=config.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
            )

        # -----------------------------------------------------
        # OpenAI fallback
        # -----------------------------------------------------

        if config.OPENAI_API_KEY and not vision:

            return AsyncOpenAI(
                api_key=config.OPENAI_API_KEY
            )

        return None

    # =========================================================
    # NORMAL TEXT MODEL
    # =========================================================

    @classmethod
    def _get_model(cls) -> str:
        """
        Normal text model.
        """

        return getattr(
            config,
            "DEFAULT_LLM_MODEL",
            "deepseek/deepseek-r1-0528:free",
        )

    # =========================================================
    # VISION MODEL
    # =========================================================

    @classmethod
    def _get_vision_model(cls) -> str:
        """
        Vision model.

        VISION_MODEL can be configured in Railway Variables.

        Default:
            openrouter/free

        OpenRouter's free router automatically chooses a free
        model capable of handling image input.
        """

        return getattr(
            config,
            "VISION_MODEL",
            "openrouter/free",
        )

    # =========================================================
    # NORMAL SYSTEM PROMPT
    # =========================================================

    @classmethod
    def _get_system_prompt(cls) -> str:

        return (
            "You are OMEX AI Assistant. "
            "Answer clearly and accurately. "
            "The user may communicate in Arabic or English. "
            "If the user writes Arabic, respond in Arabic. "
            "Do not invent information."
        )

    # =========================================================
    # PRODUCT IMAGE SYSTEM PROMPT
    # =========================================================

    @classmethod
    def _get_product_system_prompt(cls) -> str:

        return """
أنت OMEX AI Assistant، خبير محترف في:

- E-commerce Copywriting
- Product Marketing
- Consumer Psychology
- Facebook Marketing
- Instagram Marketing
- TikTok Marketing
- SEO Copywriting
- Amazon Listing Optimization

مهمتك الأساسية عند استلام صورة منتج هي تحليل الصورة بدقة ثم كتابة وصف
تسويقي احترافي مناسب للسوق اليمني.

قواعد مهمة جدًا:

1. حلل المنتج الظاهر في الصورة فقط.
2. لا تخترع اسم شركة أو علامة تجارية غير واضحة.
3. لا تخترع السعر.
4. لا تخترع المقاسات.
5. لا تخترع المواد.
6. لا تخترع المواصفات التقنية.
7. لا تدّعي وجود ميزة لا يمكن تأكيدها من الصورة أو من كلام المستخدم.
8. إذا كانت معلومة غير مؤكدة، لا تقدمها كحقيقة.
9. إذا كان اسم المنتج واضحًا في الصورة، استخدمه.
10. إذا لم يكن الاسم واضحًا، استخدم اسمًا وصفيًا مناسبًا.
11. إذا أرسل المستخدم نصًا مع الصورة، اعتبر النص معلومات إضافية من المستخدم.
12. اكتب بالعربية إذا كان المستخدم يكتب بالعربية.
13. اجعل النص مناسبًا للإعلانات ومنشورات Facebook وInstagram وWhatsApp.
14. تجنب المبالغة الكاذبة.
15. لا تستخدم عبارات طبية أو علاجية إلا إذا كانت معلومة مؤكدة من المستخدم.
16. لا تقل "الأفضل في السوق" أو "مضمون 100%" إلا إذا طلب المستخدم ذلك وكان هناك أساس واضح.
17. ركز على الفائدة العملية للعميل.
18. اجعل النص واضحًا وسهل القراءة.
19. استخدم عناوين وفقرات ونقاط.
20. لا تذكر أنك نموذج ذكاء اصطناعي داخل وصف المنتج.

هوية المتجر:

اسم المتجر:
متجر أومكس - OMEX Store

السوق:
اليمن

أسلوب الكتابة:
احترافي، مقنع، واضح، مباشر، حديث، مناسب للمستهلك اليمني.

عند تحليل صورة المنتج، أخرج النتيجة بهذا الهيكل:

━━━━━━━━━━━━━━━━━━

اسم المنتج:
[اسم المنتج]

━━━━━━━━━━━━━━━━━━

عنوان تسويقي قوي:
[Hook قصير وجذاب]

━━━━━━━━━━━━━━━━━━

وصف المنتج:
[وصف احترافي يشرح المنتج وفائدته واستخدامه اعتمادًا على المعلومات
المؤكدة فقط]

━━━━━━━━━━━━━━━━━━

أهم المميزات:
• ميزة مؤكدة من الصورة
• ميزة مؤكدة من الصورة
• ميزة مؤكدة من الصورة

━━━━━━━━━━━━━━━━━━

لماذا قد يناسبك؟
[الفائدة العملية للعميل]

━━━━━━━━━━━━━━━━━━

طريقة الاستخدام:
[اذكرها فقط إذا كانت واضحة أو مؤكدة]

━━━━━━━━━━━━━━━━━━

الفئة المناسبة:
[الفئة المحتملة بناءً على طبيعة المنتج]

━━━━━━━━━━━━━━━━━━

نص إعلاني قصير:
[نسخة مختصرة مناسبة لفيسبوك وواتساب]

━━━━━━━━━━━━━━━━━━

دعوة لاتخاذ إجراء:
[CTA مناسب للبيع]

━━━━━━━━━━━━━━━━━━

كلمات بحث SEO:
[كلمات مرتبطة بالمنتج بالعربية]

━━━━━━━━━━━━━━━━━━

ملاحظة:
إذا كانت هناك معلومات لا يمكن تحديدها من الصورة، لا تخمنها.
استخدم فقط المعلومات المؤكدة.
"""

    # =========================================================
    # TEXT CHAT
    # =========================================================

    @classmethod
    async def ask(
        cls,
        user_id: int,
        text: str,
    ) -> str:

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

        # -----------------------------------------------------
        # Create conversation
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Limit conversation
        # -----------------------------------------------------

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

                answer = (
                    "لم يتم استلام رد من نموذج الذكاء الاصطناعي."
                )

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

            # Remove failed user message
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

    # =========================================================
    # IMAGE -> BASE64
    # =========================================================

    @classmethod
    async def _download_telegram_photo(
        cls,
        botnav,
        message
    ) -> Optional[bytes]:
        """
        Download the highest-resolution Telegram photo.
        """

        if not message.photo:
            return None

        # Telegram sends multiple resolutions.
        # [-1] = highest available resolution.
        photo = message.photo[-1]

        file_info = await botnav.bot.get_file(
            photo.file_id
        )

        file_content = await botnav.bot.download_file(
            file_info.file_path
        )

        return bytes(file_content)

    # =========================================================
    # IMAGE ANALYSIS
    # =========================================================

    @classmethod
    async def analyze_product_image(
        cls,
        user_id: int,
        image_bytes: bytes,
        user_caption: str = "",
    ) -> str:
        """
        Analyze a product image using a vision-capable model.
        """

        if not image_bytes:

            return "لم أتمكن من قراءة صورة المنتج."

        # -----------------------------------------------------
        # Vision requires OpenRouter in this implementation.
        # -----------------------------------------------------

        if not config.OPENROUTER_API_KEY:

            return (
                "تحليل صور المنتجات يحتاج إلى OPENROUTER_API_KEY.\n\n"
                "أضف المفتاح في Railway Variables ثم أعد تشغيل الخدمة."
            )

        client = cls._get_client(
            vision=True
        )

        if client is None:

            return (
                "تعذر إنشاء اتصال بنموذج تحليل الصور."
            )

        # -----------------------------------------------------
        # Convert image to Base64
        # -----------------------------------------------------

        image_base64 = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        image_url = (
            "data:image/jpeg;base64,"
            + image_base64
        )

        # -----------------------------------------------------
        # User instruction
        # -----------------------------------------------------

        if user_caption and user_caption.strip():

            instruction = (
                "حلل صورة المنتج التالية واكتب وصفًا تسويقيًا احترافيًا "
                "للسوق اليمني.\n\n"
                "معلومات إضافية كتبها المستخدم:\n"
                + user_caption.strip()
            )

        else:

            instruction = (
                "حلل صورة المنتج التالية واكتب وصفًا تسويقيًا "
                "احترافيًا للسوق اليمني وفق التعليمات."
            )

        messages = [
            {
                "role": "system",
                "content": cls._get_product_system_prompt(),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": instruction,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        },
                    },
                ],
            },
        ]

        try:

            response = await client.chat.completions.create(
                model=cls._get_vision_model(),
                messages=messages,
                temperature=0.6,
            )

            answer = response.choices[0].message.content

            if not answer:

                return (
                    "تم تحليل الصورة، ولكن لم يصل نص من نموذج الذكاء الاصطناعي."
                )

            answer = answer.strip()

            # -------------------------------------------------
            # Store only textual result in conversation.
            # We don't store the huge Base64 image.
            # -------------------------------------------------

            if user_id not in cls.conversations:

                cls.conversations[user_id] = [
                    {
                        "role": "system",
                        "content": cls._get_system_prompt(),
                    }
                ]

            cls.conversations[user_id].append(
                {
                    "role": "user",
                    "content": (
                        "[تم إرسال صورة منتج للتحليل]"
                        + (
                            "\nملاحظة المستخدم: "
                            + user_caption.strip()
                            if user_caption
                            else ""
                        )
                    ),
                }
            )

            cls.conversations[user_id].append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            # Keep conversation under control
            history = cls.conversations[user_id]

            if len(history) > 21:

                cls.conversations[user_id] = [
                    history[0]
                ] + history[-20:]

            return answer

        except Exception as exc:

            logger.exception(
                "Product image analysis failed: %s",
                exc,
            )

            raise

        finally:

            try:
                await client.close()
            except Exception:
                pass

    # =========================================================
    # TELEGRAM HANDLER
    # =========================================================

    @classmethod
    async def run(
        cls,
        botnav,
        message
    ):
        """
        Main Telegram handler.

        Supports:
        - text
        - product photos
        """

        user_id = message.from_user.id

        # =====================================================
        # CLEAR COMMAND
        # =====================================================

        if (
            message.content_type == "text"
            and message.text == "/clear"
        ):

            cls.conversations.pop(
                user_id,
                None
            )

            await botnav.bot.send_message(
                message.chat.id,
                "تم مسح المحادثة والذاكرة الحالية."
            )

            return

        # =====================================================
        # PRODUCT IMAGE
        # =====================================================

        if message.content_type == "photo":

            try:

                await botnav.bot.send_message(
                    message.chat.id,
                    "جاري تحليل صورة المنتج وكتابة الوصف الاحترافي..."
                )

                image_bytes = (
                    await cls._download_telegram_photo(
                        botnav,
                        message
                    )
                )

                if not image_bytes:

                    await botnav.bot.send_message(
                        message.chat.id,
                        "تعذر تحميل صورة المنتج."
                    )

                    return

                caption = (
                    message.caption
                    if message.caption
                    else ""
                )

                try:

                    answer = await botnav.await_coro_sending_action(
                        message.chat.id,

                        cls.analyze_product_image(
                            user_id,
                            image_bytes,
                            caption,
                        ),

                        "typing",
                    )

                except AttributeError:

                    answer = await cls.analyze_product_image(
                        user_id,
                        image_bytes,
                        caption,
                    )

                # -------------------------------------------------
                # Telegram message limit
                # -------------------------------------------------

                max_length = 4096

                if len(answer) <= max_length:

                    await botnav.bot.send_message(
                        message.chat.id,
                        answer,
                    )

                    return

                # -------------------------------------------------
                # Split long response
                # -------------------------------------------------

                for i in range(
                    0,
                    len(answer),
                    max_length
                ):

                    chunk = answer[
                        i:i + max_length
                    ]

                    await botnav.bot.send_message(
                        message.chat.id,
                        chunk,
                    )

                return

            except Exception as exc:

                logger.exception(
                    "Product image Telegram handler failed: %s",
                    exc,
                )

                await botnav.bot.send_message(
                    message.chat.id,
                    "حدث خطأ أثناء تحليل صورة المنتج.\n"
                    "راجع Railway Logs لمعرفة الخطأ الحقيقي."
                )

                return

        # =====================================================
        # TEXT
        # =====================================================

        if message.content_type != "text":

            return

        text = message.text or ""

        if not text.strip():

            return

        try:

            try:

                answer = await botnav.await_coro_sending_action(
                    message.chat.id,

                    cls.ask(
                        user_id,
                        text
                    ),

                    "typing",
                )

            except AttributeError:

                answer = await cls.ask(
                    user_id,
                    text
                )

            # -------------------------------------------------
            # Telegram message limit
            # -------------------------------------------------

            max_length = 4096

            if len(answer) <= max_length:

                await botnav.bot.send_message(
                    message.chat.id,
                    answer,
                )

                return

            # -------------------------------------------------
            # Split long responses
            # -------------------------------------------------

            for i in range(
                0,
                len(answer),
                max_length
            ):

                chunk = answer[
                    i:i + max_length
                ]

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