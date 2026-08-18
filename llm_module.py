import base64
import logging
import functools
from urllib.parse import quote_plus
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

import config


logger = logging.getLogger(__name__)


class LLMRouter:
    """
    OMEX AI Assistant

    يدعم:
    1. المحادثة النصية.
    2. استقبال صور المنتجات.
    3. تحليل صور المنتجات.
    4. كتابة وصف تسويقي احترافي بالعربية.
    5. OpenRouter.
    6. OpenAI كخيار احتياطي للمحادثة النصية.
    """

    conversations: Dict[int, List[dict]] = {}

    # Reuse API clients instead of creating/closing one for every request.
    # This reduces connection setup overhead on Railway.
    _openrouter_client = None
    _openai_client = None

    # Latest analyzed product per Telegram user.
    # Stores only the image needed for re-generation and the generated text/search terms.
    product_sessions: Dict[int, dict] = {}

    # =========================================================
    # CLIENT
    # =========================================================

    @classmethod
    def _get_client(cls, vision: bool = False):
        """Return a reusable API client."""

        if config.OPENROUTER_API_KEY:
            if cls._openrouter_client is None:
                cls._openrouter_client = AsyncOpenAI(
                    api_key=config.OPENROUTER_API_KEY,
                    base_url="https://openrouter.ai/api/v1",
                    timeout=45.0,
                    max_retries=1,
                )
            return cls._openrouter_client

        if config.OPENAI_API_KEY and not vision:
            if cls._openai_client is None:
                cls._openai_client = AsyncOpenAI(
                    api_key=config.OPENAI_API_KEY,
                    timeout=45.0,
                    max_retries=1,
                )
            return cls._openai_client

        return None

    # =========================================================
    # NORMAL TEXT MODEL
    # =========================================================

    @classmethod
    def _get_model(cls) -> str:
        """
        النموذج المستخدم للمحادثة النصية.
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
        النموذج المستخدم لتحليل صور المنتجات.

        يمكن تغييره من Railway Variables بإضافة:

        VISION_MODEL

        إذا لم يتم وضع المتغير، يستخدم نموذج رؤية مجانيًا ثابتًا يدعم الصور.
        """

        return getattr(
            config,
            "VISION_MODEL",
            "google/gemma-4-26b-a4b-it:free",
        )

    # =========================================================
    # NORMAL SYSTEM PROMPT
    # =========================================================

    @classmethod
    def _get_system_prompt(cls) -> str:

        return """
أنت OMEX AI Assistant.

أجب بشكل واضح ودقيق.

إذا كان المستخدم يكتب بالعربية، أجب بالعربية.

إذا كان المستخدم يكتب بالإنجليزية، أجب بالإنجليزية.

لا تخترع معلومات غير مؤكدة.

لا تدّعي معرفة معلومات غير متوفرة.
"""

    # =========================================================
    # PRODUCT SYSTEM PROMPT
    # =========================================================

    @classmethod
    def _get_product_system_prompt(cls) -> str:

        return """
أنت كاتب محتوى تجاري محترف ومتخصص في وصف المنتجات لمتجر OMEX Store في السوق اليمني.

مهمتك:

عند استلام صورة منتج، حلل المنتج الظاهر في الصورة بدقة، ثم اكتب وصفًا تجاريًا عربيًا احترافيًا ومنظمًا وجاهزًا للنشر.

==================================================
قواعد التحليل والدقة
==================================================

1. اعتمد فقط على المعلومات الظاهرة بوضوح في الصورة أو المعلومات التي يكتبها المستخدم.

2. ممنوع اختراع أي مواصفة غير مؤكدة، مثل:

- قوة الجهاز
- سرعة الجهاز
- سعة البطارية
- مدة التشغيل
- مدة الشحن
- نوع المادة
- المقاسات
- الوزن
- عدد الملحقات
- قوة الشفط
- قوة النفخ
- مقاومة الماء
- الضمان
- السعر

3. إذا كانت المواصفة غير واضحة، لا تذكرها.

4. لا تستخدم عبارات مبالغًا فيها مثل:

"الأقوى"
"الأفضل"
"الأكثر قوة"
"غير مسبوق"
"مضمون 100%"
"أفضل منتج في السوق"

إلا إذا كانت المعلومة مؤكدة ومذكورة من المستخدم.

5. لا تستنتج خصائص تقنية من شكل المنتج فقط.

6. إذا ظهر اسم المنتج على العبوة، استخدم الاسم كما هو.

7. إذا كان الاسم غير واضح، استخدم اسمًا وصفيًا بسيطًا ولا تخترع اسمًا تجاريًا.

8. إذا كانت هناك علامة تجارية واضحة في الصورة، لا تغيّر اسمها.

==================================================
قواعد اللغة
==================================================

1. اكتب باللغة العربية السليمة.

2. لا تخلط العربية بالإنجليزية داخل الجمل.

3. الاستثناء الوحيد:

اسم العلامة التجارية أو اسم الموديل الأصلي.

4. ممنوع إدخال كلمات إنجليزية عشوائية داخل الجمل العربية مثل:

cleaning
charge
portable
powerful
smart
premium

5. استخدم المصطلح العربي المناسب بدلًا منها.

6. لا تستخدم أخطاء لغوية أو كلمات عشوائية.

7. لا تستخدم لغات مختلطة.

8. اجعل النص طبيعيًا وكأنه مكتوب بواسطة كاتب إعلانات عربي محترف.

9. لا تستخدم رموزًا زخرفية أو إيموجي.

10. لا تضع هاشتاقات داخل الوصف.

في قسم "نص إعلاني جاهز للنشر" أضف في النهاية:
هاشتاقات
[8 إلى 15 هاشتاقًا مناسبة للمنتج، عربية وإنجليزية، بدون اختراع أسماء أو مواصفات.]

==================================================
أسلوب الكتابة
==================================================

الأسلوب المطلوب:

احترافي
واضح
مقنع
منظم
واقعي
مختصر
مناسب للسوق اليمني
جاهز للنشر على فيسبوك وإنستغرام وواتساب.

لا تكرر نفس المعلومة أكثر من مرة.

لا تستخدم كلامًا مبالغًا فيه.

ركز على:

- ما هو المنتج؟
- ما وظيفته؟
- أين يستخدم؟
- كيف يفيد المستخدم؟
- ما الذي يظهر في الصورة؟
- ما الملحقات الظاهرة؟
- ما المميزات التي يمكن تأكيدها؟

==================================================
تنسيق النتيجة
==================================================

اكتب النتيجة بهذا الترتيب بالضبط:

اسم المنتج

[اسم واضح ومختصر]

الوصف

[فقرة احترافية من 3 إلى 5 أسطر تشرح المنتج ووظيفته واستخداماته.]

أبرز المميزات

• [ميزة مؤكدة]
• [ميزة مؤكدة]
• [ميزة مؤكدة]
• [ميزة مؤكدة]

الاستخدامات

• [استخدام مناسب]
• [استخدام مناسب]
• [استخدام مناسب]

مناسب لـ

[الفئة المناسبة للمنتج في جملة قصيرة.]

طريقة الاستخدام

[اذكر خطوات الاستخدام فقط إذا كانت واضحة من المنتج أو المعلومات التي قدمها المستخدم.]

الخلاصة

[جملة تسويقية قصيرة وواقعية توضح فائدة المنتج.]

نص إعلاني جاهز للنشر

[إعلان قصير من 3 إلى 5 أسطر، واضح ومقنع وطبيعي.]

==================================================
قواعد النص الإعلاني
==================================================

اجعل الإعلان طبيعيًا وليس مثل تقرير تقني.

ركز على فائدة المنتج للمستخدم.

لا تكرر قائمة المميزات بالكامل.

لا تخترع سعرًا.

لا تخترع عرضًا.

لا تخترع ضمانًا.

لا تخترع معلومات التوصيل.

لا تستخدم مبالغة غير مؤكدة.

==================================================
معلومات المتجر
==================================================

اسم المتجر:

متجر أومكس

بالإنجليزية:

OMEX Store

السوق:

اليمن

إذا لم يعطِ المستخدم السعر:

لا تذكر السعر.

إذا لم يعطِ المستخدم معلومات التوصيل:

لا تذكر معلومات التوصيل.

إذا لم يعطِ المستخدم الضمان:

لا تذكر الضمان.

إذا لم يعطِ المستخدم مواصفات تقنية:

لا تخترع المواصفات.

==================================================
النتيجة النهائية
==================================================

يجب أن تكون النتيجة:

عربية سليمة
منظمة
واضحة
احترافية
دقيقة
مقنعة
واقعية
بدون كلمات عشوائية
بدون خلط لغات
بدون مبالغة
بدون معلومات غير مؤكدة
جاهزة للنسخ والنشر مباشرة.

في نهاية الرد أضف سطرين تقنيين فقط لاستخدام البوت:
ENGLISH_SEARCH: [عبارة بحث إنجليزية قصيرة ودقيقة عن المنتج]
CHINESE_SEARCH: [عبارة بحث صينية قصيرة ودقيقة عن المنتج]

لا تضع أي شرح آخر بعد هذين السطرين.
"""

    # =========================================================
    # NORMAL TEXT CHAT
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
        # إنشاء المحادثة
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
        # منع زيادة حجم الذاكرة
        # -----------------------------------------------------

        if len(history) > 21:

            cls.conversations[user_id] = (
                [history[0]]
                + history[-20:]
            )

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

            # حذف الرسالة الفاشلة من المحادثة
            if (
                history
                and history[-1].get("role") == "user"
                and history[-1].get("content") == text
            ):

                history.pop()

            raise

    # =========================================================
    # DOWNLOAD TELEGRAM PHOTO
    # =========================================================

    @classmethod
    async def _download_telegram_photo(
        cls,
        botnav,
        message
    ) -> Optional[bytes]:
        """
        تحميل أعلى دقة متوفرة من صورة Telegram.
        """

        if not message.photo:

            return None

        # آخر عنصر عادةً هو أعلى دقة
        photo = message.photo[-1]

        file_info = await botnav.bot.get_file(
            photo.file_id
        )

        file_content = await botnav.bot.download_file(
            file_info.file_path
        )

        return bytes(file_content)

    # =========================================================
    # ANALYZE PRODUCT IMAGE
    # =========================================================

    @classmethod
    async def analyze_product_image(
        cls,
        user_id: int,
        image_bytes: bytes,
        user_caption: str = "",
    ) -> str:
        """
        تحليل صورة المنتج وكتابة وصف احترافي.
        """

        if not image_bytes:

            return "لم أتمكن من قراءة صورة المنتج."

        # -----------------------------------------------------
        # تحليل الصور يحتاج OpenRouter
        # -----------------------------------------------------

        if not config.OPENROUTER_API_KEY:

            return (
                "تحليل صور المنتجات يحتاج إلى "
                "OPENROUTER_API_KEY.\n\n"
                "أضف المفتاح في Railway Variables "
                "ثم أعد تشغيل الخدمة."
            )

        client = cls._get_client(
            vision=True
        )

        if client is None:

            return (
                "تعذر إنشاء اتصال بنموذج تحليل الصور."
            )

        # -----------------------------------------------------
        # تحويل الصورة إلى Base64
        # -----------------------------------------------------

        image_base64 = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        image_url = (
            "data:image/jpeg;base64,"
            + image_base64
        )

        # -----------------------------------------------------
        # تعليمات المستخدم
        # -----------------------------------------------------

        if user_caption and user_caption.strip():

            instruction = (
                "حلل صورة المنتج التالية واكتب وصفًا "
                "تسويقيًا احترافيًا باللغة العربية "
                "للسوق اليمني.\n\n"
                "معلومات إضافية قدمها المستخدم:\n"
                + user_caption.strip()
                + "\n\n"
                "اعتبر هذه المعلومات صحيحة من المستخدم، "
                "لكن لا تضف مواصفات أخرى غير مؤكدة من الصورة."
            )

        else:

            instruction = (
                "حلل صورة المنتج التالية واكتب وصفًا "
                "تسويقيًا احترافيًا باللغة العربية "
                "للسوق اليمني وفق التعليمات الموجودة في "
                "تعليمات النظام."
            )

        # -----------------------------------------------------
        # الرسالة التي تحتوي الصورة
        # -----------------------------------------------------

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
                            "url": image_url,
                        },
                    },
                ],
            },
        ]

        try:

            # Use only models that are documented as vision-capable.
            # The free router can change its selected model, so it is not used
            # as the default for product-image analysis.
            primary_model = cls._get_vision_model()
            models = []
            for model_name in (
                primary_model,
                "google/gemma-4-31b-it:free",
                "google/gemma-3-27b-it:free",
            ):
                if model_name and model_name not in models:
                    models.append(model_name)

            response = None
            last_error = None

            for model_name in models:
                try:
                    logger.info(
                        "Product vision request using model=%s",
                        model_name,
                    )

                    response = await client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0.4,
                        max_tokens=1400,
                        provider={
                            "sort": "latency",
                            "allow_fallbacks": True,
                        },
                    )
                    break

                except Exception as exc:
                    last_error = exc
                    logger.exception(
                        "Vision model failed (%s): %s",
                        model_name,
                        exc,
                    )

            if response is None:
                raise RuntimeError(
                    "تعذر تشغيل أي نموذج رؤية. آخر خطأ: "
                    + str(last_error)
                )

            answer = response.choices[0].message.content

            if not answer:

                return (
                    "تم تحليل الصورة، ولكن لم يصل "
                    "نص من نموذج الذكاء الاصطناعي."
                )

            answer = answer.strip()

            # -------------------------------------------------
            # حفظ النتيجة في الذاكرة النصية
            #
            # لا نحفظ Base64 حتى لا تكبر الذاكرة.
            # -------------------------------------------------

            if user_id not in cls.conversations:

                cls.conversations[user_id] = [
                    {
                        "role": "system",
                        "content": cls._get_system_prompt(),
                    }
                ]

            image_memory_text = (
                "[تم إرسال صورة منتج للتحليل]"
            )

            if user_caption:

                image_memory_text += (
                    "\nملاحظة المستخدم: "
                    + user_caption.strip()
                )

            cls.conversations[user_id].append(
                {
                    "role": "user",
                    "content": image_memory_text,
                }
            )

            cls.conversations[user_id].append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            # -------------------------------------------------
            # تحديد حجم الذاكرة
            # -------------------------------------------------

            history = cls.conversations[user_id]

            if len(history) > 21:

                cls.conversations[user_id] = (
                    [history[0]]
                    + history[-20:]
                )

            return answer

        except Exception as exc:

            logger.exception(
                "Product image analysis failed: %s",
                exc,
            )

            raise

    # =========================================================
    # PRODUCT SESSION / SEARCH HELPERS
    # =========================================================

    @classmethod
    def _parse_product_metadata(cls, answer: str) -> tuple[str, str, str]:
        """
        Extract the two internal search lines from the model response.
        Returns: clean_text, english_query, chinese_query.
        """
        english_query = ""
        chinese_query = ""
        clean_lines = []

        for line in answer.splitlines():
            stripped = line.strip()

            if stripped.upper().startswith("ENGLISH_SEARCH:"):
                english_query = stripped.split(":", 1)[1].strip()
                continue

            if stripped.upper().startswith("CHINESE_SEARCH:"):
                chinese_query = stripped.split(":", 1)[1].strip()
                continue

            clean_lines.append(line)

        clean_text = "\n".join(clean_lines).strip()

        # Safe fallbacks. These are search phrases, not claimed specifications.
        if not english_query:
            english_query = "product"
        if not chinese_query:
            chinese_query = "商品"

        return clean_text, english_query, chinese_query

    @classmethod
    def _search_urls(cls, english_query: str, chinese_query: str) -> dict[str, str]:
        en = quote_plus(english_query)
        zh = quote_plus(chinese_query)

        return {
            # Web search pages; no login or API key is required to open them.
            "tiktok": f"https://www.tiktok.com/search?q={en}",
            "douyin": f"https://www.douyin.com/search/{zh}",
            "rednote": (
                "https://www.xiaohongshu.com/search_result"
                f"?keyword={zh}"
            ),
        }

    @classmethod
    async def _show_product_menu(cls, botnav, message) -> None:
        user_id = message.from_user.id
        session = cls.product_sessions.get(user_id)

        if not session:
            await botnav.bot.send_message(
                message.chat.id,
                "لا توجد صورة منتج محفوظة. أرسل صورة المنتج أولًا."
            )
            return

        urls = cls._search_urls(
            session["english_search"],
            session["chinese_search"],
        )

        markup = InlineKeyboardMarkup(row_width=2)

        # Callback buttons are registered in TeleBotNav so they work with its
        # existing callback handler.
        description_cb = functools.partial(cls._send_product_description)
        regenerate_cb = functools.partial(cls._regenerate_product)
        facebook_cb = functools.partial(cls._prepare_facebook_post)

        callbacks = {
            description_cb: "وصف احترافي + هاشتاقات",
            regenerate_cb: "إعادة تحليل الصورة",
            facebook_cb: "تجهيز منشور Facebook",
        }

        for callback, label in callbacks.items():
            key = str(callback.__hash__())
            botnav.buttons[key] = callback
            markup.add(InlineKeyboardButton(label, callback_data=key))

        # Direct URL buttons: one tap opens the platform search.
        markup.add(
            InlineKeyboardButton("بحث TikTok", url=urls["tiktok"]),
            InlineKeyboardButton("بحث Douyin الصيني", url=urls["douyin"]),
        )
        markup.add(
            InlineKeyboardButton("بحث REDnote", url=urls["rednote"]),
        )

        text = (
            "تم تحليل المنتج.\n\n"
            f"كلمة البحث الإنجليزية:\n{session['english_search']}\n\n"
            f"كلمة البحث الصينية:\n{session['chinese_search']}\n\n"
            "اختر العملية المطلوبة:"
        )

        await botnav.bot.send_message(
            message.chat.id,
            text,
            reply_markup=markup,
        )

    @classmethod
    async def _send_product_description(cls, botnav, message) -> None:
        session = cls.product_sessions.get(message.from_user.id)

        if not session:
            await botnav.bot.send_message(
                message.chat.id,
                "انتهت جلسة المنتج. أرسل الصورة مرة أخرى."
            )
            return

        await cls._send_long_message(
            botnav,
            message.chat.id,
            session["description"],
        )

    @classmethod
    async def _prepare_facebook_post(cls, botnav, message) -> None:
        """
        Phase 1: prepare a Facebook-ready post.
        Direct Page publishing will be added after Meta credentials are configured.
        """
        session = cls.product_sessions.get(message.from_user.id)

        if not session:
            await botnav.bot.send_message(
                message.chat.id,
                "انتهت جلسة المنتج. أرسل الصورة مرة أخرى."
            )
            return

        await botnav.bot.send_message(
            message.chat.id,
            "منشور Facebook جاهز:\n\n" + session["description"],
        )

        # Re-send the original product image with the generated caption so the
        # user can save/share it directly from Telegram.
        try:
            from io import BytesIO
            photo = BytesIO(session["image_bytes"])
            photo.name = "omex_product.jpg"
            await botnav.bot.send_photo(
                message.chat.id,
                photo,
                caption=session["facebook_caption"][:1024],
            )
        except Exception as exc:
            logger.exception("Failed to prepare Facebook image: %s", exc)

    @classmethod
    async def _regenerate_product(cls, botnav, message) -> None:
        session = cls.product_sessions.get(message.from_user.id)

        if not session:
            await botnav.bot.send_message(
                message.chat.id,
                "انتهت جلسة المنتج. أرسل الصورة مرة أخرى."
            )
            return

        try:
            await botnav.bot.send_message(
                message.chat.id,
                "جاري إعادة تحليل الصورة..."
            )

            answer = await botnav.await_coro_sending_action(
                message.chat.id,
                cls.analyze_product_image(
                    user_id=message.from_user.id,
                    image_bytes=session["image_bytes"],
                    user_caption=session.get("caption", ""),
                ),
                "typing",
            )

            clean_text, english_query, chinese_query = cls._parse_product_metadata(answer)

            session["description"] = clean_text
            session["english_search"] = english_query
            session["chinese_search"] = chinese_query
            session["facebook_caption"] = clean_text

            await cls._show_product_menu(botnav, message)

        except Exception as exc:
            logger.exception("Product regeneration failed: %s", exc)
            await botnav.bot.send_message(
                message.chat.id,
                "حدث خطأ أثناء إعادة تحليل الصورة."
            )

    # =========================================================
    # SEND LONG TELEGRAM MESSAGE
    # =========================================================

    @classmethod
    async def _send_long_message(
        cls,
        botnav,
        chat_id: int,
        text: str,
    ) -> None:

        max_length = 4096

        if len(text) <= max_length:

            await botnav.bot.send_message(
                chat_id,
                text,
            )

            return

        for i in range(
            0,
            len(text),
            max_length
        ):

            chunk = text[
                i:i + max_length
            ]

            await botnav.bot.send_message(
                chat_id,
                chunk,
            )

    # =========================================================
    # TELEGRAM ROUTER
    # =========================================================

    @classmethod
    async def run(
        cls,
        botnav,
        message
    ):
        """
        معالج Telegram الرئيسي.

        يقبل:

        TEXT
        PHOTO

        النص:
            محادثة عادية.

        الصورة:
            تحليل المنتج وكتابة وصف احترافي.
        """

        user_id = message.from_user.id

        # =====================================================
        # CLEAR
        # =====================================================

        if (
            message.content_type == "text"
            and message.text
            and message.text.strip() == "/clear"
        ):

            cls.conversations.pop(
                user_id,
                None
            )
            cls.product_sessions.pop(
                user_id,
                None
            )

            await botnav.bot.send_message(
                message.chat.id,
                "تم مسح المحادثة والذاكرة الحالية."
            )

            return

        # =====================================================
        # PHOTO
        # =====================================================

        if message.content_type == "photo":

            try:

                # -------------------------------------------------
                # رسالة انتظار
                # -------------------------------------------------

                await botnav.bot.send_message(
                    message.chat.id,
                    "جاري تحليل صورة المنتج وكتابة الوصف الاحترافي..."
                )

                # -------------------------------------------------
                # تحميل الصورة
                # -------------------------------------------------

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

                # -------------------------------------------------
                # Caption
                # -------------------------------------------------

                caption = (
                    message.caption.strip()
                    if message.caption
                    else ""
                )

                # -------------------------------------------------
                # تحليل الصورة
                # -------------------------------------------------

                try:

                    answer = (
                        await botnav.await_coro_sending_action(
                            message.chat.id,

                            cls.analyze_product_image(
                                user_id=user_id,
                                image_bytes=image_bytes,
                                user_caption=caption,
                            ),

                            "typing",
                        )
                    )

                except AttributeError:

                    answer = (
                        await cls.analyze_product_image(
                            user_id=user_id,
                            image_bytes=image_bytes,
                            user_caption=caption,
                        )
                    )

                # -------------------------------------------------
                # حفظ نتيجة تحليل المنتج
                # -------------------------------------------------

                clean_text, english_query, chinese_query = (
                    cls._parse_product_metadata(answer)
                )

                cls.product_sessions[user_id] = {
                    "image_bytes": image_bytes,
                    "caption": caption,
                    "description": clean_text,
                    "facebook_caption": clean_text,
                    "english_search": english_query,
                    "chinese_search": chinese_query,
                }

                # -------------------------------------------------
                # إظهار خيارات المنتج
                # -------------------------------------------------

                await cls._show_product_menu(
                    botnav,
                    message,
                )

                return

            except Exception as exc:

                logger.exception(
                    "Product image handler failed: %s",
                    exc,
                )

                error_text = str(exc).strip()
                if len(error_text) > 700:
                    error_text = error_text[:700] + "..."

                await botnav.bot.send_message(
                    message.chat.id,
                    "فشل تحليل الصورة.\n\n"
                    f"الخطأ: {error_text}"
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

            # -------------------------------------------------
            # طلب نصي عادي
            # -------------------------------------------------

            try:

                answer = (
                    await botnav.await_coro_sending_action(
                        message.chat.id,

                        cls.ask(
                            user_id,
                            text
                        ),

                        "typing",
                    )
                )

            except AttributeError:

                answer = await cls.ask(
                    user_id,
                    text
                )

            # -------------------------------------------------
            # إرسال الرد
            # -------------------------------------------------

            await cls._send_long_message(
                botnav,
                message.chat.id,
                answer,
            )

        except Exception as exc:

            logger.exception(
                "LLM Telegram handler failed: %s",
                exc,
            )

            await botnav.bot.send_message(
                message.chat.id,
                "حدث خطأ أثناء الاتصال بنموذج الذكاء الاصطناعي.\n\n"
                "راجع Railway Logs لمعرفة الخطأ الحقيقي."
            )