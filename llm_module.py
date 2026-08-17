import base64
import logging
from typing import Dict, List, Optional

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

    # =========================================================
    # CLIENT
    # =========================================================

    @classmethod
    def _get_client(cls, vision: bool = False):
        """
        إنشاء اتصال مع مزود الذكاء الاصطناعي.

        تحليل الصور يستخدم OpenRouter.
        المحادثة النصية تستخدم OpenRouter أولًا ثم OpenAI.
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
        # OpenAI
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

        إذا لم يتم وضع المتغير، يستخدم:
        openrouter/free
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

        finally:

            try:
                await client.close()
            except Exception:
                pass

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

            response = await client.chat.completions.create(
                model=cls._get_vision_model(),
                messages=messages,
                temperature=0.6,
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

        finally:

            try:
                await client.close()
            except Exception:
                pass

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
                # إرسال النتيجة
                # -------------------------------------------------

                await cls._send_long_message(
                    botnav,
                    message.chat.id,
                    answer,
                )

                return

            except Exception as exc:

                logger.exception(
                    "Product image handler failed: %s",
                    exc,
                )

                await botnav.bot.send_message(
                    message.chat.id,
                    "حدث خطأ أثناء تحليل صورة المنتج.\n\n"
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