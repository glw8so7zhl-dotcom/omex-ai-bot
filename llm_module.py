import base64
import logging
import functools
from urllib.parse import quote_plus
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from typing import Dict, List, Optional

from openai import AsyncOpenAI
import aiohttp

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

    # Latest analyzed product per Telegram user.
    # Stores only the image needed for re-generation and the generated text/search terms.
    product_sessions: Dict[int, dict] = {}

    PRODUCT_SESSION_KEY = "omex_product_session"

    # Reuse HTTP connections instead of creating/closing an OpenAI client
    # for every message. This materially reduces latency on Railway.
    _clients = {}

    # =========================================================
    # CLIENT
    # =========================================================

    @classmethod
    def _get_client(cls, vision: bool = False):
        """Return a cached async client so TCP/TLS connections are reused."""
        key = "openrouter" if config.OPENROUTER_API_KEY else ("openai" if config.OPENAI_API_KEY and not vision else None)
        if not key:
            return None

        if key in cls._clients:
            return cls._clients[key]

        timeout = float(getattr(config, "AI_TIMEOUT_SECONDS", 45))
        max_retries = int(getattr(config, "AI_MAX_RETRIES", 1))

        if key == "openrouter":
            client = AsyncOpenAI(
                api_key=config.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
                timeout=timeout,
                max_retries=max_retries,
            )
        else:
            client = AsyncOpenAI(
                api_key=config.OPENAI_API_KEY,
                timeout=timeout,
                max_retries=max_retries,
            )

        cls._clients[key] = client
        return client

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
            "openai/gpt-oss-20b:free",
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
            "nvidia/nemotron-nano-12b-v2-vl:free",
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
                temperature=0.4,
                max_tokens=int(getattr(config, "TEXT_MAX_TOKENS", 900)),
                extra_body={"reasoning": {"enabled": False}},
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
            # Client is intentionally kept open for connection reuse.
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

        # Use the next-to-largest Telegram size when available.
        # It is usually much smaller than the original while still sharp
        # enough for product recognition and text on packaging.
        photo = message.photo[-2] if len(message.photo) >= 2 else message.photo[-1]

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
                temperature=0.35,
                max_tokens=int(getattr(config, "VISION_MAX_TOKENS", 1200)),
                extra_body={"reasoning": {"enabled": False}},
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

            answer = answer.strip()

            # بعض نماذج الرؤية المجانية قد تتجاهل بيانات البحث التقنية.
            # إذا حدث ذلك، نستخرجها من الوصف في طلب نصي منفصل.
            _, english_query, chinese_query = cls._parse_product_metadata(answer)
            if not english_query or not chinese_query:
                generated_en, generated_zh = await cls._generate_search_queries(answer)
                english_query = english_query or generated_en
                chinese_query = chinese_query or generated_zh
                answer = (
                    answer.rstrip()
                    + "\n\nENGLISH_SEARCH: " + (english_query or "")
                    + "\nCHINESE_SEARCH: " + (chinese_query or "")
                )

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
            # Shared client remains alive for subsequent requests.
            pass

    @classmethod
    async def _generate_search_queries(cls, description: str) -> tuple[str, str]:
        """Generate useful English/Chinese product-search terms when the vision model omits them."""
        if (
            not description
            or not config.OPENROUTER_API_KEY
            or not getattr(config, "GENERATE_SEARCH_QUERIES_FALLBACK", False)
        ):
            return "", ""

        client = cls._get_client(vision=False)
        if client is None:
            return "", ""

        prompt = f"""
أنت متخصص في استخراج كلمات البحث للمنتجات من وصف عربي.
استخرج عبارة بحث قصيرة ودقيقة للمنتج نفسه، وليست كلمات عامة.
يجب أن تتضمن العلامة التجارية أو الموديل إذا كانا معروفين من النص.
لا تضف مواصفات غير موجودة.

أعد سطرين فقط وبنفس الصيغة التالية:
ENGLISH_SEARCH: [2 إلى 6 كلمات إنجليزية]
CHINESE_SEARCH: [2 إلى 8 كلمات صينية]

وصف المنتج:
{description}
"""

        try:
            response = await client.chat.completions.create(
                model=cls._get_model(),
                messages=[
                    {"role": "system", "content": "استخرج كلمات بحث دقيقة فقط. لا تشرح."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            text = (response.choices[0].message.content or "").strip()
            _, english, chinese = cls._parse_product_metadata(text)
            return english, chinese
        except Exception as exc:
            logger.warning("Search query generation failed: %s", exc)
            return "", ""
        finally:
            pass

    # =========================================================
    # PRODUCT SESSION / SEARCH HELPERS
    # =========================================================

    @classmethod
    def _local_search_fallback(cls, text: str, language: str) -> str:
        """Fast, offline fallback when the vision model omits search metadata."""
        import re

        lines = [x.strip(" -*•:") for x in (text or "").splitlines() if x.strip()]
        name = ""
        for i, line in enumerate(lines):
            if line.startswith("اسم المنتج") and i + 1 < len(lines):
                name = lines[i + 1].strip()
                break
        if not name and lines:
            name = lines[0]

        # Remove long prose and keep a compact phrase. Preserve model numbers.
        name = re.sub(r"[\[\]{}()\"']", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        if language == "en":
            # If the model name is already Latin/number-heavy, it is useful as-is.
            latin = re.findall(r"[A-Za-z0-9][A-Za-z0-9+._-]*", name)
            if latin:
                return " ".join(latin[:6])
            return "product"

        # A safe Chinese fallback is intentionally generic rather than inventing a translation.
        return "商品"

    @classmethod
    def _parse_product_metadata(cls, answer: str) -> tuple[str, str, str]:
        """Extract internal search metadata without exposing it to the user."""
        english_query = ""
        chinese_query = ""
        clean_lines = []

        for line in (answer or "").splitlines():
            stripped = line.strip()
            upper = stripped.upper()

            if upper.startswith("ENGLISH_SEARCH:"):
                english_query = stripped.split(":", 1)[1].strip().strip("`*[]")
                continue

            if upper.startswith("CHINESE_SEARCH:"):
                chinese_query = stripped.split(":", 1)[1].strip().strip("`*[]")
                continue

            # Also tolerate Arabic labels if the model translates the metadata labels.
            if stripped.startswith("كلمة البحث الإنجليزية:"):
                english_query = stripped.split(":", 1)[1].strip()
                continue
            if stripped.startswith("كلمة البحث الصينية:"):
                chinese_query = stripped.split(":", 1)[1].strip()
                continue

            clean_lines.append(line)

        clean_text = "\n".join(clean_lines).strip()

        # Never send generic placeholders to search engines.
        if english_query.lower() in {"", "product", "item", "goods"}:
            english_query = ""
        if chinese_query in {"", "商品", "产品"}:
            chinese_query = ""

        return clean_text, english_query, chinese_query

    @classmethod
    def _search_urls(cls, english_query: str, chinese_query: str) -> dict[str, str]:
        """Build direct search URLs. Empty terms are handled safely."""
        en = quote_plus(english_query.strip() or "product")
        zh = quote_plus(chinese_query.strip() or "商品")

        # Android deep links: open the installed apps directly instead of
        # sending the user to a browser. These are app URL schemes, not web URLs.
        # TikTok international app commonly registers snssdk1233; Douyin uses
        # snssdk1128; Xiaohongshu documents xhsdiscover search deep links.
        return {
            "tiktok": f"snssdk1233://search?keyword={en}",
            "douyin": f"snssdk1128://search?keyword={zh}",
            "rednote": f"xhsdiscover://search/result?keyword={zh}",
        }

    @classmethod
    def _get_product_session(cls, botnav, message) -> Optional[dict]:
        """Read the product session from Telegram state first, then class memory."""
        state_data = getattr(message, "state_data", None) or {}
        session = state_data.get(cls.PRODUCT_SESSION_KEY)
        if session:
            return session
        return cls.product_sessions.get(message.from_user.id)

    @classmethod
    def _save_product_session(cls, botnav, message, session: dict) -> None:
        """Persist the session in TeleBotNav state so callback buttons cannot lose it."""
        state_data = getattr(message, "state_data", None)
        if state_data is not None:
            state_data[cls.PRODUCT_SESSION_KEY] = session
        cls.product_sessions[message.from_user.id] = session

    @classmethod
    async def _show_product_menu(cls, botnav, message) -> None:
        session = cls._get_product_session(botnav, message)

        if not session:
            await botnav.bot.send_message(
                message.chat.id,
                "لا توجد جلسة منتج محفوظة. أرسل صورة المنتج مرة أخرى."
            )
            return

        urls = cls._search_urls(
            session.get("english_search", ""),
            session.get("chinese_search", ""),
        )

        markup = InlineKeyboardMarkup(row_width=2)

        # Keep callback handlers in TeleBotNav, but store the product itself in
        # per-user Telegram state instead of relying only on a module-level dict.
        callbacks = {
            functools.partial(cls._send_product_description): "وصف احترافي + هاشتاقات",
            functools.partial(cls._regenerate_product): "إعادة تحليل الصورة",
            functools.partial(cls._prepare_facebook_post): "تجهيز منشور Facebook",
        }

        for callback, label in callbacks.items():
            key = str(callback.__hash__())
            botnav.buttons[key] = callback
            markup.add(InlineKeyboardButton(label, callback_data=key))

        markup.add(
            InlineKeyboardButton("بحث TikTok", url=urls["tiktok"]),
            InlineKeyboardButton("بحث Douyin الصيني", url=urls["douyin"]),
        )
        markup.add(
            InlineKeyboardButton("بحث REDnote", url=urls["rednote"]),
        )

        english = session.get("english_search") or "لم يتم استخراج كلمة بحث إنجليزية"
        chinese = session.get("chinese_search") or "لم يتم استخراج كلمة بحث صينية"

        text = (
            "تم تحليل المنتج.\n\n"
            f"كلمة البحث الإنجليزية:\n{english}\n\n"
            f"كلمة البحث الصينية:\n{chinese}\n\n"
            "اختر العملية المطلوبة:"
        )

        await botnav.bot.send_message(
            message.chat.id,
            text,
            reply_markup=markup,
        )

    @classmethod
    async def _send_product_description(cls, botnav, message) -> None:
        session = cls._get_product_session(botnav, message)

        if not session:
            await botnav.bot.send_message(
                message.chat.id,
                "لا توجد جلسة منتج محفوظة. أرسل صورة المنتج مرة أخرى."
            )
            return

        await cls._send_long_message(
            botnav,
            message.chat.id,
            session["description"],
        )

    @classmethod
    async def _publish_facebook_page(cls, botnav, message, session: dict) -> bool:
        """Publish the product image to a configured Facebook Page."""
        page_id = getattr(config, "FACEBOOK_PAGE_ID", "")
        access_token = getattr(config, "FACEBOOK_PAGE_ACCESS_TOKEN", "")
        graph_version = getattr(config, "FACEBOOK_GRAPH_VERSION", "v23.0")

        if not page_id or not access_token:
            return False

        image_bytes = await cls._get_session_image_bytes(botnav, session)
        if not image_bytes:
            raise ValueError("Product image is no longer available")

        url = f"https://graph.facebook.com/{graph_version}/{page_id}/photos"
        data = aiohttp.FormData()
        data.add_field("access_token", access_token)
        data.add_field(
            "source",
            image_bytes,
            filename="omex_product.jpg",
            content_type="image/jpeg",
        )
        data.add_field("caption", session.get("facebook_caption", ""))

        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as http:
            async with http.post(url, data=data) as response:
                body = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"Facebook API {response.status}: {body[:500]}")

        return True

    @classmethod
    async def _prepare_facebook_post(cls, botnav, message) -> None:
        """Publish to a configured Facebook Page, or prepare the post if credentials are absent."""
        session = cls._get_product_session(botnav, message)

        if not session:
            await botnav.bot.send_message(
                message.chat.id,
                "لا توجد جلسة منتج محفوظة. أرسل صورة المنتج مرة أخرى."
            )
            return

        try:
            published = await cls._publish_facebook_page(botnav, message, session)
        except Exception as exc:
            logger.exception("Facebook publishing failed: %s", exc)
            await botnav.bot.send_message(
                message.chat.id,
                "تعذر النشر على Facebook.\n\n" + str(exc)[:700],
            )
            published = False

        if published:
            await botnav.bot.send_message(
                message.chat.id,
                "تم نشر صورة المنتج مع الوصف على صفحة Facebook بنجاح.",
            )
            return

        await botnav.bot.send_message(
            message.chat.id,
            "منشور Facebook جاهز.\n\n" + session["description"]
            + "\n\nلم يتم النشر المباشر لأن بيانات Facebook غير مضافة في Railway.",
        )

        # Re-send the original product image with the generated caption so the
        # user can save/share it directly from Telegram.
        try:
            from io import BytesIO
            image_bytes = await cls._get_session_image_bytes(botnav, session)
            if not image_bytes:
                raise ValueError("Product image is no longer available")
            photo = BytesIO(image_bytes)
            photo.name = "omex_product.jpg"
            await botnav.bot.send_photo(
                message.chat.id,
                photo,
                caption=session["facebook_caption"][:1024],
            )
        except Exception as exc:
            logger.exception("Failed to prepare Facebook image: %s", exc)

    @classmethod
    async def _get_session_image_bytes(cls, botnav, session: dict) -> Optional[bytes]:
        """Download the original Telegram photo again using its file_id."""
        image_bytes = session.get("image_bytes")
        if image_bytes:
            return image_bytes

        file_id = session.get("file_id")
        if not file_id:
            return None

        file_info = await botnav.bot.get_file(file_id)
        content = await botnav.bot.download_file(file_info.file_path)
        return bytes(content)

    @classmethod
    async def _regenerate_product(cls, botnav, message) -> None:
        session = cls._get_product_session(botnav, message)

        if not session:
            await botnav.bot.send_message(
                message.chat.id,
                "لا توجد جلسة منتج محفوظة. أرسل صورة المنتج مرة أخرى."
            )
            return

        try:
            await botnav.bot.send_message(
                message.chat.id,
                "جاري إعادة تحليل الصورة..."
            )

            image_bytes = await cls._get_session_image_bytes(botnav, session)
            if not image_bytes:
                raise ValueError("Product image is no longer available")

            answer = await botnav.await_coro_sending_action(
                message.chat.id,
                cls.analyze_product_image(
                    user_id=message.from_user.id,
                    image_bytes=image_bytes,
                    user_caption=session.get("caption", ""),
                ),
                "typing",
            )

            clean_text, english_query, chinese_query = cls._parse_product_metadata(answer)

            session["description"] = clean_text
            session["english_search"] = english_query
            session["chinese_search"] = chinese_query
            session["facebook_caption"] = clean_text
            cls._save_product_session(botnav, message, session)

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

                # Do not make a second AI request just to create search terms.
                # The vision prompt already asks for them. If a model omits them,
                # use deterministic local fallbacks so the bot stays fast.
                if not english_query:
                    english_query = cls._local_search_fallback(clean_text, "en")
                if not chinese_query:
                    chinese_query = cls._local_search_fallback(clean_text, "zh")

                session = {
                    # Telegram keeps the original file available; storing file_id
                    # is safer than keeping raw image bytes in RAM.
                    "file_id": message.photo[-1].file_id,
                    "caption": caption,
                    "description": clean_text,
                    "facebook_caption": clean_text,
                    "english_search": english_query,
                    "chinese_search": chinese_query,
                }
                cls._save_product_session(botnav, message, session)

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