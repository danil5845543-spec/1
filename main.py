import json
import os
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conint, confloat

from openai import OpenAI

# ----------------
# Settings
# ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # добавь в Render Env Vars
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# Создаем клиента только если ключ задан (чтобы сервис не падал при старте)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title="Menu Generator API", version="1.0")

# CORS: на MVP можно разрешить все, позже ограничишь доменом лендинга
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # лучше: ["https://your-landing.netlify.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------
# Request schemas
# ----------------
Goal = Literal["budget_and_routine", "weight_loss", "muscle_gain", "just_menu"]
TimeProfile = Literal["15", "40", "60", "mix"]
AgeGroup = Literal["0_3", "4_7", "8_12", "13_plus"]
Allergy = Literal["nuts", "dairy", "eggs", "gluten", "fish_seafood", "other"]
Avoid = Literal["pork", "beef", "chicken", "fish_seafood", "spicy", "other"]


class Dietary(BaseModel):
    allergies: List[Allergy] = Field(default_factory=list)
    avoid: List[Avoid] = Field(default_factory=list)
    special_requests: Optional[str] = Field(
        default="",
        description="Свободный текст: предпочтения кухни/запреты по вкусу/культурные пожелания"
    )


class MenuRequest(BaseModel):
    city: str = Field(min_length=2, max_length=60)
    goal: Goal
    budget_week_eur: confloat(gt=0, lt=2000)
    adults: conint(ge=1, le=6)
    children: conint(ge=0, le=6) = 0
    children_age_groups: List[AgeGroup] = Field(default_factory=list)

    dietary: Dietary = Field(default_factory=Dietary)

    time_profile: TimeProfile
    favorite_dishes_text: str = Field(default="", max_length=400)


# ----------------
# Output schema
# ----------------
class MenuResponse(BaseModel):
    currency: str = "EUR"
    estimated_total_eur: float
    menu: list
    shopping_list: list
    notes: list


# ----------------
# Prompt helpers
# ----------------
def build_system_prompt() -> str:
    return (
        "Ты — помощник по планированию меню на неделю с учетом бюджета. "
        "Верни результат СТРОГО в JSON (без markdown, без пояснений). "
        "Меню должно быть реалистичным для домашней готовки. "
        "Определи страну/контекст по городу пользователя и адаптируй продукты под местную доступность. "
        "Если пользователь просит кухню (например, русскую в Испании) — сохрани стиль кухни, "
        "но используй продукты, которые реально купить в местных супермаркетах."
    )


def build_user_prompt(data: MenuRequest) -> str:
    return f"""
Данные пользователя:
- Город: {data.city}
- Цель: {data.goal}
- Бюджет на 7 дней: {data.budget_week_eur} EUR
- Семья: взрослых {data.adults}, детей {data.children}, возрастные группы детей: {data.children_age_groups}
- Аллергии: {data.dietary.allergies}
- Исключить/не есть: {data.dietary.avoid}
- Особые пожелания: {data.dietary.special_requests}
- Формат готовки (время): {data.time_profile}
- Привычные/любимые блюда (якоря вкуса): {data.favorite_dishes_text}

Задача:
Составь меню на 7 дней (завтрак/обед/ужин) и список покупок.
Старайся уложиться в бюджет.
Избегай аллергенов и исключений.
Меню должно быть выполнимым по времени готовки под выбранный формат.

Формат ответа (строго JSON):
{{
  "currency": "EUR",
  "estimated_total_eur": number,
  "menu": [
    {{
      "day": "Mon",
      "meals": {{
        "breakfast": {{"title": "...", "short": "..."}},
        "lunch": {{"title": "...", "short": "..."}},
        "dinner": {{"title": "...", "short": "..."}}
      }},
      "day_cost_eur": number
    }}
  ],
  "shopping_list": [
    {{
      "category": "produce|meat|dairy|dry|frozen|other",
      "items": [{{"name":"...", "qty":"..."}}]
    }}
  ],
  "notes": ["..."]
}}
""".strip()


# ----------------
# API endpoints
# ----------------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/ping")
def ping():
    return {"status": "ok", "message": "API is alive 🚀"}


@app.post("/test-generate")
def test_generate(payload: dict):
    return {
        "status": "ok",
        "received_payload": payload,
        "message": "Test endpoint works. AI was NOT called."
    }


@app.post("/generate-menu", response_model=MenuResponse)
def generate_menu(payload: MenuRequest):
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured (set it in Render Environment Variables)."
        )

    try:
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(payload)

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )

        raw = resp.choices[0].message.content

        # Иногда модель может добавить пробелы/текст — попробуем аккуратно вытащить JSON
        raw_stripped = raw.strip()
        # быстрый хак: если вдруг пришло с текстом, вырезаем от первой '{' до последней '}'
        if not raw_stripped.startswith("{"):
            start = raw_stripped.find("{")
            end = raw_stripped.rfind("}")
            if start != -1 and end != -1 and end > start:
                raw_stripped = raw_stripped[start:end + 1]

        data = json.loads(raw_stripped)

        # Базовая валидация ожидаемых полей
        for k in ("estimated_total_eur", "menu", "shopping_list", "notes"):
            if k not in data:
                raise ValueError(f"Missing key in model JSON: {k}")

        data.setdefault("currency", "EUR")
        return data

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502,
            detail="AI returned non-JSON response. Try again or tighten the prompt."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
