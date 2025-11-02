"""Demonstration of structured data extraction with OpenRouter and Pydantic."""
from __future__ import annotations

import json
import os
from typing import List, Optional

from openai import OpenAI
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

load_dotenv()


MODEL_ID = "openai/gpt-4o"
REFERER = "http://localhost"
SITE_TITLE = "Structured Extraction Demo"


class SpeakerContact(BaseModel):
    name: str
    organization: Optional[str]
    email: Optional[str]


class TalkDatum(BaseModel):
    topic: str
    speaker: SpeakerContact
    start_time: Optional[str]


class EventSummary(BaseModel):
    title: str
    location: Optional[str]
    talks: List[TalkDatum]
    published_at: Optional[str]


print(EventSummary.model_json_schema())


def build_system_prompt() -> str:
    schema = EventSummary.model_json_schema()
    return (
        "You extract structured data from event announcements. "
        "Answer using JSON that exactly matches this schema: "
        f"{json.dumps(schema, indent=2)}"
    )


def call_openrouter(system_prompt: str, text_to_extract: str) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is required")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    completion = client.chat.completions.create(
        model=MODEL_ID,
        response_format={"type": "json_object"},
        extra_headers={
            "HTTP-Referer": REFERER,
            "X-Title": SITE_TITLE,
        },
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text_to_extract},
        ],
    )

    return completion.choices[0].message.content


def extract_event_summary(text_to_extract: str) -> EventSummary:
    system_prompt = build_system_prompt()
    raw_response = call_openrouter(system_prompt, text_to_extract)

    try:
        structured = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model response was not valid JSON: {raw_response}") from exc

    try:
        return EventSummary.model_validate(structured)
    except ValidationError as exc:
        raise ValueError(f"Model JSON did not match schema: {exc}") from exc


def main() -> None:
    announcement = (
        "Join us this Thursday at the Mission Street Hub for the 2024 Health Tech Meetup! "
        "Doors open at 5:00 PM. Theresa Lee from BioSense will present 'AI in Clinical Trials' at 5:30, "
        "followed by Samir Gupta of HealthFirst covering 'Scaling Remote Care' at 6:15. "
        "RSVP via events@missionhub.org."
    )

    summary = extract_event_summary(announcement)
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
