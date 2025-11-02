"""Demo of calling OpenRouter without structured extraction enforcement."""
from __future__ import annotations

import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "openai/gpt-4o"
REFERER = "http://localhost"
SITE_TITLE = "Unstructured Extraction Demo"


def build_system_prompt() -> str:
    return (
        "You extract event details from announcements."
        "Provide the event title, location, each talk's topic, speaker name, organization, and start time, "
        "along with any publication date you notice."
    )


def call_openrouter(prompt: str) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is required")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    completion = client.chat.completions.create(
        model=MODEL_ID,
        extra_headers={
            "HTTP-Referer": REFERER,
            "X-Title": SITE_TITLE,
        },
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": prompt},
        ],
    )

    return completion.choices[0].message.content


def main() -> None:
    announcement = (
        "Join us this Thursday at the Mission Street Hub for the 2024 Health Tech Meetup! "
        "Doors open at 5:00 PM. Theresa Lee from BioSense will present 'AI in Clinical Trials' at 5:30, "
        "followed by Samir Gupta of HealthFirst covering 'Scaling Remote Care' at 6:15. "
        "RSVP via events@missionhub.org."
    )

    response = call_openrouter(announcement)
    print(response)


if __name__ == "__main__":
    main()
