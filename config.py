import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    default_source: str
    default_target: str
    history_limit: int
    gemini_api_keys: list[str]


def get_settings() -> Settings:
    # Support both comma-separated list and single key
    keys_str = os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", ""))
    keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    
    return Settings(
        default_source=os.getenv("DEFAULT_SOURCE_LANG", "en"),
        default_target=os.getenv("DEFAULT_TARGET_LANG", "ur"),
        history_limit=int(os.getenv("HISTORY_LIMIT", "20")),
        gemini_api_keys=keys,
    )
