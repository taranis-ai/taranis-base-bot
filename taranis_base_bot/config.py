from pydantic import field_validator, ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict
from datetime import datetime


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    MODULE_ID: str = "TaranisBot"
    DEBUG: bool = False
    API_KEY: str = ""

    COLORED_LOGS: bool = True
    BUILD_DATE: datetime = datetime.now()
    GIT_INFO: dict[str, str] | None = None
    CACHE_TYPE: str = "SimpleCache"
    CACHE_DEFAULT_TIMEOUT: int = 300

    @field_validator("API_KEY", mode="before")
    def check_non_empty_string(cls, v: str, info: ValidationInfo) -> str:
        if not isinstance(v, str) or not v.strip():
            print("API_KEY is not set or empty, disabling API key requirement")
        return v


Config = Settings()
