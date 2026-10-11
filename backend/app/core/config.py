import os
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    PROJECT_NAME: str = "Blog Notebook API"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"

    # Supabase config
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""

    # Security
    ADMIN_API_KEY: str = "dev_secret_key_123"

    # Server & CORS
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_ORIGINS: str = "*"

    @property
    def cors_origins(self) -> List[str]:
        if self.ALLOWED_ORIGINS.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

