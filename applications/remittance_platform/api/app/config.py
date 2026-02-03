from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg2://wallet:wallet@localhost:5432/wallet"
    api_key_internal: str = "dev-internal-key"

    class Config:
        env_prefix = ""
        case_sensitive = False

settings = Settings()
