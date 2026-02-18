from pydantic import BaseSettings

class LedgerSettings(BaseSettings):
    LEDGER_PROVIDER: str = "fineract"
    FINERACT_BASE_URL: str = "http://fineract:8080/fineract-provider"
    FINERACT_TENANT: str = "default"
    FINERACT_USER: str = "mifos"
    FINERACT_PASSWORD: str = "password"
    FINERACT_OFFICE_ID: int = 1

    class Config:
        env_file = ".env"
