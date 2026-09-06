import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    port: int = int(os.getenv("PORT", "8000"))
    mongo_uri: str = os.getenv("MONGO_URI", "mongodb://mongo:27017/prism")
    env: str = os.getenv("ENV", "development")


settings = Settings()
