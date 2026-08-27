from pathlib import Path

from dotenv import load_dotenv


def load_settings() -> None:
    """Load a local .env once; real environment variables retain precedence."""
    load_dotenv(Path.cwd() / ".env", override=False)
