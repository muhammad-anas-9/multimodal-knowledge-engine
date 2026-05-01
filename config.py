import os
from pathlib import Path

from dotenv import load_dotenv


REQUIRED_ENV_VARS = ("LLAMA_CLOUD_API_KEY", "GOOGLE_API_KEY")


def load_environment(
    env_file: str = ".env", required_env_vars: tuple[str, ...] = REQUIRED_ENV_VARS
) -> None:
    """Load .env values into process environment before LlamaIndex imports."""
    env_path = Path(env_file).resolve()
    if not env_path.exists():
        raise FileNotFoundError(
            f"Missing environment file: {env_path}. Create it from .env.example."
        )

    load_dotenv(dotenv_path=env_path, override=False)

    missing = [key for key in required_env_vars if not os.getenv(key)]
    if missing:
        raise EnvironmentError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Add them to your .env file."
        )

    for key in required_env_vars:
        os.environ[key] = os.environ[key]


if __name__ == "__main__":
    load_environment()
    print("Environment loaded successfully.")
