import requests


def get_modelinfo(model_name: str) -> dict:
    """
    Fetch model metadata from Hugging Face.
    If anything fails, return a simple fallback.
    """
    try:
        url = f"https://huggingface.co/api/models/{model_name}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"model": model_name, "error": str(e)}
