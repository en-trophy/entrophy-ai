import requests
import os

VISION_ENDPOINT = os.getenv("VISION_ENDPOINT")
VISION_KEY = os.getenv("VISION_KEY")

def analyze_image_with_vision(image_bytes: bytes):
    params = {
        "api-version": "2024-02-01",
        "features": "objects,people"
    }

    headers = {
        "Ocp-Apim-Subscription-Key": VISION_KEY,
        "Content-Type": "application/octet-stream"
    }

    response = requests.post(
        VISION_ENDPOINT,
        params=params,
        headers=headers,
        data=image_bytes
    )

    response.raise_for_status()
    return response.json()

file = open("test.jpg", "rb")
print(analyze_image_with_vision(file))
file.close()