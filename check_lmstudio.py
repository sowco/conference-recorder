import requests

LMSTUDIO_URL = "http://localhost:1234"

def check_lmstudio():
    # Проверяем список моделей
    try:
        r = requests.get(f"{LMSTUDIO_URL}/v1/models", timeout=5)
        r.raise_for_status()
        models = r.json().get("data", [])
        if not models:
            print("⚠️ Модели не найдены. Загрузи модель в LM Studio GUI.")
            return
        print("✅ LM Studio доступен. Доступные модели:")
        for m in models:
            print(" -", m["id"])
    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка подключения: {e}")
        return

    # Тестовый запрос completions
    try:
        model_id = models[0]["id"]
        payload = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": "Привет! Скажи одно предложение о себе."}
            ]
        }
        r = requests.post(f"{LMSTUDIO_URL}/v1/chat/completions", json=payload, timeout=360)
        r.raise_for_status()
        data = r.json()
        print("\n✅ Ответ completions:")
        print(data["choices"][0]["message"]["content"].strip())
    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка при запросе completions: {e}")

if __name__ == "__main__":
    check_lmstudio()
