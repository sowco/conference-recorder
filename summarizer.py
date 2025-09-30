import os
import re
import requests
import json
from pathlib import Path


def get_lmstudio_model_name():
    try:
        response = requests.get("http://localhost:1234/v1/models")
        response.raise_for_status()
        models = response.json().get("data", [])
        if models:
            return models[0]["id"] 
    except Exception as e:
        print(f"⚠️ Ошибка при получении списка моделей: {e}")
    return None

def clean_text(text: str) -> str:
    """Очищает текст от спецсимволов, лишних пробелов и переносов."""
    text = text.replace("\r", " ").replace("\n", " ")  # убираем переносы
    text = re.sub(r"\s+", " ", text)  # пробелы
    text = re.sub(r"[^\w\s.,!?-]", "", text)  # оставляем только буквы, цифры, знаки препинания
    return text.strip()



def generate_summary(text: str, method: str = "lmstudio", api_url: str = None, api_key: str = None) -> str:
    """
    Generate summary and key points for transcript.
    """
    try:
        if method == "lmstudio":
            model_name = get_lmstudio_model_name()
            print("Используем модель:", model_name)
            url = api_url or "http://localhost:1234/v1/chat/completions"
            clean = clean_text(text)
            payload = {
                "model": model_name,  
                "messages": [
                    {"role": "system", "content": "Ты помощник, который делает краткое саммари и ключевые пункты по расшифровке совещания."},
                    {"role": "user", "content": f"Текст совещания:\n\n{clean}\n\nСделай краткое саммари (несколько абзацев) и список ключевых моментов."}
                ],
                "temperature": 0.3,
                "stream": False
            }
            print("📤 Отправка запроса в LM Studio:")
            print(json.dumps(payload, ensure_ascii=False, indent=2))

            headers = {"Content-Type": "application/json"}
            response = requests.post(url, headers=headers, json=payload, timeout=1200)
            response.raise_for_status()
            data = response.json()

            # Попробуем разные форматы ответа
            try:
                return data["choices"][0]["message"]["content"].strip()
            except KeyError:
                return data["choices"][0]["messages"][0]["content"].strip()


        elif method == "openai":
            url = api_url or "https://api.openai.com/v1/chat/completions"
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "Ты помощник, который делает краткое саммари и ключевые пункты по расшифровке совещания."},
                    {"role": "user", "content": f"Текст совещания:\n\n{text}\n\nСделай краткое саммари (несколько абзацев) и список ключевых моментов."}
                ],
                "temperature": 0.3
            }
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        elif method == "deepseek":
            url = api_url or "https://api.deepseek.com/chat/completions"
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "Ты помощник, который делает краткое саммари и ключевые пункты по расшифровке совещания."},
                    {"role": "user", "content": f"Текст совещания:\n\n{text}\n\nСделай краткое саммари (несколько абзацев) и список ключевых моментов."}
                ],
                "temperature": 0.3,
                "stream": False
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        elif method == "huggingface":
            url = api_url or "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.post(url, headers=headers, json={"inputs": text})
            response.raise_for_status()
            data = response.json()
            return data[0]["summary_text"]

        elif method == "dummy":
            return "⚠️ Саммари не сгенерировано (используйте LM Studio или API)."

        else:
            raise ValueError(f"Unknown summarization method: {method}")

    except Exception as e:
        return f"⚠️ Ошибка при генерации саммари: {e}"


def summarize_transcripts(input_folder: str, method: str = "lmstudio", api_url: str = None, api_key: str = None):
    """
    Process all transcript .txt files in folder and generate summaries.
    Saves each summary as <filename>.summary.txt
    """
    for file_name in os.listdir(input_folder):
        # пропускаем уже готовые summary
        if not file_name.endswith(".txt") or file_name.endswith(".summary.txt"):
            continue

        file_path = os.path.join(input_folder, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            transcript = f.read().strip()

        if not transcript:
            print(f"⚠️ Empty transcript: {file_name}")
            continue

        print(f"📝 Summarizing: {file_name}")
        summary = generate_summary(transcript, method=method, api_url=api_url, api_key=api_key)

        summary_path = os.path.join(input_folder, f"{Path(file_name).stem}.summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)

        print(f"✅ Summary saved: {summary_path}")

