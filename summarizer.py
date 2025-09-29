import os
import requests
import json
from pathlib import Path

def generate_summary(text: str, method: str = "lmstudio", api_url: str = None, api_key: str = None) -> str:
    """
    Generate summary and key points for transcript.
    """
    try:
        if method == "lmstudio":
            url = api_url or "http://localhost:1234/v1/chat/completions"
            payload = {
                "model": "local-model",
                "messages": [
                    {"role": "system", "content": "Ты помощник, который делает краткое саммари и ключевые пункты по расшифровке совещания. Ничего не добавляй."},
                    {"role": "user", "content": f"Текст совещания:\n\n{text}\n\nСделай краткое саммари (несколько абзацев) и список ключевых моментов. Ничего не добавляй."}
                ],
                "temperature": 0.3
            }
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

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

