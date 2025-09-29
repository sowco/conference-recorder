import os
import keyboard
from speech_analytics import transcribe_audio_files
from summarizer import summarize_transcripts
from dotenv import load_dotenv

load_dotenv()
DS_API_KEY = os.getenv("DeepSeek_API")

def main():
    folder_path = input("Введите путь к папке с файлами: ").strip()
    if not os.path.exists(folder_path):
        print("❌ Папка не найдена.")
        return

    print("Нажмите F для транскрибации + саммари, или S для саммари только по текстовым файлам.")
    while True:
        if keyboard.is_pressed('f'):
            print("🎧 Запуск транскрибации + саммари...")
            transcribe_audio_files(folder_path, folder_path, model_name="large")
            summarize_transcripts(folder_path, method="lmstudio", api_key=DS_API_KEY)
            print("✅ Обработка завершена.")
            break
        elif keyboard.is_pressed('s'):
            print("📝 Саммари только по текстовым файлам...")
            summarize_transcripts(folder_path, method="lmstudio", api_key=DS_API_KEY)
            print("✅ Саммари завершено.")
            break

if __name__ == "__main__":
    main()
