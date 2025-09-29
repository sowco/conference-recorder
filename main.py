import os
import subprocess
import keyboard
from datetime import datetime
from speech_analytics import transcribe_audio_files
from summarizer import summarize_transcripts
from dotenv import load_dotenv

load_dotenv()

DS_API_KEY = os.getenv("DeepSeek_API")

# путь для сохранения
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = f"recordings/{current_time}"
os.makedirs(folder_name, exist_ok=True)

mic_device = 'audio=Microphone (Thronmax Stream Go Pro Microphone)'
system_device = 'audio=Stereo Mix (Realtek(R) Audio)'

cmd = [
    "ffmpeg",
    "-f", "dshow", "-i", mic_device,
    "-f", "dshow", "-i", system_device,
    "-map", "0:a", "-c:a", "aac", "-metadata:s:a:0", "title=Microphone",
    "-map", "1:a", "-c:a", "aac", "-metadata:s:a:1", "title=System Audio",
    f"{folder_name}/recording.mka"
]

process = subprocess.Popen(cmd)
print("🎙️ Запись начата. Для остановки нажмите Ctrl+Q.")
keyboard.wait('ctrl+q')  # Можно использовать 'ctrl+shift+q' и т.д.

process.communicate(input=b"q")
# process.terminate()
print("🛑 Запись остановлена.")

# Можно передать только нужную дорожку ...
# Здесь пока что передаём всё
transcribe_audio_files(folder_name, folder_name, model_name="large")

summarize_transcripts(folder_name, method="lmstudio", api_key=DS_API_KEY) #, "deepseek"