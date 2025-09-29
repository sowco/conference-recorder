import os
import whisper
from pathlib import Path
import torch
import shutil
import re
import subprocess

def format_transcript(text: str, max_sentences_per_paragraph: int = 3) -> str:
    """
    Format transcript text into readable paragraphs.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    paragraphs = [
        ' '.join(sentences[i:i + max_sentences_per_paragraph])
        for i in range(0, len(sentences), max_sentences_per_paragraph)
    ]
    return '\n\n'.join(paragraphs)

def preprocess_audio(input_path: str) -> str:
    """
    Preprocess audio to improve quality (e.g., normalize, convert to WAV).
    Returns path to cleaned audio file.
    """
    cleaned_path = str(Path(input_path).with_suffix('.cleaned.wav'))
    
    command = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-ac', '1',              # mono
        '-ar', '16000',          # sample rate
        '-af', 'highpass=f=200, lowpass=f=3000, dynaudnorm',  # фильтры
        cleaned_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return cleaned_path
    except subprocess.CalledProcessError:
        print(f"❌ Failed to preprocess audio: {input_path}")
        return input_path  

def transcribe_audio_files(input_folder: str, output_folder: str, model_name: str = "large"):
    """
    Transcribe audio files in a folder using the Whisper model and save the text output.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📦 Using device: {device}")

    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("❌ ffmpeg is not installed or not in PATH.")

    print(f"🔁 Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name).to(device)

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)

        if not file_name.lower().endswith(('.mka', '.mkv', '.mp3', '.wav', '.m4a', '.flac', '.ogg')):
            print(f"⏩ Skipping non-audio file: {file_name}")
            continue

        print(f"🎧 Processing: {file_name}")
        try:
            # Предобработка аудио
            cleaned_path = preprocess_audio(file_path)

            # Транскрипция
            result = model.transcribe(cleaned_path)
            transcript = result.get('text', '').strip()

            if not transcript:
                print(f"⚠️ Empty transcript for: {file_name}")
                continue

            # Форматирование текста
            formatted_text = format_transcript(transcript)

            # Сохранение
            output_file_path = os.path.join(output_folder, f"{Path(file_name).stem}.txt")
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(formatted_text)

            print(f"✅ Saved transcript to: {output_file_path}")

            # Удаление временного файла
            if cleaned_path != file_path and os.path.exists(cleaned_path):
                os.remove(cleaned_path)

        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")

    print("🏁 Transcription completed.")
