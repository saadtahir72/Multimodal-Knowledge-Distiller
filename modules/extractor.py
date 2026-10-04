import os
import re
import pdfplumber
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi

def extract_text_from_pdf(pdf_file):
    """
    Extracts raw text from an uploaded PDF file and estimates text clarity score.
    """
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    
    total_chars = len(text)
    if total_chars == 0:
        return "", 0.0
    
    alphanumeric_chars = sum(c.isalnum() or c.isspace() for c in text)
    extraction_accuracy = round((alphanumeric_chars / total_chars) * 100, 2)
    return text, extraction_accuracy


def extract_text_from_youtube(url: str):
    """
    Extracts transcript from a YouTube URL.
    Attempts subtitle extraction via API first (instant), falling back to yt-dlp + Whisper if needed.
    """
    video_id_match = re.search(r"(?:v=|\/([0-9A-Za-z_-]{11})|youtu\.be\/)([0-9A-Za-z_-]{11})", url)
    if not video_id_match:
        raise ValueError("Invalid YouTube URL")
    
    video_id = video_id_match.group(1) or video_id_match.group(2)
    
    # Method 1: Instant YouTube Transcript API
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.fetch(video_id)
        
        text = " ".join([t['text'] for t in transcript_list])
        if text.strip():
            return text, 100.0, "Official Subtitles API"
    except Exception:
        # Fallback for older static method calls
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join([t['text'] for t in transcript_list])
            if text.strip():
                return text, 100.0, "Official Subtitles API"
        except Exception as api_err:
            print(f"Subtitle API unavailable ({api_err}), falling back to local Whisper processing...")

    # Method 2: Heavy Local Whisper Fallback with 403 Bypass
    try:
        import yt_dlp
        import whisper

        audio_path = f"temp_{video_id}.mp3"
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': f'temp_{video_id}',
            'quiet': True,
            # Bypasses 403 Forbidden errors from YouTube signature enforcement
            'extractor_args': {'youtube': {'player_client': ['android']}},
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        model = whisper.load_model("tiny")
        result = model.transcribe(audio_path)
        
        avg_logprob = np.mean([segment.get("avg_logprob", -1.0) for segment in result.get("segments", [])])
        model_accuracy = round(min(max(np.exp(avg_logprob) * 100, 50.0), 99.0), 2)

        if os.path.exists(audio_path):
            os.remove(audio_path)

        return result["text"], model_accuracy, "Whisper Local ML"

    except Exception as e:
        raise Exception(f"Failed to process video audio: {str(e)}")