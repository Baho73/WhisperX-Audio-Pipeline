# WhisperX Audio Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![WhisperX](https://img.shields.io/badge/WhisperX-large--v3-orange)](https://github.com/m-bain/whisperX)

End-to-end audio processing pipeline: speech recognition, speaker diarization, and emotion analysis for Russian-language recordings.

## Pipeline

```
Audio File (mp3/wav/m4a/...)
    │
    ▼
┌─────────────────────────┐
│  1. Transcription        │  WhisperX (large-v3)
│     Speech → Text        │  Character-level alignment
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  2. Speaker Diarization  │  Pyannote 3.1
│     Who said what?       │  or Stereo channel split
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  3. Emotion Analysis     │  DUSHA HuBERT (5 emotions)
│     How they said it     │  Aniemore HuBERT (7 emotions)
└──────────┬──────────────┘
           │
           ▼
  Output: text / HTML / aligned
```

## Output Format

```
[00:00:05.123 - 00:00:10.456][Positive][Happiness] -> Speaker 1: Hello, how are you?
[00:00:10.789 - 00:00:15.321][Neutral][Neutral]    -> Speaker 2: Fine, let's discuss the project.
```

Each segment includes timestamps, emotion predictions from both models, speaker label, and transcribed text. HTML output adds color-coded emotions with an interactive legend.

## Emotion Models

| Model | Emotions | Source |
|-------|----------|--------|
| DUSHA HuBERT | neutral, angry, positive, sad, other | [xbgoose/dusha-asr-chunk-transformer](https://huggingface.co/xbgoose/dusha-asr-chunk-transformer) |
| Aniemore HuBERT | anger, disgust, enthusiasm, fear, happiness, neutral, sadness | [aniemore/wav2vec2-xlsr-53-russian-emotion-recognition](https://huggingface.co/Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition) |

## Components

| File | Description |
|------|-------------|
| `audio_pipeline.py` | Main pipeline — transcription, diarization, emotion analysis |
| `audio_converter.py` | Audio format converter (any format → WAV 16kHz mono via ffmpeg) |
| `emotion_xbgoose_05_prod.py` | Standalone emotion analyzer for pre-existing transcriptions |
| `check.py` | Environment verification (Python, CUDA, ffmpeg, libraries) |

## Usage

```bash
# Process all audio files in a directory
python audio_pipeline.py --input-dir ./recordings --output-dir ./results

# Process a single file
python audio_pipeline.py --single-file meeting.mp3

# Transcription only (no emotion analysis)
python audio_pipeline.py --single-file call.wav --no-emotions

# Convert audio to WAV
python audio_converter.py recording.m4a -o ./wav_output
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended, falls back to CPU)
- ffmpeg and ffprobe in PATH
- [Hugging Face token](https://huggingface.co/settings/tokens) for Pyannote diarization

### Key Dependencies

```
whisperx
pyannote.audio
torch + torchaudio
transformers
soundfile
rich (optional, for terminal preview)
```

## Tech Stack

`Python` `WhisperX` `Pyannote` `PyTorch` `HuBERT` `DUSHA` `speaker diarization` `emotion analysis` `ffmpeg`

## License

[MIT](LICENSE)
