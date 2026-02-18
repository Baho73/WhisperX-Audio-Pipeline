#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audio_pipeline_complete.py

ПОЛНЫЙ МОНОЛИТНЫЙ PIPELINE ДЛЯ ОБРАБОТКИ АУДИО
Включает: транскрипцию, диаризацию, анализ эмоций и форматирование результатов

Версия: 2.0.0
Автор: AI Assistant
"""

import os
import sys
import gc
import re
import json
import torch
import torchaudio
import torch.nn.functional as F
import logging
import warnings
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# Подавление предупреждений
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("pyannote.audio").setLevel(logging.ERROR)

# ================================================================================================
# БЛОК НАСТРОЕК - ВСЕ НАСТРОЙКИ В ОДНОМ МЕСТЕ
# ================================================================================================

class Config:
    """Центральная конфигурация всего pipeline"""
    
    # === ПУТИ К ДИРЕКТОРИЯМ ===
    INPUT_DIR = r"D:/Python/YT_DL/all"          # Папка с исходными аудиофайлами
    OUTPUT_DIR = r"./output_6"                     # Папка для сохранения результатов
    MODEL_DIR = r"./models"                      # Папка для кэширования моделей
    
    # === РЕЖИМ ОБРАБОТКИ ===
    PROCESSING_MODE = 'DIARIZE'                  # 'DIARIZE' или 'SPLIT_STEREO'
    
    # === HUGGING FACE ТОКЕН ===
    # Получите на https://huggingface.co/settings/tokens
    HF_TOKEN = 'hf_YOUR_TOKEN_HERE'             # Замените на ваш токен!
    
    # === НАСТРОЙКИ WHISPER ===
    WHISPER_MODEL = "large-v3"                   # "large-v3", "medium", "small", "base"
    WHISPER_LANGUAGE = "ru"                      # Язык для транскрипции
    BATCH_SIZE = 4                                # Размер батча (меньше = меньше памяти)
    COMPUTE_TYPE = "float16"                     # "float16" или "int8"
    
    # === НАСТРОЙКИ ЭМОЦИЙ ===
    ENABLE_EMOTIONS = True                       # Включить анализ эмоций
    EMOTION_MODELS = ['dusha', 'aniemore_hubert'] # Какие модели использовать
    
    # === УПРАВЛЕНИЕ ПАМЯТЬЮ ===
    CLEAR_MEMORY_EVERY = 2                       # Очищать память каждые N файлов
    FORCE_RELOAD_MODELS = 5                      # Перезагружать модели каждые N файлов
    
    # === ФОРМАТИРОВАНИЕ РЕЗУЛЬТАТОВ ===
    AUTO_CREATE_HTML = True                      # Автоматически создавать HTML версию
    AUTO_CREATE_ALIGNED = True                   # Автоматически создавать выровненную версию
    SHOW_TERMINAL_PREVIEW = True                 # Показывать превью в терминале
    TERMINAL_PREVIEW_LINES = 10                  # Количество строк для превью
    
    # === РЕЖИМ SPLIT_STEREO (если используется) ===
    SPEAKER_LABEL_LEFT = "Спикер_1"
    SPEAKER_LABEL_RIGHT = "Спикер_2"
    
    # === ОТЛАДКА ===
    DEBUG_MODE = False                            # Включить подробное логирование
    SKIP_EXISTING = True                         # Пропускать уже обработанные файлы

# ================================================================================================
# EMOTION PROPERTIES AND DATA STRUCTURES
# ================================================================================================

@dataclass
class EmotionProperties:
    """Свойства эмоции для визуализации"""
    color: str
    rich_color: str
    symbol: str

@dataclass
class Segment:
    """Сегмент транскрипции"""
    timestamp: str
    emotions: List[str]
    text: str
    line_number: Optional[int] = None

# Цветовая схема для эмоций
EMOTION_COLORS: Dict[str, EmotionProperties] = {
    'Нейтральная': EmotionProperties('#808080', 'bright_black', '○'),
    'Злость': EmotionProperties('#FF4444', 'red', '▲'),
    'Позитивная': EmotionProperties('#44FF44', 'green', '♦'),
    'Грусть': EmotionProperties('#4444FF', 'blue', '▼'),
    'Прочее': EmotionProperties('#FFA500', 'yellow', '■'),
    'Отвращение': EmotionProperties('#8B4513', 'magenta', '×'),
    'Энтузиазм': EmotionProperties('#FFD700', 'bright_yellow', '★'),
    'Страх': EmotionProperties('#9370DB', 'bright_magenta', '!'),
    'Счастье': EmotionProperties('#00FF00', 'bright_green', '♥'),
    'Ошибка': EmotionProperties('#FF0000', 'bright_red', '✗'),
}

# ================================================================================================
# ИМПОРТЫ МОДЕЛЕЙ (ленивая загрузка)
# ================================================================================================

# WhisperX
whisperx = None
# Pyannote
Pipeline = None
# Transformers для эмоций
HubertForSequenceClassification = None
Wav2Vec2FeatureExtractor = None
# Rich для красивого вывода
Console = None
Table = None
RICH_AVAILABLE = False

def lazy_imports():
    """Ленивая загрузка тяжелых библиотек"""
    global whisperx, Pipeline, HubertForSequenceClassification, Wav2Vec2FeatureExtractor
    global Console, Table, RICH_AVAILABLE
    
    try:
        import whisperx as wx
        whisperx = wx
    except ImportError:
        print("❌ WhisperX не установлен. Установите: pip install whisperx")
        sys.exit(1)
    
    try:
        from pyannote.audio import Pipeline as PA_Pipeline
        Pipeline = PA_Pipeline
    except ImportError:
        print("❌ Pyannote не установлен. Установите: pip install pyannote.audio")
        sys.exit(1)
    
    try:
        from transformers import (
            HubertForSequenceClassification as HubertModel,
            Wav2Vec2FeatureExtractor as WavExtractor
        )
        HubertForSequenceClassification = HubertModel
        Wav2Vec2FeatureExtractor = WavExtractor
    except ImportError:
        if Config.ENABLE_EMOTIONS:
            print("⚠️ Transformers не установлен. Эмоции будут отключены")
            Config.ENABLE_EMOTIONS = False
    
    try:
        from rich.console import Console as RichConsole
        from rich.table import Table as RichTable
        Console = RichConsole
        Table = RichTable
        RICH_AVAILABLE = True
    except ImportError:
        print("ℹ️ Rich не установлен. Красивый вывод недоступен")

# ================================================================================================
# ЛОГИРОВАНИЕ
# ================================================================================================

def setup_logging():
    """Настройка системы логирования"""
    level = logging.DEBUG if Config.DEBUG_MODE else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=level, format=log_format, stream=sys.stdout)
    return logging.getLogger(__name__)

logger = setup_logging()

# ================================================================================================
# EMOTION ANALYZER
# ================================================================================================

class EmotionAnalyzer:
    """Анализатор эмоций для аудио сегментов"""
    
    def __init__(self, device=None, models_to_use=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.all_models_config = {
            'dusha': {
                'name': 'DUSHA HuBERT',
                'short_name': 'DUSHA',
                'model_path': 'xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned',
                'feature_extractor': 'facebook/hubert-large-ls960-ft',
                'emotions': ['neutral', 'angry', 'positive', 'sad', 'other'],
                'emotions_ru': ['Нейтральная', 'Злость', 'Позитивная', 'Грусть', 'Прочее'],
            },
            'aniemore_hubert': {
                'name': 'Aniemore HuBERT',
                'short_name': 'Aniemore',
                'model_path': 'Aniemore/hubert-emotion-russian-resd',
                'feature_extractor': 'Aniemore/hubert-emotion-russian-resd',
                'emotions': ['anger', 'disgust', 'enthusiasm', 'fear', 'happiness', 'neutral', 'sadness'],
                'emotions_ru': ['Злость', 'Отвращение', 'Энтузиазм', 'Страх', 'Счастье', 'Нейтральная', 'Грусть'],
            }
        }
        
        if models_to_use is None:
            self.models_to_use = list(self.all_models_config.keys())
        else:
            self.models_to_use = [m for m in models_to_use if m in self.all_models_config]
        
        self.models_config = {
            k: v for k, v in self.all_models_config.items() 
            if k in self.models_to_use
        }
        
        self._loaded_models = {}
        self.sample_rate = 16000
        self.max_audio_length = 10
        
        logger.info(f"Анализатор эмоций инициализирован на {self.device}")
    
    def load_model(self, model_key):
        """Загрузка модели эмоций"""
        if model_key in self._loaded_models:
            return self._loaded_models[model_key]
        
        config = self.models_config[model_key]
        logger.info(f"Загрузка {config['name']}...")
        
        try:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                config['feature_extractor']
            )
            
            if model_key == 'dusha':
                model = HubertForSequenceClassification.from_pretrained(
                    config['model_path']
                )
            elif model_key == 'aniemore_hubert':
                model = HubertForSequenceClassification.from_pretrained(
                    config['model_path'],
                    num_labels=len(config['emotions']),
                    ignore_mismatched_sizes=True
                )
            
            model.to(self.device)
            model.eval()
            
            self._loaded_models[model_key] = {
                'model': model,
                'feature_extractor': feature_extractor,
                'config': config
            }
            
            logger.info(f"✅ {config['name']} загружена")
            return self._loaded_models[model_key]
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки {config['name']}: {e}")
            raise
    
    def extract_audio_segment(self, audio_path, start_time, end_time):
        """Извлечение сегмента аудио"""
        waveform, original_sample_rate = torchaudio.load(str(audio_path))
        
        if original_sample_rate != self.sample_rate:
            transform = torchaudio.transforms.Resample(original_sample_rate, self.sample_rate)
            waveform = transform(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        
        start_sample = max(0, start_sample)
        end_sample = min(waveform.shape[1], end_sample)
        
        segment = waveform[:, start_sample:end_sample]
        
        min_length = int(0.1 * self.sample_rate)
        if segment.shape[1] < min_length:
            padding = min_length - segment.shape[1]
            segment = F.pad(segment, (0, padding))
        
        return segment
    
    def predict_segment_emotion(self, audio_segment, model_key):
        """Предсказание эмоции для сегмента"""
        try:
            model_data = self.load_model(model_key)
            model = model_data['model']
            feature_extractor = model_data['feature_extractor']
            config = model_data['config']
            
            inputs = feature_extractor(
                audio_segment.squeeze().numpy(),
                sampling_rate=feature_extractor.sampling_rate,
                return_tensors="pt",
                padding=True,
                max_length=16000 * self.max_audio_length,
                truncation=True
            )
            
            input_values = inputs['input_values'].to(self.device)
            
            with torch.no_grad():
                outputs = model(input_values)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
            
            if self.device.type == "cuda":
                predictions = predictions.cpu()
                
            predicted_id = predictions.numpy()[0]
            
            num_emotions = len(config['emotions'])
            if predicted_id >= num_emotions:
                predicted_id = 0
            
            return config['emotions_ru'][predicted_id]
            
        except Exception as e:
            logger.error(f"Ошибка модели {model_key}: {e}")
            return "Ошибка"
    
    def clear_models(self):
        """Очистка загруженных моделей"""
        for model_key in list(self._loaded_models.keys()):
            del self._loaded_models[model_key]
        self._loaded_models = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ================================================================================================
# FORMATTER
# ================================================================================================

class ResultFormatter:
    """Форматирование результатов"""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.emotion_stats = {}
    
    def find_max_emotion_length(self, emotions_list):
        """Находит максимальную длину эмоции для выравнивания"""
        max_lengths = {}
        for emotions in emotions_list:
            for i, emotion in enumerate(emotions):
                if i not in max_lengths:
                    max_lengths[i] = 0
                max_lengths[i] = max(max_lengths[i], len(emotion))
        return max_lengths
    
    def create_html(self, segments, output_file, source_file=""):
        """Создает HTML файл с подсветкой эмоций"""
        html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Анализ эмоций - {source_file}</title>
    <style>
        body {{
            font-family: 'Consolas', 'Monaco', monospace;
            background-color: #1e1e1e;
            color: #d4d4d4;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{
            color: #569cd6;
            border-bottom: 2px solid #3e3e3e;
            padding-bottom: 10px;
        }}
        .timestamp {{
            color: #569cd6;
            font-weight: bold;
        }}
        .emotion {{
            padding: 2px 6px;
            margin: 0 2px;
            border-radius: 3px;
            font-weight: bold;
            display: inline-block;
        }}
        .text {{
            color: #ffffff;
            margin-left: 10px;
        }}
        .segment {{
            margin: 5px 0;
            padding: 8px;
            border-left: 3px solid #3e3e3e;
            transition: all 0.2s;
        }}
        .segment:hover {{
            background-color: #2d2d2d;
            border-left-color: #569cd6;
        }}
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 20px 0;
            padding: 15px;
            background-color: #2d2d2d;
            border-radius: 5px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
    </style>
</head>
<body>
    <h1>🎭 Анализ эмоций</h1>
    <p>Файл: {source_file} | Сегментов: {len(segments)}</p>
    
    <div class="legend">
        <strong style="width: 100%; color: #569cd6;">Легенда эмоций:</strong>"""
        
        for emotion, props in EMOTION_COLORS.items():
            html += f"""
        <div class="legend-item">
            <span class="emotion" style="background-color: {props.color};">
                {props.symbol} {emotion}
            </span>
        </div>"""
        
        html += """
    </div>
    <div class="segments">"""
        
        for seg in segments:
            html += f"""
        <div class="segment">
            <span class="timestamp">{seg['timestamp']}</span>"""
            
            for emotion in seg.get('emotions', []):
                emotion_clean = emotion.strip()
                props = EMOTION_COLORS.get(emotion_clean, EmotionProperties('#808080', 'white', '●'))
                html += f"""
            <span class="emotion" style="background-color: {props.color};">
                {props.symbol} {emotion_clean}
            </span>"""
            
            html += f"""
            <span class="text">→ {seg['text']}</span>
        </div>"""
        
        html += """
    </div>
</body>
</html>"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"HTML файл создан: {output_file}")
    
    def create_aligned_text(self, segments, output_file):
        """Создает выровненный текстовый файл"""
        all_emotions = [seg.get('emotions', []) for seg in segments]
        max_lengths = self.find_max_emotion_length(all_emotions)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Транскрипция с анализом эмоций (выровненная версия)\n")
            f.write(f"# Всего сегментов: {len(segments)}\n\n")
            
            for seg in segments:
                aligned_emotions = ''.join(
                    f"[{emotion.ljust(max_lengths.get(i, 11))}]"
                    for i, emotion in enumerate(seg.get('emotions', []))
                )
                f.write(f"{seg['timestamp']}{aligned_emotions} -> {seg['text']}\n")
        
        logger.info(f"Выровненный файл создан: {output_file}")
    
    def show_terminal_preview(self, segments, limit=10):
        """Показывает превью в терминале"""
        if not RICH_AVAILABLE or not self.console:
            # Простой вывод без Rich
            print("\n=== Превью результатов ===")
            for i, seg in enumerate(segments[:limit]):
                print(f"{seg['timestamp']} {seg.get('emotions', [])} -> {seg['text'][:80]}...")
            if len(segments) > limit:
                print(f"... и еще {len(segments) - limit} сегментов")
            return
        
        # Красивый вывод с Rich
        table = Table(title="🎭 Превью результатов анализа")
        table.add_column("Время", style="cyan", no_wrap=True)
        table.add_column("Эмоции", style="magenta")
        table.add_column("Текст", style="white")
        
        for seg in segments[:limit]:
            time_clean = seg['timestamp'].strip('[]')
            emotions_str = ' | '.join(seg.get('emotions', []))
            text = seg['text'][:60] + "..." if len(seg['text']) > 60 else seg['text']
            table.add_row(time_clean, emotions_str, text)
        
        self.console.print(table)
        if len(segments) > limit:
            self.console.print(f"[yellow]Показаны первые {limit} из {len(segments)} сегментов[/yellow]")

# ================================================================================================
# AUDIO PROCESSING
# ================================================================================================

def clear_gpu_memory():
    """Очистка памяти GPU"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def convert_to_wav_if_needed(input_path, output_dir):
    """Конвертация аудио в WAV если нужно"""
    file_ext = Path(input_path).suffix.lower()
    formats_to_convert = ['.m4a', '.mp3', '.aac', '.flac', '.ogg']
    
    if file_ext in formats_to_convert:
        base_name = Path(input_path).stem
        wav_path = os.path.join(output_dir, f"{base_name}_converted.wav")
        logger.info(f"Конвертация {file_ext} в WAV...")
        
        command = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            wav_path, "-loglevel", "error"
        ]
        
        try:
            subprocess.run(command, check=True)
            logger.info(f"Файл сконвертирован: {wav_path}")
            return wav_path, True
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка конвертации: {e}")
            raise
    
    return input_path, False

def parse_timestamp(timestamp_str):
    """Парсинг таймстампа"""
    pattern = r'\[(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})\]'
    match = re.match(pattern, timestamp_str.strip())
    if not match:
        raise ValueError(f"Неверный формат: {timestamp_str}")
    
    start_h, start_m, start_s, start_ms = map(int, match.groups()[:4])
    end_h, end_m, end_s, end_ms = map(int, match.groups()[4:])
    
    start_seconds = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000
    end_seconds = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000
    
    return start_seconds, end_seconds

# ================================================================================================
# MAIN PROCESSING PIPELINE
# ================================================================================================

class AudioProcessor:
    """Основной класс обработки аудио"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_loaded = False
        self.model_tx = None
        self.model_align = None
        self.metadata = None
        self.pipeline_diarize = None
        self.emotion_analyzer = None
        self.formatter = ResultFormatter()
    
    def load_models(self):
        """Загрузка всех моделей"""
        if self.models_loaded:
            return
        
        logger.info("Загрузка моделей...")
        
        # WhisperX
        self.model_tx = whisperx.load_model(
            Config.WHISPER_MODEL, 
            self.device, 
            compute_type=Config.COMPUTE_TYPE, 
            download_root=Config.MODEL_DIR
        )
        self.model_align, self.metadata = whisperx.load_align_model(
            language_code=Config.WHISPER_LANGUAGE, 
            device=self.device, 
            model_dir=Config.MODEL_DIR
        )
        
        # Pyannote для диаризации
        if Config.PROCESSING_MODE == 'DIARIZE':
            self.pipeline_diarize = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", 
                use_auth_token=Config.HF_TOKEN
            )
            self.pipeline_diarize.to(torch.device(self.device))
        
        # Эмоции
        if Config.ENABLE_EMOTIONS:
            self.emotion_analyzer = EmotionAnalyzer(
                device=self.device, 
                models_to_use=Config.EMOTION_MODELS
            )
            for model_key in self.emotion_analyzer.models_to_use:
                self.emotion_analyzer.load_model(model_key)
        
        self.models_loaded = True
        logger.info("Все модели загружены")
    
    def process_audio(self, input_path):
        """Основной метод обработки одного аудиофайла"""
        base_name = Path(input_path).stem
        
        # Проверка на существующие файлы
        final_file = Path(Config.OUTPUT_DIR) / f"{base_name}_final.txt"
        if Config.SKIP_EXISTING and final_file.exists():
            logger.info(f"Файл уже обработан: {base_name}")
            return str(final_file)
        
        logger.info(f"Обработка: {base_name}")
        
        # Конвертация в WAV если нужно
        wav_path, was_converted = convert_to_wav_if_needed(input_path, Config.OUTPUT_DIR)
        
        try:
            # 1. Транскрипция
            logger.info("Этап 1/3: Транскрипция...")
            sentences_file = Path(Config.OUTPUT_DIR) / f"{base_name}_sentences.txt"
            
            audio = whisperx.load_audio(input_path)
            result = self.model_tx.transcribe(
                audio, 
                batch_size=Config.BATCH_SIZE, 
                language=Config.WHISPER_LANGUAGE
            )
            aligned_result = whisperx.align(
                result["segments"], 
                self.model_align, 
                self.metadata, 
                audio, 
                self.device, 
                return_char_alignments=False
            )
            
            # Сохранение транскрипции
            segments = []
            with open(sentences_file, "w", encoding="utf-8") as f:
                for segment in aligned_result["segments"]:
                    start, end, text = segment['start'], segment['end'], segment['text'].strip()
                    if not text:
                        continue
                    start_str = f"{int(start // 3600):02}:{int((start % 3600) // 60):02}:{start % 60:06.3f}"
                    end_str = f"{int(end // 3600):02}:{int((end % 3600) // 60):02}:{end % 60:06.3f}"
                    timestamp = f"[{start_str} - {end_str}]"
                    f.write(f"{timestamp} -> {text}\n")
                    segments.append({
                        'timestamp': timestamp,
                        'start': start,
                        'end': end,
                        'text': text,
                        'emotions': []
                    })
            
            del audio, result, aligned_result
            gc.collect()
            
            # 2. Диаризация (опционально)
            if Config.PROCESSING_MODE == 'DIARIZE' and self.pipeline_diarize:
                logger.info("Этап 2/3: Диаризация...")
                diarization_result = self.pipeline_diarize(wav_path)
                # Здесь можно добавить обработку результатов диаризации
                del diarization_result
                gc.collect()
            
            # 3. Анализ эмоций
            if Config.ENABLE_EMOTIONS and self.emotion_analyzer:
                logger.info("Этап 3/3: Анализ эмоций...")
                audio_file = Path(wav_path) if was_converted else Path(input_path)
                
                for i, seg in enumerate(segments, 1):
                    if i % 10 == 0:
                        logger.info(f"  Обработано сегментов: {i}/{len(segments)}")
                    
                    try:
                        audio_segment = self.emotion_analyzer.extract_audio_segment(
                            audio_file, seg['start'], seg['end']
                        )
                        
                        emotions = []
                        for model_key in self.emotion_analyzer.models_to_use:
                            emotion = self.emotion_analyzer.predict_segment_emotion(
                                audio_segment, model_key
                            )
                            emotions.append(emotion)
                        
                        seg['emotions'] = emotions
                        
                        if i % 10 == 0:
                            del audio_segment
                            gc.collect()
                            
                    except Exception as e:
                        logger.error(f"Ошибка анализа эмоций: {e}")
                        seg['emotions'] = ['Ошибка'] * len(self.emotion_analyzer.models_to_use)
            
            # 4. Сохранение финального файла
            with open(final_file, 'w', encoding='utf-8') as f:
                if Config.ENABLE_EMOTIONS:
                    model_names = [cfg['short_name'] 
                                 for cfg in self.emotion_analyzer.models_config.values()]
                    f.write(f"# Анализ эмоций | Модели: {' | '.join(model_names)}\n")
                else:
                    f.write("# Транскрипция\n")
                f.write(f"# Файл: {base_name}\n")
                f.write(f"# Всего сегментов: {len(segments)}\n\n")
                
                for seg in segments:
                    emotions_str = ''.join(f"[{e}]" for e in seg['emotions'])
                    f.write(f"{seg['timestamp']}{emotions_str} -> {seg['text']}\n")
            
            logger.info(f"✅ Финальный файл: {final_file}")
            
            # 5. Создание дополнительных форматов
            if Config.AUTO_CREATE_HTML:
                html_file = Path(Config.OUTPUT_DIR) / f"{base_name}_colored.html"
                self.formatter.create_html(segments, html_file, base_name)
            
            if Config.AUTO_CREATE_ALIGNED:
                aligned_file = Path(Config.OUTPUT_DIR) / f"{base_name}_aligned.txt"
                self.formatter.create_aligned_text(segments, aligned_file)
            
            if Config.SHOW_TERMINAL_PREVIEW:
                self.formatter.show_terminal_preview(segments, Config.TERMINAL_PREVIEW_LINES)
            
            return str(final_file)
            
        finally:
            # Удаление временного WAV файла
            if was_converted and os.path.exists(wav_path):
                os.remove(wav_path)
                logger.info("Временный файл удален")
    
    def process_directory(self):
        """Обработка всех файлов в директории"""
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(Path(Config.INPUT_DIR).glob(f"*{ext}"))
        
        if not audio_files:
            logger.warning(f"Не найдено аудиофайлов в {Config.INPUT_DIR}")
            return
        
        logger.info(f"Найдено файлов: {len(audio_files)}")
        
        processed = 0
        for i, audio_file in enumerate(audio_files, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Файл {i}/{len(audio_files)}: {audio_file.name}")
            logger.info('='*60)
            
            try:
                self.process_audio(str(audio_file))
                processed += 1
                
                # Управление памятью
                if processed % Config.CLEAR_MEMORY_EVERY == 0:
                    logger.info("Очистка памяти...")
                    clear_gpu_memory()
                
                if Config.FORCE_RELOAD_MODELS > 0 and processed % Config.FORCE_RELOAD_MODELS == 0:
                    logger.info("Перезагрузка моделей...")
                    self.reload_models()
                    
            except Exception as e:
                logger.error(f"Ошибка обработки {audio_file.name}: {e}")
                if Config.DEBUG_MODE:
                    import traceback
                    traceback.print_exc()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Обработка завершена: {processed}/{len(audio_files)} файлов")
    
    def reload_models(self):
        """Перезагрузка моделей для освобождения памяти"""
        self.models_loaded = False
        
        if self.model_tx:
            del self.model_tx
        if self.model_align:
            del self.model_align
        if self.metadata:
            del self.metadata
        if self.pipeline_diarize:
            del self.pipeline_diarize
        if self.emotion_analyzer:
            self.emotion_analyzer.clear_models()
            del self.emotion_analyzer
        
        clear_gpu_memory()
        self.load_models()

# ================================================================================================
# MAIN ENTRY POINT
# ================================================================================================

def main():
    """Главная функция программы"""
    parser = argparse.ArgumentParser(
        description='Полный pipeline обработки аудио',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input-dir', 
        default=Config.INPUT_DIR,
        help='Папка с аудиофайлами'
    )
    parser.add_argument(
        '--output-dir', 
        default=Config.OUTPUT_DIR,
        help='Папка для результатов'
    )
    parser.add_argument(
        '--batch-size', 
        type=int,
        default=Config.BATCH_SIZE,
        help='Размер батча'
    )
    parser.add_argument(
        '--no-emotions', 
        action='store_true',
        help='Отключить анализ эмоций'
    )
    parser.add_argument(
        '--no-html', 
        action='store_true',
        help='Не создавать HTML файлы'
    )
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Режим отладки'
    )
    parser.add_argument(
        '--single-file',
        help='Обработать только один файл'
    )
    
    args = parser.parse_args()
    
    # Обновление конфигурации из аргументов
    if args.input_dir:
        Config.INPUT_DIR = args.input_dir
    if args.output_dir:
        Config.OUTPUT_DIR = args.output_dir
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.no_emotions:
        Config.ENABLE_EMOTIONS = False
    if args.no_html:
        Config.AUTO_CREATE_HTML = False
    if args.debug:
        Config.DEBUG_MODE = True
        setup_logging()  # Пересоздаем логгер с DEBUG уровнем
    
    # Создание директорий
    os.makedirs(Config.INPUT_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    # Проверка токена
    if Config.PROCESSING_MODE == 'DIARIZE' and 'ВАШ_ТОКЕН' in Config.HF_TOKEN:
        logger.error("❌ Не указан Hugging Face токен!")
        logger.info("Получите токен на https://huggingface.co/settings/tokens")
        logger.info("И замените 'hf_ВАШ_ТОКЕН_ЗДЕСЬ' в настройках Config.HF_TOKEN")
        return 1
    
    # Информация о конфигурации
    logger.info("="*60)
    logger.info("КОНФИГУРАЦИЯ:")
    logger.info(f"  Входная папка: {Config.INPUT_DIR}")
    logger.info(f"  Выходная папка: {Config.OUTPUT_DIR}")
    logger.info(f"  Модель Whisper: {Config.WHISPER_MODEL}")
    logger.info(f"  Размер батча: {Config.BATCH_SIZE}")
    logger.info(f"  Анализ эмоций: {'Да' if Config.ENABLE_EMOTIONS else 'Нет'}")
    logger.info(f"  Устройство: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    logger.info("="*60)
    
    # Ленивая загрузка библиотек
    logger.info("Загрузка библиотек...")
    lazy_imports()
    
    # Создание процессора
    processor = AudioProcessor()
    processor.load_models()
    
    # Обработка
    if args.single_file:
        # Обработка одного файла
        if not os.path.exists(args.single_file):
            logger.error(f"Файл не найден: {args.single_file}")
            return 1
        processor.process_audio(args.single_file)
    else:
        # Обработка всей директории
        processor.process_directory()
    
    logger.info("\n✨ Готово!")
    return 0

if __name__ == "__main__":
    sys.exit(main())