# -*- coding: utf-8 -*-
"""
ИСПРАВЛЕННЫЙ компактный анализатор эмоций
Формат: [timestamp][model1][model2][modelN] -> text
"""

import torch
import torch.nn.functional as F
import torchaudio
import logging
import time
import re
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
from transformers import (
    HubertForSequenceClassification, 
    Wav2Vec2FeatureExtractor
)

logger = logging.getLogger(__name__)

class FixedCompactEmotionAnalyzer:
    """
    Исправленный компактный анализатор эмоций
    """
    
    def __init__(self, device: Optional[str] = None, models_to_use: Optional[List[str]] = None):
        """Инициализация анализатора"""
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Конфигурация моделей
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
        
        # Выбор моделей
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
        self.max_audio_length = 10  # секунд
        
        logger.info(f"Инициализация на {self.device}")
        logger.info(f"Модели: {[config['short_name'] for config in self.models_config.values()]}")
    
    def load_model(self, model_key: str) -> Dict:
        """Загрузка модели"""
        if model_key in self._loaded_models:
            return self._loaded_models[model_key]
        
        config = self.models_config[model_key]
        logger.info(f"Загрузка {config['name']}...")
        
        try:
            # Feature extractor
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                config['feature_extractor']
            )
            
            # Модель
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
            
            logger.info(f"✅ {config['name']} готова")
            return self._loaded_models[model_key]
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки {config['name']}: {e}")
            raise
    
    def parse_timestamp(self, timestamp_str: str) -> Tuple[float, float]:
        """Парсинг таймстампа"""
        pattern = r'\[(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})\]'
        match = re.match(pattern, timestamp_str.strip())
        
        if not match:
            raise ValueError(f"Неверный формат таймстампа: {timestamp_str}")
        
        start_h, start_m, start_s, start_ms = map(int, match.groups()[:4])
        end_h, end_m, end_s, end_ms = map(int, match.groups()[4:])
        
        start_seconds = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000
        end_seconds = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000
        
        return start_seconds, end_seconds
    
    def parse_sentences_file(self, sentences_file: Union[str, Path]) -> List[Dict]:
        """Парсинг файла с предложениями"""
        sentences_file = Path(sentences_file)
        if not sentences_file.exists():
            raise FileNotFoundError(f"Файл с предложениями не найден: {sentences_file}")
        
        sentences = []
        
        with open(sentences_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                if ' -> ' in line:
                    timestamp_part, text_part = line.split(' -> ', 1)
                else:
                    logger.warning(f"Строка {line_num} не содержит ' -> ': {line}")
                    continue
                
                try:
                    start_time, end_time = self.parse_timestamp(timestamp_part)
                    sentences.append({
                        'timestamp': timestamp_part.strip(),
                        'start': start_time,
                        'end': end_time,
                        'text': text_part.strip(),
                        'line_number': line_num
                    })
                except ValueError as e:
                    logger.warning(f"Ошибка парсинга таймстампа в строке {line_num}: {e}")
                    continue
        
        logger.info(f"Парсинг завершен: {len(sentences)} предложений")
        return sentences
    
    def extract_audio_segment(self, audio_path: Union[str, Path], 
                             start_time: float, end_time: float) -> torch.Tensor:
        """Извлечение сегмента аудио"""
        # Загрузка аудио
        waveform, original_sample_rate = torchaudio.load(str(audio_path))
        
        # Ресэмплинг до 16kHz
        if original_sample_rate != self.sample_rate:
            transform = torchaudio.transforms.Resample(original_sample_rate, self.sample_rate)
            waveform = transform(waveform)
        
        # Конвертация в моно
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Извлечение сегмента
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        
        start_sample = max(0, start_sample)
        end_sample = min(waveform.shape[1], end_sample)
        
        segment = waveform[:, start_sample:end_sample]
        
        # Проверка минимальной длины
        min_length = int(0.1 * self.sample_rate)
        if segment.shape[1] < min_length:
            padding = min_length - segment.shape[1]
            segment = torch.nn.functional.pad(segment, (0, padding))
        
        return segment
    
    def predict_segment_emotion(self, audio_segment: torch.Tensor, model_key: str) -> str:
        """Предсказание эмоции для сегмента"""
        try:
            model_data = self.load_model(model_key)
            model = model_data['model']
            feature_extractor = model_data['feature_extractor']
            config = model_data['config']
            
            # Подготовка входных данных - ИСПРАВЛЕНО!
            inputs = feature_extractor(
                audio_segment.squeeze().numpy(),
                sampling_rate=feature_extractor.sampling_rate,
                return_tensors="pt",
                padding=True,
                max_length=16000 * self.max_audio_length,  # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ!
                truncation=True
            )
            
            input_values = inputs['input_values'].to(self.device)
            
            # Инференс
            with torch.no_grad():
                outputs = model(input_values)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
            
            # Обработка результатов
            if self.device.type == "cuda":
                predictions = predictions.cpu()
                
            predicted_id = predictions.numpy()[0]
            
            # Ограничение ID
            num_emotions = len(config['emotions'])
            if predicted_id >= num_emotions:
                predicted_id = 0
            
            return config['emotions_ru'][predicted_id]
            
        except Exception as e:
            logger.error(f"Ошибка модели {model_key}: {e}")
            return "Ошибка"
    
    def analyze_and_create_transcription(self, wav_filename: Union[str, Path], 
                                       output_file: Optional[Union[str, Path]] = None) -> Path:
        """Анализ файла и создание транскрибации с эмоциями"""
        wav_path = Path(wav_filename)
        
        # Обработка имени файла
        if wav_path.suffix.lower() == '.wav':
            base_name = wav_path.stem
            audio_file = wav_path
        else:
            base_name = str(wav_path)
            audio_file = wav_path.with_suffix('.wav')
        
        # Поиск файла с предложениями
        sentences_file = wav_path.parent / f"{base_name}_sentences.txt"
        
        if not audio_file.exists():
            raise FileNotFoundError(f"Аудиофайл не найден: {audio_file}")
        
        if not sentences_file.exists():
            raise FileNotFoundError(f"Файл с предложениями не найден: {sentences_file}")
        
        # Создание имени выходного файла
        if output_file is None:
            output_file = wav_path.parent / f"{base_name}_emotions.txt"
        
        output_file = Path(output_file)
        
        logger.info(f"🎵 Анализ файла: {audio_file.name}")
        logger.info(f"📄 Предложения: {sentences_file.name}")
        logger.info(f"💾 Выходной файл: {output_file.name}")
        
        # Парсинг предложений
        sentences = self.parse_sentences_file(sentences_file)
        
        if not sentences:
            raise ValueError("Не найдено ни одного предложения для анализа")
        
        # Предварительная загрузка всех моделей
        logger.info(f"🔄 Загрузка {len(self.models_to_use)} моделей...")
        for model_key in self.models_to_use:
            self.load_model(model_key)
        
        # Создание транскрибации с эмоциями
        with open(output_file, 'w', encoding='utf-8') as f:
            # Заголовочный комментарий
            model_names = [config['short_name'] for config in self.models_config.values()]
            f.write(f"# Анализ эмоций | Модели: {' | '.join(model_names)}\n")
            f.write(f"# Файл: {audio_file.name}\n")
            f.write(f"# Всего сегментов: {len(sentences)}\n")
            f.write("\n")
            
            for i, sentence in enumerate(sentences, 1):
                logger.info(f"Анализ сегмента {i}/{len(sentences)}")
                
                try:
                    # Извлечение аудиосегмента
                    audio_segment = self.extract_audio_segment(
                        audio_file, 
                        sentence['start'], 
                        sentence['end']
                    )
                    
                    # Анализ каждой моделью
                    emotion_results = []
                    for model_key in self.models_to_use:
                        emotion = self.predict_segment_emotion(audio_segment, model_key)
                        emotion_results.append(emotion)
                    
                    # Формирование строки результата
                    emotion_brackets = ''.join(f"[{emotion}]" for emotion in emotion_results)
                    result_line = f"{sentence['timestamp']}{emotion_brackets} -> {sentence['text']}\n"
                    
                    f.write(result_line)
                    
                    # Логирование результата
                    emotions_str = " | ".join(emotion_results)
                    logger.info(f"  Результат: {emotions_str}")
                    
                except Exception as e:
                    logger.error(f"Ошибка обработки сегмента {i}: {e}")
                    # Запись ошибки в файл
                    error_brackets = ''.join(f"[Ошибка]" for _ in self.models_to_use)
                    error_line = f"{sentence['timestamp']}{error_brackets} -> {sentence['text']}\n"
                    f.write(error_line)
        
        logger.info(f"✅ Транскрибация с эмоциями сохранена: {output_file}")
        return output_file

# Удобная функция
def create_emotion_transcription_fixed(wav_filename: Union[str, Path], 
                                     models: Optional[List[str]] = None,
                                     output_file: Optional[Union[str, Path]] = None) -> Path:
    """
    ИСПРАВЛЕННАЯ функция создания транскрибации с эмоциями
    
    Args:
        wav_filename: Имя WAV файла
        models: Список моделей ['dusha', 'aniemore_hubert'] или None для всех
        output_file: Путь к выходному файлу
        
    Returns:
        Путь к созданному файлу
    """
    analyzer = FixedCompactEmotionAnalyzer(models_to_use=models)
    return analyzer.analyze_and_create_transcription(wav_filename, output_file)

def main():
    """Пример использования"""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='ИСПРАВЛЕННЫЙ анализ эмоций по сегментам')
    parser.add_argument('wav_file', help='Путь к WAV файлу')
    parser.add_argument('-o', '--output', help='Путь к выходному файлу', default=None)
    parser.add_argument('-m', '--models', nargs='+', 
                       choices=['dusha', 'aniemore_hubert'],
                       help='Модели для использования', default=None)
    
    args = parser.parse_args()
    
    try:
        output_file = create_emotion_transcription_fixed(args.wav_file, args.models, args.output)
        print(f"✅ Исправленная транскрибация создана: {output_file}")
        
        # Показать пример содержимого
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"\n📄 Первые строки результата:")
            for line in lines[:10]:  # Показать первые 10 строк
                print(line.rstrip())
            if len(lines) > 10:
                print("...")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Тестирование
    test_wav = r"D:\Python\WhisperX_noGUI_w_02\output_5\Мария коучинг 24.06.25.wav"
    
    if Path(test_wav).exists():
        try:
            logging.basicConfig(level=logging.INFO)
            output_file = create_emotion_transcription_fixed(test_wav)
            print(f"\n✅ ИСПРАВЛЕННАЯ транскрибация создана: {output_file}")
            
            # Показать результат
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"\n📄 Содержимое файла:")
                print(content[:1000] + "..." if len(content) > 1000 else content)
                
        except Exception as e:
            print(f"❌ Ошибка тестирования: {e}")
    else:
        main()

        