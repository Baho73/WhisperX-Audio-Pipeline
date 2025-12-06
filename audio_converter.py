# -*- coding: utf-8 -*-
"""
Модуль для анализа и конвертации аудиофайлов в WAV 16kHz с использованием ffprobe и ffmpeg

ИСПОЛЬЗОВАНИЕ:

1. Как модуль в Python:
   from audio_converter import convert_to_wav_16khz
   wav_file = convert_to_wav_16khz("input.m4a")
   wav_file = convert_to_wav_16khz("input.m4a", output_dir="./wav_files")

2. Запуск из командной строки:
   python audio_converter.py input_file.m4a
   python audio_converter.py input_file.m4a --output ./wav_files
   python audio_converter.py --help

3. Настройка логирования:
   import logging
   logging.basicConfig(level=logging.DEBUG)  # DEBUG, INFO, WARNING, ERROR
   logging.getLogger('audio_converter').setLevel(logging.DEBUG)

ТРЕБОВАНИЯ:
- ffmpeg и ffprobe должны быть установлены и доступны в PATH
- Python 3.7+

Сохраните как: audio_converter.py
"""

# ========================================================================
# НАСТРОЙКИ И КОНФИГУРАЦИЯ
# ========================================================================

# Аудио параметры конвертации
TARGET_SAMPLE_RATE = 16000      # Целевая частота дискретизации (Гц)
TARGET_CHANNELS = 1             # Количество каналов (1=моно, 2=стерео)
TARGET_CODEC = 'pcm_s16le'      # Аудио кодек (PCM 16-bit little-endian)
TARGET_SAMPLE_FORMAT = 's16'    # Формат сэмпла (16-bit)

# Параметры качества ресэмплинга (совместимые с большинством версий ffmpeg)
RESAMPLER_ALGORITHM = 'soxr'        # Алгоритм ресэмплинга (soxr, libswresample)
RESAMPLER_PRECISION = 28            # Точность ресэмплинга (20-33)
RESAMPLER_CUTOFF = 0.99             # Частота среза (0.8-0.99)
USE_ADVANCED_RESAMPLER = True       # Использовать продвинутый ресэмплер (может не работать на старых ffmpeg)

# Если продвинутые параметры не работают, будет использован базовый ресэмплер
FALLBACK_RESAMPLER = 'libswresample' # Запасной алгоритм ресэмплинга

# Поддерживаемые расширения файлов
SUPPORTED_AUDIO_EXTENSIONS = (
    '.mp3', '.m4a', '.wav', '.flac', '.aac', '.ogg', 
    '.wma', '.mp4', '.avi', '.mov', '.webm', '.3gp',
    '.opus', '.amr', '.au', '.ra'
)

# Суффикс для выходных файлов
OUTPUT_FILE_SUFFIX = '_16khz_mono'  # Добавляется к имени файла

# Таймауты (в секундах)
FFPROBE_TIMEOUT = 30            # Таймаут для анализа файла
FFMPEG_TIMEOUT = 600            # Таймаут для конвертации (10 минут)

# Настройки ffmpeg
FFMPEG_LOG_LEVEL = 'error'      # Уровень логов ffmpeg (quiet, error, warning, info, debug)
SHOW_FFMPEG_PROGRESS = True     # Показывать прогресс конвертации

# Проверки и валидация
CHECK_OUTPUT_FILE = True        # Проверять созданный файл после конвертации
MIN_OUTPUT_FILE_SIZE = 1024     # Минимальный размер выходного файла (байт)

# ========================================================================

import os
import json
import subprocess
import logging
import argparse
import sys
from pathlib import Path
from typing import Union, Optional, Dict, Any

# Настройка логгера
logger = logging.getLogger(__name__)

class AudioConverterError(Exception):
    """Базовый класс исключений для аудио конвертера"""
    pass

class FFmpegNotFoundError(AudioConverterError):
    """Исключение когда ffmpeg не найден"""
    pass

class FileAnalysisError(AudioConverterError):
    """Исключение при ошибке анализа файла"""
    pass

class ConversionError(AudioConverterError):
    """Исключение при ошибке конвертации"""
    pass

def check_ffmpeg_availability():
    """
    Проверка наличия ffmpeg и ffprobe
    
    Raises:
        FFmpegNotFoundError: Если ffmpeg или ffprobe не найдены
    """
    logger.debug("Проверка наличия ffmpeg и ffprobe...")
    
    for tool in ['ffmpeg', 'ffprobe']:
        try:
            result = subprocess.run([tool, '-version'], 
                                  capture_output=True, 
                                  check=True, 
                                  timeout=10)
            logger.debug(f"{tool} найден")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            error_msg = f"{tool} не найден или недоступен: {e}"
            logger.error(error_msg)
            raise FFmpegNotFoundError(error_msg)
    
    logger.info("ffmpeg и ffprobe доступны")

def analyze_audio_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Анализ аудиофайла с помощью ffprobe
    
    Args:
        file_path: Путь к аудиофайлу
        
    Returns:
        Словарь с характеристиками файла
        
    Raises:
        FileNotFoundError: Если файл не найден
        FileAnalysisError: Если не удалось проанализировать файл
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        error_msg = f"Файл не найден: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Анализ файла: {file_path}")
    
    # Команда ffprobe для получения информации в JSON формате
    command = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        str(file_path)
    ]
    
    try:
        result = subprocess.run(command, 
                              capture_output=True, 
                              text=True, 
                              check=True, 
                              timeout=FFPROBE_TIMEOUT)
        
        # Парсинг JSON ответа
        probe_data = json.loads(result.stdout)
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Ошибка ffprobe: {e.stderr}"
        logger.error(error_msg)
        raise FileAnalysisError(error_msg)
    except subprocess.TimeoutExpired:
        error_msg = f"Таймаут при анализе файла (более {FFPROBE_TIMEOUT} секунд)"
        logger.error(error_msg)
        raise FileAnalysisError(error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"Ошибка парсинга JSON от ffprobe: {e}"
        logger.error(error_msg)
        raise FileAnalysisError(error_msg)
    
    # Извлечение информации из JSON
    try:
        format_info = probe_data.get('format', {})
        streams = probe_data.get('streams', [])
        
        # Ищем первый аудио поток
        audio_stream = None
        for stream in streams:
            if stream.get('codec_type') == 'audio':
                audio_stream = stream
                break
        
        if not audio_stream:
            error_msg = "Аудио поток не найден в файле"
            logger.error(error_msg)
            raise FileAnalysisError(error_msg)
        
        # Формирование результата
        analysis = {
            # Основные характеристики
            'filename': file_path.name,
            'filepath': str(file_path),
            'file_size': file_path.stat().st_size,
            
            # Формат и контейнер
            'container_format': format_info.get('format_name', 'unknown'),
            'container_long_name': format_info.get('format_long_name', 'unknown'),
            
            # Аудио параметры
            'audio_codec': audio_stream.get('codec_name', 'unknown'),
            'audio_codec_long': audio_stream.get('codec_long_name', 'unknown'),
            'sample_rate': int(audio_stream.get('sample_rate', 0)),
            'channels': int(audio_stream.get('channels', 0)),
            'channel_layout': audio_stream.get('channel_layout', 'unknown'),
            'bit_rate': int(audio_stream.get('bit_rate', 0)) if audio_stream.get('bit_rate') else None,
            
            # Длительность
            'duration_seconds': float(audio_stream.get('duration', 0)) if audio_stream.get('duration') else float(format_info.get('duration', 0)),
            
            # Дополнительные параметры
            'bits_per_sample': int(audio_stream.get('bits_per_raw_sample', 0)) if audio_stream.get('bits_per_raw_sample') else None,
            'sample_fmt': audio_stream.get('sample_fmt', 'unknown'),
        }
        
        # Вычисляем читаемую длительность
        duration = analysis['duration_seconds']
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = duration % 60
        analysis['duration_formatted'] = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
        
        # Вычисляем размер файла в читаемом формате
        size_bytes = analysis['file_size']
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                analysis['file_size_formatted'] = f"{size_bytes:.1f} {unit}"
                break
            size_bytes /= 1024.0
        
        logger.debug(f"Анализ завершен: {analysis['audio_codec']} {analysis['sample_rate']}Hz {analysis['channels']}ch")
        return analysis
        
    except (KeyError, ValueError, TypeError) as e:
        error_msg = f"Ошибка извлечения данных из ffprobe: {e}"
        logger.error(error_msg)
        raise FileAnalysisError(error_msg)

def print_audio_info(analysis: Dict[str, Any]) -> None:
    """
    Вывод характеристик аудиофайла в консоль
    
    Args:
        analysis: Результат анализа файла
    """
    print("\n" + "="*60)
    print(" ХАРАКТЕРИСТИКИ АУДИОФАЙЛА")
    print("="*60)
    
    print(f"📁 Файл: {analysis['filename']}")
    print(f"📏 Размер: {analysis['file_size_formatted']}")
    print(f"⏱️  Длительность: {analysis['duration_formatted']}")
    
    print(f"\n📦 КОНТЕЙНЕР:")
    print(f"   Формат: {analysis['container_format']}")
    print(f"   Описание: {analysis['container_long_name']}")
    
    print(f"\n🎵 АУДИО ПАРАМЕТРЫ:")
    print(f"   Кодек: {analysis['audio_codec']}")
    print(f"   Описание кодека: {analysis['audio_codec_long']}")
    print(f"   Частота дискретизации: {analysis['sample_rate']} Гц")
    print(f"   Каналы: {analysis['channels']} ({analysis['channel_layout']})")
    
    if analysis['bit_rate']:
        bitrate_kbps = analysis['bit_rate'] / 1000
        print(f"   Битрейт: {bitrate_kbps:.0f} кбит/с")
    else:
        print(f"   Битрейт: Не определен")
    
    if analysis['bits_per_sample']:
        print(f"   Разрядность: {analysis['bits_per_sample']} бит")
    
    print(f"   Формат сэмпла: {analysis['sample_fmt']}")
    print("="*60)

def convert_to_wav_16khz(input_path: Union[str, Path], 
                        output_dir: Optional[Union[str, Path]] = None) -> str:
    """
    Конвертация аудиофайла в WAV 16kHz моно с максимальным качеством
    
    Args:
        input_path: Путь к входному файлу
        output_dir: Директория для сохранения (если None - сохраняется рядом с исходным файлом)
        
    Returns:
        Путь к сконвертированному WAV файлу
        
    Raises:
        FileNotFoundError: Если входной файл не найден
        ConversionError: Если конвертация не удалась
        FFmpegNotFoundError: Если ffmpeg недоступен
    """
    input_path = Path(input_path)
    
    # Проверка ffmpeg
    try:
        check_ffmpeg_availability()
    except FFmpegNotFoundError:
        raise
    
    # Проверка входного файла
    if not input_path.exists():
        error_msg = f"Входной файл не найден: {input_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    if input_path.is_dir():
        error_msg = f"Передана директория, ожидался файл: {input_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Анализ входного файла
    try:
        analysis = analyze_audio_file(input_path)
        print_audio_info(analysis)
    except (FileNotFoundError, FileAnalysisError):
        raise
    
    # Определение выходной директории
    if output_dir is None:
        # Если output_dir не задан - сохраняем в ту же папку, где исходный файл
        output_dir = input_path.parent
        logger.info(f"Выходная директория не задана, сохранение рядом с исходным файлом: {output_dir}")
    else:
        output_dir = Path(output_dir)
        logger.info(f"Выходная директория задана: {output_dir}")
        
    # Создание выходной директории если не существует
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Выходная директория готова: {output_dir}")
    except OSError as e:
        error_msg = f"Не удалось создать выходную директорию {output_dir}: {e}"
        logger.error(error_msg)
        raise ConversionError(error_msg)
    
    # Проверка доступности для записи
    if not os.access(output_dir, os.W_OK):
        error_msg = f"Нет прав для записи в директорию: {output_dir}"
        logger.error(error_msg)
        raise ConversionError(error_msg)
    
    # Формирование имени выходного файла
    base_name = input_path.stem
    output_filename = f"{base_name}{OUTPUT_FILE_SUFFIX}.wav"
    output_path = output_dir / output_filename
    
    logger.info(f"Конвертация: {input_path} -> {output_path}")
    
    # Построение команды ffmpeg с совместимыми параметрами
    base_command = [
        'ffmpeg',
        '-y',  # Перезаписать выходной файл
        '-i', str(input_path),  # Входной файл
        
        # Аудио параметры из конфигурации
        '-ar', str(TARGET_SAMPLE_RATE),
        '-ac', str(TARGET_CHANNELS),
        '-c:a', TARGET_CODEC,
        '-sample_fmt', TARGET_SAMPLE_FORMAT,
    ]
    
    # Пробуем продвинутый ресэмплер если включен
    if USE_ADVANCED_RESAMPLER:
        # Сначала пробуем soxr с базовыми параметрами (без beta)
        resampler_filter = f"aresample=resampler={RESAMPLER_ALGORITHM}:precision={RESAMPLER_PRECISION}:cutoff={RESAMPLER_CUTOFF}"
    else:
        # Используем базовый ресэмплер
        resampler_filter = f"aresample=resampler={FALLBACK_RESAMPLER}"
    
    command = base_command + [
        '-af', resampler_filter,
        '-loglevel', FFMPEG_LOG_LEVEL,
    ]
    
    if SHOW_FFMPEG_PROGRESS:
        command.append('-stats')
    else:
        command.append('-nostats')
    
    command.append(str(output_path))  # Выходной файл
    
    logger.debug(f"Команда ffmpeg: {' '.join(command)}")
    
    # Пробуем выполнить с продвинутыми параметрами
    conversion_success = False
    error_msg = ""
    
    try:
        print(f"\n🔄 Конвертация в WAV {TARGET_SAMPLE_RATE}Гц {'моно' if TARGET_CHANNELS == 1 else 'стерео'}...")
        print(f"   📂 Исходный файл: {input_path}")
        print(f"   📁 Сохранение в: {output_dir}")
        print(f"   📄 Выходной файл: {output_path.name}")
        
        if USE_ADVANCED_RESAMPLER:
            print(f"   🔧 Ресэмплер: {RESAMPLER_ALGORITHM} (высокое качество)")
        else:
            print(f"   🔧 Ресэмплер: {FALLBACK_RESAMPLER} (базовое качество)")
        
        # Выполнение конвертации
        result = subprocess.run(
            command,
            text=True,
            check=True,
            timeout=FFMPEG_TIMEOUT,
            capture_output=True
        )
        conversion_success = True
        
    except subprocess.CalledProcessError as e:
        error_msg = str(e)
        
        # Если ошибка связана с ресэмплером и мы используем продвинутые настройки
        if USE_ADVANCED_RESAMPLER and ("aresample" in e.stderr or "Option" in e.stderr):
            logger.warning(f"Продвинутый ресэмплер не работает, пробуем базовый: {e.stderr}")
            print(f"   ⚠️ Продвинутый ресэмплер недоступен, переходим к базовому...")
            
            # Пробуем с базовым ресэмплером
            fallback_command = base_command + [
                '-af', f"aresample=resampler={FALLBACK_RESAMPLER}",
                '-loglevel', FFMPEG_LOG_LEVEL,
            ]
            
            if SHOW_FFMPEG_PROGRESS:
                fallback_command.append('-stats')
            else:
                fallback_command.append('-nostats')
                
            fallback_command.append(str(output_path))
            
            logger.debug(f"Запасная команда ffmpeg: {' '.join(fallback_command)}")
            
            try:
                print(f"   🔧 Ресэмплер: {FALLBACK_RESAMPLER} (совместимый режим)")
                result = subprocess.run(
                    fallback_command,
                    text=True,
                    check=True,
                    timeout=FFMPEG_TIMEOUT,
                    capture_output=True
                )
                conversion_success = True
                
            except subprocess.CalledProcessError as e2:
                error_msg = f"Ошибка и с базовым ресэмплером: {e2.stderr}"
        
        if not conversion_success:
            logger.error(f"Ошибка ffmpeg при конвертации: {error_msg}")
            raise ConversionError(f"Ошибка ffmpeg при конвертации: {error_msg}")
            
    except subprocess.TimeoutExpired:
        timeout_min = FFMPEG_TIMEOUT // 60
        error_msg = f"Таймаут конвертации (более {timeout_min} минут)"
        logger.error(error_msg)
        raise ConversionError(error_msg)
        
    except OSError as e:
        error_msg = f"Системная ошибка при конвертации: {e}"
        logger.error(error_msg)
        raise ConversionError(error_msg)
        
    # Проверка результата (если включена в настройках)
    if CHECK_OUTPUT_FILE and conversion_success:
        if not output_path.exists():
            error_msg = "Выходной файл не был создан"
            logger.error(error_msg)
            raise ConversionError(error_msg)
        
        file_size = output_path.stat().st_size
        if file_size < MIN_OUTPUT_FILE_SIZE:
            error_msg = f"Файл слишком мал ({file_size} байт, минимум {MIN_OUTPUT_FILE_SIZE})"
            logger.error(error_msg)
            raise ConversionError(error_msg)
    
    # Анализ результата
    if conversion_success:
        try:
            result_analysis = analyze_audio_file(output_path)
            print(f"\n✅ КОНВЕРТАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
            print(f"📁 Результат: {output_path}")
            print(f"📏 Размер: {result_analysis['file_size_formatted']}")
            print(f"🎵 Параметры: {result_analysis['sample_rate']}Гц, {result_analysis['channels']} канал(ов)")
            
        except FileAnalysisError as e:
            logger.warning(f"Не удалось проанализировать результат: {e}")
        
        logger.info(f"Конвертация завершена успешно: {output_path}")
        return str(output_path)
    else:
        raise ConversionError("Конвертация не удалась")

def process_audio_directory(input_dir: Union[str, Path], 
                           output_dir: Optional[Union[str, Path]] = None,
                           audio_extensions: tuple = SUPPORTED_AUDIO_EXTENSIONS) -> list:
    """
    Обработка всех аудиофайлов в директории
    
    Args:
        input_dir: Входная директория
        output_dir: Выходная директория (по умолчанию та же)
        audio_extensions: Кортеж расширений аудиофайлов (берется из настроек)
        
    Returns:
        Список путей к сконвертированным файлам
        
    Raises:
        FileNotFoundError: Если входная директория не найдена
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        error_msg = f"Входная директория не найдена: {input_dir}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    if not input_dir.is_dir():
        error_msg = f"Путь не является директорией: {input_dir}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Поиск аудиофайлов
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_dir.glob(f"*{ext}"))
        audio_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not audio_files:
        logger.warning(f"Аудиофайлы не найдены в директории: {input_dir}")
        print(f"⚠️ Аудиофайлы не найдены в директории: {input_dir}")
        print(f"Поддерживаемые форматы: {', '.join(audio_extensions)}")
        return []
    
    logger.info(f"Найдено {len(audio_files)} аудиофайлов для обработки")
    
    # Обработка файлов
    results = []
    errors = []
    
    for i, audio_file in enumerate(audio_files, 1):
        try:
            print(f"\n{'='*60}")
            print(f"ФАЙЛ {i}/{len(audio_files)}: {audio_file.name}")
            print(f"{'='*60}")
            
            output_file = convert_to_wav_16khz(audio_file, output_dir)
            results.append(output_file)
            
        except Exception as e:
            error_msg = f"Ошибка при обработке {audio_file}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            results.append(None)
    
    # Итоговая статистика
    successful = len([r for r in results if r is not None])
    print(f"\n{'='*60}")
    print(f"ОБРАБОТКА ЗАВЕРШЕНА")
    print(f"{'='*60}")
    print(f"✅ Успешно: {successful}/{len(audio_files)}")
    print(f"❌ Ошибок: {len(errors)}")
    
    if errors:
        print(f"\nОШИБКИ:")
        for error in errors:
            print(f"   {error}")
    
    return results

def print_configuration():
    """Вывод текущей конфигурации"""
    print("\n" + "="*60)
    print(" ТЕКУЩАЯ КОНФИГУРАЦИЯ")
    print("="*60)
    print(f"🎵 Параметры аудио:")
    print(f"   Частота дискретизации: {TARGET_SAMPLE_RATE} Гц")
    print(f"   Каналы: {TARGET_CHANNELS} ({'моно' if TARGET_CHANNELS == 1 else 'стерео'})")
    print(f"   Кодек: {TARGET_CODEC}")
    print(f"   Формат сэмпла: {TARGET_SAMPLE_FORMAT}")
    
    print(f"\n🔧 Параметры ресэмплинга:")
    print(f"   Продвинутый режим: {'включен' if USE_ADVANCED_RESAMPLER else 'отключен'}")
    print(f"   Основной алгоритм: {RESAMPLER_ALGORITHM}")
    print(f"   Запасной алгоритм: {FALLBACK_RESAMPLER}")
    if USE_ADVANCED_RESAMPLER:
        print(f"   Точность: {RESAMPLER_PRECISION}")
        print(f"   Частота среза: {RESAMPLER_CUTOFF}")
    else:
        print(f"   Режим: базовая совместимость")
    
    print(f"\n📁 Файлы:")
    print(f"   Суффикс выходных файлов: {OUTPUT_FILE_SUFFIX}")
    print(f"   Поддерживаемые форматы: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}")
    
    print(f"\n⏱️ Таймауты:")
    print(f"   ffprobe: {FFPROBE_TIMEOUT} сек")
    print(f"   ffmpeg: {FFMPEG_TIMEOUT} сек ({FFMPEG_TIMEOUT//60} мин)")
    
    print(f"\n🛠️ ffmpeg:")
    print(f"   Уровень логов: {FFMPEG_LOG_LEVEL}")
    print(f"   Показывать прогресс: {SHOW_FFMPEG_PROGRESS}")
    print("="*60)

def main():
    """Основная функция для запуска из командной строки"""
    parser = argparse.ArgumentParser(
        description="Конвертер аудиофайлов в WAV 16kHz с анализом через ffprobe",
        epilog="""
Примеры использования:
  %(prog)s "input file.m4a"                       # Конвертация файла с пробелами в имени
  %(prog)s input.m4a                              # Конвертация простого файла
  %(prog)s "input.m4a" -o "./wav files"           # С указанием папки вывода (с пробелами)
  %(prog)s -d "./audio files"                     # Обработка всей директории
  %(prog)s -d ./audio_files -o ./converted        # Обработка директории с выводом в другую папку
  %(prog)s --config                               # Показать текущую конфигурацию
  
Важно: Пути с пробелами нужно брать в кавычки!
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input', nargs='?', 
                       help='Путь к аудиофайлу или директории (используйте кавычки для путей с пробелами)')
    parser.add_argument('-o', '--output', type=str, 
                       help='Директория для сохранения результатов (используйте кавычки для путей с пробелами)')
    parser.add_argument('-d', '--directory', action='store_true', 
                       help='Обработать все аудиофайлы в указанной директории')
    parser.add_argument('--config', action='store_true', 
                       help='Показать текущую конфигурацию и выйти')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                       help='Уровень детализации логов (-v, -vv, -vvv)')
    parser.add_argument('--log-file', type=str, help='Файл для сохранения логов')
    
    args = parser.parse_args()
    
    # Настройка логирования в зависимости от уровня детализации
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = log_levels[min(args.verbose, len(log_levels) - 1)]
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if args.log_file:
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(args.log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=log_level, format=log_format)
    
    # Показать конфигурацию и выйти
    if args.config:
        print_configuration()
        return 0
    
    # Проверка аргументов
    if not args.input:
        parser.print_help()
        print(f"\n❌ Ошибка: Не указан входной файл или директория")
        print(f"💡 Совет: Используйте кавычки для путей с пробелами:")
        print(f'   python audio_converter.py "путь с пробелами/файл.mp3"')
        return 1
    
    try:
        input_path = Path(args.input)
        
        if not input_path.exists():
            print(f"❌ Путь не найден: {input_path}")
            print(f"💡 Совет: Проверьте правильность пути и используйте кавычки для путей с пробелами")
            print(f'   Пример: python audio_converter.py "D:\\Music\\My Song.mp3"')
            return 1
        
        # Определение режима работы
        if args.directory or input_path.is_dir():
            # Режим обработки директории
            print(f"📁 Режим: Обработка директории")
            print(f"📂 Входная директория: {input_path}")
            if args.output:
                print(f"📂 Выходная директория: {args.output}")
            
            results = process_audio_directory(input_path, args.output)
            successful = len([r for r in results if r is not None])
            
            if successful > 0:
                print(f"\n🎉 Успешно обработано {successful} файлов")
                return 0
            else:
                print(f"\n❌ Не удалось обработать ни одного файла")
                return 1
                
        else:
            # Режим обработки одного файла
            print(f"📄 Режим: Обработка одного файла")
            print(f"📂 Входной файл: {input_path}")
            if args.output:
                print(f"📂 Выходная директория: {args.output}")
            
            result = convert_to_wav_16khz(input_path, args.output)
            print(f"\n🎉 Файл успешно конвертирован: {result}")
            return 0
            
    except KeyboardInterrupt:
        print(f"\n❌ Операция прервана пользователем")
        return 1
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        print(f"\n❌ Ошибка: {e}")
        
        # Специальная обработка для файлов с пробелами
        if "не найден" in str(e).lower() or "not found" in str(e).lower():
            print(f"💡 Возможно проблема с пробелами в пути. Попробуйте:")
            print(f'   python audio_converter.py "{args.input}"')
            
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
        return 1

# Пример использования
def example_usage():
    """Примеры использования модуля в коде"""
    print("🎵 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ МОДУЛЯ")
    print("="*50)
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n1. Показ текущей конфигурации:")
    print_configuration()
    
    print("\n2. Примеры кода:")
    print("""
# Простая конвертация одного файла
from audio_converter import convert_to_wav_16khz
output_file = convert_to_wav_16khz("input.m4a")

# Конвертация с указанием выходной папки
output_file = convert_to_wav_16khz("input.m4a", output_dir="./wav_files")

# Обработка всей директории
from audio_converter import process_audio_directory
results = process_audio_directory("./audio_files", output_dir="./converted")

# Настройка детального логирования
import logging
logging.getLogger('audio_converter').setLevel(logging.DEBUG)
    """)
    
    # Если есть тестовые файлы, можно их обработать
    test_files = ["example.m4a", "test.mp3", "audio.wav"]
    existing_files = [f for f in test_files if Path(f).exists()]
    
    if existing_files:
        print(f"\n3. Обработка найденных тестовых файлов:")
        for test_file in existing_files:
            try:
                print(f"\nОбработка: {test_file}")
                output_file = convert_to_wav_16khz(test_file, output_dir="./example_output")
                print(f"Результат: {output_file}")
                
            except Exception as e:
                print(f"Ошибка при обработке {test_file}: {e}")
    else:
        print(f"\n3. Тестовые файлы не найдены")
        print(f"Создайте файлы {test_files} для тестирования")

if __name__ == "__main__":
    # При запуске скрипта напрямую - используем main() для командной строки
    exit_code = main()
    sys.exit(exit_code)

"""
===============================================================================
ПОДРОБНАЯ ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ
===============================================================================

1. ЗАПУСК ИЗ КОМАНДНОЙ СТРОКИ:

   Базовые команды:
   python audio_converter.py input.m4a                    # Конвертация файла
   python audio_converter.py input.m4a -o ./wav_files     # Указать папку вывода
   python audio_converter.py -d ./audio_folder            # Обработать папку
   python audio_converter.py --config                     # Показать настройки
   python audio_converter.py --help                       # Показать справку

   Дополнительные опции:
   python audio_converter.py input.m4a -v                 # Больше логов
   python audio_converter.py input.m4a -vv                # Ещё больше логов 
   python audio_converter.py input.m4a --log-file log.txt # Сохранить логи

2. ИСПОЛЬЗОВАНИЕ КАК МОДУЛЬ:

   # Простая конвертация
   from audio_converter import convert_to_wav_16khz
   wav_file = convert_to_wav_16khz("audio.m4a")
   
   # С указанием папки
   wav_file = convert_to_wav_16khz("audio.m4a", output_dir="./wav_files")
   
   # Обработка папки
   from audio_converter import process_audio_directory
   results = process_audio_directory("./audio", output_dir="./converted")

3. НАСТРОЙКА ЛОГИРОВАНИЯ:

   import logging
   
   # Базовое логирование
   logging.basicConfig(level=logging.INFO)
   
   # Детальное логирование только для аудио конвертера
   logging.getLogger('audio_converter').setLevel(logging.DEBUG)
   
   # Сохранение логов в файл
   logging.basicConfig(
       level=logging.DEBUG,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler('converter.log', encoding='utf-8'),
           logging.StreamHandler()
       ]
   )

4. ИЗМЕНЕНИЕ НАСТРОЕК:

   Отредактируйте константы в начале файла:
   
   TARGET_SAMPLE_RATE = 44100        # Изменить частоту на 44.1 кГц
   TARGET_CHANNELS = 2               # Изменить на стерео
   OUTPUT_FILE_SUFFIX = '_converted'  # Изменить суффикс файлов
   FFMPEG_TIMEOUT = 1200             # Увеличить таймаут до 20 минут
   USE_ADVANCED_RESAMPLER = False    # Отключить продвинутый ресэмплер для совместимости

5. ОБРАБОТКА ОШИБОК:

   try:
       result = convert_to_wav_16khz("audio.m4a")
       print(f"Успешно: {result}")
   except FFmpegNotFoundError:
       print("ffmpeg не найден, установите его")
   except FileNotFoundError as e:
       print(f"Файл не найден: {e}")
   except ConversionError as e:
       print(f"Ошибка конвертации: {e}")
   except Exception as e:
       print(f"Неожиданная ошибка: {e}")

6. ПОДДЕРЖИВАЕМЫЕ ФОРМАТЫ:

   Входные: mp3, m4a, wav, flac, aac, ogg, wma, mp4, avi, mov, webm, 3gp, opus
   Выходной: wav (PCM 16-bit, по умолчанию 16kHz моно)

7. ТРЕБОВАНИЯ:

   - Python 3.7+
   - ffmpeg и ffprobe в PATH
   - Свободное место на диске для выходных файлов

===============================================================================
"""