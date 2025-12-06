# env_check_v1.py
# Полная проверка окружения для проектов на torch/transformers
# (c) 2025, MIT License

import importlib
import sys
import platform
import subprocess
import shutil
from datetime import datetime

try:
    import torch
except ImportError:
    torch = None

COLOR_OK   = "\033[92m"
COLOR_WARN = "\033[93m"
COLOR_ERR  = "\033[91m"
RESET      = "\033[0m"

def head(msg):      # крупный раздел
    print(f"\n{COLOR_OK}{'='*4} {msg} {'='*4}{RESET}")

def line(k,v):
    print(f"{k:<32}: {v}")

def check_pkg(name):
    spec = importlib.util.find_spec(name)
    if spec is None:
        line(name, f"{COLOR_ERR}НЕ УСТАНОВЛЕН{RESET}")
        return None
    mod = importlib.import_module(name)
    line(name, f"{COLOR_OK}{mod.__version__}{RESET}")
    return mod

def check_ffmpeg():
    path = shutil.which("ffmpeg")
    if path:
        try:
            out = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
            line("ffmpeg", f"{COLOR_OK}{out}{RESET}")
        except subprocess.SubprocessError:
            line("ffmpeg", f"{COLOR_WARN}обнаружен, но не отвечает{RESET}")
    else:
        line("ffmpeg", f"{COLOR_WARN}не найден в PATH{RESET}")

def gpu_info():
    n = torch.cuda.device_count()
    line("CUDA устройств", n)
    for i in range(n):
        prop = torch.cuda.get_device_properties(i)
        mem_total = prop.total_memory / 1024**3
        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
        mem_alloc    = torch.cuda.memory_allocated(i) / 1024**3
        print(f" └─ GPU[{i}] {prop.name} (CC {prop.major}.{prop.minor})")
        print(f"    Драйвер/Runtime      : {torch.version.cuda}/{torch.version.cuda_runtime}")
        print(f"    Память total/res/alloc: {mem_total:.1f}/{mem_reserved:.1f}/{mem_alloc:.1f} ГБ")

def main():
    head("Общее")
    line("Дата / время", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    line("Python", platform.python_version())
    line("Платформа", f"{platform.system()} {platform.release()} ({platform.machine()})")

    head("Библиотеки")
    check_pkg("torch")
    check_pkg("torchaudio")
    check_pkg("transformers")
    check_pkg("soundfile")

    head("CUDA / GPU")
    if torch is None:
        print(f"{COLOR_ERR}torch не установлен ― пропускаем CUDA-проверку{RESET}")
    elif not torch.cuda.is_available():
        print(f"{COLOR_WARN}CUDA недоступна ― будет использоваться CPU{RESET}")
    else:
        gpu_info()
        try:
            line("cuDNN", torch.backends.cudnn.version())
        except AttributeError:
            line("cuDNN", f"{COLOR_WARN}не найден{RESET}")

    head("Сторонние утилиты")
    check_ffmpeg()

    head("Готово")
    print("Если всё зелёным — окружение готово к работе!")

if __name__ == "__main__":
    main()
