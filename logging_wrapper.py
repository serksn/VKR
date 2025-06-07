# logging_wrapper.py

import time
import functools

def log_calls(func):
    """
    Декоратор для логирования начала и конца вызова функции,
    а также времени её выполнения.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        name = func.__name__
        print(f">>> Запуск {name} с args={args}, kwargs={kwargs}")
        t0 = time.time()
        result = func(*args, **kwargs)
        dt = time.time() - t0
        print(f"<<< Завершено {name} за {dt:.2f}s, возвращено {result!r}")
        return result
    return wrapper
