# -*- coding: utf-8 -*-
"""
Загружает данные из for_ansys_X.npy, запускает CPWING.py, geom_ansys.py,
Interpolador.py, nagryzka_ansys.py, ansys_raschet.py и thickness_optimizer.py
"""
import numpy as np
import time
import subprocess
import sys
import os

# Получаем номер файла из аргументов командной строки (по умолчанию 0)
file_number = 1
if len(sys.argv) > 1:
    try:
        file_number = int(sys.argv[1])
    except ValueError:
        print("⚠ Некорректный аргумент, используется номер 0")

def run_script(script_name):
    """Запускает указанный Python-скрипт и возвращает время выполнения."""
    print("\n" + "="*50)
    print(f"ЗАПУСК {script_name}")
    print("="*50)
    inicio = time.perf_counter()
    try:
        # Для nagryzka_ansys.py не передаём аргументы
        if script_name == "nagryzka_ansys.py":
            args = []
        else:
            args = [str(file_number)]

        result = subprocess.run([sys.executable, script_name] + args,
                              capture_output=True, text=True, encoding='utf-8')
        fin = time.perf_counter()
        tiempo = fin - inicio
        print(result.stdout)
        if result.stderr:
            print(f"Ошибки {script_name}:")
            print(result.stderr)
        return tiempo
    except Exception as e:
        print(f"❌ Ошибка при запуске {script_name}: {e}")
        return 0

# ====== ОСНОВНАЯ ПРОГРАММА ======
inicio_total = time.perf_counter()

print("="*50)
print("ЗАГРУЗКА КОНФИГУРАЦИЙ ИЗ ФАЙЛА")
print("="*50)

data_file = f"for_ansys_{file_number}.npy"
if not os.path.exists(data_file):
    print(f"❌ Ошибка: Файл {data_file} не найден.")
    exit(1)

# Загружаем данные (просто для информации)
config_data = np.load(data_file)
print(f"✅ Загружено {config_data.shape[0]} конфигураций из {data_file}")
print(f"   Размерность: {config_data.shape} (строк, столбцов)")

tiempo_generacion = time.perf_counter() - inicio_total
print(f"\nВремя загрузки: {tiempo_generacion:.4f} секунд")

# Запуск 
#tiempo_cpwing = run_script("run_aero_batch.py")

# Запуск остальных скриптов
#tiempo_geom = run_script("geom_ansys.py")
#tiempo_interp = run_script("Interpolador.py")
#tiempo_nagryzka = run_script("nagryzka_ansys.py")
#tiempo_ansys = run_script("ansys_raschet.py")

# Запуск thickness_optimizer.py
tiempo_spar = run_script("spar_optimizer.py")
#tiempo_optimizer = run_script("thickness_optimizer.py")

# Общее время
fin_total = time.perf_counter()
tiempo_total = fin_total-inicio_total

print("\n" + "="*50)
print("ИТОГ ВРЕМЕНИ ВЫПОЛНЕНИЯ")
print("="*50)
print(f"Время загрузки (parametros.py):               {tiempo_generacion:.4f} секунд")
#print(f"Время geom_ansys.py:                           {tiempo_geom:.4f} секунд")
#print(f"Время Interpolador.py:                         {tiempo_interp:.4f} секунд")
#print(f"Время nagryzka_ansys.py:                       {tiempo_nagryzka:.4f} секунд")
#print(f"Время ansys_raschet.py:                        {tiempo_ansys:.4f} секунд")
#print(f"Время spar_optimizer.py:                        {tiempo_spar:.4f} секунд")
#print(f"Время оптимизации (thickness_optimizer.py):    {tiempo_optimizer:.4f} секунд")
print(f"Общее время выполнения:                         {tiempo_total:.4f} секунд")
print("="*50)