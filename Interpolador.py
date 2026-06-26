# -*- coding: utf-8 -*-
"""
Интерполятор давления по узлам сетки с нормализацией координат
- Загружает силы из базовых файлов (0{iteration}_{case}.txt)
- Переводит силу в давление (деление на площадь ячейки)
- Интерполирует давление на целевую сетку (файлы {iteration}-{iteration}.txt) с нормализацией
- Сравнивает интеграл давления до и после интерполяции (суммарную силу)
- Сохраняет результат с координатами X, Y, Z и интерполированным давлением
- Поддерживает дробные и отрицательные случаи
- Исключает из интерполяции точки с Y = 0 и Y = Y_max
"""
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import os
import re
import glob
import matplotlib.pyplot as plt
import time
import warnings
import sys

# Игнорируем предупреждения openpyxl
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# Получаем номер файла из аргументов командной строки (оставлен для совместимости)
file_number = 1
target_iteration = None
if len(sys.argv) > 1:
    try:
        file_number = int(sys.argv[1])
    except ValueError:
        print("⚠ Некорректный аргумент, используется номер файла 1")
if len(sys.argv) > 2:
    try:
        target_iteration = int(sys.argv[2])
    except ValueError:
        pass

def leer_archivo(ruta, es_fuerza=False):
    """
    Читает TXT файлы без заголовков или с комментарием в первой строке.
    Для базовых файлов (с силой) третий столбец — сила.
    Для целевых файлов третий столбец — координата Z.
    """
    df = pd.read_csv(ruta, sep='\t', decimal=',', header=None, dtype=str, comment='#')
    
    # Преобразуем в числа
    for col in range(min(3, df.shape[1])):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=[0, 1, 2])
    
    if es_fuerza:
        df = df.rename(columns={0: 'X', 1: 'Y', 2: 'force'})
    else:
        df = df.rename(columns={0: 'X', 1: 'Y', 2: 'Z'})
    
    return df

def encontrar_archivos_pareados(target_iteration=None):
    """Находит все пары файлов для интерполяции."""
    todos_archivos = glob.glob("*.txt")
    
    patron_base = re.compile(r'^(\d+)_(-?\d+(?:\.\d+)?)\.txt$')
    patron_target = re.compile(r'^(\d+)-(\d+)\.txt$')
    
    archivos_base = {}
    archivos_target = {}
    
    for archivo in todos_archivos:
        match_base = patron_base.match(archivo)
        if match_base:
            iteration = int(match_base.group(1))
            case_str = match_base.group(2)
            case = float(case_str)
            if target_iteration is None or iteration == target_iteration:
                key = f"{iteration}_{case}"
                archivos_base[key] = (archivo, iteration, case)
        
        match_target = patron_target.match(archivo)
        if match_target:
            iteration = int(match_target.group(1))
            iteration2 = int(match_target.group(2))
            if iteration == iteration2:
                if target_iteration is None or iteration == target_iteration:
                    archivos_target[iteration] = archivo
    
    pares = []
    for key in archivos_base:
        archivo_base, iteration, case = archivos_base[key]
        if iteration in archivos_target:
            archivo_target = archivos_target[iteration]
            pares.append((archivo_base, archivo_target, iteration, case))
    
    pares.sort(key=lambda x: (x[2], x[3]))
    return pares

def guardar_resultados_excel(resultados_interpolacion, filename="resultados_verificacion.xlsx"):
    """Сохраняет результаты интерполяции в Excel."""
    if not resultados_interpolacion:
        print("  📊 Нет данных для сохранения в Excel")
        return
    
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df_interpolacion = pd.DataFrame(resultados_interpolacion)
            columnas_rusas = {
                'iteracion': 'Итерация',
                'caso': 'Случай',
                'archivo_base': 'Базовый файл',
                'archivo_objetivo': 'Целевой файл',
                'suma_fuerza_original': 'Сумма сил исходная',
                'suma_fuerza_integrada': 'Сумма сил после интерполяции (интеграл давления)',
                'error_relativo_porcentaje': 'Относительная ошибка, %',
                'metodo_interpolacion': 'Метод интерполяции',
                'tiempo_procesamiento_segundos': 'Время обработки, с',
                'estado': 'Статус'
            }
            df_ruso = df_interpolacion.rename(columns=columnas_rusas)
            df_ruso.to_excel(writer, sheet_name='Интерполяция', index=False)
            print(f"  💾 Сохранено {len(df_ruso)} записей интерполяции")
        print(f"  📊 Excel файл с результатами сохранен: {filename}")
    except Exception as e:
        print(f"  ❌ Ошибка при сохранении Excel файла: {e}")

def visualizar_coordenadas(base_points, target_points, iteration, case, output_dir="visualizaciones"):
    """Визуализация координат базовой и целевой сеток."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if case.is_integer():
        case_str = str(int(case))
    else:
        case_str = str(case)
    
    plt.figure(figsize=(12, 10))
    plt.scatter(base_points[:, 0], base_points[:, 1], 
                c='blue', alpha=0.7, s=30, label='Исходные точки (базовые)', marker='o')
    plt.scatter(target_points[:, 0], target_points[:, 1], 
                c='red', alpha=0.7, s=20, label='Целевые точки', marker='x')
    plt.title(f'Визуализация координат X-Y (Итерация {iteration}, Случай {case})')
    plt.xlabel('Координата X')
    plt.ylabel('Координата Y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    output_filename = os.path.join(output_dir, f"coordenadas_{iteration}_{case_str}.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Визуализация сохранена: {output_filename}")

def compute_cell_areas(points):
    """
    Вычисляет площадь ячейки, ассоциированную с каждым узлом.
    Для структурированной сетки (группировка по Y).
    Возвращает массив площадей той же длины, что и points.
    """
    # Группируем точки по Y (с учётом погрешности)
    eps = 1e-9
    unique_y = np.sort(np.unique(points[:, 1]))
    y_to_idx = {y: i for i, y in enumerate(unique_y)}
    groups = [[] for _ in range(len(unique_y))]
    for i, (x, y) in enumerate(points):
        idx = np.argmin(np.abs(unique_y - y))
        groups[idx].append((x, i))
    
    # Для каждого Y-сечения сортируем по X
    for g in groups:
        g.sort(key=lambda t: t[0])
    
    # Вычисляем шаги по X для каждого узла
    areas = np.zeros(len(points))
    dy = np.zeros(len(unique_y))
    # Шаги по Y (полусумма расстояний до соседних сечений)
    for j in range(len(unique_y)):
        if j == 0:
            dy[j] = (unique_y[1] - unique_y[0])
        elif j == len(unique_y)-1:
            dy[j] = (unique_y[-1] - unique_y[-2])
        else:
            dy[j] = 0.5 * (unique_y[j+1] - unique_y[j-1])
    
    # Для каждого сечения вычисляем шаги по X и площади
    for j, g in enumerate(groups):
        xs = np.array([t[0] for t in g])
        indices = [t[1] for t in g]
        n = len(xs)
        dx = np.zeros(n)
        for i in range(n):
            if n == 1:
                dx[i] = 1.0  # условная единица, но в реальности сетка должна быть хотя бы 2x2
            elif i == 0:
                dx[i] = xs[1] - xs[0]
            elif i == n-1:
                dx[i] = xs[-1] - xs[-2]
            else:
                dx[i] = 0.5 * (xs[i+1] - xs[i-1])
        # Площадь = dx * dy
        for i, idx in enumerate(indices):
            areas[idx] = dx[i] * dy[j]
    
    # Если остались нули (например, из-за одного узла в сечении), заменяем на среднее
    if np.any(areas == 0):
        mean_area = np.mean(areas[areas > 0]) if np.any(areas > 0) else 1.0
        areas[areas == 0] = mean_area
    
    return areas

def normalize_points(base_points, target_points):
    """
    Нормализует координаты по методу:
        Y_nor = (Y - Y_min_global) / (Y_max_global - Y_min_global)
        X_nor = (X - X_local_min(Y)) / (X_local_max(Y) - X_local_min(Y))
    Использует базовые точки для определения локальных min/max X по Y.
    Возвращает нормализованные массивы для base и target.
    """
    # Глобальные min/max Y
    y_min_global = np.min(base_points[:, 1])
    y_max_global = np.max(base_points[:, 1])
    span = y_max_global - y_min_global
    if span == 0:
        span = 1.0
    
    # Группируем базовые точки по Y
    eps = 1e-9
    unique_y_base = np.sort(np.unique(base_points[:, 1]))
    y_groups = {}
    for y in unique_y_base:
        mask = np.abs(base_points[:, 1] - y) < eps
        xs = base_points[mask, 0]
        y_groups[y] = (np.min(xs), np.max(xs))
    
    # Нормализация базовых точек
    base_norm = np.zeros_like(base_points)
    for i, (x, y) in enumerate(base_points):
        # Находим ближайший Y в базовых данных
        y_key = unique_y_base[np.argmin(np.abs(unique_y_base - y))]
        x_min, x_max = y_groups[y_key]
        chord = x_max - x_min
        if chord == 0:
            chord = 1.0
        base_norm[i, 0] = (x - x_min) / chord
        base_norm[i, 1] = (y - y_min_global) / span
    
    # Нормализация целевых точек: для каждого Y используем локальные min/max из базовых,
    # но Y может не совпадать с базовыми – интерполируем min/max по Y
    # Сначала создаём массивы Y_base, minX_base, maxX_base
    y_base_vals = np.array(list(y_groups.keys()))
    minX_base = np.array([v[0] for v in y_groups.values()])
    maxX_base = np.array([v[1] for v in y_groups.values()])
    
    target_norm = np.zeros_like(target_points)
    for i, (x, y) in enumerate(target_points):
        # Интерполируем локальные min/max по Y (линейно)
        # Если y выходит за пределы, экстраполируем с помощью ближайшего
        if y <= y_base_vals[0]:
            x_min = minX_base[0]
            x_max = maxX_base[0]
        elif y >= y_base_vals[-1]:
            x_min = minX_base[-1]
            x_max = maxX_base[-1]
        else:
            idx = np.searchsorted(y_base_vals, y)
            y0, y1 = y_base_vals[idx-1], y_base_vals[idx]
            t = (y - y0) / (y1 - y0)
            x_min = minX_base[idx-1] * (1 - t) + minX_base[idx] * t
            x_max = maxX_base[idx-1] * (1 - t) + maxX_base[idx] * t
        chord = x_max - x_min
        if chord == 0:
            chord = 1.0
        target_norm[i, 0] = (x - x_min) / chord
        target_norm[i, 1] = (y - y_min_global) / span
    
    return base_norm, target_norm

def procesar_interpolacion(target_iteration=None):
    """Основная функция интерполяции давления с нормализацией."""
    print("\n" + "=" * 60)
    print("ИНТЕРПОЛЯЦИЯ ДАВЛЕНИЯ ПО УЗЛАМ СЕТКИ (С НОРМАЛИЗАЦИЕЙ)")
    if target_iteration is not None:
        print(f"Обработка только для итерации: {target_iteration}")
    print("=" * 60)

    resultados_interpolacion = []
    pares_archivos = encontrar_archivos_pareados(target_iteration)

    if not pares_archivos:
        print("Не найдены пары файлов для интерполяции")
        return []

    print(f"Найдено {len(pares_archivos)} пар файлов:")

    for base_file, target_file, iteration, case in pares_archivos:
        print(f"\nОбрабатываем пару: {base_file} -> {target_file} (итерация {iteration}, случай {case})")
        start_time_total = time.time()

        try:
            base_df = leer_archivo(base_file, es_fuerza=True)
            if 'force' not in base_df.columns:
                print(f"  Ошибка: в базовом файле нет столбца force")
                continue
        except Exception as e:
            print(f"  Ошибка при чтении базового файла {base_file}: {e}")
            continue

        try:
            target_df = leer_archivo(target_file, es_fuerza=False)
            if 'Z' not in target_df.columns:
                print(f"  Ошибка: в целевом файле нет столбца Z")
                continue
        except Exception as e:
            print(f"  Ошибка при чтении целевого файла {target_file}: {e}")
            continue

        if base_df.empty or target_df.empty:
            print(f"  Ошибка: файлы пустые после фильтрации")
            continue

        base_points = base_df[['X', 'Y']].to_numpy()
        base_force = base_df['force'].to_numpy()
        
        # Сохраняем исходные целевые точки для визуализации
        target_points_original = target_df[['X', 'Y']].to_numpy()
        target_Z_original = target_df['Z'].values
        
        target_points = target_points_original
        target_Z = target_Z_original
        target_df_filtered = target_df.copy()

        # Визуализация координат с исходными точками (включая исключенные)
        visualizar_coordenadas(base_points, target_points_original, iteration, case)

        # --- Вычисление площадей ячеек для исходных точек ---
        base_areas = compute_cell_areas(base_points)
        # Переводим силу в давление
        base_pressure = base_force / base_areas

        # Исходная сумма сил (для контроля)
        original_force_sum = np.sum(base_force)

        print(f"  Сумма сил исходных данных: {original_force_sum:.8f}")
        print(f"  Среднее давление исходных данных: {np.mean(base_pressure):.8f}")

        # --- Нормализация координат ---
        base_norm, target_norm = normalize_points(base_points, target_points)

        # --- Интерполяция давления на нормализованных координатах ---
        # Используем griddata с методом linear
        try:
            interpolated_pressure = griddata(base_norm, base_pressure, target_norm, method='linear', fill_value=np.nan)
            # Заполняем NaN ближайшими
            nan_mask = np.isnan(interpolated_pressure)
            if np.any(nan_mask):
                print(f"    Найдено {np.sum(nan_mask)} NaN после linear, заполняем nearest")
                interpolated_pressure[nan_mask] = griddata(base_norm, base_pressure, target_norm[nan_mask], method='nearest')
            method_used = 'linear+nearest'
        except Exception as e:
            print(f"    Ошибка при linear, используем nearest: {e}")
            interpolated_pressure = griddata(base_norm, base_pressure, target_norm, method='nearest')
            method_used = 'nearest'

        # --- Вычисление площадей ячеек для целевых точек ---
        target_areas = compute_cell_areas(target_points)

        # --- Интеграл давления по целевой сетке (суммарная сила) ---
        integrated_force = np.sum(interpolated_pressure * target_areas)

        # Относительная ошибка
        if abs(original_force_sum) > 1e-10:
            rel_error = abs(integrated_force - original_force_sum) / abs(original_force_sum) * 100
        else:
            rel_error = 0.0

        total_time = time.time() - start_time_total
        print(f"  Сумма сил после интерполяции (интеграл давления): {integrated_force:.8f}")
        print(f"  Относительная ошибка: {rel_error:.4f}%")
        print(f"  Метод интерполяции: {method_used}")
        print(f"  Общее время обработки: {total_time:.2f}с")

        estado = '✅ EXITO' if rel_error <= 3.0 else '⚠ ALERTA'

        resultado_interpolacion = {
            'iteracion': iteration,
            'caso': case,
            'archivo_base': base_file,
            'archivo_objetivo': target_file,
            'suma_fuerza_original': original_force_sum,
            'suma_fuerza_integrada': integrated_force,
            'error_relativo_porcentaje': rel_error,
            'metodo_interpolacion': method_used,
            'tiempo_procesamiento_segundos': total_time,
            'estado': estado
        }
        resultados_interpolacion.append(resultado_interpolacion)

        # --- Сохранение результата в файл ---
        if case.is_integer():
            case_str = str(int(case))
        else:
            case_str = str(case)

        # Собираем результат: X, Y, Z, давление
        result_df = pd.DataFrame({
            'X': target_df_filtered['X'],
            'Y': target_df_filtered['Y'],
            'Z': target_Z,
            'pressure': interpolated_pressure
        })

        output_file = f"resultados_interpolacion_{iteration}_{case_str}.txt"
        if os.path.exists(output_file):
            os.remove(output_file)
        # Сохраняем без заголовков, разделитель табуляция, десятичная запятая
        result_df.to_csv(output_file, index=False, sep='\t', decimal=',', header=False, float_format='%.8f')

        info_file = f"info_interpolacion_{iteration}_{case_str}.txt"
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(f"Итерация: {iteration}\n")
            f.write(f"Случай: {case}\n")
            f.write(f"Базовый файл: {base_file}\n")
            f.write(f"Целевой файл: {target_file}\n")
            f.write(f"Метод интерполяции: {method_used}\n")
            f.write(f"Сумма сил исходных данных: {original_force_sum:.8f}\n")
            f.write(f"Сумма сил после интерполяции (интеграл давления): {integrated_force:.8f}\n")
            f.write(f"Относительная ошибка: {rel_error:.8f}%\n")
            f.write(f"Время обработки: {total_time:.2f}с\n")
            f.write(f"Статус: {estado}\n")

        print(f"Интерполяция завершена для итерации {iteration}, случай {case}")

    return resultados_interpolacion

def main():
    """Основная функция."""
    resultados_interpolacion = procesar_interpolacion(target_iteration)
    guardar_resultados_excel(resultados_interpolacion)
    
    print("\n" + "=" * 60)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 60)
    print("✅ ВСЕ ОПЕРАЦИИ ЗАВЕРШЕНЫ!")
    if target_iteration is not None:
        print(f"📊 Обработана итерация: {target_iteration}")
    else:
        print(f"📊 Обработано пар интерполяции: {len(resultados_interpolacion)}")
    print(f"📊 Результаты сохранены в Excel: resultados_verificacion.xlsx")
    print("\n🎯 Целевая точность: ошибка <= 3%")
    print("=" * 60)

if __name__ == "__main__":
    main()