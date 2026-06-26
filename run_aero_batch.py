import matlab.engine
import numpy as np
import os
import sys
import time
import glob

def run_matlab_aero(config_idx, params_dict, target_cy, output_force_file):
    """
    Запускает MATLAB функцию aero_solver для одной конфигурации.
    Возвращает словарь с результатами.
    """
    eng = matlab.engine.start_matlab()
    try:
        # Преобразуем словарь Python в MATLAB struct
        matlab_params = eng.struct()
        for key, value in params_dict.items():
            matlab_params[key] = value

        # Вызываем aero_solver
        results = eng.aero_solver(matlab_params, target_cy, output_force_file, nargout=1)

        # results - это словарь Python
        py_results = {}
        for key, val in results.items():
            # Преобразуем matlab.double в стандартные типы Python
            if isinstance(val, matlab.double):
                if val.size == 1:
                    val = float(val)
                else:
                    val = list(val)
            py_results[key] = val
        return py_results
    finally:
        eng.quit()

def main():
    # Получаем номер файла из аргументов командной строки (по умолчанию 1)
    file_number = 1
    if len(sys.argv) > 1:
        try:
            file_number = int(sys.argv[1])
        except ValueError:
            print("⚠ Некорректный аргумент, используется номер 1")

    # Очистка старых txt-файлов, созданных скриптом
    patterns = [
        "temp_forces_*.txt",      # временные файлы сил
        "config_*_alpha_CL.txt",  # файлы с углом атаки и Cy
        "[0-9]*_*.txt"            # итоговые файлы сил (начинаются с цифры)
    ]
    print("Очистка старых txt-файлов...")
    for pattern in patterns:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
                print(f"  Удалён: {f}")
            except OSError as e:
                print(f"  Не удалось удалить {f}: {e}")

    data_file = f"for_ansys_{file_number}.npy"
    if not os.path.exists(data_file):
        print(f"❌ Ошибка: Файл {data_file} не найден.")
        sys.exit(1)

    all_data = np.load(data_file)
    print(f"✅ Загружено {len(all_data)} конфигураций из {data_file}")

    aerodata_target_list = []

    for idx, row in enumerate(all_data):
        wing_area = float(row[0])
        aspect_ratio = float(row[1])
        taper_ratio = float(row[2])
        sweep_angle = float(row[3])
        thickness_raw = float(row[4])
        flight_speed = float(row[6])
        target_cy = float(row[12])
        air_density = float(row[14])

        # Если нужно округлить входные параметры до 3 знаков, раскомментируйте:
        # wing_area = round(wing_area, 3)
        # aspect_ratio = round(aspect_ratio, 3)
        # taper_ratio = round(taper_ratio, 3)
        # sweep_angle = round(sweep_angle, 3)

        # Округляем толщину профиля до целого процента
        thickness_percent = round(thickness_raw * 100)  # целое число процентов
        thickness = thickness_percent / 100.0           # доля, переданная в расчёт

        print(f"\n{'#'*60}")
        print(f"ОБРАБОТКА КОНФИГУРАЦИИ #{idx}")
        print(f"{'#'*60}")
        print(f"Площадь: {wing_area:.3f} м² | Удлинение: {aspect_ratio:.3f} | Сужение: {taper_ratio:.3f}")
        print(f"Стреловидность: {sweep_angle:.3f}° | Толщина: {thickness_percent}%")
        print(f"Скорость: {flight_speed:.2f} м/с | Плотность: {air_density:.3f} кг/м³")
        print(f"Целевой Cy: {target_cy:.4f}")

        # Формируем словарь параметров для MATLAB (все в СИ)
        params = {
            'wing_area_m2': wing_area,
            'aspect_ratio': aspect_ratio,
            'taper_ratio': taper_ratio,
            'sweep_angle_deg': sweep_angle,
            'thickness': thickness,               # передаём округлённую долю
            'flight_speed_ms': flight_speed,
            'air_density_kgm3': air_density
        }

        temp_force_file = f"temp_forces_{idx}.txt"

        t0 = time.time()
        results = run_matlab_aero(idx, params, target_cy, temp_force_file)
        elapsed = time.time() - t0

        if results is None:
            print(f"❌ Ошибка при расчёте конфигурации #{idx}")
            continue

        alpha_target = results['alpha_target']
        CL = results['Cy']
        CD = results.get('Cx', 0.0)
        Lift = 0.5 * air_density * flight_speed**2 * wing_area * CL

        print(f"\n--> Результаты (время: {elapsed:.2f} с):")
        print(f"    alpha_target = {alpha_target:.3f}°")
        print(f"    CL = {CL:.4f}, CD = {CD:.4f}, Lift = {Lift:.2f} Н")

        # Сохраняем округлённые до 3 знаков результаты
        aerodata_target_list.append([
            round(alpha_target, 3),
            round(CL, 3),
            round(CD, 3),
            round(Lift, 3)
        ])

        # Переименовываем файл с силами
        final_force = f"{idx}_{alpha_target:.2f}.txt"
        if os.path.exists(temp_force_file):
            os.rename(temp_force_file, final_force)
            print(f"    ✅ Файл сил сохранён как: {final_force}")
        else:
            print(f"    ⚠ Файл {temp_force_file} не найден")

        # Записываем alpha и CL с округлением до 3 знаков
        with open(f"config_{idx}_alpha_CL.txt", 'w') as f:
            f.write(f"{round(alpha_target, 3):.3f} {round(CL, 3):.3f}\n")

    if aerodata_target_list:
        aerodata_target = np.array(aerodata_target_list)
        np.save('aerodata_target.npy', aerodata_target)
        print(f"\n✅ Аэродинамические данные сохранены: aerodata_target.npy (форма {aerodata_target.shape})")
        print("   Столбцы: [alpha_target, CL, CD, Lift] (все округлены до 3 знаков)")

if __name__ == "__main__":
    main()