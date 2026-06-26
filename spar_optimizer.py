# spar_optimizer_bayesian.py – байесовская оптимизация положений лонжеронов и относительного расстояния между нервюрами
# Используется Gaussian Process (GP) с acquisition function EI.
# Количество вычислений целевой функции = 100.
# Теперь относительное расстояние между нервюрами задаётся в долях от размаха (0..1).

import os
import sys
import math
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from thickness_optimizer import WingOptimizerManualSpars

# ----------------------------------------------------------------------
# (УДАЛЕНО) Вспомогательная функция get_max_ribs больше не нужна,
# так как количество нервюр вычисляется напрямую из rel_spacing.
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Компонент для вычисления массы крыла (передаёт все параметры оптимизатору)
# ----------------------------------------------------------------------
class WingMassComponent:
    def __init__(self, file_number, config_index,
                 total_thickness_min=0.0005, total_thickness_max=0.1,
                 face_ratio=0.40, core_ratio=0.20, face_min_thickness=0.00025,
                 buck_t_max=0.1, buck_energy_threshold=1e-10, buck_gain=10000.0,
                 buck_power_alpha=0.5, buck_base_energy=0.1,
                 buck_power_alpha_A=0.05, buck_power_alpha_B=0.5,
                 max_buck_iter=30, buck_thick_increase=1.0, energy_threshold_factor=1.0,
                 buck_adapt_factor=5.0, n_buck_modes=5, buck_density_cutoff=0.1,
                 buck_filter_radius=1.5, use_buck_sensitivity_filter=True,
                 nproc=None):
        self.file_number = file_number
        self.config_index = config_index
        self.params = {
            'total_thickness_min': total_thickness_min,
            'total_thickness_max': total_thickness_max,
            'face_ratio': face_ratio,
            'core_ratio': core_ratio,
            'face_min_thickness': face_min_thickness,
            'buck_t_max': buck_t_max,
            'buck_energy_threshold': buck_energy_threshold,
            'buck_gain': buck_gain,
            'buck_power_alpha': buck_power_alpha,
            'buck_base_energy': buck_base_energy,
            'buck_power_alpha_A': buck_power_alpha_A,
            'buck_power_alpha_B': buck_power_alpha_B,
            'max_buck_iter': max_buck_iter,
            'buck_thick_increase': buck_thick_increase,
            'energy_threshold_factor': energy_threshold_factor,
            'buck_adapt_factor': buck_adapt_factor,
            'n_buck_modes': n_buck_modes,
            'buck_density_cutoff': buck_density_cutoff,
            'buck_filter_radius': buck_filter_radius,
            'use_buck_sensitivity_filter': use_buck_sensitivity_filter,
            'nproc': nproc
        }
        # Загружаем конфигурацию для получения wing_area и aspect_ratio
        data_file = f"for_ansys_{file_number}.npy"
        if os.path.exists(data_file):
            all_data = np.load(data_file)
            if config_index < len(all_data):
                row = all_data[config_index]
                self.wing_area = float(row[0])
                self.aspect_ratio = float(row[1])
            else:
                raise ValueError(f"config_index {config_index} вне диапазона")
        else:
            raise FileNotFoundError(f"Файл {data_file} не найден")

    def compute(self, inputs, outputs):
        spar1 = inputs['spar1']
        spar2 = inputs['spar2']
        rel_spacing = inputs['relative_rib_spacing']
        rel_spacing = max(1e-6, min(1.0, rel_spacing))

        actual_wing_area = self.wing_area / 2.0
        actual_aspect_ratio = self.aspect_ratio / 2.0
        wing_length = math.sqrt(actual_wing_area * actual_aspect_ratio)

        # ---- ИЗМЕНЕНИЕ: вычисляем количество нервюр ТОЛЬКО из rel_spacing ----
        if rel_spacing <= 0:
            rib_count = 2
        else:
            n_intervals = int(round(1.0 / rel_spacing))
            rib_count = max(2, n_intervals + 1)
        # (Убрано ограничение через get_max_ribs)

        print(f"\n{'='*60}")
        print(f"ЗАПУСК РАСЧЁТА ДЛЯ spar1={spar1:.4f}, spar2={spar2:.4f}, rel_spacing={rel_spacing:.4f}")
        print(f"  wing_length = {wing_length:.3f} м -> rib_count = {rib_count}")
        print(f"{'='*60}")

        optimizer = WingOptimizerManualSpars(
            file_number=self.file_number,
            objective='energy',
            uz_max_limit=None,
            spar1_pos=spar1,
            spar2_pos=spar2,
            rib_count=rib_count,
            total_thickness_min=self.params['total_thickness_min'],
            total_thickness_max=self.params['total_thickness_max'],
            face_ratio=self.params['face_ratio'],
            core_ratio=self.params['core_ratio'],
            face_min_thickness=self.params['face_min_thickness'],
            buck_t_max=self.params['buck_t_max'],
            buck_energy_threshold=self.params['buck_energy_threshold'],
            buck_gain=self.params['buck_gain'],
            buck_power_alpha=self.params['buck_power_alpha'],
            buck_base_energy=self.params['buck_base_energy'],
            buck_power_alpha_A=self.params['buck_power_alpha_A'],
            buck_power_alpha_B=self.params['buck_power_alpha_B'],
            max_buck_iter=self.params['max_buck_iter'],
            buck_thick_increase=self.params['buck_thick_increase'],
            energy_threshold_factor=self.params['energy_threshold_factor'],
            buck_adapt_factor=self.params['buck_adapt_factor'],
            n_buck_modes=self.params['n_buck_modes'],
            buck_density_cutoff=self.params['buck_density_cutoff'],
            buck_filter_radius=self.params['buck_filter_radius'],
            use_buck_sensitivity_filter=self.params['use_buck_sensitivity_filter'],
            nproc=self.params['nproc']
        )

        mass = optimizer.optimize_configuration_manual(self.config_index)

        if mass is None:
            print("⚠ optimize_configuration_manual вернул None, причина неизвестна")
            best_dir = os.path.join(optimizer.results_dir, f"config_{self.config_index}_best")
            mass_file = os.path.join(best_dir, f"mass_config_{self.config_index}.txt")
            if os.path.exists(mass_file):
                try:
                    with open(mass_file, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                        import re
                        match = re.search(r'([\d.]+)', first_line)
                        if match:
                            mass = float(match.group(1))
                            print(f"   Удалось извлечь массу {mass:.4f} кг")
                        else:
                            mass = 1e10
                except Exception:
                    mass = 1e10
            else:
                mass = 1e10
        else:
            print(f"✅ Масса = {mass:.4f} кг")

        outputs['mass'] = mass
        return mass

# ----------------------------------------------------------------------
# Байесовская оптимизация (без изменений)
# ----------------------------------------------------------------------
def bayesian_optimization(objective_func, bounds, n_calls=100, random_state=42):
    space = [Real(low, high, name=f'x{i}') for i, (low, high) in enumerate(bounds)]

    @use_named_args(space)
    def wrapped_objective(**kwargs):
        x = [kwargs[f'x{i}'] for i in range(len(bounds))]
        return objective_func(x)

    result = gp_minimize(
        wrapped_objective,
        space,
        n_calls=n_calls,
        random_state=random_state,
        verbose=True,
        n_initial_points=10,
        acq_func='EI'
    )
    return result.x, result.fun

# ----------------------------------------------------------------------
# Функция, запускающая оптимизацию для одной конфигурации (байесовская)
# ----------------------------------------------------------------------
def optimize_configuration(file_number, config_index, spar1_init, spar2_init,
                           rib_count_init, wing_area, aspect_ratio,
                           opt_params, n_calls=100):
    mass_comp = WingMassComponent(file_number, config_index, **opt_params)

    def objective(x):
        spar1, spar2, rel_spacing = x
        inputs = {
            'spar1': spar1,
            'spar2': spar2,
            'relative_rib_spacing': rel_spacing
        }
        outputs = {}
        mass = mass_comp.compute(inputs, outputs)
        return mass

    actual_wing_area = wing_area / 2.0
    actual_aspect_ratio = aspect_ratio / 2.0
    wing_length = math.sqrt(actual_wing_area * actual_aspect_ratio)

    # ---- ИЗМЕНЕНИЕ: начальное rel_spacing вычисляется из rib_count_init ----
    if rib_count_init > 1:
        rel_spacing_init = 1.0 / (rib_count_init - 1)
    else:
        rel_spacing_init = 1.0
    rel_spacing_init = max(1e-6, min(1.0, rel_spacing_init))

    # ---- ИЗМЕНЕНИЕ: границы для rel_spacing – фиксированный диапазон [0.01, 1.0] ----
    # Нижняя граница выбрана 0.01, чтобы избежать слишком большого числа нервюр (макс. 101).
    # При необходимости её можно изменить (например, 0.001 даст до 1001 нервюры).
    bounds = [
        (0.10, 0.25),   # spar1
        (0.50, 0.90),   # spar2
        (0.07, 1.0)     # relative spacing (теперь чисто относительное)
    ]

    print(f"\nНачальные параметры: spar1={spar1_init:.4f}, spar2={spar2_init:.4f}, rel_spacing={rel_spacing_init:.4f}")
    print(f"Границы: spar1 {bounds[0]}, spar2 {bounds[1]}, rel_spacing {bounds[2]}")
    print(f"Запуск байесовской оптимизации (GP, {n_calls} итераций)...")

    best_x, best_mass = bayesian_optimization(
        objective,
        bounds=bounds,
        n_calls=n_calls,
        random_state=config_index
    )

    print("\nБайесовская оптимизация завершена. Лучшие параметры:")
    print(f"  spar1 = {best_x[0]:.4f}")
    print(f"  spar2 = {best_x[1]:.4f}")
    print(f"  relative spacing = {best_x[2]:.4f}")
    print(f"  Лучшая масса (из вызовов) = {best_mass:.4f} кг")

    # ------------------------------------------------------------------
    # ФИНАЛЬНЫЙ РАСЧЁТ С ЛУЧШИМИ ПАРАМЕТРАМИ
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("ЗАПУСК ФИНАЛЬНОГО РАСЧЁТА С ЛУЧШИМИ ПАРАМЕТРАМИ")
    print("="*60)

    final_optimizer = WingOptimizerManualSpars(
        file_number=file_number,
        objective='energy',
        uz_max_limit=None,
        spar1_pos=best_x[0],
        spar2_pos=best_x[1],
        rib_count=None,
        **opt_params
    )
    # Вычисляем финальное количество нервюр из best_x[2]
    rel_spacing_best = best_x[2]
    if rel_spacing_best <= 0:
        final_rib_count = 2
    else:
        n_intervals = int(round(1.0 / rel_spacing_best))
        final_rib_count = max(2, n_intervals + 1)
    final_optimizer.manual_rib_count = final_rib_count

    final_mass = final_optimizer.optimize_configuration_manual(config_index)

    if final_mass is None:
        print("⚠ Финальный расчёт не удался, используется масса из оптимизации")
        final_mass = best_mass
    else:
        print(f"✅ Финальный расчёт завершён, масса = {final_mass:.4f} кг")

    opt_rel_spacing = best_x[2]
    opt_rib_count = final_rib_count
    best_x_final = best_x
    best_mass_final = final_mass

    print("\n" + "="*60)
    print(f"ОПТИМИЗАЦИЯ КОНФИГУРАЦИИ {config_index} ЗАВЕРШЕНА")
    print("="*60)
    print(f"Оптимальное положение первого лонжерона: {best_x_final[0]:.4f}")
    print(f"Оптимальное положение второго лонжерона: {best_x_final[1]:.4f}")
    print(f"Оптимальное относительное расстояние между нервюрами: {opt_rel_spacing:.4f}")
    print(f"Оптимальное количество нервюр: {opt_rib_count}")
    print(f"Минимальная масса (финальный расчёт): {best_mass_final:.4f} кг")
    print("="*60)

    results_dir = os.path.join(os.getcwd(), "optimization_results_bayesian")
    os.makedirs(results_dir, exist_ok=True)
    summary_file = os.path.join(results_dir, f"spar_optimization_config_{config_index}.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Конфигурация: {config_index}\n")
        f.write(f"Оптимальное положение первого лонжерона: {best_x_final[0]:.6f}\n")
        f.write(f"Оптимальное положение второго лонжерона: {best_x_final[1]:.6f}\n")
        f.write(f"Оптимальное относительное расстояние между нервюрами: {opt_rel_spacing:.6f}\n")
        f.write(f"Оптимальное количество нервюр: {opt_rib_count}\n")
        f.write(f"Минимальная масса (финальный расчёт): {best_mass_final:.6f} кг\n")
        f.write(f"Масса из байесовской оптимизации (справочно): {best_mass:.6f} кг\n")

    return best_x_final, best_mass_final

# ----------------------------------------------------------------------
# Основная функция: загрузка конфигураций и запуск оптимизации для всех
# ----------------------------------------------------------------------
def main():
    # ------------------- Разбор параметров командной строки -------------------
    if len(sys.argv) < 2:
        print("Ошибка: необходимо указать номер файла")
        print("Использование: python spar_optimizer_bayesian.py <file_number> [параметры...] [n_calls]")
        return

    try:
        file_number = int(sys.argv[1])
    except ValueError:
        print("Ошибка: номер файла должен быть целым числом")
        return

    # Параметры по умолчанию (как в thickness_optimizer.py)
    objective = 'energy'
    uz_max_limit = None
    spar1_init = None
    spar2_init = None
    max_buck_iter = 30
    buck_thick_increase = 1.0
    energy_threshold_factor = 1.0
    buck_t_max = 0.1
    buck_adapt_factor = 5.0
    n_buck_modes = 5
    buck_density_cutoff = 0.1
    buck_filter_radius = 1.5
    use_buck_sensitivity_filter = 1
    rib_count_init = None
    nproc = None
    total_thickness_min = 0.0005
    total_thickness_max = 0.1
    core_ratio = 1e-9
    face_ratio = (1.0 - core_ratio) / 2.0
    face_min_thickness = 0.00025
    buck_energy_threshold = 1e-10
    buck_gain = 10000.0
    buck_power_alpha = 0.5
    buck_base_energy = 0.1
    buck_power_alpha_A = 0.05
    buck_power_alpha_B = 0.5
    n_calls = 100

    # Разбор аргументов (без изменений, кроме последнего параметра n_calls)
    idx = 2
    if len(sys.argv) > idx and sys.argv[idx].lower() in ['energy', 'g']:
        objective = sys.argv[idx].lower()
        idx += 1
    if len(sys.argv) > idx:
        try:
            uz_max_limit = float(sys.argv[idx])
            idx += 1
        except:
            pass
    if len(sys.argv) > idx:
        try:
            spar1_init = float(sys.argv[idx])
            idx += 1
        except:
            pass
    if len(sys.argv) > idx:
        try:
            spar2_init = float(sys.argv[idx])
            idx += 1
        except:
            pass
    if len(sys.argv) > idx:
        try:
            max_buck_iter = int(sys.argv[idx])
            idx += 1
        except:
            pass
    if len(sys.argv) > idx:
        try:
            buck_thick_increase = float(sys.argv[idx])
            idx += 1
        except:
            pass
    if len(sys.argv) > idx:
        try:
            energy_threshold_factor = float(sys.argv[idx])
            idx += 1
        except:
            pass
    if len(sys.argv) > idx:
        try:
            buck_t_max = float(sys.argv[idx])
            idx += 1
        except:
            pass
    if len(sys.argv) > idx:
        try:
            buck_adapt_factor = float(sys.argv[idx])
            idx += 1
        except:
            pass
    if len(sys.argv) > idx:
        try:
            n_buck_modes = int(sys.argv[idx])
            idx += 1
        except:
            pass
    if len(sys.argv) > idx:
        try:
            buck_density_cutoff = float(sys.argv[idx])
            idx += 1
        except:
            pass
    if len(sys.argv) > idx:
        try:
            buck_filter_radius = float(sys.argv[idx])
            idx += 1
        except:
            pass
    if len(sys.argv) > idx:
        try:
            use_buck_sensitivity_filter = int(sys.argv[idx])
            idx += 1
        except:
            pass
    if len(sys.argv) > idx:
        try:
            rib_count_init = int(sys.argv[idx])
            idx += 1
        except:
            pass
    if len(sys.argv) > idx:
        try:
            nproc = int(sys.argv[idx])
            idx += 1
        except:
            pass
    if len(sys.argv) > idx:
        try:
            total_thickness_min = float(sys.argv[idx])
            idx += 1
        except:
            pass
    if len(sys.argv) > idx:
        try:
            total_thickness_max = float(sys.argv[idx])
            idx += 1
        except:
            pass
    if len(sys.argv) > idx:
        try:
            face_ratio = float(sys.argv[idx])
            core_ratio = 1.0 - 2*face_ratio
            if core_ratio < 0:
                print("Предупреждение: face_ratio слишком велик, устанавливаем face_ratio=0.5, core_ratio=0")
                face_ratio = 0.5
                core_ratio = 0.0
            idx += 1
        except:
            pass
    if len(sys.argv) > idx:
        try:
            core_ratio = float(sys.argv[idx])
            face_ratio = (1.0 - core_ratio) / 2.0
            if face_ratio < 0:
                print("Предупреждение: core_ratio слишком велик, устанавливаем core_ratio=0, face_ratio=0.5")
                core_ratio = 0.0
                face_ratio = 0.5
            idx += 1
        except:
            pass
    if len(sys.argv) > idx:
        try:
            face_min_thickness = float(sys.argv[idx])
            idx += 1
        except:
            pass
    if len(sys.argv) > idx:
        try:
            buck_energy_threshold = float(sys.argv[idx])
            idx += 1
        except:
            pass
    if len(sys.argv) > idx:
        try:
            buck_gain = float(sys.argv[idx])
            idx += 1
        except:
            pass
    if len(sys.argv) > idx:
        try:
            buck_power_alpha = float(sys.argv[idx])
            buck_power_alpha_A = buck_power_alpha
            buck_power_alpha_B = buck_power_alpha
            idx += 1
        except:
            pass
    if len(sys.argv) > idx:
        try:
            buck_base_energy = float(sys.argv[idx])
            idx += 1
        except:
            pass
    if len(sys.argv) > idx:
        try:
            buck_power_alpha_A = float(sys.argv[idx])
            idx += 1
        except:
            pass
    if len(sys.argv) > idx:
        try:
            buck_power_alpha_B = float(sys.argv[idx])
            idx += 1
        except:
            pass
    if len(sys.argv) > idx:
        try:
            n_calls = int(sys.argv[idx])
            idx += 1
        except:
            pass

    total = 2*face_ratio + core_ratio
    if abs(total - 1.0) > 1e-6:
        core_ratio = 1.0 - 2*face_ratio
        if core_ratio < 0:
            core_ratio = 0.0
            face_ratio = 0.5

    opt_params = {
        'total_thickness_min': total_thickness_min,
        'total_thickness_max': total_thickness_max,
        'face_ratio': face_ratio,
        'core_ratio': core_ratio,
        'face_min_thickness': face_min_thickness,
        'buck_t_max': buck_t_max,
        'buck_energy_threshold': buck_energy_threshold,
        'buck_gain': buck_gain,
        'buck_power_alpha': buck_power_alpha,
        'buck_base_energy': buck_base_energy,
        'buck_power_alpha_A': buck_power_alpha_A,
        'buck_power_alpha_B': buck_power_alpha_B,
        'max_buck_iter': max_buck_iter,
        'buck_thick_increase': buck_thick_increase,
        'energy_threshold_factor': energy_threshold_factor,
        'buck_adapt_factor': buck_adapt_factor,
        'n_buck_modes': n_buck_modes,
        'buck_density_cutoff': buck_density_cutoff,
        'buck_filter_radius': buck_filter_radius,
        'use_buck_sensitivity_filter': bool(use_buck_sensitivity_filter),
        'nproc': nproc
    }

    data_file = f"for_ansys_{file_number}.npy"
    if not os.path.exists(data_file):
        print(f"❌ Ошибка: файл {data_file} не найден.")
        return

    all_data = np.load(data_file)
    if len(all_data) == 0:
        print("❌ Файл не содержит данных.")
        return

    print(f"✅ Загружено {len(all_data)} конфигураций из {data_file}")

    for config_index, row in enumerate(all_data):
        if len(row) < 16:
            print(f"⚠ Конфигурация {config_index} имеет недостаточно столбцов ({len(row)}), пропускаем.")
            continue

        if spar1_init is None:
            spar1 = row[10]
        else:
            spar1 = spar1_init
        if spar2_init is None:
            spar2 = row[11]
        else:
            spar2 = spar2_init
        if rib_count_init is None:
            rib_cnt = int(round(row[15]))
        else:
            rib_cnt = rib_count_init
        wing_area = row[0]
        aspect_ratio = row[1]

        # (Убрано вычисление max_ribs через get_max_ribs)
        rib_cnt = max(2, rib_cnt)   # просто гарантируем минимум 2

        print(f"\nНачальные параметры для конфигурации {config_index}:")
        print(f"  spar1 = {spar1:.3f}, spar2 = {spar2:.3f}, rib_count = {rib_cnt}")
        print(f"  (Максимальное количество нервюр не ограничено, определяется только rel_spacing)")

        optimize_configuration(file_number, config_index,
                               spar1, spar2,
                               rib_cnt,
                               wing_area, aspect_ratio,
                               opt_params, n_calls=n_calls)

    print("\n" + "="*60)
    print("ОПТИМИЗАЦИЯ ВСЕХ КОНФИГУРАЦИЙ ЗАВЕРШЕНА")
    print("="*60)

if __name__ == "__main__":
    main()