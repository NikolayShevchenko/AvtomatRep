# thickness_optimizer.py – оптимизация трёхслойных панелей (обшивка-заполнитель-обшивка)
# Суммарная толщина total = 2*t_face + t_core.
# Добавлена возможность задавать минимальную толщину несущего слоя (face_min_thickness)
# Первая фаза: оптимизация total по напряжениям (все элементы).
# Вторая фаза: дооптимизация только перегруженных элементов.
# Третья фаза: баклинг-оптимизация (увеличение t_core) со степенным множителем
#               и нормализацией по максимальной энергии (ускоренная сходимость).
# Сохраняет массы по типам элементов и толщины (total, t_face, t_core) для каждого элемента.
# ДОБАВЛЕНО: вычисление упругой энергии каждого слоя (bot, mid, top) по формуле 0.5*σ*ε,
#            с созданием таблиц SENE_LAY_B, SENE_LAY_M, SENE_LAY_T.
# ДОБАВЛЕНО: сохранение поэлементной энергии среднего слоя (заполнителя) в файл core_energy_elem_config_X.txt
# ДОБАВЛЕНО (2026-05-22): ИСПРАВЛЕННОЕ вычисление силового фактора G и безразмерного коэффициента Ck
#            до и после баклинг-оптимизации. G = ∫ σ_eqv dV вычисляется как сумма вкладов каждого слоя,
#            Ck = G / (P * sqrt(S)), где P – максимальная суммарная аэродинамическая сила по всем load steps,
#            S – площадь крыла.
#
# Особенности задания смещений сечений (SECOFFSET) в зависимости от типа элемента:
#   тип 1 (обшивка)   – SECOFFSET,BOT  (нижняя грань фиксирована, панель растёт вверх)
#   тип 2 (лонжероны) – SECOFFSET,MID (средняя линия фиксирована, симметричное утолщение)
#   тип 3 (нервюры)   – SECOFFSET,MID (средняя линия фиксирована)
#   тип 4 (задняя стенка) – SECOFFSET,TOP (верхняя грань фиксирована, панель растёт внутрь)
# Структура слоёв для всех типов одинакова:
#   слой 1 (нижний несущий) – смещение 0 (BOT), материал 1
#   слой 2 (заполнитель)    – смещение 0 (BOT), материал 2
#   слой 3 (верхний несущий) – смещение 1 (TOP), материал 1

import os
import sys
import subprocess
import shutil
import numpy as np
from datetime import datetime
import glob
import re
import time
import multiprocessing

class WingOptimizerManualSpars:
    """
    Оптимизатор крыла с ручным заданием позиций лонжеронов и количества нервюр.
    Трёхслойная оболочка:
      - суммарная толщина total = 2*t_face + t_core
      - доли: face_ratio = t_face / total (для одного несущего слоя)
      - core_ratio = t_core / total
    Статическая оптимизация: изменяется total, затем пересчитываются t_face и t_core.
    Баклинг-оптимизация: увеличивается t_core (и соответственно total) с использованием
                         степенного множителя и нормализации энергии.
    ДОБАВЛЕНО: вычисление упругой энергии каждого слоя.
    ДОБАВЛЕНО: ИСПРАВЛЕННОЕ вычисление силового фактора G и безразмерного коэффициента Ck.
    """
    def __init__(self, file_number=0, objective='energy', uz_max_limit=None,
                 spar1_pos=None, spar2_pos=None,
                 max_buck_iter=20, buck_thick_increase=1,
                 energy_threshold_factor=1, buck_t_max=0.1,
                 buck_adapt_factor=5, n_buck_modes=5,
                 buck_density_cutoff=0.1, buck_filter_radius=1.5,
                 buck_energy_threshold=1e-10, buck_gain=1000.0,
                 use_buck_sensitivity_filter=True,
                 rib_count=None, nproc=None,
                 total_thickness_min=0.0005, total_thickness_max=0.1,
                 face_ratio=0.40, core_ratio=0.20,
                 face_min_thickness=0.00025,
                 buck_power_alpha=0.5, buck_base_energy=0.1,
                 buck_power_alpha_A=0.05, buck_power_alpha_B=0.5):
        self.file_number = file_number
        self.base_dir = os.getcwd()
        self.results_dir = os.path.join(self.base_dir, "optimization_results_manual_spars")
        self.best_results = {}
        self.results_history = []
        self.baseline_results = {}
        self.function_evaluations = 0
        self.current_evaluation_data = {}

        self.objective = objective.lower()
        self.uz_max_limit = uz_max_limit
        self.start_time = None
        self.end_time = None
        self.total_duration = None
        self.spar1_bounds = [0.382372159921237, 0.382372159921238]
        self.spar2_bounds = [0.743231850462974, 0.743231850462975]
        self.user_spar1 = spar1_pos
        self.user_spar2 = spar2_pos
        self.manual_rib_count = rib_count
        self.phase2_iterations = 10          # итерации второй фазы (только перегруженные)
        # Параметры баклинг-оптимизации
        self.max_buck_iter = max_buck_iter
        self.buck_thick_increase = buck_thick_increase
        self.energy_threshold_factor = energy_threshold_factor
        self.buck_t_max = buck_t_max          # максимальная толщина заполнителя
        self.buck_adapt_factor = buck_adapt_factor
        self.n_buck_modes = n_buck_modes
        self.buck_density_cutoff = buck_density_cutoff
        self.buck_filter_radius = buck_filter_radius
        self.use_buck_sensitivity_filter = use_buck_sensitivity_filter
        self.buck_energy_threshold = buck_energy_threshold   # порог чувствительности
        self.buck_gain = buck_gain                           # устаревший, не используется
        self.buck_power_alpha = buck_power_alpha             # общий (для обратной совместимости)
        self.buck_base_energy = buck_base_energy             # базовая энергия для всех элементов
        # Новые параметры для раздельной оптимизации групп
        self.buck_power_alpha_A = buck_power_alpha_A         # показатель для группы A (обшивка+зад. стенка)
        self.buck_power_alpha_B = buck_power_alpha_B         # показатель для группы B (лонжероны+нервюры)
        self.buck_energy_threshold = buck_energy_threshold
        self.buck_gain = buck_gain
        # Параметры трёхслойной панели (суммарная толщина и доли)
        self.total_thickness_min = total_thickness_min   # общая минимальная толщина (м)
        self.total_thickness_max = total_thickness_max   # общая максимальная толщина (м)
        self.face_ratio = face_ratio                     # доля одного несущего слоя (от общей)
        self.core_ratio = core_ratio                     # доля заполнителя
        self.face_min_thickness = face_min_thickness     # минимальная толщина одного несущего слоя (м)

        # Корректировка долей: если core_ratio слишком мал (почти 0), то face_ratio пересчитывается
        # чтобы сумма 2*face_ratio+core_ratio = 1
        if self.core_ratio < 1e-8:
            self.core_ratio = 1e-9   # минимальная ненулевая толщина заполнителя для численной устойчивости
            self.face_ratio = (1.0 - self.core_ratio) / 2.0
        total = 2*self.face_ratio + self.core_ratio
        if abs(total - 1.0) > 1e-6:
            # Корректируем core_ratio
            self.core_ratio = 1.0 - 2*self.face_ratio
            if self.core_ratio < 0:
                self.face_ratio = 0.5
                self.core_ratio = 0.0
            if self.core_ratio < 1e-9:
                self.core_ratio = 1e-9
        # Улучшенная начальная минимальная толщина заполнителя (ускорение сходимости баклинга)
        # вместо 1e-9 используем разумный минимум, пропорциональный минимальной толщине обшивки
        self.core_thickness_min = max(1e-4, self.face_min_thickness * 0.5)

        self.rho_face = 2700                  # плотность несущего слоя (кг/м³)
        self.rho_core = 75                   # плотность заполнителя

        # Минимальные и максимальные суммарные толщины для разных типов элементов (опционально)
        # Здесь пока единые для всех, но можно расширить
        self.min_total_by_type = [self.total_thickness_min] * 5   # индексы 1..4
        self.max_total_by_type = [self.total_thickness_max] * 5

        # Многопоточность
        if nproc is None:
            nproc = max(1, multiprocessing.cpu_count() - 1)
        self.nproc = min(nproc, 16)
        self.log(f"Инициализация оптимизатора: количество потоков ANSYS = {self.nproc}")

        self.configurations = self.load_configurations()
        os.makedirs(self.results_dir, exist_ok=True)

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def load_configurations(self):
        data_file = f"for_ansys_{self.file_number}.npy"
        try:
            data = np.load(data_file)
            self.log(f"✅ Загружено {len(data)} конфигураций из {data_file}")
            return data
        except FileNotFoundError:
            self.log(f"❌ Ошибка: Файл {data_file} не найден")
            return np.array([])
        except Exception as e:
            self.log(f"❌ Ошибка при загрузке {data_file}: {e}")
            return np.array([])

    def get_ansys_path(self):
        possible_paths = [
            r"C:\Program Files\ANSYS Inc\v182\ansys\bin\winx64\ANSYS182.exe",
            r"C:\Program Files\ANSYS Inc\v202\ansys\bin\winx64\ANSYS202.exe",
            r"C:\Program Files\ANSYS Inc\v211\ansys\bin\winx64\ANSYS211.exe",
            r"C:\Program Files\ANSYS Inc\v221\ansys\bin\winx64\ANSYS221.exe",
            r"C:\Program Files\ANSYS Inc\v231\ansys\bin\winx64\ANSYS231.exe",
            r"C:\Program Files\ANSYS Inc\v241\ansys\bin\winx64\ANSYS241.exe",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        self.log("⚠ ANSYS not found. Please install or provide correct path.")
        return None

    def get_rib_configuration(self, config_index):
        if self.manual_rib_count is not None:
            return int(round(self.manual_rib_count))
        if config_index < len(self.configurations) and self.configurations.shape[1] >= 16:
            return int(round(self.configurations[config_index, 15]))
        return 6

    def get_all_cases_for_config(self, config_index):
        cases = []
        cpwing_pattern = f"{config_index}_*.txt"
        for file_path in glob.glob(cpwing_pattern):
            match = re.match(rf'^{re.escape(str(config_index))}_(-?\d+(?:\.\d+)?)\.txt$', os.path.basename(file_path))
            if match:
                case_str = match.group(1)
                try:
                    case_float = float(case_str)
                    cases.append((case_str, case_float))
                except ValueError:
                    continue
        if not cases:
            load_pattern = f"resultados_interpolacion_{config_index}_*.txt"
            for file_path in glob.glob(load_pattern):
                match = re.match(rf'^resultados_interpolacion_{re.escape(str(config_index))}_(-?\d+(?:\.\d+)?)\.txt$', os.path.basename(file_path))
                if match:
                    case_str = match.group(1)
                    try:
                        case_float = float(case_str)
                        cases.append((case_str, case_float))
                    except ValueError:
                        continue
        if not cases:
            cases = [("5", 5.0)]
        cases.sort(key=lambda x: x[1])
        return [case_str for case_str, _ in cases]

    def get_all_cases_for_config_with_float(self, config_index):
        cases = []
        cpwing_pattern = f"{config_index}_*.txt"
        for file_path in glob.glob(cpwing_pattern):
            match = re.match(rf'^{re.escape(str(config_index))}_(-?\d+(?:\.\d+)?)\.txt$', os.path.basename(file_path))
            if match:
                case_str = match.group(1)
                try:
                    case_float = float(case_str)
                    cases.append((case_str, case_float))
                except ValueError:
                    continue
        if not cases:
            load_pattern = f"resultados_interpolacion_{config_index}_*.txt"
            for file_path in glob.glob(load_pattern):
                match = re.match(rf'^resultados_interpolacion_{re.escape(str(config_index))}_(-?\d+(?:\.\d+)?)\.txt$', os.path.basename(file_path))
                if match:
                    case_str = match.group(1)
                    try:
                        case_float = float(case_str)
                        cases.append((case_str, case_float))
                    except ValueError:
                        continue
        if not cases:
            cases = [("5", 5.0)]
        cases.sort(key=lambda x: x[1])
        return cases

    def is_single_case_configuration(self, config_index):
        cases = self.get_all_cases_for_config(config_index)
        return len(cases) == 1

    def extract_max_uz(self, config_index, case):
        try:
            case_str = str(case)
            candidates = [
                f"max_uz_{config_index}_{case_str}.txt",
                f"max_uz_{config_index}_{case_str.replace('.', 'p')}.txt",
                f"max_uz_{config_index}_{case_str.replace('.', 'p').replace('-', 'm')}.txt"
            ]
            uz_file = None
            for fname in candidates:
                if os.path.exists(fname):
                    uz_file = fname
                    break
            if not uz_file:
                return None
            with open(uz_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                patterns = [
                    r'Максимальное перемещение по Z \(UZ_MAX\)\s*=\s*([\d.E+-]+)',
                    r'UZ_MAX\s*=\s*([\d.E+-]+)',
                    r'([\d.E+-]+)'
                ]
                for pattern in patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        return float(match.group(1))
            return None
        except Exception:
            return None

    def extract_G(self, config_index, case):
        try:
            case_str = str(case)
            candidates = [
                f"G_output_{config_index}_{case_str}.txt",
                f"G_output_{config_index}_{case_str.replace('.', 'p')}.txt",
                f"G_output_{config_index}_{case_str.replace('.', 'p').replace('-', 'm')}.txt"
            ]
            g_file = None
            for fname in candidates:
                if os.path.exists(fname):
                    g_file = fname
                    break
            if not g_file:
                return None
            with open(g_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                patterns = [
                    r'Интеграл G\s*\(σ_eqv \* V\)\s*=\s*([\d.E+-]+)',
                    r'Интеграл G .*?=\s*([\d.E+-]+)',
                    r'G\s*=\s*([\d.E+-]+)',
                    r'([\d.E+-]+)'
                ]
                for pattern in patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        return float(match.group(1))
            return None
        except Exception:
            return None

    def extract_elastic_strain_energy(self, config_index, case="0"):
        try:
            case_str = str(case)
            energy_file = f"elastic_strain_energy_{config_index}_{case_str}.txt"
            if os.path.exists(energy_file):
                with open(energy_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    patterns = [
                        r'Суммарная энергия эластичных нагрузок\s*\(TOTAL_SE\)\s*=\s*([\d.E+-]+)',
                        r'Total elastic strain energy\s*=\s*([\d.E+-]+)',
                        r'Энергия\s*=\s*([\d.E+-]+)',
                        r'Energy\s*=\s*([\d.E+-]+)'
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, content, re.IGNORECASE)
                        if match:
                            return float(match.group(1))
            return None
        except Exception:
            return None

    def cleanup_working_directory(self):
        extensions_to_remove = ['.db', '.DSP', '.esav', '.mntr',
                              '.err', '.full', '.log', '.stat', '.out', '.apdl']
        for ext in extensions_to_remove:
            for file in glob.glob(f"*{ext}"):
                try:
                    os.remove(file)
                except:
                    pass
        rst_files = glob.glob("*.rst")
        for rst_file in rst_files:
            try:
                if os.path.getmtime(rst_file) < time.time() - 7200:
                    os.remove(rst_file)
            except:
                pass

    def safe_rmtree(self, path, max_attempts=5, delay=1.0):
        if not os.path.exists(path):
            return True
        for attempt in range(max_attempts):
            try:
                shutil.rmtree(path)
                self.log(f"✓ Удалена директория: {path}")
                return True
            except PermissionError as e:
                self.log(f"⚠ Попытка {attempt+1}/{max_attempts}: не удалось удалить {path} — {e}")
                if attempt < max_attempts - 1:
                    time.sleep(delay)
                    continue
                backup_dir = path + f"_old_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                try:
                    os.rename(path, backup_dir)
                    self.log(f"⚠ Переименована директория {path} -> {backup_dir}")
                    return True
                except Exception as rename_err:
                    self.log(f"✗ Критическая ошибка: не удалось удалить или переименовать {path} — {rename_err}")
                    return False
        return False

    def create_configuration_file(self, config_index, spar1_pos, spar2_pos):
        try:
            with open('spar_positions.txt', 'w') as f:
                f.write(f"{spar1_pos:.15f}\n{spar2_pos:.15f}")
            time.sleep(0.05)
            return True
        except Exception as e:
            self.log(f"Ошибка при создании spar_positions.txt: {e}")
            return False

    def create_rib_count_file(self, rib_count):
        try:
            with open('rib_count.txt', 'w') as f:
                f.write(str(rib_count))
            self.log(f"Создан rib_count.txt со значением {rib_count}")
            return True
        except Exception as e:
            self.log(f"Ошибка при создании rib_count.txt: {e}")
            return False

    def verify_spar_positions(self, config_index, expected_spar1, expected_spar2):
        try:
            time.sleep(0.1)
            if os.path.exists('spar_positions.txt'):
                with open('spar_positions.txt', 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        actual_spar1 = float(lines[0].strip())
                        actual_spar2 = float(lines[1].strip())
                        if abs(actual_spar1 - expected_spar1) < 1e-10 and abs(actual_spar2 - expected_spar2) < 1e-10:
                            return True
                        else:
                            return self.create_configuration_file(config_index, expected_spar1, expected_spar2)
            else:
                return self.create_configuration_file(config_index, expected_spar1, expected_spar2)
        except Exception:
            return self.create_configuration_file(config_index, expected_spar1, expected_spar2)

    def run_geometry_generation(self, config_index, spar1_pos, spar2_pos, case="0"):
        try:
            rib_count = self.get_rib_configuration(config_index)
            self.log(f"Генерация геометрии для конфигурации {config_index}, нервюр = {rib_count}")
            if not self.create_configuration_file(config_index, spar1_pos, spar2_pos):
                return False
            if not self.create_rib_count_file(rib_count):
                return False
            result = subprocess.run([sys.executable, 'geom_ansys.py', str(self.file_number), str(config_index)],
                                  capture_output=True, text=True, cwd=self.base_dir, timeout=300)
            if result.returncode == 0:
                cdb_file = f"wing_mesh_config_{config_index}.cdb"
                return os.path.exists(cdb_file)
            else:
                self.log(f"Ошибка генерации геометрии для конфигурации {config_index}")
                return False
        except subprocess.TimeoutExpired:
            self.log(f"Таймаут генерации геометрии для конфигурации {config_index}")
            return False
        except Exception as e:
            self.log(f"Исключение при генерации геометрии: {e}")
            return False

    def run_pressure_interpolation(self, config_index, case="0"):
        try:
            self.log(f"Интерполяция давления для конфигурации {config_index}, случай {case}")
            case_str = str(case)
            result_presion = subprocess.run([sys.executable, 'Interpolador.py', str(self.file_number), str(config_index)],
                                          capture_output=True, text=True, cwd=self.base_dir, timeout=300)
            if result_presion.returncode != 0:
                self.log(f"Ошибка преобразования давления для конфигурации {config_index}")
                return False
            result_nagryzka = subprocess.run([sys.executable, 'nagryzka_ansys.py', f"{config_index}", case_str],
                                           capture_output=True, text=True, cwd=self.base_dir, timeout=300)
            return result_nagryzka.returncode == 0
        except subprocess.TimeoutExpired:
            self.log(f"Таймаут интерполяции давления для конфигурации {config_index}")
            return False
        except Exception as e:
            self.log(f"Исключение при интерполяции давления: {e}")
            return False

    def run_apdl_script_generation(self, config_index, case="0"):
        try:
            self.log(f"Генерация APDL скриптов для конфигурации {config_index}, случай {case}")
            case_str = str(case)
            result = subprocess.run([sys.executable, 'nagryzka_ansys.py', f"{config_index}", case_str],
                                  capture_output=True, text=True, cwd=self.base_dir, timeout=300)
            if result.returncode == 0:
                apdl_file = f"simple_load_{config_index}_{case_str}.apdl"
                return os.path.exists(apdl_file)
            return False
        except subprocess.TimeoutExpired:
            self.log(f"Таймаут генерации APDL скриптов для конфигурации {config_index}")
            return False
        except Exception as e:
            self.log(f"Исключение при генерации APDL скриптов: {e}")
            return False

    def run_ansys_calculation(self, config_index, case="0"):
        try:
            self.log(f"Расчет в ANSYS для конфигурации {config_index}, случай {case}")
            case_str = str(case)
            result = subprocess.run([sys.executable, 'ansys_raschet.py', f"{config_index}", case_str],
                                  capture_output=True, text=True, cwd=self.base_dir, timeout=600)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            self.log(f"Таймаут расчета в ANSYS для конфигурации {config_index}")
            return False
        except Exception as e:
            self.log(f"Исключение при расчете в ANSYS: {e}")
            return False

    def run_baseline_calculation(self, config_index):
        self.log(f"\n{'='*60}")
        self.log(f"БАЗОВЫЙ РАСЧЕТ ДЛЯ КОНФИГУРАЦИИ {config_index}")
        self.log(f"{'='*60}")
        if self.user_spar1 is not None and self.user_spar2 is not None:
            spar1_base = self.user_spar1
            spar2_base = self.user_spar2
        else:
            spar1_base = (self.spar1_bounds[0] + self.spar1_bounds[1]) / 2
            spar2_base = (self.spar2_bounds[0] + self.spar2_bounds[1]) / 2
        all_cases = self.get_all_cases_for_config(config_index)
        case_results = {}
        for case in all_cases:
            self.cleanup_working_directory()
            if not self.create_configuration_file(config_index, spar1_base, spar2_base):
                continue
            if not self.verify_spar_positions(config_index, spar1_base, spar2_base):
                continue
            if not self.run_geometry_generation(config_index, spar1_base, spar2_base, case):
                continue
            if not self.run_pressure_interpolation(config_index, case):
                continue
            if not self.run_apdl_script_generation(config_index, case):
                continue
            if not self.run_ansys_calculation(config_index, case):
                continue
            energy = self.extract_elastic_strain_energy(config_index, case)
            g_val = self.extract_G(config_index, case)
            uz = self.extract_max_uz(config_index, case)
            if energy is not None or g_val is not None:
                case_results[case] = {'energy': energy, 'g': g_val, 'uz_max': uz}
        if not case_results:
            return None, None, {}
        positive_case = self._get_positive_case(all_cases)
        return positive_case, None, case_results

    def _get_positive_case(self, all_cases):
        positive = None
        max_pos = -float('inf')
        for case_str in all_cases:
            try:
                val = float(case_str)
                if val > 0 and val > max_pos:
                    max_pos = val
                    positive = case_str
            except:
                continue
        if positive is None and all_cases:
            positive = all_cases[0]
        return positive

    # ----------------------------------------------------------------------
    # ГЕНЕРАЦИЯ APDL-КОДА ДЛЯ РАСЧЁТА ЭНЕРГИИ ПО СЛОЯМ
    # ----------------------------------------------------------------------
    def _generate_layer_energy_code(self, config_index):
        """Генерирует APDL-код для вычисления упругой энергии каждого слоя
           и максимальных эквивалентных напряжений по всем слоям.
           Имена таблиц:
             SENE_B, SENE_M, SENE_T – энергия нижнего, среднего, верхнего слоёв,
             SENE_TOTAL – суммарная энергия всех слоёв,
             SEQV_MAX   – максимальное эквивалентное напряжение среди трёх слоёв.
           Дополнительно сохраняет поэлементную энергию среднего слоя (заполнителя)
           в текстовый файл core_energy_elem_config_{config_index}.txt.
        """
        return f"""
        ! === ВЫЧИСЛЕНИЕ УПРУГОЙ ЭНЕРГИИ И МАКСИМАЛЬНЫХ НАПРЯЖЕНИЙ ПО СЛОЯМ ===
        *del,energy_bot_total,,nopr
        *del,energy_mid_total,,nopr
        *del,energy_top_total,,nopr
        *del,max_seqv_elem,,nopr
        *dim,energy_bot_total,array,ne
        *dim,energy_mid_total,array,ne
        *dim,energy_top_total,array,ne
        *dim,max_seqv_elem,array,ne
        *do,i,1,ne
            energy_bot_total(i)=0
            energy_mid_total(i)=0
            energy_top_total(i)=0
            max_seqv_elem(i)=0
        *enddo
        
        *do,ls,1,LS_COUNT
            SET,ls
            ! ---------- нижний слой ----------
            shell,bot
            etable,_sx_bot,S,X
            etable,_sy_bot,S,Y
            etable,_sz_bot,S,Z
            etable,_sxy_bot,S,XY
            etable,_syz_bot,S,YZ
            etable,_sxz_bot,S,XZ
            etable,_ex_bot,EPEL,X
            etable,_ey_bot,EPEL,Y
            etable,_ez_bot,EPEL,Z
            etable,_exy_bot,EPEL,XY
            etable,_eyz_bot,EPEL,YZ
            etable,_exz_bot,EPEL,XZ
            ! Эквивалентное напряжение для нижнего слоя
            etable,_seqv_bot,S,EQV
        
            ! ---------- средний слой ----------
            shell,mid
            etable,_sx_mid,S,X
            etable,_sy_mid,S,Y
            etable,_sz_mid,S,Z
            etable,_sxy_mid,S,XY
            etable,_syz_mid,S,YZ
            etable,_sxz_mid,S,XZ
            etable,_ex_mid,EPEL,X
            etable,_ey_mid,EPEL,Y
            etable,_ez_mid,EPEL,Z
            etable,_exy_mid,EPEL,XY
            etable,_eyz_mid,EPEL,YZ
            etable,_exz_mid,EPEL,XZ
            etable,_seqv_mid,S,EQV
        
            ! ---------- верхний слой ----------
            shell,top
            etable,_sx_top,S,X
            etable,_sy_top,S,Y
            etable,_sz_top,S,Z
            etable,_sxy_top,S,XY
            etable,_syz_top,S,YZ
            etable,_sxz_top,S,XZ
            etable,_ex_top,EPEL,X
            etable,_ey_top,EPEL,Y
            etable,_ez_top,EPEL,Z
            etable,_exy_top,EPEL,XY
            etable,_eyz_top,EPEL,YZ
            etable,_exz_top,EPEL,XZ
            etable,_seqv_top,S,EQV
        
            *do,i,1,ne
                *get,area_val,elem,i,area
        
                *get,sx_bot,elem,i,etab,_sx_bot
                *get,sy_bot,elem,i,etab,_sy_bot
                *get,sz_bot,elem,i,etab,_sz_bot
                *get,sxy_bot,elem,i,etab,_sxy_bot
                *get,syz_bot,elem,i,etab,_syz_bot
                *get,sxz_bot,elem,i,etab,_sxz_bot
                *get,ex_bot,elem,i,etab,_ex_bot
                *get,ey_bot,elem,i,etab,_ey_bot
                *get,ez_bot,elem,i,etab,_ez_bot
                *get,exy_bot,elem,i,etab,_exy_bot
                *get,eyz_bot,elem,i,etab,_eyz_bot
                *get,exz_bot,elem,i,etab,_exz_bot
        
                t1 = sx_bot*ex_bot + sy_bot*ey_bot + sz_bot*ez_bot
                t2 = sxy_bot*exy_bot + syz_bot*eyz_bot + sxz_bot*exz_bot
                ed_bot = 0.5*(t1 + t2)
                energy_bot = ed_bot*area_val*t_face(i)
                energy_bot_total(i) = energy_bot_total(i) + energy_bot
        
                *get,sx_mid,elem,i,etab,_sx_mid
                *get,sy_mid,elem,i,etab,_sy_mid
                *get,sz_mid,elem,i,etab,_sz_mid
                *get,sxy_mid,elem,i,etab,_sxy_mid
                *get,syz_mid,elem,i,etab,_syz_mid
                *get,sxz_mid,elem,i,etab,_sxz_mid
                *get,ex_mid,elem,i,etab,_ex_mid
                *get,ey_mid,elem,i,etab,_ey_mid
                *get,ez_mid,elem,i,etab,_ez_mid
                *get,exy_mid,elem,i,etab,_exy_mid
                *get,eyz_mid,elem,i,etab,_eyz_mid
                *get,exz_mid,elem,i,etab,_exz_mid
        
                t1 = sx_mid*ex_mid + sy_mid*ey_mid + sz_mid*ez_mid
                t2 = sxy_mid*exy_mid + syz_mid*eyz_mid + sxz_mid*exz_mid
                ed_mid = 0.5*(t1 + t2)
                energy_mid = ed_mid*area_val*t_core(i)
                energy_mid_total(i) = energy_mid_total(i) + energy_mid
        
                *get,sx_top,elem,i,etab,_sx_top
                *get,sy_top,elem,i,etab,_sy_top
                *get,sz_top,elem,i,etab,_sz_top
                *get,sxy_top,elem,i,etab,_sxy_top
                *get,syz_top,elem,i,etab,_syz_top
                *get,sxz_top,elem,i,etab,_sxz_top
                *get,ex_top,elem,i,etab,_ex_top
                *get,ey_top,elem,i,etab,_ey_top
                *get,ez_top,elem,i,etab,_ez_top
                *get,exy_top,elem,i,etab,_exy_top
                *get,eyz_top,elem,i,etab,_eyz_top
                *get,exz_top,elem,i,etab,_exz_top
        
                t1 = sx_top*ex_top + sy_top*ey_top + sz_top*ez_top
                t2 = sxy_top*exy_top + syz_top*eyz_top + sxz_top*exz_top
                ed_top = 0.5*(t1 + t2)
                energy_top = ed_top*area_val*t_face(i)
                energy_top_total(i) = energy_top_total(i) + energy_top
        
                ! ---- максимальное эквивалентное напряжение среди слоёв ----
                *get,seqv_bot,elem,i,etab,_seqv_bot
                *get,seqv_mid,elem,i,etab,_seqv_mid
                *get,seqv_top,elem,i,etab,_seqv_top
                seqv_max_cur = seqv_bot
                *if,seqv_mid,gt,seqv_max_cur,then
                    seqv_max_cur = seqv_mid
                *endif
                *if,seqv_top,gt,seqv_max_cur,then
                    seqv_max_cur = seqv_top
                *endif
                *if,seqv_max_cur,gt,max_seqv_elem(i),then
                    max_seqv_elem(i) = seqv_max_cur
                *endif
            *enddo
        
            *del,_sx_bot,,nopr
            *del,_sy_bot,,nopr
            *del,_sz_bot,,nopr
            *del,_sxy_bot,,nopr
            *del,_syz_bot,,nopr
            *del,_sxz_bot,,nopr
            *del,_ex_bot,,nopr
            *del,_ey_bot,,nopr
            *del,_ez_bot,,nopr
            *del,_exy_bot,,nopr
            *del,_eyz_bot,,nopr
            *del,_exz_bot,,nopr
            *del,_seqv_bot,,nopr
        
            *del,_sx_mid,,nopr
            *del,_sy_mid,,nopr
            *del,_sz_mid,,nopr
            *del,_sxy_mid,,nopr
            *del,_syz_mid,,nopr
            *del,_sxz_mid,,nopr
            *del,_ex_mid,,nopr
            *del,_ey_mid,,nopr
            *del,_ez_mid,,nopr
            *del,_exy_mid,,nopr
            *del,_eyz_mid,,nopr
            *del,_exz_mid,,nopr
            *del,_seqv_mid,,nopr
        
            *del,_sx_top,,nopr
            *del,_sy_top,,nopr
            *del,_sz_top,,nopr
            *del,_sxy_top,,nopr
            *del,_syz_top,,nopr
            *del,_sxz_top,,nopr
            *del,_ex_top,,nopr
            *del,_ey_top,,nopr
            *del,_ez_top,,nopr
            *del,_exy_top,,nopr
            *del,_eyz_top,,nopr
            *del,_exz_top,,nopr
            *del,_seqv_top,,nopr
        *enddo
        
        total_energy_bot = 0
        total_energy_mid = 0
        total_energy_top = 0
        total_energy_skin = 0
        total_energy_all = 0
        *do,i,1,ne
            total_energy_bot = total_energy_bot + energy_bot_total(i)
            total_energy_mid = total_energy_mid + energy_mid_total(i)
            total_energy_top = total_energy_top + energy_top_total(i)
            total_energy_skin = total_energy_skin + energy_bot_total(i) + energy_top_total(i)
            total_energy_all = total_energy_all + energy_bot_total(i) + energy_mid_total(i) + energy_top_total(i)
        *enddo
        
        ! === СОЗДАНИЕ ТАБЛИЦ ЭНЕРГИИ ПО СЛОЯМ ===
        *del,SENE_B,,nopr
        *del,SENE_M,,nopr
        *del,SENE_T,,nopr
        *del,SENE_TOTAL,,nopr
        etable,SENE_B,VOLU
        etable,SENE_M,VOLU
        etable,SENE_T,VOLU
        etable,SENE_TOTAL,VOLU
        *do,i,1,ne
            detab,i,SENE_B,energy_bot_total(i)
            detab,i,SENE_M,energy_mid_total(i)
            detab,i,SENE_T,energy_top_total(i)
            detab,i,SENE_TOTAL,energy_bot_total(i)+energy_mid_total(i)+energy_top_total(i)
        *enddo
        
        ! === СОЗДАНИЕ ТАБЛИЦЫ МАКСИМАЛЬНЫХ ЭКВИВАЛЕНТНЫХ НАПРЯЖЕНИЙ ===
        *del,SEQV_MAX,,nopr
        etable,SEQV_MAX,VOLU
        *do,i,1,ne
            detab,i,SEQV_MAX,max_seqv_elem(i)
        *enddo
        
        ! === СОХРАНЕНИЕ ГЛОБАЛЬНЫХ ЗНАЧЕНИЙ ЭНЕРГИИ В ФАЙЛЫ ===
        *cfopen,'strain_energy_bot_layer_config_{config_index}','txt'
        *vwrite,total_energy_bot
        (E15.8)
        *cfclose
        *cfopen,'strain_energy_core_layer_config_{config_index}','txt'
        *vwrite,total_energy_mid
        (E15.8)
        *cfclose
        *cfopen,'strain_energy_top_layer_config_{config_index}','txt'
        *vwrite,total_energy_top
        (E15.8)
        *cfclose
        *cfopen,'strain_energy_skin_total_config_{config_index}','txt'
        *vwrite,total_energy_skin
        (E15.8)
        *cfclose
        *cfopen,'strain_energy_total_all_layers_config_{config_index}','txt'
        *vwrite,total_energy_all
        (E15.8)
        *cfclose
        
        SAVE,'optimized_plate_after_buckling',db
        SAVE,'optimized_plate_after_buckling',rst
        
        *msg,UI
        Упругая энергия: нижний слой = %total_energy_bot% Дж, заполнитель = %total_energy_mid% Дж, верхний слой = %total_energy_top% Дж, суммарная энергия обшивок = %total_energy_skin% Дж, общая энергия = %total_energy_all% Дж
        """

    # ----------------------------------------------------------------------
    # ОСНОВНОЙ МЕТОД: ОПТИМИЗАЦИЯ ТОЛЩИН ДЛЯ ВСЕХ СЛУЧАЕВ
    # ----------------------------------------------------------------------
    def run_thickness_optimization_for_config(self, config_index, spar1_pos, spar2_pos, allowable_stress):
        self.log(f"\n{'='*60}")
        self.log(f"ОПТИМИЗАЦИЯ ТОЛЩИН ДЛЯ КОНФИГУРАЦИИ {config_index} (ТРЁХСЛОЙНАЯ ПАНЕЛЬ)")
        self.log(f"{'='*60}")
        all_cases = self.get_all_cases_for_config(config_index)
        self.log(f"Расчётные случаи: {all_cases}")

        if not os.path.exists(f"wing_mesh_config_{config_index}.cdb"):
            if not self.run_geometry_generation(config_index, spar1_pos, spar2_pos):
                self.log("✗ Ошибка генерации геометрии")
                return False

        self.log("Создание единого APDL-файла со всеми load steps...")
        result = subprocess.run([sys.executable, 'nagryzka_ansys.py', str(config_index)],
                                capture_output=True, text=True, cwd=self.base_dir, timeout=300)
        if result.returncode != 0:
            self.log("✗ Ошибка при запуске nagryzka_ansys.py")
            return False

        template_apdl = f"simple_load_{config_index}.apdl"
        if not os.path.exists(template_apdl):
            self.log(f"✗ APDL-файл {template_apdl} не создан")
            return False

        with open(template_apdl, 'r', encoding='utf-8') as f:
            content = f.read()
        ls_count = len(re.findall(r'LSWRITE,\d+', content))
        if ls_count == 0:
            ls_count = len(all_cases)
        self.log(f"Найдено load steps: {ls_count}")

        # ---- Сбор суммарных сил для каждого load step из info_interpolacion ----
        self.wing_area_for_ck = self.configurations[config_index, 0]  # площадь крыла S
        total_force_per_ls = {}
        max_force = 0.0
        for case_str in all_cases:
            info_file = f"info_interpolacion_{config_index}_{case_str}.txt"
            if os.path.exists(info_file):
                with open(info_file, 'r', encoding='utf-8') as f:
                    content_info = f.read()
                    match = re.search(r'Сумма сил исходных данных:\s*([\d.E+-]+)', content_info)
                    if match:
                        force = abs(float(match.group(1)))
                        total_force_per_ls[case_str] = force
                        if force > max_force:
                            max_force = force
            else:
                # если файла нет – приближённо оцениваем по файлу нагрузок
                txt_file = f"resultados_interpolacion_{config_index}_{case_str}.txt"
                if os.path.exists(txt_file):
                    total_force_per_ls[case_str] = 1.0  # заглушка
                else:
                    total_force_per_ls[case_str] = 1.0
        if max_force == 0.0:
            max_force = 1.0
        self.log(f"Максимальная суммарная сила P = {max_force:.3f} Н (используется для Ck)")

        opt_dir = os.path.join(self.results_dir, f"config_{config_index}_best", "thickness_optimization")
        if not self.safe_rmtree(opt_dir):
            self.log(f"✗ Не удалось подготовить директорию {opt_dir}")
            return False
        os.makedirs(opt_dir, exist_ok=True)

        shutil.copy2(template_apdl, os.path.join(opt_dir, os.path.basename(template_apdl)))
        shutil.copy2(f"wing_mesh_config_{config_index}.cdb", opt_dir)
        shutil.copy2("spar_positions.txt", opt_dir)
        if os.path.exists("rib_count.txt"):
            shutil.copy2("rib_count.txt", opt_dir)

        apdl_path = os.path.join(opt_dir, os.path.basename(template_apdl))
        if not self._add_thickness_optimization_to_apdl(apdl_path, config_index, ls_count, all_cases,
                                                        allowable_stress, max_force, self.wing_area_for_ck):
            self.log("✗ Не удалось добавить код оптимизации в APDL")
            return False

        ansys_path = self.get_ansys_path()
        if not ansys_path:
            self.log("✗ ANSYS не найден")
            return False

        original_dir = os.getcwd()
        os.chdir(opt_dir)
        try:
            output_file = f"ansys_thickness_optimization_config_{config_index}.out"
            command = [ansys_path, "-b", "-smp", "-np", str(self.nproc),
                       "-i", os.path.basename(apdl_path), "-o", output_file]
            self.log(f"Запуск ANSYS в {opt_dir} с {self.nproc} потоками")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(timeout=3600)
            time.sleep(0.5)
            if process.returncode == 0:
                self.log("✓ ANSYS завершился успешно")
            else:
                self.log(f"⚠ ANSYS завершился с кодом {process.returncode}")
                with open("ansys_stderr.log", "w") as f:
                    f.write(stderr)
        except subprocess.TimeoutExpired:
            self.log("⚠ Таймаут ANSYS (60 мин)")
            process.kill()
        except Exception as e:
            self.log(f"✗ Ошибка при запуске ANSYS: {e}")
        finally:
            os.chdir(original_dir)

        # Проверка результатов
        mass_file = os.path.join(opt_dir, "mass.txt")
        uz_files = glob.glob(os.path.join(opt_dir, f"max_uz_{config_index}_*.txt"))
        g_files = glob.glob(os.path.join(opt_dir, f"G_*_buckling_config_{config_index}.txt"))
        if os.path.exists(mass_file) and uz_files and g_files:
            self.log("✓ Оптимизация толщин завершена, найдены файлы результатов")
            # Копирование файлов шагов нагружения
            base_src = f"Wing_Load_{config_index}"
            best_dir = os.path.dirname(opt_dir)
            for file in os.listdir(opt_dir):
                if file.startswith(base_src) and re.match(rf"{re.escape(base_src)}\.s\d{{2}}$", file):
                    src_path = os.path.join(opt_dir, file)
                    dst1 = os.path.join(opt_dir, file.replace(base_src, f"thickness_optimized_results_{config_index}"))
                    shutil.copy2(src_path, dst1)
                    dst2 = os.path.join(opt_dir, file.replace(base_src, f"thickness_optimization_complete_{config_index}"))
                    shutil.copy2(src_path, dst2)
                    shutil.copy2(dst1, os.path.join(best_dir, os.path.basename(dst1)))
                    shutil.copy2(dst2, os.path.join(best_dir, os.path.basename(dst2)))
            self._save_optimization_results_txt(opt_dir, config_index)
            self.save_mass_to_txt(config_index, opt_dir, [opt_dir])
            self.save_masses_by_type_to_global(config_index, opt_dir)
            self.save_thickness_by_type_to_global(config_index, opt_dir)
            return True
        else:
            self.log("✗ Не найдены файлы результатов оптимизации")
            return False

    def _add_thickness_optimization_to_apdl(self, apdl_path, config_index, ls_count, all_cases,
                                            allowable_stress, max_force, wing_area):
        try:
            with open(apdl_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            # Находим блок /SOLU ... LSSOLVE
            solu_start = -1
            solu_end = -1
            for i, line in enumerate(lines):
                if line.strip().upper().startswith('/SOLU'):
                    solu_start = i
                if solu_start != -1 and line.strip().upper() == 'FINISH':
                    solu_end = i
                    break
            if solu_start == -1 or solu_end == -1:
                self.log("Не найден блок /SOLU в APDL-файле")
                return False
            lssolve_index = -1
            pre_solve_lines = []
            for i in range(solu_start + 1, solu_end):
                line = lines[i]
                pre_solve_lines.append(line)
                if 'LSSOLVE' in line.upper():
                    lssolve_index = i
                    break
            if lssolve_index == -1:
                self.log("Не найдена команда LSSOLVE в блоке /SOLU")
                return False
            load_step_lines = pre_solve_lines[:len(pre_solve_lines)-1]
            ls_count_actual = sum(1 for line in load_step_lines if 'LSWRITE' in line.upper())
            self.log(f"Найдено load steps: {ls_count_actual}")

            opt_code = self._generate_optimization_code(config_index, ls_count_actual, all_cases,
                                                        allowable_stress, max_force, wing_area, ls_count_actual)
            new_block = ''.join(load_step_lines) + '\n' + opt_code + '\n'
            new_lines = lines[:solu_start] + [new_block] + lines[solu_end+1:]
            with open(apdl_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            self.log("✓ APDL-файл успешно модифицирован для трёхслойной оптимизации с учётом смещений секций и расчёта энергии слоёв")
            return True
        except Exception as e:
            self.log(f"Ошибка при добавлении оптимизации в APDL: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _generate_phase2_code_for_sandwich(self, config_index, ls_count):
        return f"""
    ! === ВТОРАЯ ФАЗА: ОПТИМИЗАЦИЯ ТОЛЬКО ПЕРЕГРУЖЕННЫХ ЭЛЕМЕНТОВ ===
    *if,high_stress_count,gt,0,then
        *msg,UI
        Запускается вторая фаза оптимизации для %high_stress_count% элементов
    
        phase2_it={self.phase2_iterations}
        s_lim_phase2=s_lim
    
        *do,j_phase2,1,phase2_it
            /solu
                LSSOLVE,1,LS_COUNT
            finish
    
            /post1
                ! ----- Очистка старых таблиц -----
                *del,stress_max_phase2,,nopr
                *del,high_stress_elements_phase2,,nopr
                *dim,stress_max_phase2,array,ne
                *dim,high_stress_elements_phase2,array,ne
    
                *do,i,1,ne
                    stress_max_phase2(i)=0
                    high_stress_elements_phase2(i)=0
                *enddo
    
                ! ----- Сбор максимальных напряжений по всем load steps -----
                *do,ls,1,LS_COUNT
                    SET,ls
                    shell,top
                    etable,stress_top_p2,s,eqv
                    shell,mid
                    etable,stress_mid_p2,s,eqv
                    shell,bot
                    etable,stress_bot_p2,s,eqv
    
                    *do,i,1,ne
                        *get,stress_top_val,elem,i,etab,stress_top_p2
                        *get,stress_mid_val,elem,i,etab,stress_mid_p2
                        *get,stress_bot_val,elem,i,etab,stress_bot_p2
                        stress_max_local=stress_top_val
                        *if,stress_mid_val,gt,stress_max_local,then
                            stress_max_local=stress_mid_val
                        *endif
                        *if,stress_bot_val,gt,stress_max_local,then
                            stress_max_local=stress_bot_val
                        *endif
                        *if,stress_max_local,gt,stress_max_phase2(i),then
                            stress_max_phase2(i)=stress_max_local
                        *endif
                    *enddo
                *enddo
    
                ! ----- Определение перегруженных элементов -----
                high_stress_count_phase2=0
                *do,i,1,ne
                    *if,stress_max_phase2(i),gt,s_lim_phase2,then
                        high_stress_elements_phase2(i)=1
                        high_stress_count_phase2=high_stress_count_phase2+1
                    *endif
                *enddo
    
                *cfopen,'phase2_debug_%j_phase2%','txt'
                *vwrite,high_stress_count_phase2
                ('High stress count = ',I8)
                *cfclose
    
                *if,high_stress_count_phase2,eq,0,then
                    *msg,UI
                    Все перегруженные элементы оптимизированы. Вторая фаза завершена.
                    *exit
                *endif
    
                ! ----- Обновление толщин только для перегруженных элементов -----
                *do,i,1,ne
                    *if,high_stress_elements_phase2(i),eq,1,then
                        total_old=total_thick(i)
                        new_total=total_old*stress_max_phase2(i)/s_lim_phase2
                        *get,elem_type,elem,i,attr,type
                        *if,elem_type,eq,1,then
                            new_total=max(min_total_1, min(max_total_1, new_total))
                        *elseif,elem_type,eq,2,then
                            new_total=max(min_total_2, min(max_total_2, new_total))
                        *elseif,elem_type,eq,3,then
                            new_total=max(min_total_3, min(max_total_3, new_total))
                        *elseif,elem_type,eq,4,then
                            new_total=max(min_total_4, min(max_total_4, new_total))
                        *else
                            new_total=max(1e-6, new_total)
                        *endif
                        total_thick(i)=new_total
                        t_face(i)=total_thick(i)*face_ratio
                        t_core(i)=total_thick(i)*core_ratio
                        ! === Применение минимальной толщины несущего слоя ===
                        *if,t_face(i),lt,face_min,then
                            t_face(i)=face_min
                            total_thick(i)=2*t_face(i)+t_core(i)
                        *endif
                    *endif
                *enddo
    
            finish
    
            /prep7
                *do,i,1,ne
                    *if,high_stress_elements_phase2(i),eq,1,then
                        *get,elem_type,elem,i,attr,type
                        *if,elem_type,eq,1,then
                            secoffset_val='BOT'
                        *elseif,elem_type,eq,2,or,elem_type,eq,3,then
                            secoffset_val='MID'
                        *elseif,elem_type,eq,4,then
                            secoffset_val='TOP'
                        *else
                            secoffset_val='MID'
                        *endif
                        sectype,i,shell
                        secdata,t_face(i),1,0,0
                        secdata,t_core(i),2,0,0
                        secdata,t_face(i),1,0,1
                        secoffset,%secoffset_val%
                        emodif,i,secnum,i
                    *endif
                *enddo
            finish
        *enddo
    *else
        *msg,UI
        Вторая фаза не требуется: нет элементов с напряжениями выше предельных
    *endif
    """

    def _generate_optimization_code(self, config_index, ls_count, all_cases, allowable_stress,
                                    max_force, wing_area, actual_ls_count):
        """Генерирует APDL-код с исправленным вычислением G и Ck до и после баклинга."""
        # Блок создания etable для напряжений по трём слоям для всех load steps
        etable_commands = ''
        for i in range(1, actual_ls_count+1):
            etable_commands += f"""
                SET,{i}
                shell,top
                etable,stress_top_{i},s,eqv
                shell,mid
                etable,stress_mid_{i},s,eqv
                shell,bot
                etable,stress_bot_{i},s,eqv
            """
        max_stress_loop = """
                *dim,stress_max_elem,array,ne
                *do,i,1,ne
                    stress_max_elem(i)=0
                *enddo
                *do,ls,1,LS_COUNT
                    SET,ls
                    *do,i,1,ne
                        *get,stress_top_val,elem,i,etab,stress_top_%ls%
                        *get,stress_mid_val,elem,i,etab,stress_mid_%ls%
                        *get,stress_bot_val,elem,i,etab,stress_bot_%ls%
                        stress_max_local=stress_top_val
                        *if,stress_mid_val,gt,stress_max_local,then
                            stress_max_local=stress_mid_val
                        *endif
                        *if,stress_bot_val,gt,stress_max_local,then
                            stress_max_local=stress_bot_val
                        *endif
                        *if,stress_max_local,gt,stress_max_elem(i),then
                            stress_max_elem(i)=stress_max_local
                        *endif
                    *enddo
                *enddo
            """
        # Обновление суммарной толщины total (на основе напряжений)
        thickness_update_total = f"""
                *do,i,1,ne
                    *get,elem_type,elem,i,attr,type
                    total_old=total_thick(i)
                    new_total=total_old*stress_max_elem(i)/s_lim
                    *if,elem_type,eq,1,then
                        new_total = max(min_total_1, min(max_total_1, new_total))
                    *elseif,elem_type,eq,2,then
                        new_total = max(min_total_2, min(max_total_2, new_total))
                    *elseif,elem_type,eq,3,then
                        new_total = max(min_total_3, min(max_total_3, new_total))
                    *elseif,elem_type,eq,4,then
                        new_total = max(min_total_4, min(max_total_4, new_total))
                    *else
                        new_total = max(1e-6, new_total)
                    *endif
                    total_thick(i)=alpha*total_old+(1-alpha)*new_total
                    t_face(i)=total_thick(i)*face_ratio
                    t_core(i)=total_thick(i)*core_ratio
                    *if,t_face(i),lt,face_min,then
                        t_face(i)=face_min
                        total_thick(i)=2*t_face(i)+t_core(i)
                    *endif
                *enddo
            """
        # ---- ИСПРАВЛЕННЫЙ блок вычисления G и Ck ДО баклинга ----
        g_before_block = f"""
            ! === ПРАВИЛЬНОЕ ВЫЧИСЛЕНИЕ G (σ_eqv * V) ДО БАКЛИНГА ===
            /POST1
            ! Создаём таблицы напряжений для всех слоёв и всех load steps
            *do,ls,1,LS_COUNT
                SET,ls
                shell,bot
                etable,stress_bot_%ls%,s,eqv
                shell,mid
                etable,stress_mid_%ls%,s,eqv
                shell,top
                etable,stress_top_%ls%,s,eqv
            *enddo
        
            *del,G_before,,nopr
            G_before = 0
            *do,ls,1,LS_COUNT
                G_ls = 0
                *do,i,1,ne
                    *get,area_val,elem,i,area
                    *get,stress_bot_val,elem,i,etab,stress_bot_%ls%
                    *get,stress_mid_val,elem,i,etab,stress_mid_%ls%
                    *get,stress_top_val,elem,i,etab,stress_top_%ls%
                    vol_bot = area_val*t_face(i)
                    vol_mid = area_val*t_core(i)
                    vol_top = area_val*t_face(i)
                    G_ls = G_ls + stress_bot_val*vol_bot + stress_mid_val*vol_mid + stress_top_val*vol_top
                *enddo
                *if,G_ls,gt,G_before,then
                    G_before = G_ls
                *endif
            *enddo
        
            *cfopen,'G_before_buckling_config_{config_index}','txt'
            *vwrite,G_before
            ('G до баклинга (σ_eqv*V) =', E16.8)
            *cfclose
        
            ! ---- ВЫЧИСЛЕНИЕ БЕЗРАЗМЕРНОГО КОЭФФИЦИЕНТА Ck ДО БАКЛИНГА ----
            P_MAX = {max_force}
            wing_area = {wing_area}
            L = sqrt(wing_area)
            Ck_before = G_before/(P_MAX*L)
            *cfopen,'Ck_before_buckling_config_{config_index}','txt'
            *vwrite,Ck_before
            ('Ck до баклинга =', E16.8)
            *cfclose
            FINISH
        """
    
        # ---- БАКЛИНГ-ОПТИМИЗАЦИЯ (оригинальный код, без изменений) ----
        buckling_code = f"""
            ! ======================================================================
            ! БАКЛИНГ-ОПТИМИЗАЦИЯ (УСКОРЕННАЯ СХОДИМОСТЬ) - РАЗДЕЛЬНЫЕ ФОРМУЛЫ
            ! Группа A (обшивка+зад.стенка): effective = BASE_ENERGY + norm_energy**alpha_A
            ! Группа B (лонжероны+нервюры): effective = norm_energy**alpha_B
            ! ======================================================================
            /POST1
                SET,LAST
                SAVE,'optimized_plate_before_buckling',db
                SAVE,'optimized_plate_before_buckling',rst
            FINISH
        
            MAX_BUCK_ITER    = {self.max_buck_iter}
            BUCK_T_MAX       = {self.buck_t_max}
            N_BUCK_MODES     = {self.n_buck_modes}
            TARGET_BLF       = 1.0
            MAX_REQ_RATIO    = 50000.0
            DAMP_FACTOR      = 0.8
            DAMP_EXP         = 0.7
            MAX_FACTOR       = 50000.0
            MAX_INCREASE_RATIO = 100000.0
            ENERGY_THRESHOLD = {self.buck_energy_threshold}
            BASE_ENERGY      = {self.buck_base_energy}
            ALPHA_A          = {self.buck_power_alpha_A}
            ALPHA_B          = {self.buck_power_alpha_B}
        
            /SOLU
                ANTYPE,BUCKLE
                BUCOPT,LANB,N_BUCK_MODES
                MXPAND,N_BUCK_MODES,,,YES
                SOLVE
            FINISH
        
            /POST1
                *dim,blf_curr,array,N_BUCK_MODES
                *do,imode,1,N_BUCK_MODES
                    SET,1,imode
                    *get,temp_val,FREQ,imode
                    blf_curr(imode) = abs(temp_val)
                *enddo
                *cfopen,'buckling_factors_initial','txt'
                *vwrite,blf_curr(1),blf_curr(2),blf_curr(3),blf_curr(4),blf_curr(5)
                (5E15.8)
                *cfclose
        
                rho_face={self.rho_face}
                rho_core={self.rho_core}
                total_mass_before=0
                *do,i,1,ne
                    *get,area_val,elem,i,area
                    total_mass_before=total_mass_before+area_val*(2*rho_face*t_face(i)+rho_core*t_core(i))
                *enddo
                *cfopen,'mass_before_buckling','txt'
                *vwrite,total_mass_before
                (E15.8)
                *cfclose
            FINISH
        
            *do,buck_iter,1,MAX_BUCK_ITER
        
                min_blf = blf_curr(1)
                active_mode = 1
                *do,imode,2,N_BUCK_MODES
                    *if,blf_curr(imode),lt,min_blf,then
                        min_blf = blf_curr(imode)
                        active_mode = imode
                    *endif
                *enddo
        
                TARGET_BLF_TOL = 0.995
                *if,min_blf,ge,TARGET_BLF_TOL,then
                    *msg,UI
                    Баклинг-оптимизация завершена: BLF=%min_blf% >= %TARGET_BLF_TOL%
                    *exit
                *endif
        
                /POST1
                    SET,1,active_mode
                    ETABLE,SENE_MODE,SENE
                    *dim,sene_elem,array,ne
                    *dim,max_sene_group,array,3   ! группы: 1=A, 2=B, 3=прочие
        
                    *do,igroup,1,3
                        max_sene_group(igroup) = 0
                    *enddo
        
                    *do,i,1,ne
                        *get,sene_elem(i),elem,i,etab,SENE_MODE
                        *get,elem_type_i,elem,i,attr,type
                        *if,elem_type_i,eq,1,or,elem_type_i,eq,4,then
                            group_id = 1
                        *elseif,elem_type_i,eq,2,or,elem_type_i,eq,3,then
                            group_id = 2
                        *else
                            group_id = 3
                        *endif
                        *if,sene_elem(i),gt,max_sene_group(group_id),then
                            max_sene_group(group_id) = sene_elem(i)
                        *endif
                    *enddo
        
                    total_sene = 0
                    *do,i,1,ne
                        total_sene = total_sene + sene_elem(i)
                    *enddo
                    avg_sene = total_sene/ne
                FINISH
        
                *do,i,1,ne
                    *get,elem_type_i,elem,i,attr,type
                    *if,elem_type_i,eq,1,or,elem_type_i,eq,4,then
                        group_id = 1
                    *elseif,elem_type_i,eq,2,or,elem_type_i,eq,3,then
                        group_id = 2
                    *else
                        group_id = 3
                    *endif
        
                    group_max = max_sene_group(group_id)
        
                    *if,group_max,gt,0,then
                        norm_energy = sene_elem(i)/group_max
                    *else
                        norm_energy = 0
                    *endif
        
                    *if,group_id,eq,1,then
                        effective_energy = BASE_ENERGY + (norm_energy**ALPHA_A)
                    *elseif,group_id,eq,2,then
                        effective_energy = norm_energy**ALPHA_B
                    *else
                        effective_energy = norm_energy**ALPHA_B
                    *endif
        
                    req_ratio = TARGET_BLF/min_blf
                    *if,req_ratio,gt,MAX_REQ_RATIO,then
                        req_ratio = MAX_REQ_RATIO
                    *endif
        
                    *if,buck_iter,le,5,and,min_blf,lt,0.2,then
                        factor = sqrt(req_ratio)
                    *else
                        factor = 1+(req_ratio-1)*effective_energy*DAMP_FACTOR
                    *endif
        
                    factor = factor**DAMP_EXP
        
                    *if,factor,gt,MAX_FACTOR,then
                        factor = MAX_FACTOR
                    *endif
                    *if,factor,gt,MAX_INCREASE_RATIO,then
                        factor = MAX_INCREASE_RATIO
                    *endif
        
                    new_t_core = t_core(i)*factor
                    *if,new_t_core,gt,BUCK_T_MAX,then
                        new_t_core = BUCK_T_MAX
                    *endif
                    t_core(i) = new_t_core
        
                    *get,elem_type,elem,i,attr,type
                    *if,elem_type,eq,1,then
                        max_allowed = max_total_1
                    *elseif,elem_type,eq,2,then
                        max_allowed = max_total_2
                    *elseif,elem_type,eq,3,then
                        max_allowed = max_total_3
                    *elseif,elem_type,eq,4,then
                        max_allowed = max_total_4
                    *else
                        max_allowed = {self.total_thickness_max}
                    *endif
                    current_total = 2*t_face(i)+t_core(i)
                    *if,current_total,gt,max_allowed,then
                        t_core(i) = max_allowed-2*t_face(i)
                        *if,t_core(i),lt,0,then
                            t_core(i) = 0
                        *endif
                    *endif
        
                    total_thick(i) = 2*t_face(i)+t_core(i)
                *enddo
        
                /PREP7
                    *do,i,1,ne
                        *get,elem_type,elem,i,attr,type
                        *if,elem_type,eq,1,then
                            secoffset_val = 'BOT'
                        *elseif,elem_type,eq,2,or,elem_type,eq,3,then
                            secoffset_val = 'MID'
                        *elseif,elem_type,eq,4,then
                            secoffset_val = 'TOP'
                        *else
                            secoffset_val = 'MID'
                        *endif
                        sectype,i,shell
                        secdata,t_face(i),1,0,0
                        secdata,t_core(i),2,0,0
                        secdata,t_face(i),1,0,1
                        secoffset,%secoffset_val%
                        emodif,i,secnum,i
                    *enddo
                FINISH
        
                /SOLU
                    LSSOLVE,1,LS_COUNT
                FINISH
        
                /SOLU
                    ANTYPE,BUCKLE
                    BUCOPT,LANB,N_BUCK_MODES
                    MXPAND,N_BUCK_MODES,,,YES
                    SOLVE
                FINISH
        
                /POST1
                    *do,imode,1,N_BUCK_MODES
                        SET,1,imode
                        *get,temp_val,FREQ,imode
                        blf_curr(imode) = abs(temp_val)
                    *enddo
                    *cfopen,'buckling_factors_iter_%buck_iter%','txt'
                    *vwrite,blf_curr(1),blf_curr(2),blf_curr(3),blf_curr(4),blf_curr(5)
                    (5E15.8)
                    *cfclose
                FINISH
        
            *enddo
        
            /SOLU
                LSSOLVE,1,LS_COUNT
            FINISH
        
            /POST1
                SET,LAST
        
                etable,t_face_final,smisc,1
                *do,i,1,ne
                    detab,i,t_face_final,t_face(i)
                *enddo
                etable,t_core_final,smisc,1
                *do,i,1,ne
                    detab,i,t_core_final,t_core(i)
                *enddo
                etable,total_thick_final,smisc,1
                *do,i,1,ne
                    detab,i,total_thick_final,total_thick(i)
                *enddo
        
                total_mass = 0
                mass_skin = 0
                mass_spar = 0
                mass_rib  = 0
                mass_rear = 0
                *do,i,1,ne
                    *get,area_val,elem,i,area
                    elem_mass = area_val*(2*rho_face*t_face(i)+rho_core*t_core(i))
                    total_mass = total_mass + elem_mass
                    *get,elem_type,elem,i,attr,type
                    *if,elem_type,eq,1,then
                        mass_skin = mass_skin + elem_mass
                    *elseif,elem_type,eq,2,then
                        mass_spar = mass_spar + elem_mass
                    *elseif,elem_type,eq,3,then
                        mass_rib = mass_rib + elem_mass
                    *elseif,elem_type,eq,4,then
                        mass_rear = mass_rear + elem_mass
                    *endif
                *enddo
                *cfopen,'mass','txt'
                *vwrite,total_mass
                (E15.8)
                *cfclose
                *cfopen,'masses_by_type_{config_index}','txt'
                *vwrite,mass_skin,mass_spar,mass_rib,mass_rear
                (4E15.8)
                *cfclose
        
                SAVE,'optimized_plate_after_buckling',db
                SAVE,'optimized_plate_after_buckling',rst
        
                *msg,UI
                Баклинг-оптимизация завершена. Итоговая масса = %total_mass% кг
            FINISH
        """
        
        pre_buckling_fix = f"""
            ! === ФИКСАЦИЯ НЕСУЩИХ СЛОЁВ И УСТАНОВКА МИНИМАЛЬНОГО ЗАПОЛНИТЕЛЯ ===
            /PREP7
            core_min = {self.core_thickness_min:.6e}
            face_min = {self.face_min_thickness:.6e}
            *do,i,1,ne
                *if,t_core(i),lt,core_min,then
                    t_core(i) = core_min
                *endif
                *if,t_face(i),lt,face_min,then
                    t_face(i) = face_min
                *endif
                total_thick(i)=2*t_face(i)+t_core(i)
            *enddo
            *do,i,1,ne
                *get,elem_type,elem,i,attr,type
                *if,elem_type,eq,1,then
                    secoffset_val = 'BOT'
                *elseif,elem_type,eq,2,or,elem_type,eq,3,then
                    secoffset_val = 'MID'
                *elseif,elem_type,eq,4,then
                    secoffset_val = 'TOP'
                *else
                    secoffset_val = 'MID'
                *endif
                sectype,i,shell
                secdata,t_face(i),1,0,0
                secdata,t_core(i),2,0,0
                secdata,t_face(i),1,0,1
                secoffset,%secoffset_val%
                emodif,i,secnum,i
            *enddo
            FINISH
        """
    
        # ---- ИСПРАВЛЕННЫЙ блок вычисления G и Ck ПОСЛЕ БАКЛИНГА (с пересозданием таблиц) ----
        g_after_block = f"""
            ! === ПЕРЕСЧЁТ G ПОСЛЕ БАКЛИНГА (С ПЕРЕСОЗДАНИЕМ ТАБЛИЦ НАПРЯЖЕНИЙ) ===
            /POST1
            *do,ls,1,LS_COUNT
                SET,ls
                shell,bot
                etable,stress_bot_%ls%,s,eqv
                shell,mid
                etable,stress_mid_%ls%,s,eqv
                shell,top
                etable,stress_top_%ls%,s,eqv
            *enddo
        
            *del,G_after,,nopr
            G_after = 0
            *do,ls,1,LS_COUNT
                G_ls = 0
                *do,i,1,ne
                    *get,area_val,elem,i,area
                    *get,stress_bot_val,elem,i,etab,stress_bot_%ls%
                    *get,stress_mid_val,elem,i,etab,stress_mid_%ls%
                    *get,stress_top_val,elem,i,etab,stress_top_%ls%
                    vol_bot = area_val*t_face(i)
                    vol_mid = area_val*t_core(i)
                    vol_top = area_val*t_face(i)
                    G_ls = G_ls + stress_bot_val*vol_bot + stress_mid_val*vol_mid + stress_top_val*vol_top
                *enddo
                *if,G_ls,gt,G_after,then
                    G_after = G_ls
                *endif
            *enddo
        
            *cfopen,'G_after_buckling_config_{config_index}','txt'
            *vwrite,G_after
            ('G после баклинга (σ_eqv*V) =', E16.8)
            *cfclose
        
            ! ---- ВЫЧИСЛЕНИЕ БЕЗРАЗМЕРНОГО КОЭФФИЦИЕНТА Ck ПОСЛЕ БАКЛИНГА ----
            P_MAX = {max_force}
            wing_area = {wing_area}
            L = sqrt(wing_area)
            Ck_after = G_after/(P_MAX*L)
            *cfopen,'Ck_after_buckling_config_{config_index}','txt'
            *vwrite,Ck_after
            ('Ck после баклинга =', E16.8)
            *cfclose
            FINISH
        """
    
        # Основной код (первая и вторая фазы) с добавлением G до баклинга
        code = f"""
            ! === ПЕРВАЯ ФАЗА: ОПТИМИЗАЦИЯ СУММАРНОЙ ТОЛЩИНЫ (ПО НАПРЯЖЕНИЯМ) ===
            /prep7
            *get,ne,elem,0,count
        
            it=10
            s_lim={allowable_stress:.15e}
            alpha=0
        
            ! Минимальные и максимальные суммарные толщины для каждого типа элемента (м)
            min_total_1={self.min_total_by_type[1]:.8f}
            max_total_1={self.max_total_by_type[1]:.8f}
            min_total_2={self.min_total_by_type[2]:.8f}
            max_total_2={self.max_total_by_type[2]:.8f}
            min_total_3={self.min_total_by_type[3]:.8f}
            max_total_3={self.max_total_by_type[3]:.8f}
            min_total_4={self.min_total_by_type[4]:.8f}
            max_total_4={self.max_total_by_type[4]:.8f}
        
            face_ratio={self.face_ratio:.6f}
            core_ratio={self.core_ratio:.6f}
            face_min={self.face_min_thickness:.6e}
        
            *del,total_thick,,nopr
            *del,t_face,,nopr
            *del,t_core,,nopr
            *dim,total_thick,array,ne
            *dim,t_face,array,ne
            *dim,t_core,array,ne
        
            init_total = (min_total_1+max_total_1)/2.0
            *do,i,1,ne
                total_thick(i)=init_total
                t_face(i)=total_thick(i)*face_ratio
                t_core(i)=total_thick(i)*core_ratio
                *if,t_face(i),lt,face_min,then
                    t_face(i) = face_min
                    total_thick(i) = 2*t_face(i)+t_core(i)
                *endif
            *enddo
        
            *do,i,1,ne
                *get,elem_type,elem,i,attr,type
                *if,elem_type,eq,1,then
                    secoffset_val = 'BOT'
                *elseif,elem_type,eq,2,or,elem_type,eq,3,then
                    secoffset_val = 'MID'
                *elseif,elem_type,eq,4,then
                    secoffset_val = 'TOP'
                *else
                    secoffset_val = 'MID'
                *endif
                sectype,i,shell
                secdata,t_face(i),1,0,0
                secdata,t_core(i),2,0,0
                secdata,t_face(i),1,0,1
                secoffset,%secoffset_val%
                emodif,i,secnum,i
            *enddo
            finish
        
            LS_COUNT={actual_ls_count}
        
            *do,j,1,it
                /solu
                    LSSOLVE,1,LS_COUNT
                finish
        
                /post1
            {etable_commands}
            {max_stress_loop}
                    high_stress_count=0
                    *dim,high_stress_elements,array,ne
                    *do,i,1,ne
                        *if,stress_max_elem(i),gt,s_lim,then
                            high_stress_elements(i)=1
                            high_stress_count=high_stress_count+1
                        *else
                            high_stress_elements(i)=0
                        *endif
                    *enddo
                    *cfopen,'high_stress_elements_info_{config_index}','txt'
                        *vwrite,high_stress_count
                        ('Элементов с напряжениями выше предельных: ',I8)
                    *cfclose
                finish
        
                /prep7
            {thickness_update_total}
        
                *do,i,1,ne
                    *get,elem_type,elem,i,attr,type
                    *if,elem_type,eq,1,then
                        secoffset_val = 'BOT'
                    *elseif,elem_type,eq,2,or,elem_type,eq,3,then
                        secoffset_val = 'MID'
                    *elseif,elem_type,eq,4,then
                        secoffset_val = 'TOP'
                    *else
                        secoffset_val = 'MID'
                    *endif
                    sectype,i,shell
                    secdata,t_face(i),1,0,0
                    secdata,t_core(i),2,0,0
                    secdata,t_face(i),1,0,1
                    secoffset,%secoffset_val%
                    emodif,i,secnum,i
                *enddo
                finish
            *enddo
        
            /solu
                LSSOLVE,1,LS_COUNT
            finish
        
            /post1
                etable,eras
                *do,i,1,LS_COUNT
                    SET,i
                    shell,top
                    etable,seqv_top_final_%i%,s,eqv
                    shell,mid
                    etable,seqv_mid_final_%i%,s,eqv
                    shell,bot
                    etable,seqv_bot_final_%i%,s,eqv
                *enddo
                SAVE,'thickness_optimized_results_{config_index}',db
                SAVE,'thickness_optimized_results_{config_index}',rst
            finish
        
            {self._generate_phase2_code_for_sandwich(config_index, actual_ls_count)}
        
            {g_before_block}
        
            {pre_buckling_fix}
        
            {buckling_code}
        
            {g_after_block}
        
            /POST1
            {self._generate_layer_energy_code(config_index)}
            FINISH
        """
        return code

    def _save_optimization_results_txt(self, opt_dir, config_index):
        best_dir = os.path.dirname(opt_dir)
        mass_src = os.path.join(opt_dir, "mass.txt")
        if os.path.exists(mass_src):
            shutil.copy2(mass_src, os.path.join(best_dir, f"mass_config_{config_index}.txt"))
        self.log(f"✓ Результаты сохранены в {best_dir}")

    def save_mass_to_txt(self, config_index, excel_dir, target_dirs=None):
        try:
            mass_file = os.path.join(excel_dir, "mass.txt")
            if not os.path.exists(mass_file):
                self.log(f"⚠ Не найден файл mass.txt в {excel_dir}")
                return False
            with open(mass_file, 'r', encoding='utf-8') as f:
                mass_value = float(f.read().strip())
            if target_dirs is None:
                target_dirs = [excel_dir]
            elif isinstance(target_dirs, str):
                target_dirs = [target_dirs]
            for target_dir in target_dirs:
                out_file = os.path.join(target_dir, f"mass_config_{config_index}.txt")
                with open(out_file, 'w', encoding='utf-8') as f:
                    f.write(f"Масса конструкции для конфигурации {config_index}: {mass_value:.6f} кг\n")
            all_mass_file = os.path.join(self.results_dir, "all_masses.txt")
            with open(all_mass_file, 'a', encoding='utf-8') as f:
                f.write(f"{config_index}\t{mass_value:.6f}\n")
            return True
        except Exception as e:
            self.log(f"✗ Ошибка при сохранении массы: {e}")
            return False

    def save_masses_by_type_to_global(self, config_index, opt_dir):
        masses_file = os.path.join(opt_dir, f"masses_by_type_{config_index}.txt")
        if not os.path.exists(masses_file):
            self.log(f"⚠ Файл масс по типам {masses_file} не найден")
            return False
        try:
            with open(masses_file, 'r') as f:
                line = f.readline().strip()
                parts = line.split()
                if len(parts) >= 4:
                    mass_skin = float(parts[0])
                    mass_spar = float(parts[1])
                    mass_rib = float(parts[2])
                    mass_rear = float(parts[3])
                else:
                    return False
        except Exception as e:
            self.log(f"Ошибка чтения {masses_file}: {e}")
            return False
        global_file = os.path.join(self.results_dir, "all_masses_by_type.txt")
        if not os.path.exists(global_file):
            with open(global_file, 'w', encoding='utf-8') as f:
                f.write("config_index\tmass_skin_kg\tmass_spar_kg\tmass_rib_kg\tmass_rearwall_kg\n")
        with open(global_file, 'a', encoding='utf-8') as f:
            f.write(f"{config_index}\t{mass_skin:.6f}\t{mass_spar:.6f}\t{mass_rib:.6f}\t{mass_rear:.6f}\n")
        self.log(f"✓ Массы по типам для конфигурации {config_index} добавлены в {global_file}")
        return True

    def save_thickness_by_type_to_global(self, config_index, opt_dir):
        best_dir = os.path.dirname(opt_dir)
        type_names = {1: 'skin', 2: 'spar', 3: 'rib', 4: 'rearwall'}
        copied_files = []
        for typ, name in type_names.items():
            src = os.path.join(opt_dir, f'thickness_{name}_{config_index}.txt')
            if os.path.exists(src):
                dst = os.path.join(best_dir, f'thickness_{name}_{config_index}.txt')
                shutil.copy2(src, dst)
                copied_files.append(dst)
                self.log(f"✓ Файл толщин для типа '{name}' скопирован: {dst}")
        global_log = os.path.join(self.results_dir, 'all_thickness_files.txt')
        with open(global_log, 'a', encoding='utf-8') as f:
            f.write(f"Конфигурация {config_index}:\n")
            for dst in copied_files:
                f.write(f"  {dst}\n")
        return len(copied_files) > 0

    # ----------------------------------------------------------------------
    # Основной процесс оптимизации для одной конфигурации
    # ----------------------------------------------------------------------
    def optimize_configuration_manual(self, config_index):
        self.log(f"\n{'='*60}")
        self.log(f"ОПТИМИЗАЦИЯ КОНФИГУРАЦИИ {config_index} С ЗАДАННЫМИ ПАРАМЕТРАМИ")
        self.log(f"{'='*60}")

        if config_index >= len(self.configurations):
            self.log(f"Ошибка: индекс {config_index} выходит за пределы")
            return None

        if self.user_spar1 is not None and self.user_spar2 is not None:
            spar1_manual = self.user_spar1
            spar2_manual = self.user_spar2
        else:
            spar1_manual = (self.spar1_bounds[0] + self.spar1_bounds[1]) / 2
            spar2_manual = (self.spar2_bounds[0] + self.spar2_bounds[1]) / 2

        if self.configurations.shape[1] <= 9:
            self.log(f"❌ Ошибка: в файле конфигураций недостаточно столбцов (нужен 9-й столбец)")
            return None
        allowable_stress = self.configurations[config_index, 9] * 1e6
        self.log(f"Допустимое напряжение из конфигурации: {allowable_stress:.3e} Па")

        if self.manual_rib_count is not None:
            rib_cnt = int(round(self.manual_rib_count))
            rib_cnt = max(2, rib_cnt)
            self.create_rib_count_file(rib_cnt)
            self.log(f"Установлено количество нервюр из оптимизатора: {rib_cnt}")

        cdb_file = f"wing_mesh_config_{config_index}.cdb"
        if os.path.exists(cdb_file):
            os.remove(cdb_file)

        if not self.run_geometry_generation(config_index, spar1_manual, spar2_manual):
            self.log("✗ Не удалось сгенерировать геометрию")
            return None

        success = self.run_thickness_optimization_for_config(config_index, spar1_manual, spar2_manual, allowable_stress)

        if success:
            best_dir = os.path.join(self.results_dir, f"config_{config_index}_best")
            mass_file = os.path.join(best_dir, f"mass_config_{config_index}.txt")
            if not os.path.exists(mass_file):
                mass_file = os.path.join(best_dir, "thickness_optimization", f"mass_config_{config_index}.txt")
            try:
                with open(mass_file, 'r', encoding='utf-8') as f:
                    line = f.readline().strip()
                    match = re.search(r'([\d.]+)', line)
                    if match:
                        mass = float(match.group(1))
                        self.best_results[config_index] = {
                            'spar1_position': spar1_manual,
                            'spar2_position': spar2_manual,
                            'rib_count': self.get_rib_configuration(config_index),
                            'mass': mass,
                            'case': 'all'
                        }
                        self.log(f"✓ Оптимизация конфигурации {config_index} завершена, масса = {mass:.4f} кг")
                        return mass
                    else:
                        self.log(f"Не удалось извлечь массу из файла {mass_file}")
                        return None
            except Exception as e:
                self.log(f"Ошибка чтения файла массы {mass_file}: {e}")
                return None
        else:
            self.log(f"✗ Оптимизация конфигурации {config_index} не удалась")
            return None

    def collect_all_results_npy(self, output_file="final_results.npy"):
        self.log("\n" + "="*60)
        self.log("СБОР ИТОГОВЫХ РЕЗУЛЬТАТОВ В .npy")
        self.log("="*60)
        if self.configurations.size == 0:
            self.log("Нет загруженных конфигураций.")
            return
        max_cases = 0
        cases_per_config = {}
        for idx in range(len(self.configurations)):
            cases_float = self.get_all_cases_for_config_with_float(idx)
            cases_per_config[idx] = sorted(cases_float, key=lambda x: x[1], reverse=True)
            max_cases = max(max_cases, len(cases_per_config[idx]))
        if max_cases == 0:
            self.log("Не найдено ни одного расчётного случая.")
            return
        all_rows = []
        for idx in range(len(self.configurations)):
            config = self.configurations[idx]
            wing_area = config[0]
            if config.shape[0] > 13:
                q = config[13]
            else:
                continue
            mass_file = os.path.join(self.results_dir, f"config_{idx}_best", f"mass_config_{idx}.txt")
            if not os.path.exists(mass_file):
                mass = np.nan
            else:
                try:
                    with open(mass_file, 'r') as f:
                        line = f.readline().strip()
                        numbers = re.findall(r"[-+]?\d*\.?\d+E?[-+]?\d*", line)
                        mass = float(numbers[0]) if numbers else np.nan
                except:
                    mass = np.nan
            row_data = [mass]
            cases_for_this = cases_per_config.get(idx, [])
            for case_str, case_float in cases_for_this:
                reac_file = self._find_result_file("reactions", idx, case_float, case_str)
                reaction = self._extract_reaction_from_file(reac_file) if reac_file else np.nan
                target_alpha, target_cy = self._read_config_alpha_cy(idx)
                if target_alpha is not None and abs(case_float - target_alpha) < 1e-3:
                    cy = target_cy
                else:
                    cy = reaction / (q * wing_area) if (not np.isnan(reaction) and q > 0 and wing_area > 0) else np.nan
                g_file = self._find_result_file("G_output", idx, case_float, case_str)
                g_val = self._extract_G_from_file(g_file) if g_file else np.nan
                uz_file = self._find_result_file("max_uz", idx, case_float, case_str)
                uz_val = self._extract_uz_from_file(uz_file) if uz_file else np.nan
                row_data.extend([case_float, reaction, cy, g_val, uz_val])
            missing = max_cases - len(cases_for_this)
            if missing > 0:
                row_data.extend([np.nan] * (5 * missing))
            all_rows.append(row_data)
        if not all_rows:
            return
        final_array = np.array(all_rows, dtype=float)
        np.save(output_file, final_array)
        self.log(f"✅ Итоговые результаты сохранены в {output_file}")

    def _find_result_file(self, base_name, config_index, case_float, case_str):
        opt_dir = os.path.join(self.results_dir, f"config_{config_index}_best", "thickness_optimization")
        candidates = self._generate_filename_candidates(base_name, config_index, case_float, case_str)
        for fname in candidates:
            if opt_dir and os.path.exists(os.path.join(opt_dir, fname)):
                return os.path.join(opt_dir, fname)
            if os.path.exists(fname):
                return fname
        return None

    def _generate_filename_candidates(self, base_name, config_index, case_float, case_str):
        candidates = [f"{base_name}_{config_index}_{case_str}.txt"]
        if '.' in case_str and len(case_str.split('.')[1]) == 1:
            candidates.append(f"{base_name}_{config_index}_{case_float:.2f}.txt")
        if '.' in case_str and len(case_str.split('.')[1]) == 2:
            candidates.append(f"{base_name}_{config_index}_{case_float:.1f}.txt")
        if case_float.is_integer():
            candidates.append(f"{base_name}_{config_index}_{int(case_float)}.txt")
        case_safe = case_str.replace('.', 'p').replace('-', 'm')
        candidates.append(f"{base_name}_{config_index}_{case_safe}.txt")
        return candidates

    def _extract_reaction_from_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if 'TOTAL VALUES' in line:
                        if i+1 < len(lines) and 'VALUE' in lines[i+1]:
                            values = re.findall(r'[-\d.E]+', lines[i+1])
                            if len(values) >= 4:
                                return float(values[3])
            return None
        except:
            return None

    def _extract_G_from_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                patterns = [
                    r'Интеграл G\s*\(σ_eqv \* V\)\s*=\s*([\d.E+-]+)',
                    r'Интеграл G .*?=\s*([\d.E+-]+)',
                    r'G\s*=\s*([\d.E+-]+)'
                ]
                for pattern in patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        return float(match.group(1))
            return None
        except:
            return None

    def _extract_uz_from_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                patterns = [
                    r'Максимальное перемещение по Z \(UZ_MAX\)\s*=\s*([\d.E+-]+)',
                    r'UZ_MAX\s*=\s*([\d.E+-]+)'
                ]
                for pattern in patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        return float(match.group(1))
            return None
        except:
            return None

    def _read_config_alpha_cy(self, config_index):
        filename = f"config_{config_index}_alpha_CL.txt"
        if not os.path.exists(filename):
            return None, None
        try:
            with open(filename, 'r') as f:
                line = f.readline().strip()
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[0]), float(parts[1])
        except:
            pass
        return None, None

    def generate_final_report(self):
        self.end_time = datetime.now()
        self.total_duration = self.end_time - self.start_time
        self.log("\n" + "="*60)
        self.log("ОПТИМИЗАЦИЯ С РУЧНЫМ ЗАДАНИЕМ ПАРАМЕТРОВ ЗАВЕРШЕНА")
        self.log("="*60)
        self.log(f"Время начала: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Время окончания: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Общее время выполнения: {self.total_duration}")
        if self.best_results:
            self.log(f"\nРЕЗУЛЬТАТЫ:")
            for config_index, result in self.best_results.items():
                best_dir = os.path.join(self.results_dir, f"config_{config_index}_best")
                self.log(f"\n  Конфигурация {config_index}:")
                self.log(f"    Позиция лонжерона 1: {result['spar1_position']:.3f}")
                self.log(f"    Позиция лонжерона 2: {result['spar2_position']:.3f}")
                self.log(f"    Количество нервюр: {result['rib_count']}")
                self.log(f"    Масса: {result['mass']:.4f} кг")
                self.log(f"    Файлы сохранены в: {best_dir}")
        else:
            self.log("\n✗ Не удалось найти успешные конфигурации")
        report_file = os.path.join(self.results_dir, 'manual_optimization_summary.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("ОТЧЕТ ПО ОПТИМИЗАЦИИ ТРЁХСЛОЙНОЙ КОНСТРУКЦИИ\n")
            f.write("=" * 50 + "\n")
            f.write(f"Целевая метрика: {self.objective}\n")
            f.write(f"Дата выполнения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Время начала: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Время окончания: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Общее время выполнения: {str(self.total_duration)}\n")
            f.write(f"Всего конфигураций: {len(self.configurations)}\n")
            f.write(f"Всего вычислений: {len(self.results_history)}\n")
            f.write(f"Параметры трёхслойной панели:\n")
            f.write(f"  Минимальная суммарная толщина: {self.total_thickness_min} м\n")
            f.write(f"  Максимальная суммарная толщина: {self.total_thickness_max} м\n")
            f.write(f"  Доля одного несущего слоя: {self.face_ratio}\n")
            f.write(f"  Доля заполнителя: {self.core_ratio}\n")
            f.write(f"  Минимальная толщина несущего слоя: {self.face_min_thickness} м\n")
            f.write(f"  Минимальная толщина заполнителя (численная): {self.core_thickness_min} м\n")
            f.write(f"  Плотность несущего слоя: {self.rho_face} кг/м³\n")
            f.write(f"  Плотность заполнителя: {self.rho_core} кг/м³\n")
            f.write(f"  Максимальная толщина заполнителя при баклинге: {self.buck_t_max} м\n")
            f.write(f"  Порог чувствительности энергии: {self.buck_energy_threshold}\n")
            f.write(f"  Показатель степени для группы A (обшивка+зад.стенка): {self.buck_power_alpha_A}\n")
            f.write(f"  Показатель степени для группы B (лонжероны+нервюры): {self.buck_power_alpha_B}\n")
            f.write(f"  Базовая энергия для группы A: {self.buck_base_energy}\n")
            f.write(f"  Смещения секций:\n")
            f.write(f"    Обшивка (тип 1)   – SECOFFSET,BOT\n")
            f.write(f"    Лонжероны (тип 2) – SECOFFSET,MID\n")
            f.write(f"    Нервюры (тип 3)   – SECOFFSET,MID\n")
            f.write(f"    Задняя стенка (тип 4) – SECOFFSET,TOP\n")
            f.write(f"  Баклинг-оптимизация:\n")
            f.write(f"    Группа A (обшивка+зад.стенка): effective = BASE_ENERGY + norm_energy**{self.buck_power_alpha_A}\n")
            f.write(f"    Группа B (лонжероны+нервюры):   effective = norm_energy**{self.buck_power_alpha_B}\n")
            if self.best_results:
                f.write("\nРЕЗУЛЬТАТЫ:\n")
                for config_index, result in self.best_results.items():
                    f.write(f"\nКонфигурация {config_index}:\n")
                    f.write(f"  Позиция лонжерона 1: {result['spar1_position']:.3f}\n")
                    f.write(f"  Позиция лонжерона 2: {result['spar2_position']:.3f}\n")
                    f.write(f"  Количество нервюр: {result['rib_count']}\n")
                    f.write(f"  Масса: {result['mass']:.4f} кг\n")
        self.log(f"\n✓ Сводный отчет сохранен: {report_file}")

    def run_optimization(self):
        self.start_time = datetime.now()
        self.log("\n" + "="*60)
        self.log("ЗАПУСК ОПТИМИЗАЦИИ ТРЁХСЛОЙНЫХ ПАНЕЛЕЙ (СУММАРНАЯ ТОЛЩИНА)")
        self.log("="*60)
        self.log(f"Всего конфигураций: {len(self.configurations)}")
        self.log(f"Количество потоков ANSYS: {self.nproc}")
        for config_index in range(len(self.configurations)):
            self.optimize_configuration_manual(config_index)
        self.collect_all_results_npy()
        self.generate_final_report()

def main():
    if len(sys.argv) < 2:
        print("Ошибка: необходимо указать номер файла")
        print(__doc__)
        return
    try:
        file_number = int(sys.argv[1])
    except ValueError:
        print("Ошибка: номер файла должен быть целым числом")
        return

    objective = 'energy'
    uz_max_limit = None
    spar1 = None
    spar2 = None
    max_buck_iter = 20
    buck_thick_increase = 1.0
    energy_threshold_factor = 1.0
    buck_t_max = 0.1
    buck_adapt_factor = 5.0
    n_buck_modes = 5
    buck_density_cutoff = 0.1
    buck_filter_radius = 1.5
    use_buck_sensitivity_filter = 1
    rib_count = None
    nproc = None
    total_thickness_min = 0.0005
    total_thickness_max = 0.1
    core_ratio = 1e-9
    face_ratio = (1.0-core_ratio)/2.0
    face_min_thickness = 0.00025
    buck_energy_threshold = 1e-10
    buck_gain = 10000.0
    buck_power_alpha = 0.5          # не используется, для обратной совместимости
    buck_base_energy = 0.1
    buck_power_alpha_A = 0.05        # для группы A (обшивка+зад. стенка)
    buck_power_alpha_B = 0.5        # для группы B (лонжероны+нервюры)

    # Разбор аргументов командной строки
    if len(sys.argv) > 2:
        if sys.argv[2].lower() in ['energy', 'g']:
            objective = sys.argv[2].lower()
    if len(sys.argv) > 3:
        try:
            uz_max_limit = float(sys.argv[3])
        except:
            pass
    if len(sys.argv) > 4:
        try:
            spar1 = float(sys.argv[4])
        except:
            pass
    if len(sys.argv) > 5:
        try:
            spar2 = float(sys.argv[5])
        except:
            pass
    if len(sys.argv) > 6:
        try:
            max_buck_iter = int(sys.argv[6])
        except:
            pass
    if len(sys.argv) > 7:
        try:
            buck_thick_increase = float(sys.argv[7])
        except:
            pass
    if len(sys.argv) > 8:
        try:
            energy_threshold_factor = float(sys.argv[8])
        except:
            pass
    if len(sys.argv) > 9:
        try:
            buck_t_max = float(sys.argv[9])
        except:
            pass
    if len(sys.argv) > 10:
        try:
            buck_adapt_factor = float(sys.argv[10])
        except:
            pass
    if len(sys.argv) > 11:
        try:
            n_buck_modes = int(sys.argv[11])
        except:
            pass
    if len(sys.argv) > 12:
        try:
            buck_density_cutoff = float(sys.argv[12])
        except:
            pass
    if len(sys.argv) > 13:
        try:
            buck_filter_radius = float(sys.argv[13])
        except:
            pass
    if len(sys.argv) > 14:
        try:
            use_buck_sensitivity_filter = int(sys.argv[14])
        except:
            pass
    if len(sys.argv) > 15:
        try:
            rib_count = int(sys.argv[15])
        except:
            pass
    if len(sys.argv) > 16:
        try:
            nproc = int(sys.argv[16])
        except:
            pass
    if len(sys.argv) > 17:
        try:
            total_thickness_min = float(sys.argv[17])
        except:
            pass
    if len(sys.argv) > 18:
        try:
            total_thickness_max = float(sys.argv[18])
        except:
            pass
    if len(sys.argv) > 19:
        try:
            face_ratio = float(sys.argv[19])
            core_ratio = 1.0 - 2*face_ratio
            if core_ratio < 0:
                print("Предупреждение: face_ratio слишком велик, устанавливаем face_ratio=0.5, core_ratio=0")
                face_ratio = 0.5
                core_ratio = 0.0
        except:
            pass
    if len(sys.argv) > 20:
        try:
            core_ratio = float(sys.argv[20])
            face_ratio = (1.0 - core_ratio) / 2.0
            if face_ratio < 0:
                print("Предупреждение: core_ratio слишком велик, устанавливаем core_ratio=0, face_ratio=0.5")
                core_ratio = 0.0
                face_ratio = 0.5
        except:
            pass
    if len(sys.argv) > 21:
        try:
            face_min_thickness = float(sys.argv[21])
        except:
            pass
    if len(sys.argv) > 22:
        try:
            buck_energy_threshold = float(sys.argv[22])
        except:
            pass
    if len(sys.argv) > 23:
        try:
            buck_gain = float(sys.argv[23])
        except:
            pass
    if len(sys.argv) > 24:
        try:
            buck_power_alpha = float(sys.argv[24])
            buck_power_alpha_A = buck_power_alpha
            buck_power_alpha_B = buck_power_alpha
        except:
            pass
    if len(sys.argv) > 25:
        try:
            buck_base_energy = float(sys.argv[25])
        except:
            pass
    if len(sys.argv) > 26:
        try:
            buck_power_alpha_A = float(sys.argv[26])
        except:
            pass
    if len(sys.argv) > 27:
        try:
            buck_power_alpha_B = float(sys.argv[27])
        except:
            pass

    # Проверка корректности долей
    total = 2*face_ratio + core_ratio
    if abs(total - 1.0) > 1e-6:
        print(f"Предупреждение: 2*face_ratio+core_ratio={total} != 1. Корректируем core_ratio.")
        core_ratio = 1.0 - 2*face_ratio
        if core_ratio < 0:
            core_ratio = 0.0
            face_ratio = 0.5
        print(f"  Новые доли: face_ratio={face_ratio}, core_ratio={core_ratio}")

    optimizer = WingOptimizerManualSpars(
        file_number=file_number,
        objective=objective,
        uz_max_limit=uz_max_limit,
        spar1_pos=spar1,
        spar2_pos=spar2,
        max_buck_iter=max_buck_iter,
        buck_thick_increase=buck_thick_increase,
        energy_threshold_factor=energy_threshold_factor,
        buck_t_max=buck_t_max,
        buck_adapt_factor=buck_adapt_factor,
        n_buck_modes=n_buck_modes,
        buck_density_cutoff=buck_density_cutoff,
        buck_filter_radius=buck_filter_radius,
        buck_energy_threshold=buck_energy_threshold,
        buck_gain=buck_gain,
        use_buck_sensitivity_filter=bool(use_buck_sensitivity_filter),
        rib_count=rib_count,
        nproc=nproc,
        total_thickness_min=total_thickness_min,
        total_thickness_max=total_thickness_max,
        face_ratio=face_ratio,
        core_ratio=core_ratio,
        face_min_thickness=face_min_thickness,
        buck_power_alpha=buck_power_alpha,
        buck_base_energy=buck_base_energy,
        buck_power_alpha_A=buck_power_alpha_A,
        buck_power_alpha_B=buck_power_alpha_B
    )
    try:
        optimizer.run_optimization()
    except KeyboardInterrupt:
        optimizer.log("\n✗ Оптимизация прервана пользователем")
        optimizer.end_time = datetime.now()
        if optimizer.start_time:
            optimizer.total_duration = optimizer.end_time - optimizer.start_time
        optimizer.generate_final_report()
    except Exception as e:
        optimizer.log(f"\n✗ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        optimizer.end_time = datetime.now()
        if optimizer.start_time:
            optimizer.total_duration = optimizer.end_time - optimizer.start_time
        optimizer.generate_final_report()

if __name__ == "__main__":
    main()