# -*- coding: utf-8 -*-
"""
Модифицированный скрипт nagryzka_ansys.py
Генерирует один APDL-файл для каждой итерации, содержащий все случаи нагружения
как разные load steps. Постпроцессинг выполняется в едином сеансе POST1,
без лишних выходов, что исключает ошибки "command not recognized".
"""

import os
import math
import glob
import re
import sys
import numpy as np

def parse_nodes_from_cdb(cdb_path):
    print("Чтение узлов из CDB...")
    nodes = {}
    with open(cdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("N,"):
                parts = line.strip().split(",")
                if len(parts) >= 5:
                    try:
                        nid = int(parts[1])
                        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                        nodes[nid] = (x, y, z)
                    except:
                        pass
    if not nodes:
        raise ValueError("❌ Не удалось загрузить узлы из CDB.")
    print(f"✅ Загружено узлов: {len(nodes)}")
    return nodes

def parse_elements_from_cdb(cdb_path):
    print("Чтение элементов из CDB...")
    elements = {}
    element_types = {}
    
    with open(cdb_path, "r", encoding="utf-8", errors="ignore") as f:
        current_element_id = None
        
        for line in f:
            line = line.strip()
            
            if line.startswith("EN,"):
                parts = line.split(",")
                if len(parts) >= 3:
                    try:
                        eid = int(parts[1])
                        node_ids = list(map(int, parts[2:6])) if len(parts) >= 6 else list(map(int, parts[2:]))
                        elements[eid] = node_ids
                        current_element_id = eid
                    except Exception as e:
                        pass
            
            elif line.startswith("EMODIF,") and current_element_id is not None:
                parts = line.split(",")
                if len(parts) >= 4 and parts[2].strip() == "TYPE":
                    try:
                        element_type = int(parts[3].strip())
                        element_types[current_element_id] = element_type
                    except:
                        pass
    
    if not elements:
        raise ValueError("❌ Не удалось загрузить элементы из CDB.")
    
    print(f"✅ Загружено элементов: {len(elements)}")
    
    return elements, element_types

def element_center(nodes, node_ids):
    xs = [nodes[nid][0] for nid in node_ids if nid in nodes]
    ys = [nodes[nid][1] for nid in node_ids if nid in nodes]
    zs = [nodes[nid][2] for nid in node_ids if nid in nodes]
    
    if not xs or not ys or not zs:
        return (0, 0, 0)
    
    return (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))

def identify_upper_skin_elements(elements, element_types, nodes, sections_data=None):
    print("🔍 Идентификация верхних элементов обшивки...")
    
    upper_elements = set()
    
    skin_elements = {}
    for eid, node_ids in elements.items():
        element_type = element_types.get(eid, 1)
        if element_type == 1:
            skin_elements[eid] = node_ids
    
    for eid, node_ids in skin_elements.items():
        cx, cy, cz = element_center(nodes, node_ids)
        
        if sections_data:
            closest_section = None
            min_y_dist = float('inf')
            
            for y_pos, profile in sections_data:
                if abs(y_pos - cy) < min_y_dist:
                    min_y_dist = abs(y_pos - cy)
                    closest_section = profile
            
            if closest_section is not None:
                profile_mean_z = np.mean(closest_section[:,1])
                
                if cz > profile_mean_z:
                    upper_elements.add(eid)
        else:
            element_nodes_z = [nodes[nid][2] for nid in node_ids if nid in nodes]
            if element_nodes_z and sum(element_nodes_z) / len(element_nodes_z) > 0:
                upper_elements.add(eid)
    
    print(f"✅ Верхних элементов обшивки: {len(upper_elements)}")
    return upper_elements

def find_closest_upper_element(x, y, z, elements, element_types, nodes, upper_elements):
    min_dist = float('inf')
    closest_eid = None
    
    for eid, node_ids in elements.items():
        if eid not in upper_elements:
            continue
            
        cx, cy, cz = element_center(nodes, node_ids)
        dist = math.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
        if dist < min_dist:
            min_dist = dist
            closest_eid = eid
    
    return closest_eid

def find_matching_file_pairs(script_dir, target_iteration=None, target_case=None):
    """
    Находит пары файлов для обработки и группирует по итерациям.
    Возвращает список словарей: {'iteration': int, 'cases': [(case_str, case_float, txt_path), ...], 'cdb': cdb_path}
    """
    cdb_files = {}
    for file_path in glob.glob(os.path.join(script_dir, "wing_mesh_config_*.cdb")):
        filename = os.path.basename(file_path)
        match = re.search(r'wing_mesh_config_(\d+)\.cdb$', filename)
        if match:
            iteration = int(match.group(1))
            cdb_files[iteration] = file_path
    
    txt_files_by_iter = {}
    for file_path in glob.glob(os.path.join(script_dir, "resultados_interpolacion_*.txt")):
        filename = os.path.basename(file_path)
        match = re.search(r'resultados_interpolacion_(\d+)_(-?\d+(?:\.\d+)?)\.txt$', filename)
        if match:
            iteration = int(match.group(1))
            case_str = match.group(2)
            try:
                case_float = float(case_str)
            except ValueError:
                case_float = None
            if iteration not in txt_files_by_iter:
                txt_files_by_iter[iteration] = []
            txt_files_by_iter[iteration].append((case_str, case_float, file_path))
    
    groups = []
    if target_iteration is not None:
        iterations = [target_iteration]
    else:
        iterations = sorted(set(cdb_files.keys()) & set(txt_files_by_iter.keys()))
    
    for iteration in iterations:
        if iteration not in cdb_files:
            print(f"❌ Не найден файл геометрии для итерации {iteration}")
            continue
        if iteration not in txt_files_by_iter:
            print(f"❌ Не найдены файлы нагрузок для итерации {iteration}")
            continue
        
        cases = txt_files_by_iter[iteration]
        if target_case is not None:
            target_case_str = str(target_case)
            cases = [c for c in cases if c[0] == target_case_str]
        
        if cases:
            groups.append({
                'iteration': iteration,
                'cdb': cdb_files[iteration],
                'cases': cases
            })
    
    print(f"📊 Найдено групп (итераций): {len(groups)}")
    return groups

def generate_cases_order_file(iteration, cases_list, script_dir):
    """
    Сохраняет порядок следования случаев в текстовый файл.
    """
    order_file = os.path.join(script_dir, f"cases_order_{iteration}.txt")
    with open(order_file, 'w', encoding='utf-8') as f:
        for case_str, _, _ in cases_list:
            f.write(case_str + '\n')
    print(f"   📄 Файл порядка случаев сохранён: {os.path.basename(order_file)}")

def generate_combined_apdl_for_iteration(script_dir, group):
    """
    Создаёт один APDL-файл для итерации, включающий все случаи как разные load steps.
    Постпроцессинг выполняется в едином сеансе POST1.
    """
    iteration = group['iteration']
    cdb_file = group['cdb']
    cases = group['cases']
    
    output_script = os.path.join(script_dir, f"simple_load_{iteration}.apdl")
    
    print(f"🔧 Генерация объединённого APDL для итерации {iteration} (случаев: {len(cases)})")
    
    try:
        nodes = parse_nodes_from_cdb(cdb_file)
        elements, element_types = parse_elements_from_cdb(cdb_file)
        upper_elements = identify_upper_skin_elements(elements, element_types, nodes, None)
        
        load_cmds_per_case = []
        total_pressures = []
        
        for idx, (case_str, case_float, txt_file) in enumerate(cases, start=1):
            print(f"   ⚙️  Случай {idx}: {case_str}")
            load_cmds = []
            total_loads = 0
            applied_loads = 0
            total_pressure = 0.0
            
            with open(txt_file, "r", encoding="utf-8") as f:
                first_line = f.readline()
                if not any(char.isdigit() for char in first_line):
                    pass
                else:
                    f.seek(0)
                
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    line = line.replace(",", ".")
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    
                    total_loads += 1
                    try:
                        x, y, z = map(float, parts[:3])
                        pressure = float(parts[3])
                        
                        eid = find_closest_upper_element(x, y, z, elements, element_types, nodes, upper_elements)
                        
                        if eid is not None:
                            load_cmds.append(f"SFE,{eid},1,PRES,,{pressure:.6f}\n")
                            applied_loads += 1
                            total_pressure += pressure
                    except Exception as e:
                        print(f"      ⚠️ Ошибка обработки строки: {line[:50]}... - {e}")
            
            print(f"      ✅ Нагрузок: {total_loads}, применено: {applied_loads}, сумма давл.: {total_pressure:.2f} Па")
            load_cmds_per_case.append(load_cmds)
            total_pressures.append(total_pressure)
            
            # Сохраняем информацию о случае в отдельный файл (для отчёта)
            info_file = os.path.join(script_dir, f"info_pre_{iteration}_{case_str}.txt")
            with open(info_file, "w", encoding="utf-8") as f:
                f.write(f"Итерация: {iteration}\n")
                f.write(f"Случай: {case_str}\n")
                f.write(f"Файл геометрии: {os.path.basename(cdb_file)}\n")
                f.write(f"Файл нагрузки: {os.path.basename(txt_file)}\n")
                f.write(f"Всего нагрузок: {total_loads}\n")
                f.write(f"Приложено к верхним элементам: {applied_loads}\n")
                f.write(f"Суммарное давление: {total_pressure:.6f} Па\n")
        
        generate_cases_order_file(iteration, cases, script_dir)
        
        geometry_path = os.path.abspath(cdb_file).replace("\\", "/")
        
        # Формируем APDL-код
        apdl_code = f"""! APDL скрипт - объединённый расчёт для итерации {iteration}
! Включает {len(cases)} случаев нагружения как разные load steps
FINISH
/CLEAR,START
/FILNAME,Wing_Load_{iteration},1

/PREP7
CDREAD,DB,'{geometry_path}',,, 

ALLSEL,ALL

/SOLU
ANTYPE,0
"""
        # Добавляем запись load steps
        for idx, (case_str, load_cmds) in enumerate(zip([c[0] for c in cases], load_cmds_per_case), start=1):
            apdl_code += f"""
! --- Load step {idx} для случая {case_str} ---
{''.join(load_cmds)}
LSWRITE,{idx}
! Очищаем нагрузки перед следующим шагом
FCUM, ,    
SFEDELE,ALL,ALL,PRES
"""
        
        # Решаем все load steps
        apdl_code += f"""
LSSOLVE,1,{len(cases)}
FINISH

! === Единый сеанс POST1 для всех load steps ===
/POST1
"""
        # Для каждого load step выводим результаты
        for idx, (case_str, _) in enumerate(zip([c[0] for c in cases], load_cmds_per_case), start=1):
            apdl_code += f"""
! --- Load step {idx} (случай {case_str}) ---
SET,{idx}

! Реакции
/output,reactions_{iteration}_{case_str},txt
PRRSOL
/output

! Эквивалентные напряжения
ETABLE, SEQV, S, EQV
/output,element_stresses_{iteration}_{case_str},txt
PRETAB,SEQV
/output

! Энергия деформаций
ETABLE, SENE, SENE
SSUM
*GET, TOTAL_SE, SSUM, , ITEM, SENE
*CFOPEN,'elastic_strain_energy_{iteration}_{case_str}','txt',' '
*VWRITE,TOTAL_SE
('Суммарная энергия эластичных нагрузок (TOTAL_SE) =', E16.8)
*CFCLOS

! Интеграл G (σ_eqv * V) - вычисляется прямо в POST1
*get,ne,elem,0,count
*dim,elem_stress,array,ne
*dim,elem_volume,array,ne
*do,i,1,ne
    *get,elem_volume(i),elem,i,volu
*enddo
etable,stress_table,s,eqv
*do,i,1,ne
    *get,elem_stress(i),elem,i,etab,stress_table
*enddo
G=0
*do,i,1,ne
    G=G+elem_stress(i)*elem_volume(i)
*enddo
*CFOPEN,'G_output_{iteration}_{case_str}','txt',' '
*VWRITE,G
('Интеграл G (σ_eqv * V) =', E16.8)
*CFCLOS

! Максимальный прогиб по Z
ALLSEL
NSORT, U, Z, 0, 0, 0
*GET, UZ_MAX, SORT, 0, MAX
*CFOPEN,'max_uz_{iteration}_{case_str}','txt'
*VWRITE, UZ_MAX
('Максимальное перемещение по Z (UZ_MAX) =', E16.8)
*CFCLOS

"""
        # Завершаем POST1
        apdl_code += """
FINISH

! === Визуализация последнего load step (опционально) ===
/POST1
SET,LAST
PLNSOL,S,EQV
PLNSOL,U,SUM
/ESHAPE,1
/DSCALE,1
PLLSF
FINISH
"""
        
        with open(output_script, "w", encoding="utf-8") as f:
            f.write(apdl_code)
        
        print(f"   ✅ Объединённый APDL-скрипт создан: {os.path.basename(output_script)}")
        return True
        
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

def main(target_iteration=None, target_case=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"📁 Рабочая директория: {script_dir}")
    
    if len(sys.argv) > 1:
        try:
            if "_" in sys.argv[1]:
                parts = sys.argv[1].split("_")
                target_iteration = int(parts[0])
                if len(parts) > 1:
                    target_case = parts[1]
                print(f"🎯 Обработка итерации {target_iteration}, случай {target_case}")
            else:
                target_iteration = int(sys.argv[1])
                if len(sys.argv) > 2:
                    target_case = sys.argv[2]
                print(f"🎯 Обработка итерации {target_iteration}, случай {target_case}")
        except ValueError:
            print("❌ Ошибка: аргумент должен быть целым числом или в формате 'iteration_case' (например, '100_-5.73')")
            return
    
    groups = find_matching_file_pairs(script_dir, target_iteration, target_case)
    
    if not groups:
        if target_iteration is not None:
            print(f"❌ Не найдены данные для итерации {target_iteration}")
        else:
            print("❌ Не найдены пары файлов CDB и TXT")
        return
    
    successful = 0
    for group in groups:
        if generate_combined_apdl_for_iteration(script_dir, group):
            successful += 1
    
    print(f"🎯 Итог: успешно {successful} из {len(groups)}")

if __name__ == "__main__":
    main()