# -*- coding: utf-8 -*-
"""
geom_ansys.py – генерация конечно-элементной сетки крыла.
Обшивка и задняя стенка – трёхслойные (несущие слои + заполнитель).
Лонжероны и нервюры – однослойные.
Закрепление – кессонное: на корневом сечении закрепляются лонжероны
и обшивка между передним и задним лонжеронами.
"""
import numpy as np
from scipy.spatial import KDTree, Delaunay
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import csv
import math
import os
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
import sys

# Глобальные переменные для хранения узлов и элементов
nodes = []
elements = []
node_dict = {}

# Получаем номер файла из аргументов командной строки
file_number = 1
config_index = None
if len(sys.argv) > 1:
    try:
        file_number = int(sys.argv[1])
    except ValueError:
        print("⚠ Некорректный аргумент, используется номер файла 0")
if len(sys.argv) > 2:
    try:
        config_index = int(sys.argv[2])
    except ValueError:
        pass

class WingMeshGenerator:
    def __init__(self, config=None):
        # Параметры сетки (все размеры в метрах)
        self.params = {
            "skin_thickness": 0.0008,          # толщина одного несущего слоя обшивки
            "spar_thickness": 0.0008,           # толщина лонжерона (однослойного)
            "rib_thickness": 0.0008,           # толщина нервюры (однослойной)
            "core_thickness": 0.00001,           # толщина заполнителя (для трёхслойных панелей)
            "spar_points_multiplier": 15,
            "wing_area": 6,
            "aspect_ratio": 2,
            "taper_ratio": 0.2,
            "sweep_angle": 2,
            "num_sections": 60,
            "min_spar_distance": 0.1,
            "min_rib_distance": 0.08,
            "interpolation_density": 60,
            "leading_edge_density": 0.7,
            "profile_file": "naca2412-il.csv",
            "output_file": "wing_mesh_enhanced.cdb",
            "spar_count": 2,
            "rib_count": 10,
            "rib_mesh_density": 0.025,
            "rib_contour_density": 0.025,
            "spar_positions": None
        }
        
        if config is not None:
            # Новый формат (for_ansys_X.npy) – минимум 12 столбцов
            if len(config) >= 12:
                self.params["wing_area"] = config[0]
                self.params["aspect_ratio"] = config[1]
                self.params["taper_ratio"] = config[2]
                self.params["sweep_angle"] = config[3]
                thickness = config[4]
                thickness_percent = int(round(thickness * 100))
                profile_file = f"naca24{thickness_percent:02d}-il.csv"
                if not os.path.exists(profile_file):
                    print(f"⚠ Предупреждение: файл профиля {profile_file} не найден. Будет использован стандартный naca2412-il.csv")
                    profile_file = "naca2412-il.csv"
                self.params["profile_file"] = profile_file
                
                self.params["spar_count"] = 2
                # Чтение количества нервюр из 16-го столбца (индекс 15)
                if len(config) >= 16:
                    rib_cnt = int(round(config[15]))
                    self.params["rib_count"] = max(2, rib_cnt)
                else:
                    self.params["rib_count"] = 6
                
                spar_pos_1 = config[10]
                spar_pos_2 = config[11]
                self.params["spar_positions"] = sorted([spar_pos_1, spar_pos_2])
            # Старый формат (wing_configurations.npy) – для обратной совместимости
            elif len(config) >= 6:
                self.params["wing_area"] = config[0]
                self.params["aspect_ratio"] = config[1]
                self.params["taper_ratio"] = config[2]
                self.params["sweep_angle"] = config[3]
                self.params["spar_count"] = math.ceil(config[4])
                if len(config) >= 16:
                    rib_cnt = int(round(config[15]))
                    self.params["rib_count"] = max(2, rib_cnt)
                else:
                    self.params["rib_count"] = math.ceil(config[5]) if len(config) > 5 else 10
            else:
                print("⚠ Предупреждение: передан конфигурационный массив недостаточной длины, используются параметры по умолчанию.")
                
        # ---- ДОБАВЛЯЕМ ПРОВЕРКУ ФАЙЛА spar_positions.txt ----
        if os.path.exists("spar_positions.txt"):
            try:
                with open("spar_positions.txt", "r") as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        spar1 = float(lines[0].strip())
                        spar2 = float(lines[1].strip())
                        self.params["spar_positions"] = sorted([spar1, spar2])
                        print(f"✅ Используются позиции лонжеронов из файла: {self.params['spar_positions'][0]:.3f}, {self.params['spar_positions'][1]:.3f}")
                    else:
                        print("⚠ Файл spar_positions.txt содержит недостаточно данных, используются позиции из конфигурации или автоматические.")
            except Exception as e:
                print(f"⚠ Ошибка чтения spar_positions.txt: {e}. Используются позиции из конфигурации или автоматические.")
        
        if os.path.exists("rib_count.txt"):
            try:
                with open("rib_count.txt", "r") as f:
                    line = f.readline().strip()
                    rib_cnt = int(line)
                    self.params["rib_count"] = max(2, rib_cnt)
                    print(f"✅ Используется количество нервюр из файла: {self.params['rib_count']}")
            except Exception as e:
                print(f"⚠ Ошибка чтения rib_count.txt: {e}")
        # ----------------------------------------------------
        
        # Генерация позиций лонжеронов (если не заданы, создаются автоматически)
        self.params["spar_positions"] = self.generate_spar_positions()
        
        self.sections = []
        self.skin_elements = []
        self.current_unique_nodes = []
        self.spar_nodes = []
        self.rear_wall_elements = []
        self.rib_elements_list = []
        self.spar_elements = []
        
        # Смещения передней кромки (будут вычислены в generate_mesh)
        self.leading_edge_offset_x = None
        self.leading_edge_offset_z = None
    
    def generate_spar_positions(self):
        """Возвращает позиции лонжеронов (в долях хорды)."""
        if self.params.get("spar_positions") is not None:
            return self.params["spar_positions"]
        
        spar_count = self.params["spar_count"]
        if os.path.exists("spar_positions.txt"):
            try:
                with open("spar_positions.txt", "r") as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        spar1 = float(lines[0].strip())
                        spar2 = float(lines[1].strip())
                        if 0 <= spar1 <= 1 and 0 <= spar2 <= 1 and spar1 < spar2:
                            print(f"✅ Используются позиции лонжеронов из файла: {spar1}, {spar2}")
                            return [spar1, spar2]
                        else:
                            print(f"⚠️ Некорректные позиции в файле: {spar1}, {spar2}. Используются автоматические позиции.")
                    else:
                        print(f"⚠️ Файл spar_positions.txt содержит недостаточно данных, используются автоматические позиции.")
            except Exception as e:
                print(f"⚠️ Ошибка чтения файла spar_positions.txt: {e}. Используются автоматические позиции.")
        
        if spar_count <= 1:
            positions = [0.3]
        else:
            positions = np.linspace(0.1, 0.9, spar_count).tolist()
        
        print(f"⚠️ Используются автоматические позиции лонжеронов: {positions}")
        return positions
    
    def generate_rib_positions(self, wing_length):
        """Генерирует позиции нервюр в долях размаха (0..1) с учётом минимального расстояния."""
        rib_count = self.params["rib_count"]
        min_dist = self.params["min_rib_distance"]

        if wing_length < min_dist:
            max_ribs = 2
        else:
            max_ribs = int(wing_length // min_dist) + 1

        if rib_count > max_ribs:
            print(f"⚠️ Предупреждение: rib_count={rib_count} превышает максимально допустимое ({max_ribs}) "
                  f"при min_rib_distance={min_dist} м. Будет использовано {max_ribs} нервюр.")
            rib_count = max_ribs

        if rib_count <= 2:
            return [0.0, 1.0]

        step = 1.0 / (rib_count - 1)
        positions = np.linspace(0.0, 1.0, rib_count).tolist()
        actual_step_m = wing_length * step
        if actual_step_m < min_dist - 1e-6:
            max_ribs = int(wing_length // min_dist) + 1
            if max_ribs >= 2:
                step = 1.0 / (max_ribs - 1)
                positions = np.linspace(0.0, 1.0, max_ribs).tolist()
                print(f"⚠️ Автоматически уменьшено количество нервюр до {max_ribs} для соблюдения min_rib_distance={min_dist} м.")
            else:
                positions = [0.0, 1.0]
        return positions

    def find_existing_node(self, x, y, z, tolerance=1e-3):
        """Поиск существующего узла в заданной позиции с учетом допуска"""
        global nodes, node_dict
        node_key = (x, y, z)
        if node_key in node_dict:
            return node_dict[node_key]
        for existing_key, node_id in node_dict.items():
            ex, ey, ez = existing_key
            if (abs(ex - x) < tolerance and 
                abs(ey - y) < tolerance and 
                abs(ez - z) < tolerance):
                return node_id
        return None

    def remove_unused_node(self, node_id):
        """Удаляет неиспользуемый узел"""
        global nodes, node_dict, elements
        node_used = False
        for elem in elements:
            if node_id in elem:
                node_used = True
                break
        if not node_used:
            node_to_remove = None
            for key, nid in node_dict.items():
                if nid == node_id:
                    node_to_remove = key
                    break
            if node_to_remove:
                del node_dict[node_to_remove]

    def generate_rib_quad_mesh_connected(self, rib_profile, rib_y, skin_nodes_on_rib):
        """Генерация сетки нервюр с использованием узлов обшивки для соединения"""
        global nodes, node_dict
        rib_skin_nodes = []
        for skin_layer in skin_nodes_on_rib:
            for node_id in skin_layer:
                node = nodes[node_id-1]
                if abs(node[1] - rib_y) < 1e-3:
                    rib_skin_nodes.append((node_id, node[0], node[2]))
        
        if not rib_skin_nodes:
            return [], np.zeros((0, 0), dtype=int), [], []
        
        rib_skin_nodes.sort(key=lambda x: x[1])
        profile_mean_z = np.mean(rib_profile[:, 1])
        upper_nodes = [n for n in rib_skin_nodes if n[2] > profile_mean_z]
        lower_nodes = [n for n in rib_skin_nodes if n[2] < profile_mean_z]
        
        vertical_lines = []
        x_positions = sorted(list(set([n[1] for n in rib_skin_nodes])))
        
        for x in x_positions:
            upper_for_x = [n for n in upper_nodes if abs(n[1] - x) < 1e-3]
            lower_for_x = [n for n in lower_nodes if abs(n[1] - x) < 1e-3]
            vertical_line_nodes = []
            if upper_for_x and lower_for_x:
                upper_node = upper_for_x[0]
                lower_node = lower_for_x[0]
                vertical_line_nodes.append(lower_node[0])
                num_internal_layers = self.params["spar_points_multiplier"] - 2
                for j in range(1, num_internal_layers + 1):
                    z = lower_node[2] + (upper_node[2] - lower_node[2]) * j / (num_internal_layers + 1)
                    if self.is_point_inside_profile((x, z), rib_profile):
                        existing_node = self.find_existing_node(x, rib_y, z)
                        if existing_node:
                            vertical_line_nodes.append(existing_node)
                        else:
                            node_key = (x, rib_y, z)
                            if node_key not in node_dict:
                                nodes.append([x, rib_y, z])
                                node_dict[node_key] = len(nodes)
                            vertical_line_nodes.append(node_dict[node_key])
                vertical_line_nodes.append(upper_node[0])
                vertical_lines.append({'x': x, 'nodes': vertical_line_nodes})
        
        vertical_lines.sort(key=lambda line: line['x'])
        if vertical_lines:
            num_z_layers = len(vertical_lines[0]['nodes'])
            grid_nodes = np.zeros((len(vertical_lines), num_z_layers), dtype=int)
            for i, line in enumerate(vertical_lines):
                for j, node_id in enumerate(line['nodes']):
                    grid_nodes[i, j] = node_id
        else:
            grid_nodes = np.zeros((0, 0), dtype=int)
        
        rib_elements = []
        additional_nodes = []
        for i in range(len(vertical_lines) - 1):
            for j in range(num_z_layers - 1):
                n1 = grid_nodes[i, j]
                n2 = grid_nodes[i + 1, j]
                n3 = grid_nodes[i + 1, j + 1]
                n4 = grid_nodes[i, j + 1]
                if n1 and n2 and n3 and n4 and len(set([n1, n2, n3, n4])) == 4:
                    p1 = np.array(nodes[n1-1])
                    p2 = np.array(nodes[n2-1])
                    p3 = np.array(nodes[n3-1])
                    p4 = np.array(nodes[n4-1])
                    area = 0.5 * (np.linalg.norm(np.cross(p2-p1, p4-p1)) + 
                                 np.linalg.norm(np.cross(p4-p1, p3-p1)))
                    if area > 1e-8:
                        rib_elements.append((n1, n2, n3, n4))
        
        contour_nodes = [n[0] for n in rib_skin_nodes]
        return contour_nodes, grid_nodes, rib_elements, additional_nodes

    def connect_ribs_to_spars_improved(self, rib_spar_nodes, grid_nodes, rib_y):
        """Улучшенное соединение нервюр с лонжеронами"""
        global nodes
        for spar_nodes in rib_spar_nodes:
            if not spar_nodes:
                continue
            spar_x = nodes[spar_nodes[0]-1][0]
            closest_line_idx = -1
            min_x_diff = float('inf')
            for i in range(grid_nodes.shape[0]):
                if grid_nodes[i, 0] == 0:
                    continue
                line_x = nodes[grid_nodes[i, 0]-1][0]
                x_diff = abs(line_x - spar_x)
                if x_diff < min_x_diff:
                    min_x_diff = x_diff
                    closest_line_idx = i
            if closest_line_idx >= 0 and min_x_diff < 1e-4:
                for j in range(min(len(spar_nodes), grid_nodes.shape[1])):
                    if grid_nodes[closest_line_idx, j] != spar_nodes[j]:
                        old_node = grid_nodes[closest_line_idx, j]
                        new_node = spar_nodes[j]
                        for elem_idx, elem in enumerate(elements):
                            if isinstance(elem, (list, tuple)):
                                new_elem = list(elem)
                                for k in range(len(new_elem)):
                                    if new_elem[k] == old_node:
                                        new_elem[k] = new_node
                                elements[elem_idx] = tuple(new_elem)
                        grid_nodes[closest_line_idx, j] = new_node
                        self.remove_unused_node(old_node)

    def generate_mesh(self, output_filename=None):
        global nodes, elements, node_dict
        nodes = []
        elements = []
        node_dict = {}
        
        if output_filename:
            self.params["output_file"] = output_filename
        
        profile = self.read_airfoil_data(self.params["profile_file"])
        
        actual_wing_area = self.params["wing_area"] / 2
        actual_aspect_ratio = self.params["aspect_ratio"] / 2
        wing_length = math.sqrt(actual_wing_area * actual_aspect_ratio)
        actual_taper_ratio = self.params["taper_ratio"]
        tip_chord = 2 * actual_wing_area / (wing_length * (1 + actual_taper_ratio))
        root_chord = tip_chord * actual_taper_ratio
        
        self.params["rib_positions"] = self.generate_rib_positions(wing_length)
        rib_positions = self.params["rib_positions"].copy()
        spar_positions = self.params["spar_positions"].copy()
        
        all_y_positions = set(y * wing_length for y in rib_positions)
        rib_y_list = sorted(all_y_positions)
        for i in range(len(rib_y_list)-1):
            start_y = rib_y_list[i]
            end_y = rib_y_list[i+1]
            distance = end_y - start_y
            num_segments = max(2, int(distance / (wing_length / self.params["num_sections"])))
            for j in range(1, num_segments):
                y = start_y + j * (distance / num_segments)
                all_y_positions.add(y)
        
        y_positions = sorted(all_y_positions)
        sections = []
        for y in y_positions:
            chord = tip_chord + (root_chord - tip_chord) * (1 - y / wing_length)
            transformed_profile = self.transform_profile(
                profile, 
                chord, 
                self.params["sweep_angle"], 
                y, 
                wing_length
            )
            sections.append((y, transformed_profile))
        
        self.sections = sections

        root_profile = sections[0][1]
        leading_edge_offset_x = -root_profile[np.argmin(root_profile[:,0])][0]
        leading_edge_offset_z = -root_profile[np.argmin(root_profile[:,0])][1]
        self.leading_edge_offset_x = leading_edge_offset_x
        self.leading_edge_offset_z = leading_edge_offset_z

        spar_nodes = []
        for _ in range(len(spar_positions)):
            spar_nodes.append({'top': [], 'bottom': [], 'middle': []})

        spar_node_positions = {}
        for yi, profile in sections:
            min_x = np.min(profile[:,0])
            max_x = np.max(profile[:,0])
            for spar_idx, current_x_rel in enumerate(spar_positions):
                current_x = min_x + (max_x - min_x) * current_x_rel
                intersections = self.find_intersections(profile, current_x)
                if len(intersections) < 2:
                    continue
                z_top = intersections[-1]
                z_bottom = intersections[0]
                top_key = (current_x + leading_edge_offset_x, yi, z_top + leading_edge_offset_z)
                bottom_key = (current_x + leading_edge_offset_x, yi, z_bottom + leading_edge_offset_z)
                if top_key not in node_dict:
                    nodes.append([top_key[0], top_key[1], top_key[2]])
                    node_dict[top_key] = len(nodes)
                spar_nodes[spar_idx]['top'].append(node_dict[top_key])
                if bottom_key not in node_dict:
                    nodes.append([bottom_key[0], bottom_key[1], bottom_key[2]])
                    node_dict[bottom_key] = len(nodes)
                spar_nodes[spar_idx]['bottom'].append(node_dict[bottom_key])
                for j in range(1, self.params["spar_points_multiplier"]-1):
                    z_middle = z_bottom + (z_top - z_bottom) * j / (self.params["spar_points_multiplier"]-1)
                    middle_key = (current_x + leading_edge_offset_x, yi, z_middle + leading_edge_offset_z)
                    if middle_key not in node_dict:
                        nodes.append([middle_key[0], middle_key[1], middle_key[2]])
                        node_dict[middle_key] = len(nodes)
                    spar_nodes[spar_idx]['middle'].append(node_dict[middle_key])
                spar_node_positions[(current_x, yi)] = (node_dict[top_key], node_dict[bottom_key])

        skin_nodes = []
        for yi, profile in sections:
            layer_nodes = []
            min_x = np.min(profile[:,0])
            max_x = np.max(profile[:,0])
            for x, z in profile:
                adjusted_x = x + leading_edge_offset_x
                adjusted_z = z + leading_edge_offset_z
                node_key = (adjusted_x, yi, adjusted_z)
                is_spar_node = False
                for x_rel in spar_positions:
                    spar_x = min_x + (max_x - min_x) * x_rel
                    if abs(x - spar_x) < 1e-6:
                        spar_top_node, spar_bottom_node = spar_node_positions.get((spar_x, yi), (None, None))
                        if spar_top_node and abs(adjusted_z - nodes[spar_top_node-1][2]) < 1e-6:
                            layer_nodes.append(spar_top_node)
                            is_spar_node = True
                            break
                        elif spar_bottom_node and abs(adjusted_z - nodes[spar_bottom_node-1][2]) < 1e-6:
                            layer_nodes.append(spar_bottom_node)
                            is_spar_node = True
                            break
                if not is_spar_node:
                    if node_key in node_dict:
                        layer_nodes.append(node_dict[node_key])
                    else:
                        nodes.append([adjusted_x, yi, adjusted_z])
                        node_dict[node_key] = len(nodes)
                        layer_nodes.append(len(nodes))
            skin_nodes.append(layer_nodes)

        self.attach_skin_to_spars(skin_nodes, spar_nodes, sections)
        skin_elements = self.generate_skin_elements(skin_nodes)
        self.skin_elements = skin_elements

        self.spar_elements = []
        for spar in spar_nodes:
            spar_elems = self.generate_spar_elements(spar, self.params["spar_points_multiplier"])
            self.spar_elements.extend(spar_elems)

        rear_wall_elements = self.generate_rear_wall_elements(skin_nodes, sections, leading_edge_offset_x, leading_edge_offset_z)
        self.rear_wall_elements = rear_wall_elements

        rib_nodes_list = []
        rib_elements_list = []
        rib_grid_nodes_list = []
        
        for rib_y in [y for y in y_positions if y in [rib_y_rel * wing_length for rib_y_rel in self.params["rib_positions"]]]:
            rib_profile = next(profile for y, profile in sections if abs(y - rib_y) < 1e-6)
            rib_profile[:,0] += leading_edge_offset_x
            rib_profile[:,1] += leading_edge_offset_z
            min_x = np.min(rib_profile[:,0])
            max_x = np.max(rib_profile[:,0])
            rib_spar_nodes = []
            for spar_x in [min_x + (max_x - min_x) * x_rel for x_rel in spar_positions]:
                intersections = self.find_intersections(rib_profile, spar_x)
                if len(intersections) >= 2:
                    z_top = intersections[-1]
                    z_bottom = intersections[0]
                    current_spar_nodes = []
                    for j in range(self.params["spar_points_multiplier"]):
                        z = z_bottom + (z_top - z_bottom) * j / (self.params["spar_points_multiplier"] - 1)
                        existing_node = self.find_existing_node(spar_x, rib_y, z)
                        if existing_node:
                            current_spar_nodes.append(existing_node)
                        else:
                            node_key = (spar_x, rib_y, z)
                            if node_key not in node_dict:
                                nodes.append([spar_x, rib_y, z])
                                node_dict[node_key] = len(nodes)
                            current_spar_nodes.append(node_dict[node_key])
                    rib_spar_nodes.append(current_spar_nodes)
            
            skin_nodes_on_rib = []
            for skin_layer in skin_nodes:
                first_node = nodes[skin_layer[0]-1]
                if abs(first_node[1] - rib_y) < 1e-6:
                    skin_nodes_on_rib.append(skin_layer)

            contour_nodes, grid_nodes, rib_elems, additional_nodes = self.generate_rib_quad_mesh_connected(
                rib_profile, rib_y, skin_nodes_on_rib
            )
            self.connect_ribs_to_spars_improved(rib_spar_nodes, grid_nodes, rib_y)
            rib_nodes_list.append(contour_nodes)
            rib_grid_nodes_list.append(grid_nodes)
            rib_elements_list.append(rib_elems)

        self.rib_elements_list = rib_elements_list
        
        elements = skin_elements.copy()
        elements.extend(self.spar_elements)
        elements.extend(rear_wall_elements)
        for rib_elems in rib_elements_list:
            for elem in rib_elems:
                if len(elem) == 4:
                    n1, n2, n3, n4 = elem
                    elements.append((n1, n2, n3, n4))

        self.current_unique_nodes = nodes
        self.spar_nodes = spar_nodes

        self.save_mesh_to_file()
        self.export_upper_element_centers()

    def attach_skin_to_spars(self, skin_nodes, spar_nodes, sections):
        """Привязывает ближайшие узлы обшивки к верхним и нижним узлам лонжеронов"""
        global nodes, elements, node_dict
        for spar in spar_nodes:
            for top_node_id in spar['top']:
                top_node = nodes[top_node_id-1]
                same_y_nodes = []
                for layer in skin_nodes:
                    first_node_in_layer = nodes[layer[0]-1]
                    if abs(first_node_in_layer[1] - top_node[1]) < 1e-6:
                        for node_id in layer:
                            node = nodes[node_id-1]
                            if self.is_upper_surface_node(node, sections):
                                same_y_nodes.append(node_id)
                if not same_y_nodes:
                    continue
                distances = [(node_id, math.sqrt((nodes[node_id-1][0]-top_node[0])**2 + (nodes[node_id-1][2]-top_node[2])**2)) for node_id in same_y_nodes]
                distances.sort(key=lambda x: x[1])
                if distances:
                    closest_node_id = distances[0][0]
                    if closest_node_id != top_node_id:
                        for i, elem in enumerate(elements):
                            if isinstance(elem, (list, tuple)):
                                new_elem = list(elem)
                                for j in range(len(new_elem)):
                                    if new_elem[j] == closest_node_id:
                                        new_elem[j] = top_node_id
                                elements[i] = tuple(new_elem)
                        closest_node_key = (nodes[closest_node_id-1][0], nodes[closest_node_id-1][1], nodes[closest_node_id-1][2])
                        if closest_node_key in node_dict:
                            del node_dict[closest_node_key]
                        for i, layer in enumerate(skin_nodes):
                            for j in range(len(layer)):
                                if layer[j] == closest_node_id:
                                    skin_nodes[i][j] = top_node_id
            for bottom_node_id in spar['bottom']:
                bottom_node = nodes[bottom_node_id-1]
                same_y_nodes = []
                for layer in skin_nodes:
                    first_node_in_layer = nodes[layer[0]-1]
                    if abs(first_node_in_layer[1] - bottom_node[1]) < 1e-6:
                        for node_id in layer:
                            node = nodes[node_id-1]
                            if not self.is_upper_surface_node(node, sections):
                                same_y_nodes.append(node_id)
                if not same_y_nodes:
                    continue
                distances = [(node_id, math.sqrt((nodes[node_id-1][0]-bottom_node[0])**2 + (nodes[node_id-1][2]-bottom_node[2])**2)) for node_id in same_y_nodes]
                distances.sort(key=lambda x: x[1])
                if distances:
                    closest_node_id = distances[0][0]
                    if closest_node_id != bottom_node_id:
                        for i, elem in enumerate(elements):
                            if isinstance(elem, (list, tuple)):
                                new_elem = list(elem)
                                for j in range(len(new_elem)):
                                    if new_elem[j] == closest_node_id:
                                        new_elem[j] = bottom_node_id
                                elements[i] = tuple(new_elem)
                        closest_node_key = (nodes[closest_node_id-1][0], nodes[closest_node_id-1][1], nodes[closest_node_id-1][2])
                        if closest_node_key in node_dict:
                            del node_dict[closest_node_key]
                        for i, layer in enumerate(skin_nodes):
                            for j in range(len(layer)):
                                if layer[j] == closest_node_id:
                                    skin_nodes[i][j] = bottom_node_id

    def is_upper_surface_node(self, node, sections):
        """Определяет, находится ли узел на верхней поверхности крыла"""
        x, y, z = node
        closest_section = None
        min_y_dist = float('inf')
        for section_y, profile in sections:
            if abs(section_y - y) < min_y_dist:
                min_y_dist = abs(section_y - y)
                closest_section = profile
        if closest_section is not None:
            profile_mean_z = np.mean(closest_section[:,1])
            return z > profile_mean_z
        return z > 0

    def generate_rear_wall_elements(self, skin_nodes, sections, offset_x, offset_z):
        rear_wall_elements = []
        for j in range(len(skin_nodes)-1):
            layer1 = skin_nodes[j]
            layer2 = skin_nodes[j+1]
            n1_bottom = layer1[0]
            n1_top = layer1[-1]
            n2_bottom = layer2[0]
            n2_top = layer2[-1]
            if (n1_bottom != n1_top and n1_bottom != n2_bottom and n1_bottom != n2_top and
                n1_top != n2_bottom and n1_top != n2_top and n2_bottom != n2_top):
                rear_wall_elements.append((n1_bottom, n1_top, n2_top, n2_bottom))
        return rear_wall_elements

    def visualize_wing_silhouette(self, config_idx=None):
        if not hasattr(self, 'sections') or not self.sections:
            print("Ошибка: Сначала необходимо сгенерировать сечения")
            return
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        all_points = []
        for i, (y, profile) in enumerate(self.sections):
            root_profile = self.sections[0][1]
            leading_edge_offset_x = -root_profile[np.argmin(root_profile[:,0])][0]
            shifted_profile = profile.copy()
            shifted_profile[:, 0] += leading_edge_offset_x
            for point in shifted_profile:
                all_points.append([point[0], y])
        if len(all_points) > 2:
            all_points = np.array(all_points)
            hull = Delaunay(all_points)
            ax.tripcolor(all_points[:, 0], all_points[:, 1], hull.simplices.copy(), 
                        facecolors=np.ones(len(hull.simplices)), 
                        edgecolors='none', cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_aspect('equal')
        filename = f"wing_silhouette_config_{config_idx}.png" if config_idx is not None else "wing_silhouette.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"Силуэт крыла сохранен в {filename}")
        
    def plot_spars_and_ribs_top_view(self, config_idx=None):
        if not hasattr(self, 'sections') or not self.sections:
            print("Ошибка: Сначала необходимо сгенерировать сечения")
            return
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        actual_wing_area = self.params["wing_area"] / 2
        actual_aspect_ratio = self.params["aspect_ratio"] / 2
        wing_length = math.sqrt(actual_wing_area * actual_aspect_ratio)
        actual_taper_ratio = self.params["taper_ratio"]
        tip_chord = 2 * actual_wing_area / (wing_length * (1 + actual_taper_ratio))
        root_chord = tip_chord * actual_taper_ratio
        y_positions = []
        root_chords = []
        tip_chords = []
        for y, profile in self.sections:
            y_positions.append(y)
            min_x = np.min(profile[:, 0])
            max_x = np.max(profile[:, 0])
            root_chords.append(min_x)
            tip_chords.append(max_x)
        sorted_indices = np.argsort(y_positions)
        y_positions = np.array(y_positions)[sorted_indices]
        root_chords = np.array(root_chords)[sorted_indices]
        tip_chords = np.array(tip_chords)[sorted_indices]
        ax.plot(root_chords, y_positions, 'k-', linewidth=2, label='Передняя кромка')
        ax.plot(tip_chords, y_positions, 'k-', linewidth=2, label='Задняя кромка')
        ax.fill_betweenx(y_positions, root_chords, tip_chords, alpha=0.2, color='blue', label='Крыло')
        spar_positions = self.params["spar_positions"]
        for i, spar_pos in enumerate(spar_positions):
            spar_x = root_chords + (tip_chords - root_chords) * spar_pos
            if i == 0:
                ax.plot(spar_x, y_positions, 'r-', linewidth=3, label='Лонжероны')
            else:
                ax.plot(spar_x, y_positions, 'r-', linewidth=3)
        rib_positions = self.params["rib_positions"]
        for i, rib_pos in enumerate(rib_positions):
            rib_y = rib_pos * wing_length
            closest_idx = np.argmin(np.abs(y_positions - rib_y))
            if closest_idx < len(root_chords):
                x_min = root_chords[closest_idx]
                x_max = tip_chords[closest_idx]
                if i == 0:
                    ax.plot([x_min, x_max], [rib_y, rib_y], 'g-', linewidth=2, label='Нервюры')
                else:
                    ax.plot([x_min, x_max], [rib_y, rib_y], 'g-', linewidth=2)
        ax.set_xlabel('X координата (м)')
        ax.set_ylabel('Y координата (м)')
        ax.set_title('Расположение лонжеронов и нервюр (вид сверху)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
        filename = f"spars_ribs_top_view_config_{config_idx}.png" if config_idx is not None else "spars_ribs_top_view.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"Схема расположения лонжеронов и нервюр сохранена в {filename}")    

    def read_airfoil_data(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        start_idx = None
        for i, line in enumerate(lines):
            if "Airfoil surface" in line:
                start_idx = i + 2
                break
        if start_idx is None:
            raise ValueError("Не удалось найти данные профиля в файле")
        data = []
        for line in lines[start_idx:]:
            line = line.strip()
            if not line or line == ',':
                break
            parts = line.split(',')
            try:
                x = float(parts[0])
                y = float(parts[1])
                data.append((x, y))
            except ValueError:
                continue
        return np.array(data)
    
    def interpolate_profile(self, profile, num_points=100, leading_edge_density=3.0):
        leading_edge_idx = np.argmin(profile[:, 0])
        upper_profile = profile[:leading_edge_idx+1]
        lower_profile = profile[leading_edge_idx:]
        power = leading_edge_density * 2
        def nonlinear_spacing(n, power=2):
            t_linear = np.linspace(0, 1, n)
            return 1 - (1 - t_linear)**power
        if len(upper_profile) > 1:
            upper_x = upper_profile[:, 0]
            upper_z = upper_profile[:, 1]
            n_points_upper = num_points//2
            t_upper = nonlinear_spacing(n_points_upper, power=power)
            new_upper_x = upper_x[0] + t_upper*(upper_x[-1] - upper_x[0])
            upper_interp = interp1d(upper_x, upper_z, kind='cubic')
            new_upper_z = upper_interp(new_upper_x)
            upper_interpolated = np.column_stack((new_upper_x, new_upper_z))
        else:
            upper_interpolated = upper_profile
        if len(lower_profile) > 1:
            lower_x = lower_profile[:, 0][::-1]
            lower_z = lower_profile[:, 1][::-1]
            n_points_lower = num_points//2
            t_lower = nonlinear_spacing(n_points_lower, power=power)
            new_lower_x = lower_x[0] + t_lower*(lower_x[-1] - lower_x[0])
            lower_interp = interp1d(lower_x, lower_z, kind='cubic')
            new_lower_z = lower_interp(new_lower_x)
            lower_interpolated = np.column_stack((new_lower_x, new_lower_z))
            lower_interpolated = lower_interpolated[::-1]
        else:
            lower_interpolated = lower_profile
        interpolated_profile = np.vstack((upper_interpolated, lower_interpolated[1:]))
        return interpolated_profile
    
    def transform_profile(self, points, chord, sweep_angle, y_pos, wing_length):
        interpolated_profile = self.interpolate_profile(
            points, 
            self.params["interpolation_density"],
            self.params["leading_edge_density"]
        )
        leading_edge_idx = np.argmin(interpolated_profile[:, 0])
        leading_edge_x = interpolated_profile[leading_edge_idx, 0]
        min_x, max_x = np.min(interpolated_profile[:,0]), np.max(interpolated_profile[:,0])
        current_chord = max_x - min_x
        scale_factor = chord / current_chord
        scaled_points = interpolated_profile.copy()
        scaled_points[:, 0] = leading_edge_x + (interpolated_profile[:, 0] - leading_edge_x) * scale_factor
        scaled_points[:, 1] = interpolated_profile[:, 1] * scale_factor
        sweep_offset = y_pos * math.tan(math.radians(sweep_angle))
        swept_points = scaled_points.copy()
        swept_points[:, 0] += sweep_offset
        return swept_points
    
    def find_intersections(self, profile, x_val):
        intersections = []
        for i in range(len(profile)-1):
            x1, y1 = profile[i]
            x2, y2 = profile[i+1]
            if (x1 <= x_val <= x2) or (x2 <= x_val <= x1):
                t = (x_val - x1) / (x2 - x1)
                y = y1 + t * (y2 - y1)
                intersections.append(y)
        return sorted(list(set(intersections)))
    
    def generate_spar_elements(self, spar_nodes, spar_points_multiplier):
        elements = []
        num_sections = len(spar_nodes['top'])
        for i in range(num_sections - 1):
            top1 = spar_nodes['top'][i]
            bottom1 = spar_nodes['bottom'][i]
            top2 = spar_nodes['top'][i+1]
            bottom2 = spar_nodes['bottom'][i+1]
            middle_start1 = i * (spar_points_multiplier - 2)
            middle_end1 = middle_start1 + (spar_points_multiplier - 2)
            middle_nodes1 = spar_nodes['middle'][middle_start1:middle_end1]
            middle_start2 = (i + 1) * (spar_points_multiplier - 2)
            middle_end2 = middle_start2 + (spar_points_multiplier - 2)
            middle_nodes2 = spar_nodes['middle'][middle_start2:middle_end2]
            nodes1 = [bottom1] + middle_nodes1 + [top1]
            nodes2 = [bottom2] + middle_nodes2 + [top2]
            if len(nodes1) != len(nodes2):
                continue
            for j in range(len(nodes1) - 1):
                n1 = nodes1[j]
                n2 = nodes1[j+1]
                n3 = nodes2[j+1]
                n4 = nodes2[j]
                if len(set([n1, n2, n3, n4])) == 4:
                    elements.append((n1, n2, n3, n4))
        return elements
    
    def generate_skin_elements(self, skin_nodes):
        elements = []
        for j in range(len(skin_nodes)-1):
            layer1 = skin_nodes[j]
            layer2 = skin_nodes[j+1]
            for i in range(len(layer1)-1):
                n1 = layer1[i]
                n2 = layer1[i+1]
                n3 = layer2[i+1]
                n4 = layer2[i]
                if n1 == n2 or n1 == n3 or n1 == n4 or n2 == n3 or n2 == n4 or n3 == n4:
                    continue
                elements.append((n1, n2, n3, n4))
        return elements
    
    def is_point_inside_profile(self, point, profile_points):
        x, z = point
        crossings = 0
        n = len(profile_points)
        for i in range(n):
            x1, z1 = profile_points[i]
            x2, z2 = profile_points[(i+1)%n]
            if ((z1 <= z and z2 > z) or (z1 > z and z2 <= z)):
                x_intersect = (z - z1) * (x2 - x1) / (z2 - z1) + x1
                if x <= x_intersect:
                    crossings += 1
        return crossings % 2 == 1
    
    def get_all_spar_nodes(self):
        spar_nodes = []
        for spar in self.spar_nodes:
            spar_nodes.extend(spar['top'])
            spar_nodes.extend(spar['bottom'])
            spar_nodes.extend(spar['middle'])
        return spar_nodes
    
    def save_mesh_to_file(self):
        node_types = {}
        for elem in self.skin_elements:
            for node_id in elem:
                node_types[node_id] = 'skin'
        spar_node_ids = self.get_all_spar_nodes()
        for node_id in spar_node_ids:
            node_types[node_id] = 'spar'
        for elem in self.rear_wall_elements:
            for node_id in elem:
                node_types[node_id] = 'skin'
        root_y = 0.0
        for node_id in range(1, len(self.current_unique_nodes) + 1):
            node = self.current_unique_nodes[node_id - 1]
            if abs(node[1] - root_y) < 1e-6 and node_id not in node_types:
                node_types[node_id] = 'rib'
        
        with open(self.params["output_file"], 'w') as f:
            f.write("/PREP7\n")
            f.write("! Сетка крыла: обшивка и задняя стенка – трёхслойные, лонжероны и нервюры – однослойные\n")
            f.write(f"! Параметры: S={self.params['wing_area']}, AR={self.params['aspect_ratio']}, TR={self.params['taper_ratio']}, sweep={self.params['sweep_angle']}\n")
            f.write(f"! Лонжероны: {len(self.params['spar_positions'])} шт. на позициях {self.params['spar_positions']}\n")
            f.write(f"! Нервюры: {len(self.params['rib_positions'])} шт. (с учётом минимального расстояния {self.params['min_rib_distance']} м)\n")
            f.write("! Все узлы объединены для обеспечения целостности сетки\n")
            f.write(f"! Стреловидность по передней кромке: {self.params['sweep_angle']} градусов\n")
            f.write("! ЗАДНЯЯ СТЕНКА: вместо заднего лонжерона используется стенка с свойствами обшивки\n")
            f.write("! НЕРВЮРЫ: построены четырехугольными элементами с согласованными узлами\n")
            f.write("! УЗЛЫ: обеспечено совпадение высоты по Z элементов нервюр и лонжеронов\n")
            f.write("! СОЕДИНЕНИЕ: нервюры используют узлы обшивки для perfect connection\n")
            f.write("! ТРЁХСЛОЙНАЯ ОБШИВКА: слой 1 (нижний несущий), слой 2 (заполнитель), слой 3 (верхний несущий)\n")
            f.write("! ЛОНЖЕРОНЫ И НЕРВЮРЫ: однослойные\n")
            
            f.write("\n! Отключение проверки качества элементов\n")
            f.write("SHPP,OFF\n")
            
            f.write("\n! Узлы\n")
            for i, (x, y, z) in enumerate(self.current_unique_nodes):
                f.write(f"N,{i+1},{x:.6f},{y:.6f},{z:.6f}\n")
            
            f.write("\n! Материалы\n")
            # Материал 1 - несущие слои (алюминий)
            f.write("MP,EX,1,69e9\n")
            f.write("MP,NUXY,1,0.33\n")
            f.write("MP,DENS,1,2700\n")
            # Материал 2 - заполнитель (композит)
            f.write("MP,EX,2,140e6\n")
            f.write("MP,NUXY,2,0.35\n")
            f.write("MP,DENS,2,75\n")
            
            f.write("\n! Определение типов элементов\n")
            # Тип 1: обшивка (трёхслойная, смещение BOT)
            f.write("ET,1,SHELL181\n")
            f.write("KEYOPT,1,8,2\n")    # вывод напряжений в каждом слое
            f.write("SECTYPE,1,SHELL\n")
            f.write(f"SECDATA,{self.params['skin_thickness']},1,0,0\n")   # нижний несущий слой
            f.write(f"SECDATA,{self.params['core_thickness']},2,0,0\n")   # заполнитель
            f.write(f"SECDATA,{self.params['skin_thickness']},1,0,1\n")   # верхний несущий слой
            f.write("SECOFFSET,BOT\n")   # фиксируем нижнюю грань

            # Тип 2: лонжероны (однослойные, смещение MID)
            f.write("\nET,2,SHELL181\n")
            f.write("KEYOPT,2,8,2\n")
            f.write("SECTYPE,2,SHELL\n")
            f.write(f"SECDATA,{self.params['spar_thickness']},1,0,3\n")   # один слой, несущий
            f.write("SECOFFSET,MID\n")

            # Тип 3: нервюры (однослойные, смещение MID)
            f.write("\nET,3,SHELL181\n")
            f.write("KEYOPT,3,8,2\n")
            f.write("SECTYPE,3,SHELL\n")
            f.write(f"SECDATA,{self.params['rib_thickness']},1,0,3\n")   # один слой, несущий
            f.write("SECOFFSET,MID\n")

            # Тип 4: задняя стенка (трёхслойная, смещение TOP)
            f.write("\nET,4,SHELL181\n")
            f.write("KEYOPT,4,8,2\n")
            f.write("SECTYPE,4,SHELL\n")
            f.write(f"SECDATA,{self.params['skin_thickness']},1,0,0\n")   # нижний несущий (со стороны нижней поверхности)
            f.write(f"SECDATA,{self.params['core_thickness']},2,0,0\n")   # заполнитель
            f.write(f"SECDATA,{self.params['skin_thickness']},1,0,1\n")   # верхний несущий
            f.write("SECOFFSET,TOP\n")   # фиксируем верхнюю грань
            
            elem_counter = 1
            
            # Элементы обшивки
            f.write("\n! Элементы обшивки (трёхслойные, смещение BOT)\n")
            for elem in self.skin_elements:
                if len(elem) == 4:
                    n1, n2, n3, n4 = elem
                    f.write(f"EN,{elem_counter},{n1},{n2},{n3},{n4}\n")
                    f.write(f"EMODIF,{elem_counter},MAT,1\n")
                    f.write(f"EMODIF,{elem_counter},TYPE,1\n")
                    f.write(f"EMODIF,{elem_counter},SECNUM,1\n")
                    elem_counter += 1
            
            # Элементы лонжеронов (однослойные)
            f.write("\n! Элементы лонжеронов (однослойные, смещение MID)\n")
            for elem in self.spar_elements:
                if len(elem) == 4:
                    n1, n2, n3, n4 = elem
                    f.write(f"EN,{elem_counter},{n1},{n2},{n3},{n4}\n")
                    f.write(f"EMODIF,{elem_counter},MAT,1\n")
                    f.write(f"EMODIF,{elem_counter},TYPE,2\n")
                    f.write(f"EMODIF,{elem_counter},SECNUM,2\n")
                    elem_counter += 1
            
            # Элементы нервюр (однослойные)
            f.write("\n! Элементы нервюр (однослойные, смещение MID)\n")
            for rib_elems in self.rib_elements_list:
                for elem in rib_elems:
                    if len(elem) == 4:
                        n1, n2, n3, n4 = elem
                        f.write(f"EN,{elem_counter},{n1},{n2},{n3},{n4}\n")
                        f.write(f"EMODIF,{elem_counter},MAT,1\n")
                        f.write(f"EMODIF,{elem_counter},TYPE,3\n")
                        f.write(f"EMODIF,{elem_counter},SECNUM,3\n")
                        elem_counter += 1
            
            # Элементы задней стенки (трёхслойные)
            f.write("\n! Элементы задней стенки (трёхслойные, смещение TOP)\n")
            for elem in self.rear_wall_elements:
                if len(elem) == 4:
                    n1, n2, n3, n4 = elem
                    f.write(f"EN,{elem_counter},{n1},{n2},{n3},{n4}\n")
                    f.write(f"EMODIF,{elem_counter},MAT,1\n")
                    f.write(f"EMODIF,{elem_counter},TYPE,4\n")
                    f.write(f"EMODIF,{elem_counter},SECNUM,4\n")
                    elem_counter += 1
            
            # ========== КЕССОННОЕ ЗАКРЕПЛЕНИЕ ==========
            # Получаем корневое сечение (y=0)
            root_section = None
            for y, profile in self.sections:
                if abs(y) < 1e-6:
                    root_section = profile
                    break
            if root_section is None:
                # fallback: первое сечение
                root_section = self.sections[0][1]

            # Определяем границы кессона на корне в абсолютных координатах (с учетом смещения передней кромки)
            min_x_root = np.min(root_section[:, 0])
            max_x_root = np.max(root_section[:, 0])
            spar_positions = self.params["spar_positions"]
            if len(spar_positions) >= 2:
                x_front_spar = min_x_root + (max_x_root - min_x_root) * spar_positions[0]
                x_rear_spar = min_x_root + (max_x_root - min_x_root) * spar_positions[1]
            else:
                # Если по какой-то причине лонжеронов меньше двух, используем весь диапазон
                x_front_spar = min_x_root
                x_rear_spar = max_x_root

            # Собираем узлы, которые попадают в кессон (лонжероны и обшивка между ними) на корневом сечении
            root_caisson_nodes = []
            for node_id in range(1, len(self.current_unique_nodes) + 1):
                node = self.current_unique_nodes[node_id - 1]
                if abs(node[1]) < 1e-6:  # на корневом сечении
                    node_type = node_types.get(node_id, 'unknown')
                    # Лонжероны всегда закрепляем
                    if node_type == 'spar':
                        root_caisson_nodes.append(node_id)
                    # Обшивку закрепляем только между лонжеронами
                    elif node_type == 'skin':
                        if x_front_spar - 1e-6 <= node[0] <= x_rear_spar + 1e-6:
                            root_caisson_nodes.append(node_id)
                    # Узлы нервюр и задней стенки не закрепляем

            # Запись граничных условий
            f.write("\n! Кессонное закрепление на корневом сечении (лонжероны и обшивка между ними)\n")
            for n in root_caisson_nodes:
                f.write(f"D,{n},ALL,0\n")
        
        print(f"✅ Сетка сохранена в файл: {self.params['output_file']}")
        print(f"   Узлов: {len(self.current_unique_nodes)}")
        print(f"   Элементов обшивки: {len(self.skin_elements)}")
        print(f"   Элементов лонжеронов: {len(self.spar_elements)}")
        print(f"   Элементов нервюр: {sum(len(rib_elems) for rib_elems in self.rib_elements_list)}")
        print(f"   Элементов задней стенки: {len(self.rear_wall_elements)}")
        print(f"   Трёхслойные: обшивка и задняя стенка; однослойные: лонжероны и нервюры")
        print(f"   Закрепление: кессонное (лонжероны + обшивка между ними на корне)")
    
    def export_upper_element_centers(self):
        """Экспорт координат центров элементов верхней обшивки."""
        if not hasattr(self, 'skin_elements') or not self.skin_elements:
            print("Ошибка: Сначала необходимо сгенерировать сетку")
            return
        if not hasattr(self, 'current_unique_nodes'):
            print("Ошибка: Данные узлов не найдены")
            return
        if not hasattr(self, 'sections') or not self.sections:
            print("Ошибка: Данные профилей не найдены")
            return
        if not hasattr(self, 'leading_edge_offset_x'):
            print("Ошибка: Не определено смещение передней кромки")
            return
    
        unique_nodes = self.current_unique_nodes
        upper_centers = []
        from scipy.interpolate import interp1d
    
        section_data = {}
        for y, profile in self.sections:
            le_idx = np.argmin(profile[:, 0])
            upper_pts = profile[:le_idx+1]
            lower_pts = profile[le_idx:]
            upper_pts = upper_pts[np.argsort(upper_pts[:, 0])]
            lower_pts = lower_pts[np.argsort(lower_pts[:, 0])]
            interp_upper = interp1d(upper_pts[:, 0], upper_pts[:, 1],
                                     kind='linear', bounds_error=False,
                                     fill_value=(upper_pts[0, 1], upper_pts[-1, 1]))
            interp_lower = interp1d(lower_pts[:, 0], lower_pts[:, 1],
                                     kind='linear', bounds_error=False,
                                     fill_value=(lower_pts[0, 1], lower_pts[-1, 1]))
            x_min = np.min(profile[:, 0])
            x_max = np.max(profile[:, 0])
            section_data[y] = (interp_upper, interp_lower, x_min, x_max)
    
        for elem in self.skin_elements:
            if len(elem) != 4:
                continue
            node_indices = elem
            coords = [unique_nodes[n - 1] for n in node_indices]
            avg_x = sum(c[0] for c in coords) / 4
            avg_y = sum(c[1] for c in coords) / 4
            avg_z = sum(c[2] for c in coords) / 4
            profile_x = avg_x - self.leading_edge_offset_x
            closest_y = None
            min_dist = float('inf')
            for y in section_data.keys():
                if abs(y - avg_y) < min_dist:
                    min_dist = abs(y - avg_y)
                    closest_y = y
            if closest_y is not None:
                interp_upper, interp_lower, x_min, x_max = section_data[closest_y]
                if profile_x < x_min:
                    profile_x = x_min
                if profile_x > x_max:
                    profile_x = x_max
                z_upper = interp_upper(profile_x)
                z_lower = interp_lower(profile_x)
                z_mid = (z_upper + z_lower) / 2.0
                thickness = z_upper - z_lower
                chord = x_max - x_min
                if chord > 0 and (profile_x - x_min) < 0.15 * chord:
                    eps = max(0.05 * thickness, 1e-4)
                else:
                    eps = max(0.01 * thickness, 1e-5)
                if avg_z >= z_mid - eps:
                    upper_centers.append((avg_x, avg_y, avg_z))
    
        if not upper_centers:
            print("Ошибка: Не найдено элементов верхней поверхности")
            return
        centers_sorted = sorted(upper_centers, key=lambda p: (p[1], p[0]))
        output_file = f"upper_centers_S{self.params['wing_area']}_AR{self.params['aspect_ratio']}_TR{self.params['taper_ratio']}_SW{self.params['sweep_angle']}.txt"
        try:
            with open(output_file, 'w') as f:
                f.write("X\tY\tZ\n")
                for x, y, z in centers_sorted:
                    f.write(f"{x:.6f}\t{y:.6f}\t{z:.6f}\n")
            print(f"Координаты центров элементов верхней обшивки успешно экспортированы в {output_file}")
            print(f"Всего элементов: {len(centers_sorted)}")
        except Exception as e:
            print(f"Ошибка экспорта: {str(e)}")

def process_all_configurations(file_number=0):
    """Обрабатывает все конфигурации из файла for_ansys_{file_number}.npy"""
    data_file = f"for_ansys_{file_number}.npy"
    try:
        data = np.load(data_file)
        print(f"✅ Загружено {len(data)} конфигураций из {data_file}")
    except FileNotFoundError:
        print(f"❌ Ошибка: Файл {data_file} не найден")
        return
    
    for idx, row in enumerate(data):
        if len(row) < 12:
            print(f"⚠ Пропускаем конфигурацию {idx}: недостаточно столбцов (нужно минимум 12, получено {len(row)})")
            continue
        
        print(f"\n{'='*60}")
        print(f"ОБРАБОТКА КОНФИГУРАЦИИ #{idx}")
        print(f"{'='*60}")
        print(f"Площадь: {row[0]:.3f} | Удлинение: {row[1]:.3f} | Сужение: {row[2]:.3f} | Стреловидность: {row[3]:.2f}°")
        print(f"Толщина профиля: {row[4]:.3f} (→ {int(round(row[4]*100)):02d}%)")
        print(f"Лонжерон 1 позиция: {row[10]:.3f} | Лонжерон 2 позиция: {row[11]:.3f}")
        if len(row) >= 16:
            print(f"Количество нервюр: {int(round(row[15]))}")
        else:
            print("Количество нервюр: 6 (по умолчанию)")
        generator = WingMeshGenerator(row)
        output_filename = f"wing_mesh_config_{idx}.cdb"
        try:
            generator.generate_mesh(output_filename)
            print(f"✅ Сетка сохранена в {output_filename}")
            generator.visualize_wing_silhouette(idx)
            generator.plot_spars_and_ribs_top_view(idx)
            expected_centers_file = f"upper_centers_S{row[0]}_AR{row[1]}_TR{row[2]}_SW{row[3]}.txt"
            if os.path.exists(expected_centers_file):
                new_centers_filename = f"{idx}-{idx}.txt"
                if os.path.exists(new_centers_filename):
                    os.remove(new_centers_filename)
                os.rename(expected_centers_file, new_centers_filename)
                print(f"✅ Центры элементов сохранены в {new_centers_filename}")
            else:
                print(f"⚠ Файл центров {expected_centers_file} не найден")
        except Exception as e:
            print(f"❌ Ошибка при генерации сетки для конфигурации {idx}: {str(e)}")
    
    print(f"\n{'='*60}")
    print(f"ОБРАБОТКА ВСЕХ КОНФИГУРАЦИЙ ЗАВЕРШЕНА")
    print(f"{'='*60}")

def process_single_configuration(config_index, file_number=0):
    """Обрабатывает одну конфигурацию по индексу из файла for_ansys_{file_number}.npy"""
    data_file = f"for_ansys_{file_number}.npy"
    try:
        data = np.load(data_file)
        print(f"✅ Загружено {len(data)} конфигураций из {data_file}")
    except FileNotFoundError:
        print(f"❌ Ошибка: Файл {data_file} не найден")
        return
    
    if config_index >= len(data):
        print(f"❌ Ошибка: Индекс {config_index} выходит за пределы (всего {len(data)} конфигураций)")
        return
    
    row = data[config_index]
    if len(row) < 12:
        print(f"❌ Ошибка: конфигурация {config_index} имеет недостаточно столбцов ({len(row)})")
        return
    
    print(f"\n{'='*60}")
    print(f"ОБРАБОТКА КОНФИГУРАЦИИ #{config_index}")
    print(f"{'='*60}")
    print(f"Площадь: {row[0]:.3f} | Удлинение: {row[1]:.3f} | Сужение: {row[2]:.3f} | Стреловидность: {row[3]:.2f}°")
    print(f"Толщина профиля: {row[4]:.3f} (→ {int(round(row[4]*100)):02d}%)")
    print(f"Лонжерон 1 позиция: {row[10]:.3f} | Лонжерон 2 позиция: {row[11]:.3f}")
    
    generator = WingMeshGenerator(row)
    output_filename = f"wing_mesh_config_{config_index}.cdb"
    try:
        generator.generate_mesh(output_filename)
        print(f"✅ Сетка сохранена в {output_filename}")
        generator.visualize_wing_silhouette(config_index)
        generator.plot_spars_and_ribs_top_view(config_index)
        expected_centers_file = f"upper_centers_S{row[0]}_AR{row[1]}_TR{row[2]}_SW{row[3]}.txt"
        if os.path.exists(expected_centers_file):
            new_centers_filename = f"{config_index}-{config_index}.txt"
            if os.path.exists(new_centers_filename):
                os.remove(new_centers_filename)
            os.rename(expected_centers_file, new_centers_filename)
            print(f"✅ Центры элементов сохранены в {new_centers_filename}")
        else:
            print(f"⚠ Файл центров {expected_centers_file} не найден")
    except Exception as e:
        print(f"❌ Ошибка при генерации сетки: {str(e)}")
        
if __name__ == "__main__":
    if config_index is not None:
        print(f"🔧 Режим: обработка одной конфигурации {config_index} из файла for_ansys_{file_number}.npy")
        process_single_configuration(config_index, file_number)
    else:
        print(f"🔧 Режим: обработка всех конфигураций из файла for_ansys_{file_number}.npy")
        process_all_configurations(file_number)