function results = aero_solver(params, target_cy, output_file)
    % aero_solver - аэродинамический расчёт для одной конфигурации
    % Все вычисления выполняются в базовом workspace MATLAB,
    % чтобы избежать проблем со static workspace внутри функции.

    %% 1. Извлечение параметров и перевод в мм
    S_m2 = params.wing_area_m2;
    AR = params.aspect_ratio;
    taper = params.taper_ratio;
    sweep = params.sweep_angle_deg;
    thickness = params.thickness;
    V = params.flight_speed_ms;
    rho = params.air_density_kgm3;

    % Перевод в мм
    S_mm2 = S_m2 * 1e6;
    Lk_mm = 0.5 * sqrt(S_m2 * AR) * 1000;          % полуразмах, мм
    b0_mm = (2 * S_mm2 * taper) / (2 * Lk_mm * (1 + taper));   % корневая хорда, мм
    bk_mm = b0_mm / taper;                                      % концевая хорда, мм

    %% 2. Параметры по умолчанию
    T = 25;                         % температура, °C
    g = 9.8;                         % ускорение свободного падения, м/с²
    visc = 1.98e-5;                  % кинематическая вязкость, м²/с
    lzz = 1;                         % отношение сторон ячейки (размах/хорда)
    raz_chord = 1/15;                % число разбиений по хорде (обратная величина)
    sl = 20;                          % коэффициент следа
    a_k = 2;                          % угол крутки в концевом сечении, град
    n_t = 1;                          % степень закона крутки по размаху
    f_0 = 0.02;                       % кривизна в корневом сечении
    f_k = 0.02;                       % кривизна в концевом сечении
    a_t = 1;                          % степень закона изменения кривизны
    m0 = 390000;                       % взлётная масса, кг (не используется)
    P0 = 0.000625;                     % удельная нагрузка, кг/мм² (не используется)
    str_25 = 0;                        % угол стреловидности на 0,25 хорды
    PROF_type = 0;                     % тип профиля
    S_WF_ = 0.15;                      % относительная подфюзеляжная площадь

    % Производные параметры сетки
    raz_razmax = ceil(Lk_mm / (raz_chord * ((b0_mm + bk_mm) / 2) * lzz));
    shag = Lk_mm / raz_razmax;        % шаг по размаху, мм
    nraz = 2 * raz_razmax;            % число панелей по размаху

    %% 3. Запись всех переменных в базовое workspace
    assignin('base', 'T', T);
    assignin('base', 'g', g);
    assignin('base', 'visc', visc);
    assignin('base', 'S', S_mm2);
    assignin('base', 'lz', AR);
    assignin('base', 'str', sweep);
    assignin('base', 'nu', taper);
    assignin('base', 'c_', thickness);
    assignin('base', 'ck_', thickness);
    assignin('base', 'Vpol', V);
    assignin('base', 'plotn', rho);
    assignin('base', 'Lk', Lk_mm);
    assignin('base', 'b0', b0_mm);
    assignin('base', 'bk', bk_mm);
    assignin('base', 'lzz', lzz);
    assignin('base', 'raz_chord', raz_chord);
    assignin('base', 'raz_razmax', raz_razmax);
    assignin('base', 'shag', shag);
    assignin('base', 'sl', sl);
    assignin('base', 'a_k', a_k);
    assignin('base', 'n_t', n_t);
    assignin('base', 'f_0', f_0);
    assignin('base', 'f_k', f_k);
    assignin('base', 'a_t', a_t);
    assignin('base', 'm0', m0);
    assignin('base', 'P0', P0);
    assignin('base', 'str_25', str_25);
    assignin('base', 'PROF_type', PROF_type);
    assignin('base', 'S_WF_', S_WF_);

    %% 4. Выполнение скриптов построения геометрии и матриц влияния (в базовом workspace)
    evalin('base', 'A_Prof');
    evalin('base', 'B_node_aero');
    evalin('base', 'C_Elem_aero');
    evalin('base', 'D_aero_twist_aero');
    evalin('base', 'F_twist_aero');
    evalin('base', 'E_Normal_aero');
    evalin('base', 'G_dopnode_aero');
    evalin('base', 'H_aero_matrix');

    %% 5. Вспомогательная функция для решения при заданном угле
    % (используем evalin для выполнения I_solver и извлечения результатов)
    function [Cy, Cx, Cxi, Ki] = solve_alpha(alpha_deg)
        assignin('base', 'alpha', alpha_deg);
        evalin('base', 'I_solver');
        Cy   = evalin('base', 'Cy');
        Cx   = evalin('base', 'Cx');
        Cxi  = evalin('base', 'Cxi');
        Ki   = evalin('base', 'Ki');
    end

    %% 6. Подбор угла атаки по целевому Cy (линейная интерполяция)
    alpha1 = -5.0;
    alpha2 = 10.0;

    [Cy1, ~, ~, ~] = solve_alpha(alpha1);
    [Cy2, ~, ~, ~] = solve_alpha(alpha2);

    if abs(Cy2 - Cy1) < 1e-6
        alpha_target = alpha1;
    else
        alpha_target = alpha1 + (target_cy - Cy1) * (alpha2 - alpha1) / (Cy2 - Cy1);
    end

    % Финальный расчёт при найденном угле
    [Cy_target, Cx_target, Cxi_target, Ki_target] = solve_alpha(alpha_target);

    %% 7. Сохранение распределения сил в текстовый файл
    % Извлекаем необходимые переменные из базового workspace
    NODDDOW = evalin('base', 'NODDDOW');
    % Проверяем, какая переменная с силами есть: delta_Y_z или delta_Y
    if evalin('base', 'exist(''delta_Y_z'', ''var'')')
        force_var = evalin('base', 'delta_Y_z');
    else
        force_var = evalin('base', 'delta_Y');
    end
    Lk = evalin('base', 'Lk');
    sym_line_y = 2 * Lk;  % ось симметрии

    % Собираем данные для правой консоли (y > sym_line_y)
    data = [];
    for i = 1:size(NODDDOW,1)
        y_mm = NODDDOW(i,2);
        if y_mm > sym_line_y
            x_m = NODDDOW(i,1) / 1000.0;
            y_local_m = (y_mm - sym_line_y) / 1000.0;  % 0 в корне, растёт к концу
            force = -force_var(i);
            data = [data; x_m, y_local_m, force];
        end
    end
    % Сортировка: сначала по y_local (столбец 2), затем по x (столбец 1)
    data = sortrows(data, [2, 1]);

    % Запись в файл
    fid = fopen(output_file, 'w');
    fprintf(fid, '# x [м]\ty_local [м]\tforce [Н] (правая консоль)\n');
    for i = 1:size(data,1)
        fprintf(fid, '%f\t%f\t%f\n', data(i,1), data(i,2), data(i,3));
    end
    fclose(fid);

    %% 8. Формирование выходной структуры
    results = struct();
    results.alpha_target = alpha_target;
    results.Cy = Cy_target;
    results.Cx = Cx_target;
    results.Cxi = Cxi_target;
    results.Ki = Ki_target;
    % Дополнительные результаты по желанию
    try results.C_mz = evalin('base', 'C_mz'); catch, end
    try results.X_d = evalin('base', 'X_d'); catch, end
    try results.Cy_alpha = evalin('base', 'Cy_alpha'); catch, end
    try results.alpha0 = evalin('base', 'alpha0'); catch, end
    try results.X_F = evalin('base', 'X_F'); catch, end
    try results.M = evalin('base', 'M'); catch, end
end