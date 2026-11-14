import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Настройка страницы
st.set_page_config(page_title="Решение и синтез нелинейных систем", page_icon="📐", layout="wide")

st.title("📐 Программный комплекс для работы с системами нелинейных моделей")
st.markdown("Веб-сервис реализует **прямую задачу** (решение систем уравнений) и **обратную задачу** (построение нелинейной системы по заданным корням).")

# Боковая панель настроек
st.sidebar.header("⚙️ Режим работы и параметры")
app_mode = st.sidebar.radio(
    "Выберите режим:",
    ["1. Решение системы по уравнениям", "2. Построение системы по корням"]
)

num_eqs = st.sidebar.selectbox("Размерность системы (количество переменных):", [2, 3])

# Умный парсер (понимает 2x, 3y, x^2 без знака *)
transformations = (standard_transformations + (implicit_multiplication_application,))

def parse_equation(eq_str):
    eq_str = eq_str.replace("^", "**")
    if "=" in eq_str:
        left, right = eq_str.split("=")
        return parse_expr(left, transformations=transformations) - parse_expr(right, transformations=transformations)
    return parse_expr(eq_str, transformations=transformations)

# Форматирование корней для красивого текстового отчета (замена sqrt на √ и I на i)
def format_readable(expr):
    s = str(expr).replace("sqrt", "√").replace("*I", "i").replace("I", "i")
    try:
        val = complex(expr.evalf())
        if abs(val.imag) < 1e-12:
            return f"{s}  (≈ {val.real:.6f})"
        else:
            sign = "+" if val.imag >= 0 else "-"
            return f"{s}  (≈ {val.real:.4f} {sign} {abs(val.imag):.4f}i)"
    except Exception:
        return s

# ==============================================================================
# РЕЖИМ 1: ПРЯМАЯ ЗАДАЧА (РЕШЕНИЕ СИСТЕМЫ)
# ==============================================================================
if app_mode == "1. Решение системы по уравнениям":
    method = st.sidebar.radio(
        "Метод решения:",
        ["Аналитический метод (Символьный)", "Численный метод Ньютона (Итерационный)"]
    )

    filter_real = st.sidebar.checkbox("Показывать только действительные (вещественные) корни", value=False)

    st.subheader("1. Ввод системы нелинейных уравнений")
    eq_inputs = []
    cols = st.columns(num_eqs)

    default_2 = ["x^2 + y^2 = 1", "x^2 - y^2 = 2"]
    default_3 = ["x^2 + y + z = 4", "x + y^2 + z = 4", "x + y + z^2 = 4"]

    for i in range(num_eqs):
        with cols[i]:
            def_val = default_2[i] if num_eqs == 2 else default_3[i]
            eq_inputs.append(st.text_input(f"Уравнение {i+1}:", value=def_val))

    if method == "Численный метод Ньютона (Итерационный)":
        st.info("Задайте начальное приближение $X^{(0)}$, точность $\\varepsilon$ и максимум итераций:")
        p_cols = st.columns(num_eqs + 2)
        init_guess = []
        var_names = ["x", "y", "z"][:num_eqs]
        for i, v in enumerate(var_names):
            init_guess.append(p_cols[i].number_input(f"Начальное {v}₀:", value=1.0, step=0.1))
        eps = p_cols[num_eqs].number_input("Точность (ε):", value=1e-6, format="%.1e")
        max_iter = p_cols[num_eqs + 1].number_input("Макс. итераций:", value=20, min_value=1, max_value=100)

    if st.button("🚀 Решить систему", type="primary"):
        try:
            vars_sym = sp.symbols('x y') if num_eqs == 2 else sp.symbols('x y z')
            funcs = [parse_equation(eq) for eq in eq_inputs]

            st.subheader("2. Математическая постановка задачи")
            latex_lines = r" \\ ".join([sp.latex(f) + " = 0" for f in funcs])
            st.latex(r"\begin{cases} " + latex_lines + r" \end{cases}")

            report_text = f"ОТЧЕТ О РЕШЕНИИ СИСТЕМЫ НЕЛИНЕЙНЫХ УРАВНЕНИЙ\n{'='*55}\n"
            report_text += "Введенная система:\n" + "\n".join(eq_inputs) + "\n\n"

            if method == "Аналитический метод (Символьный)":
                st.subheader("3. Пошаговое аналитическое решение")
                if num_eqs == 2:
                    x, y = vars_sym
                    try:
                        sub_expr = sp.solve(funcs[0], x)
                        if sub_expr:
                            st.markdown("**Шаг 1.** Выразим переменную $x$ из первого уравнения:")
                            st.latex(rf"x = {sp.latex(sub_expr[0])}")
                            substituted = funcs[1].subs(x, sub_expr[0])
                            st.markdown("**Шаг 2.** Подставим полученное выражение во второе уравнение:")
                            st.latex(rf"{sp.latex(sp.expand(substituted))} = 0")
                    except Exception:
                        pass

                st.markdown("**Итоговые корни системы:**")
                try:
                    solutions = sp.solve(funcs, vars_sym, dict=True)
                except NotImplementedError:
                    solutions = None
                    st.warning("⚠️ Данная система является трансцендентной и не имеет точного аналитического решения. Выберите «Численный метод Ньютона».")

                if solutions is not None:
                    # Разделение на действительные и комплексные корни
                    real_sols = []
                    complex_sols = []
                    for sol in solutions:
                        is_real = all(abs(complex(sol[v].evalf()).imag) < 1e-12 for v in vars_sym)
                        if is_real:
                            real_sols.append(sol)
                        else:
                            complex_sols.append(sol)

                    display_sols = real_sols if filter_real else solutions

                    if filter_real and not real_sols:
                        st.warning("Действительных (вещественных) корней нет. Все найденные решения являются комплексными (снимите галочку слева, чтобы увидеть их).")
                        report_text += "Результат: Действительных корней нет (все решения комплексные).\n"
                    elif not display_sols:
                        st.warning("Система не имеет аналитических решений.")
                        report_text += "Результат: Система не имеет решений.\n"
                    else:
                        st.success(f"Всего решений: {len(solutions)} (действительных: {len(real_sols)}, комплексных: {len(complex_sols)})")
                        for idx, sol in enumerate(display_sols, 1):
                            is_r = all(abs(complex(sol[v].evalf()).imag) < 1e-12 for v in vars_sym)
                            tag = "Действительный корень" if is_r else "Комплексный корень"
                            
                            # Красивый вывод формулы и десятичного значения на экране
                            sol_parts = []
                            for v in vars_sym:
                                c_val = complex(sol[v].evalf())
                                dec_str = f"{c_val.real:.4f}" if is_r else f"{c_val.real:.3f} {'+' if c_val.imag>=0 else '-'} {abs(c_val.imag):.3f}i"
                                sol_parts.append(rf"{sp.latex(v)} = {sp.latex(sol[v])} \approx {dec_str}")
                            
                            st.latex(rf"\text{{{tag} }} {idx}: \quad " + r", \quad ".join(sol_parts))
                            
                            # Красивый вывод в скачиваемый отчет (с символом √ вместо sqrt)
                            report_text += f"{tag} {idx}:\n"
                            for v in vars_sym:
                                report_text += f"   {v} = {format_readable(sol[v])}\n"

            else:
                st.subheader("3. Итерационный процесс (Метод Ньютона)")
                J_sym = sp.Matrix(funcs).jacobian(vars_sym)
                st.markdown("**Матрица Якоби $J(X)$ для данной системы:**")
                st.latex(r"J = " + sp.latex(J_sym))

                F_num = sp.lambdify(vars_sym, funcs, 'numpy')
                J_num = sp.lambdify(vars_sym, J_sym, 'numpy')

                x_curr = np.array(init_guess, dtype=float)
                history = []

                for k in range(1, int(max_iter) + 1):
                    f_val = np.array(F_num(*x_curr), dtype=float)
                    j_val = np.array(J_num(*x_curr), dtype=float)
                    
                    delta = np.linalg.solve(j_val, -f_val)
                    x_next = x_curr + delta
                    error = np.linalg.norm(delta, ord=np.inf)

                    row = {"Итерация (k)": k}
                    for idx, v in enumerate(vars_sym):
                        row[f"{v}_k"] = round(x_next[idx], 7)
                    row["Погрешность (ε)"] = f"{error:.2e}"
                    history.append(row)

                    x_curr = x_next
                    if error < eps:
                        break

                df_iter = pd.DataFrame(history)
                st.dataframe(df_iter, use_container_width=True)

                res_str = ", ".join([rf"{v} \approx {x_curr[i]:.6f}" for i, v in enumerate(vars_sym)])
                st.success(f"Сходимость достигнута за {len(history)} итераций!")
                st.latex(res_str)
                report_text += f"Метод Ньютона (сходимость за {len(history)} итераций):\n" + df_iter.to_string(index=False) + "\n"

            if num_eqs == 2:
                st.subheader("4. Геометрическая интерпретация решения")
                x, y = vars_sym
                x_vals = np.linspace(-5, 5, 400)
                y_vals = np.linspace(-5, 5, 400)
                X, Y = np.meshgrid(x_vals, y_vals)

                F1_grid = sp.lambdify((x, y), funcs[0], 'numpy')(X, Y)
                F2_grid = sp.lambdify((x, y), funcs[1], 'numpy')(X, Y)

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.contour(X, Y, F1_grid, levels=[0], colors='blue', linewidths=2)
