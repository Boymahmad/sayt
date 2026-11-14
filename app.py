import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Системы нелинейных моделей", page_icon="📐", layout="wide")
st.title("📐 Решение и синтез систем нелинейных моделей")

app_mode = st.sidebar.radio("Режим работы:", ("1. Решение системы по уравнениям", "2. Построение системы по корням"))
num_eqs = st.sidebar.selectbox("Количество уравнений:", (2, 3))
trans = (standard_transformations + (implicit_multiplication_application,))

def parse_eq(s):
    s = s.replace("^", "**")
    if "=" in s:
        l, r = s.split("=")
        return parse_expr(l, transformations=trans) - parse_expr(r, transformations=trans)
    return parse_expr(s, transformations=trans)

def fmt_val(expr):
    s = str(expr).replace("sqrt", "√").replace("*I", "i").replace("I", "i")
    try:
        v = complex(expr.evalf())
        if abs(v.imag) < 1e-12:
            return f"{s} (≈ {v.real:.6f})"
        sgn = "+" if v.imag >= 0 else "-"
        return f"{s} (≈ {v.real:.4f} {sgn} {abs(v.imag):.4f}i)"
    except Exception:
        return s

if app_mode == "1. Решение системы по уравнениям":
    method = st.sidebar.radio("Метод:", ("Аналитический (Символьный)", "Численный метод Ньютона"))
    only_real = st.sidebar.checkbox("Только действительные корни", value=False)

    st.subheader("1. Ввод уравнений")
    cols = st.columns(num_eqs)
    defs = ("x^2 + y^2 = 1", "x^2 - y^2 = 2") if num_eqs == 2 else ("x^2 + y + z = 4", "x + y^2 + z = 4", "x + y + z^2 = 4")
    eqs_in = list(cols.__getitem__(i).text_input(f"Уравнение {i+1}:", value=defs.__getitem__(i)) for i in range(num_eqs))

    if method == "Численный метод Ньютона":
        p_cols = st.columns(num_eqs + 2)
        v_names = ("x", "y", "z")
        x0_list = list(p_cols.__getitem__(i).number_input(f"Начальное {v_names.__getitem__(i)}₀:", value=1.0) for i in range(num_eqs))
        eps = p_cols.__getitem__(num_eqs).number_input("Точность (ε):", value=1e-6, format="%.1e")
        max_it = p_cols.__getitem__(num_eqs + 1).number_input("Макс. итераций:", value=20, min_value=1, max_value=100)

    if st.button("🚀 Решить систему", type="primary"):
        try:
            vars_s = sp.symbols('x y') if num_eqs == 2 else sp.symbols('x y z')
            funcs = list(parse_eq(e) for e in eqs_in)

            st.subheader("2. Математическая запись")
            st.latex(r"\begin{cases} " + r" \\ ".join(list(sp.latex(f) + " = 0" for f in funcs)) + r" \end{cases}")
            rep = "ОТЧЕТ О РЕШЕНИИ СИСТЕМЫ\n" + "="*40 + "\n" + "\n".join(eqs_in) + "\n\n"

            if method == "Аналитический (Символьный)":
                st.subheader("3. Аналитическое решение")
                try:
                    sols = sp.solve(funcs, vars_s, dict=True)
                except NotImplementedError:
                    sols = None
                    st.warning("⚠️ Трансцендентная система. Выберите «Численный метод Ньютона».")

                if sols is not None:
                    real_s = list(s for s in sols if all(abs(complex(s.get(v).evalf()).imag) < 1e-12 for v in vars_s))
                    show_s = real_s if only_real else sols
                    if not show_s:
                        st.warning("Действительных решений не найдено.")
                        rep += "Действительных решений нет.\n"
                    else:
                        st.success(f"Найдено решений: {len(show_s)}")
                        for idx, s in enumerate(show_s, 1):
                            is_r = all(abs(complex(s.get(v).evalf()).imag) < 1e-12 for v in vars_s)
                            tag = "Действительный корень" if is_r else "Комплексный корень"
                            parts = list()
                            rep += f"{tag} {idx}:\n"
                            for v in vars_s:
                                ex = s.get(v)
                                cv = complex(ex.evalf())
                                dec = f"{cv.real:.4f}" if is_r else f"{cv.real:.3f}{'+' if cv.imag>=0 else '-'}{abs(cv.imag):.3f}i"
                                parts.append(rf"{sp.latex(v)} = {sp.latex(ex)} \approx {dec}")
                                rep += f"  {v} = {fmt_val(ex)}\n"
                            st.latex(rf"\text{{{tag} }} {idx}: \quad " + r", \quad ".join(parts))
            else:
                st.subheader("3. Метод Ньютона")
                J_s = sp.Matrix(funcs).jacobian(vars_s)
                st.latex(r"J(X) = " + sp.latex(J_s))
                F_n = sp.lambdify(vars_s, funcs, 'numpy')
                J_n = sp.lambdify(vars_s, J_s, 'numpy')
                xc = np.array(x0_list, dtype=float)
                hist = list()
                for k in range(1, int(max_it) + 1):
                    dx = np.linalg.solve(np.array(J_n(*xc), dtype=float), -np.array(F_n(*xc), dtype=float))
                    xn = xc + dx
                    err = np.linalg.norm(dx, ord=np.inf)
                    row = {"k": k}
                    for i, v in enumerate(vars_s):
                        row.update({f"{v}_k": round(float(xn.item(i)), 6)})
                    row.update({"Погрешность": f"{err:.2e}"})
                    hist.append(row)
                    xc = xn
                    if err < eps:
                        break
                df = pd.DataFrame(hist)
                st.dataframe(df, use_container_width=True)
                st.latex(", ".join(list(rf"{v} \approx {float(xc.item(i)):.6f}" for i, v in enumerate(vars_s))))
                rep += df.to_string(index=False) + "\n"

            if num_eqs == 2:
                st.subheader("4. График системы")
                x, y = vars_s
                xv = np.linspace(-5, 5, 300)
                X, Y = np.meshgrid(xv, xv)
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.contour(X, Y, sp.lambdify((x, y), funcs.__getitem__(0), 'numpy')(X, Y), levels=(0,), colors='blue')
                ax.contour(X, Y, sp.lambdify((x, y), funcs.__getitem__(1), 'numpy')(X, Y), levels=(0,), colors='red')
                ax.axhline(0, color='black', lw=0.8)
                ax.axvline(0, color='black', lw=0.8)
                ax.grid(True, ls='--', alpha=0.6)
                st.pyplot(fig)

            st.download_button("📥 Скачать отчет (.txt)", data=rep, file_name="otchet.txt")
        except Exception as e:
            st.error(f"Ошибка: {e}")

else:
    st.subheader("1. Ввод заданных корней")
    r_cols = st.columns(num_eqs)
    v_names = ("x", "y", "z")
    d_roots = ("2", "3", "-1")
    r_in = list(r_cols.__getitem__(i).text_input(f"Корень {v_names.__getitem__(i)}₀:", value=d_roots.__getitem__(i)) for i in range(num_eqs))
    m_type = st.selectbox("Тип модели:", ("Квадратичная", "Кубическая", "Тригонометрическая"))

    if st.button("🛠️ Построить систему", type="primary"):
        try:
            vars_s = sp.symbols('x y') if num_eqs == 2 else sp.symbols('x y z')
            subs_d = dict((v, parse_eq(r_in.__getitem__(i))) for i, v in enumerate(vars_s))
            if num_eqs == 2:
                x, y = vars_s
                tmpls = (x**2 + y**2, x*y - y) if m_type == "Квадратичная" else ((x**3 - y**2, x**2 + y**3) if m_type == "Кубическая" else (sp.sin(x) + y**2, x**2 + sp.cos(y)))
            else:
                x, y, z = vars_s
                tmpls = (x**2 + y + z, x + y**2 - z, x*y + z**2) if m_type == "Квадратичная" else ((x**3 + y - z, x - y**3 + z, x + y + z**3) if m_type == "Кубическая" else (sp.sin(x) + y + z, x + sp.cos(y) - z, x*y + z**2))

            ltx, pln = list(), list()
            for f in tmpls:
                c = sp.simplify(f.subs(subs_d))
                ltx.append(f"{sp.latex(f)} = {sp.latex(c)}")
                pln.append(f"{f} = {c}")

            st.subheader("2. Сгенерированная система")
            st.latex(r"\begin{cases} " + r" \\ ".join(ltx) + r" \end{cases}")
            st.code("\n".join(pln), language="text")

            if num_eqs == 2:
                x, y = vars_s
                x0f, y0f = float(subs_d.get(x).evalf()), float(subs_d.get(y).evalf())
                spn = max(5.0, abs(x0f) + 3.0, abs(y0f) + 3.0)
                xv = np.linspace(-spn, spn, 300)
                X, Y = np.meshgrid(xv, xv)
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.contour(X, Y, sp.lambdify((x, y), tmpls.__getitem__(0) - float(tmpls.__getitem__(0).subs(subs_d).evalf()), 'numpy')(X, Y), levels=(0,), colors='blue')
                ax.contour(X, Y, sp.lambdify((x, y), tmpls.__getitem__(1) - float(tmpls.__getitem__(1).subs(subs_d).evalf()), 'numpy')(X, Y), levels=(0,), colors='red')
                ax.scatter((x0f,), (y0f,), color='green', s=80, zorder=5, label=f"Корень ({x0f:.2f}, {y0f:.2f})")
                ax.axhline(0, color='black', lw=0.8)
                ax.axvline(0, color='black', lw=0.8)
                ax.grid(True, ls='--', alpha=0.6)
                ax.legend()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Ошибка: {e}")
