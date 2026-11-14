import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Системы нелинейных моделей", page_icon="📐", layout="wide")
st.title("📐 Решение и синтез систем нелинейных моделей")

mode = st.sidebar.radio("Режим:", ("1. Решение системы", "2. Построение по корням"))
n = st.sidebar.selectbox("Уравнений:", (2, 3))
tr = (standard_transformations + (implicit_multiplication_application,))

def p_eq(s):
    s = s.replace("^", "**")
    if "=" in s:
        l, r = s.split("=")
        return parse_expr(l, transformations=tr) - parse_expr(r, transformations=tr)
    return parse_expr(s, transformations=tr)

def fmt_v(ex):
    s = str(ex).replace("sqrt", "√").replace("*I", "i").replace("I", "i")
    if ex.is_Integer: return s
    try:
        v = complex(ex.evalf())
        if abs(v.imag) < 1e-12: return f"{s} (≈ {v.real:.6f})"
        return f"{s} (≈ {v.real:.4f}{'+' if v.imag>=0 else '-'}{abs(v.imag):.4f}i)"
    except Exception: return s

def brace(eqs):
    if len(eqs) == 2: return f"⎧ {eqs.__getitem__(0)}\n⎩ {eqs.__getitem__(1)}"
    return f"⎧ {eqs.__getitem__(0)}\n⎨ {eqs.__getitem__(1)}\n⎩ {eqs.__getitem__(2)}"

def plot_2d(f1, f2, x, y, pt=None):
    spn = max(5.0, abs(pt.__getitem__(0)) + 2.5, abs(pt.__getitem__(1)) + 2.5) if pt else 5.0
    xv = np.linspace(-spn, spn, 300)
    X, Y = np.meshgrid(xv, xv)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contour(X, Y, sp.lambdify((x, y), f1, 'numpy')(X, Y), levels=(0,), colors='blue')
    ax.contour(X, Y, sp.lambdify((x, y), f2, 'numpy')(X, Y), levels=(0,), colors='red')
    if pt:
        ax.scatter((pt.__getitem__(0),), (pt.__getitem__(1),), color='green', s=80, zorder=5, label=f"Корень ({pt.__getitem__(0):.2f}, {pt.__getitem__(1):.2f})")
        ax.legend()
    ax.axhline(0, color='black', lw=0.8); ax.axvline(0, color='black', lw=0.8); ax.grid(True, ls='--', alpha=0.6)
    st.pyplot(fig)

def plot_3d(pts):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    for p in pts:
        ax.scatter(p.__getitem__(0), p.__getitem__(1), p.__getitem__(2), color='green', s=70)
        ax.text(p.__getitem__(0), p.__getitem__(1), p.__getitem__(2), f" ({p.__getitem__(0):.2f}, {p.__getitem__(1):.2f}, {p.__getitem__(2):.2f})", fontsize=8)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    st.pyplot(fig)

if mode == "1. Решение системы":
    mth = st.sidebar.radio("Метод:", ("Аналитический", "Метод Ньютона"))
    only_r = st.sidebar.checkbox("Только действительные корни", value=False)
    cols = st.columns(n)
    defs = ("x^2 + y^2 = 1", "x^2 - y^2 = 2") if n == 2 else ("x^2 + y + z = 4", "x + y^2 + z = 4", "x + y + z^2 = 4")
    eqs = list(cols.__getitem__(i).text_input(f"Уравнение {i+1}:", value=defs.__getitem__(i)) for i in range(n))

    if mth == "Метод Ньютона":
        pc = st.columns(n + 2)
        vn = ("x", "y", "z")
        x0 = list(pc.__getitem__(i).number_input(f"Нач. {vn.__getitem__(i)}₀:", value=1.0) for i in range(n))
        eps = pc.__getitem__(n).number_input("Точность ε:", value=1e-6, format="%.1e")
        mit = pc.__getitem__(n + 1).number_input("Макс. итер.:", value=20, min_value=1, max_value=100)

    if st.button("🚀 Решить систему", type="primary"):
        try:
            vs = sp.symbols('x y') if n == 2 else sp.symbols('x y z')
            fs = list(p_eq(e) for e in eqs)
            st.latex(r"\begin{cases} " + r" \\ ".join(list(sp.latex(f) + " = 0" for f in fs)) + r" \end{cases}")
            rep = "ОТЧЕТ О РЕШЕНИИ СИСТЕМЫ\n" + "="*45 + "\n" + brace(eqs) + "\n\n"
            r_pts = list()

            if mth == "Аналитический":
                if n == 2:
                    x, y = vs
                    try:
                        sub_x = sp.solve(fs.__getitem__(0), x)
                        if sub_x:
                            sx = sub_x.__getitem__(0)
                            eq_sub = sp.expand(fs.__getitem__(1).subs(x, sx))
                            st.markdown("**Шаг 1.** Выразим $x$ из первого уравнения:")
                            st.latex(rf"x = {sp.latex(sx)}")
                            st.markdown("**Шаг 2.** Подставим во второе уравнение:")
                            st.latex(rf"{sp.latex(eq_sub)} = 0")
                            rep += f"Шаг 1 (выражение x): x = {sx}\nШаг 2 (подстановка): {eq_sub} = 0\n\n"
                    except Exception: pass

                try: sols = sp.solve(fs, vs, dict=True)
                except NotImplementedError:
                    sols = None
                    st.warning("⚠️ Выберите «Метод Ньютона» для трансцендентных систем.")
                if sols is not None:
                    rs = list(s for s in sols if all(abs(complex(s.get(v).evalf()).imag) < 1e-12 for v in vs))
                    sh = rs if only_r else sols
                    if not sh:
                        st.warning("Действительных корней нет.")
                        rep += "Действительных корней нет.\n"
                    else:
                        st.success(f"Найдено решений: {len(sh)}")
                        for idx, s in enumerate(sh, 1):
                            ir = all(abs(complex(s.get(v).evalf()).imag) < 1e-12 for v in vs)
                            tg = "Действительный корень" if ir else "Комплексный корень"
                            pts = list()
                            rep += f"{tg} {idx}:\n"
                            if ir: r_pts.append(tuple(float(complex(s.get(v).evalf()).real) for v in vs))
                            for v in vs:
                                ex = s.get(v)
                                if ex.is_Integer: pts.append(rf"{sp.latex(v)} = {sp.latex(ex)}")
                                else:
                                    cv = complex(ex.evalf())
                                    dc = f"{cv.real:.4f}" if ir else f"{cv.real:.3f}{'+' if cv.imag>=0 else '-'}{abs(cv.imag):.3f}i"
                                    pts.append(rf"{sp.latex(v)} = {sp.latex(ex)} \approx {dc}")
                                rep += f"   {v} = {fmt_v(ex)}\n"
                            st.latex(rf"\text{{{tg} }} {idx}: \quad " + r", \quad ".join(pts))
            else:
                Js = sp.Matrix(fs).jacobian(vs)
                st.latex(r"J(X) = " + sp.latex(Js))
                Fn, Jn = sp.lambdify(vs, fs, 'numpy'), sp.lambdify(vs, Js, 'numpy')
                xc, hist = np.array(x0, dtype=float), list()
                for k in range(1, int(mit) + 1):
                    dx = np.linalg.solve(np.array(Jn(*xc), dtype=float), -np.array(Fn(*xc), dtype=float))
                    xn = xc + dx
                    err = np.linalg.norm(dx, ord=np.inf)
                    row = {"k": k}
                    for i, v in enumerate(vs): row.update({f"{v}_k": round(float(xn.item(i)), 6)})
                    row.update({"Погрешность": f"{err:.2e}"})
                    hist.append(row)
                    xc = xn
                    if err < eps: break
                df = pd.DataFrame(hist)
                st.dataframe(df, use_container_width=True)
                st.latex(", ".join(list(rf"{v} \approx {float(xc.item(i)):.6f}" for i, v in enumerate(vs))))
                rep += df.to_string(index=False) + "\n"
                r_pts.append(tuple(float(xc.item(i)) for i in range(n)))

            if n == 2: plot_2d(fs.__getitem__(0), fs.__getitem__(1), vs.__getitem__(0), vs.__getitem__(1), r_pts.__getitem__(0) if r_pts else None)
            elif r_pts: plot_3d(r_pts)
            st.download_button("📥 Скачать отчет (.txt)", data=rep, file_name="otchet.txt")
        except Exception as e: st.error(f"Ошибка: {e}")

else:
    rc = st.columns(n)
    vn, dr = ("x", "y", "z"), ("2", "3", "-1")
    rin = list(rc.__getitem__(i).text_input(f"Корень {vn.__getitem__(i)}₀:", value=dr.__getitem__(i)) for i in range(n))
    mt = st.selectbox("Модель:", ("Квадратичная", "Кубическая", "Тригонометрическая"))
    if st.button("🛠️ Построить систему", type="primary"):
        try:
            vs = sp.symbols('x y') if n == 2 else sp.symbols('x y z')
            sd = dict((v, p_eq(rin.__getitem__(i))) for i, v in enumerate(vs))
            if n == 2:
                x, y = vs
                tm = (x**2+y**2, x*y-y) if mt=="Квадратичная" else ((x**3-y**2, x**2+y**3) if mt=="Кубическая" else (sp.sin(x)+y**2, x**2+sp.cos(y)))
            else:
                x, y, z = vs
                tm = (x**2+y+z, x+y**2-z, x*y+z**2) if mt=="Квадратичная" else ((x**3+y-z, x-y**3+z, x+y+z**3) if mt=="Кубическая" else (sp.sin(x)+y+z, x+sp.cos(y)-z, x*y+z**2))
            ltx, pln, f_zeros = list(), list(), list()
            for f in tm:
                c = sp.simplify(f.subs(sd))
                ltx.append(f"{sp.latex(f)} = {sp.latex(c)}"); pln.append(f"{f} = {c}")
                f_zeros.append(f - c)
            st.latex(r"\begin{cases} " + r" \\ ".join(ltx) + r" \end{cases}")
            st.code("\n".join(pln), language="text")
            pt_val = tuple(float(sd.get(v).evalf()) for v in vs)
            if n == 2: plot_2d(f_zeros.__getitem__(0), f_zeros.__getitem__(1), x, y, pt_val)
            else: plot_3d((pt_val,))
            rep_s = "СИНТЕЗ СИСТЕМЫ\n" + "="*30 + "\n" + brace(pln) + "\n"
            st.download_button("📥 Скачать систему (.txt)", data=rep_s, file_name="sintez.txt")
        except Exception as e: st.error(f"Ошибка: {e}")
        
