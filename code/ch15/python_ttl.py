# ============================================================
# python_ttl.py
# Κεφάλαιο 15 — Θεμελιώδες Θεώρημα Λογισμού
# Ματζάκος, Ν. (2026). Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
# ============================================================
# ΒΑΣΙΚΕΣ ΕΝΤΟΛΕΣ:
#   sympy.Integral(f,(t,0,x)) -> F(x) = int_0^x f(t)dt
#   sympy.diff(F, x)          -> F'(x) = f(x)  [ΘΘΛ-Α]
#   sympy.integrate(f,(x,a,b))-> F(b)-F(a)      [ΘΘΛ-Β]
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x, t = sp.symbols('x t')
print("=" * 60)
print(" Κεφάλαιο 15: Θεμελιώδες Θεώρημα Λογισμού")
print("=" * 60)

# ΘΘΛ Μέρος Α
print("\n[1] ΘΘΛ Μέρος Α: d/dx[int_a^x f(t)dt] = f(x)")
for name, f_ in [("t^2",t**2),("sin(t)",sp.sin(t)),("e^t",sp.exp(t))]:
    F = sp.Integral(f_, (t,0,x))
    dF = sp.diff(F, x)
    ok = sp.simplify(dF - f_.subs(t,x)) == 0
    print(f"  f(t)={name}  -> F'(x)={dF}  [ok={ok}]")

# ΘΘΛ Μέρος Β: Newton-Leibniz
print("\n[2] ΘΘΛ Μέρος Β: int_a^b f = F(b)-F(a)")
for name, f_, a, b in [
    ("x^2",      x**2,      1, 3),
    ("cos(x)",   sp.cos(x), 0, sp.pi),
    ("1/x",      1/x,       1, sp.E),
    ("e^x",      sp.exp(x), 0, 1),
    ("x^3",      x**3,      -1, 1)]:
    v = sp.integrate(f_,(x,a,b))
    print(f"  int({name}) [{a},{b}] = {v} ≈ {float(v):.6f}")

# Γράφημα
xs_p = np.linspace(0, 3*np.pi, 400)
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
ax1.plot(xs_p,np.sin(xs_p),'b-',lw=2)
ax1.fill_between(xs_p,np.sin(xs_p),alpha=0.2,color='blue')
ax1.set_title(r'$f(t)=\sin t$'); ax1.set_xlabel('t'); ax1.grid(True,alpha=0.3)
ax2.plot(xs_p,1-np.cos(xs_p),'r-',lw=2)
ax2.set_title(r'$F(x)=\int_0^x\sin t\,dt=1-\cos x$')
ax2.set_xlabel('x'); ax2.grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig('ch14_ftc.png',dpi=100)
print("\n  Γράφημα: ch14_ftc.png")
print("\n✓ Ολοκληρώθηκε.")
