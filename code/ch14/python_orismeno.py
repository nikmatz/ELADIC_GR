# ============================================================
# python_orismeno.py
# Κεφάλαιο 14 — Ορισμένο Ολοκλήρωμα
# Ματζάκος, Ν. (2026). Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
# ============================================================
# ΒΑΣΙΚΕΣ ΕΝΤΟΛΕΣ:
#   sympy.integrate(f,(x,a,b)) -> ακριβές ορισμένο
#   scipy.integrate.quad()     -> αριθμητικό
#   numpy.trapz(y,x)           -> κανόνας τραπεζίου
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy import integrate as sci

x = sp.symbols('x')
print("=" * 55)
print(" Κεφάλαιο 14: Ορισμένο Ολοκλήρωμα")
print("=" * 55)

# 1. Αθροίσματα Riemann
print("\n[1] Αθροίσματα Riemann: int_0^1 x^2 dx = 1/3")
f_n = lambda t: t**2
for n in [4,10,100,1000]:
    dx = 1.0/n
    xs_l = np.linspace(0,1-dx,n)
    xs_m = np.linspace(dx/2, 1-dx/2, n)
    Sl = np.sum(f_n(xs_l))*dx
    Sm = np.sum(f_n(xs_m))*dx
    print(f"  n={n:5d}  Αριστ={Sl:.6f}  Μέσο={Sm:.6f}")

# 2. Συμβολικά
print("\n[2] Συμβολικά ορισμένα ολοκληρώματα")
for name,f_,bounds in [
    ("int_0^1 x^2",x**2,(x,0,1)),
    ("int_0^pi sin(x)",sp.sin(x),(x,0,sp.pi)),
    ("int_1^e ln(x)",sp.log(x),(x,1,sp.E)),
    ("int_0^1 e^x",sp.exp(x),(x,0,1)),
    ("int_0^1 1/(1+x^2)",1/(1+x**2),(x,0,1))]:
    v = sp.integrate(f_,bounds)
    print(f"  {name:22s} = {v} ≈ {float(v):.6f}")

# 3. Γράφημα εμβαδού
xs_p = np.linspace(0,np.pi,300)
plt.figure(figsize=(6,4))
plt.fill_between(xs_p,np.sin(xs_p),alpha=0.3,color='blue')
plt.plot(xs_p,np.sin(xs_p),'b-',lw=2,label=r'$\sin x$')
plt.axhline(0,color='k',lw=0.5)
plt.title(r'$\int_0^\pi \sin x\,dx=2$'); plt.legend()
plt.tight_layout(); plt.savefig('ch13_area.png',dpi=100)
print("\n  Γράφημα: ch13_area.png")
print("\n✓ Ολοκληρώθηκε.")
