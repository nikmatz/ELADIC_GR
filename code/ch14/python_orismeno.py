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

# ============================================================
# ΣΥΜΠΛΗΡΩΜΑ — Αθροίσματα Riemann: από το άθροισμα στο όριο
#   sympy.summation, sympy.limit, matplotlib bar
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

i, n = sp.symbols('i n', positive=True, integer=True)

# f(x)=x^2 στο [0,2]:  Δx = 2/n,  x_i = 2i/n  (δεξιά άκρα)
dx  = sp.Rational(2, 1)/n
R_n = sp.summation(((2*i/n)**2) * dx, (i, 1, n))
R_n = sp.simplify(R_n)
print("R_n =", R_n)
print("lim R_n =", sp.limit(R_n, n, sp.oo), "   (αναμενόμενο 8/3)")

# Αριστερά άκρα — ίδιο όριο:
L_n = sp.simplify(sp.summation(((2*(i-1)/n)**2) * dx, (i, 1, n)))
print("L_n =", L_n, "   lim L_n =", sp.limit(L_n, n, sp.oo))

# Αριθμητικά: L_n < 8/3 < R_n και το μεσαίο συγκλίνει ταχύτερα
exact = 8/3
print(f"\n{'n':>5} {'L_n':>10} {'M_n':>10} {'R_n':>10} {'|M_n-I|':>10}")
for N in (4, 10, 50, 200):
    xs = np.linspace(0, 2, N+1); h = 2/N
    L = np.sum(xs[:-1]**2)*h
    R = np.sum(xs[1:]**2)*h
    M = np.sum(((xs[:-1]+xs[1:])/2)**2)*h
    print(f"{N:5d} {L:10.6f} {M:10.6f} {R:10.6f} {abs(M-exact):10.2e}")

# Εικόνα των ορθογωνίων (bar):
N = 10; xs = np.linspace(0, 2, N+1); h = 2/N
plt.figure(figsize=(6.5, 4))
plt.bar(xs[:-1], xs[1:]**2, width=h, align='edge',
        edgecolor='white', alpha=.55, label=f'δεξιά αθροίσματα (n={N})')
t = np.linspace(0, 2, 300); plt.plot(t, t**2, 'r', lw=2, label='$f(x)=x^2$')
plt.legend(); plt.grid(alpha=.3); plt.title('Άθροισμα Riemann και το ολοκλήρωμα')
plt.tight_layout(); plt.savefig('ch14_riemann_bars.png', dpi=100)
print("\nΓράφημα: ch14_riemann_bars.png")
