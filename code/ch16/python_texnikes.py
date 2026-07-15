# ============================================================
# python_texnikes.py
# Κεφάλαιο 16 — Τεχνικές Ολοκλήρωσης
# Ματζάκος, Ν. (2026). Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
# ============================================================
# ΒΑΣΙΚΕΣ ΕΝΤΟΛΕΣ:
#   sympy.integrate(f, x) -> ολοκλήρωση (αυτόματη τεχνική)
#   sympy.apart(f, x)     -> μερικά κλάσματα
#   sympy.trigsimp()      -> απλοποίηση τριγ.
# ============================================================
import sympy as sp

x = sp.symbols('x')
print("=" * 55)
print(" Κεφάλαιο 16: Τεχνικές Ολοκλήρωσης")
print("=" * 55)

# 1. Κατά μέρη
print("\n[1] Κατά μέρη: int(u dv) = uv - int(v du)")
for name,f_ in [("x*e^x",x*sp.exp(x)),("x^2*ln(x)",x**2*sp.log(x)),
                 ("e^x*sin(x)",sp.exp(x)*sp.sin(x)),("arctan(x)",sp.atan(x)),
                 ("x*cos(x)",x*sp.cos(x))]:
    print(f"  int({name:20s}) = {sp.simplify(sp.integrate(f_,x))} + C")

# 2. Μερικά κλάσματα
print("\n[2] Μερικά κλάσματα")
for name,f_ in [("(2x+1)/((x-1)(x+2))",(2*x+1)/((x-1)*(x+2))),
                 ("x/(x^2-5x+6)",x/(x**2-5*x+6)),
                 ("1/(x^3-x)",1/(x**3-x))]:
    pf = sp.apart(f_,x)
    print(f"  {name}")
    print(f"    = {pf}")
    print(f"    int = {sp.simplify(sp.integrate(f_,x))} + C")

# 3. Τριγωνομετρική αντικατάσταση
print("\n[3] Τριγωνομετρική αντικατάσταση")
for name,f_ in [("1/sqrt(1-x^2)",1/sp.sqrt(1-x**2)),
                 ("sqrt(1-x^2)",sp.sqrt(1-x**2)),
                 ("1/sqrt(x^2+4)",1/sp.sqrt(x**2+4))]:
    try:
        r = sp.integrate(f_,x)
        print(f"  int({name:25s}) = {sp.simplify(r)} + C")
    except Exception:
        print(f"  int({name:25s}) = (αριθμητικά)")

# 4. Τριγωνομετρικά
print("\n[4] Τριγωνομετρικά ολοκληρώματα")
for name,f_ in [("sin^2(x)",sp.sin(x)**2),("cos^3(x)",sp.cos(x)**3),
                 ("sin(x)*cos(x)",sp.sin(x)*sp.cos(x)),("tan(x)",sp.tan(x))]:
    print(f"  int({name:18s}) = {sp.trigsimp(sp.integrate(f_,x))} + C")
print("\n✓ Ολοκληρώθηκε.")
