# ============================================================
# python_texnikes.py
# Κεφάλαιο 16 — Τεχνικές Ολοκλήρωσης
# Ματζάκος, Ν. (2026). Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
# ------------------------------------------------------------
# ΒΑΣΙΚΕΣ ΕΝΤΟΛΕΣ:
#   sympy.integrate(f, x)                  -> ολοκλήρωση
#   sympy.apart(f, x)                      -> μερικά κλάσματα
#   sympy.integrals.manualintegrate(f, x)  -> «σχολικές» τεχνικές
#   sympy.integrals.integral_steps(f, x)   -> ποιον κανόνα διάλεξε
#   sympy.trigsimp(), sympy.simplify()     -> απλοποίηση
#   scipy.integrate.quad(f, a, b)          -> αριθμητικό ολοκλήρωμα
# ============================================================
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy.integrals.manualintegrate import manualintegrate, integral_steps
from scipy.integrate import quad

x = sp.symbols('x', positive=True)

print("=" * 62)
print(" Κεφάλαιο 16: Τεχνικές Ολοκλήρωσης")
print("=" * 62)


# ------------------------------------------------------------
# Βοηθητική συνάρτηση επαλήθευσης (ζητείται στο ερώτημα (α)):
#   η F είναι αντιπαράγωγος της f  <=>  simplify(F' - f) == 0
# ------------------------------------------------------------
def check(F, f):
    """True αν η F είναι αντιπαράγωγος της f."""
    return sp.simplify(sp.diff(F, x) - f) == 0


# ============================================================
# Α. ΑΝΤΙΚΑΤΑΣΤΑΣΗ
#    Τα ίδια ολοκληρώματα με τη δραστηριότητα Maxima.
# ============================================================
print("\n=== Α. ΑΝΤΙΚΑΤΑΣΤΑΣΗ ===")
print("f(x)                  αντικατάσταση   int f dx")

antikat = [
    (x * sp.exp(x**2),      "u = x^2",    sp.exp(x**2) / 2),
    (x / sp.sqrt(x**2 + 4), "u = x^2+4",  sp.sqrt(x**2 + 4)),
    (sp.log(x) / x,         "u = ln x",   sp.log(x)**2 / 2),
]

for f, subst, anamenomeno in antikat:
    F = sp.integrate(f, x)
    print(f"  {str(f):20s}  {subst:11s}  {F}  + C")
    print(f"      check(F, f) = {check(F, f)}"
          f"   |  αναμενόμενο {anamenomeno}: "
          f"{sp.simplify(F - anamenomeno) == 0}")


# ============================================================
# Β. ΚΑΤΑ ΠΑΡΑΓΟΝΤΕΣ — ποια τεχνική διαλέγει η SymPy;
#    Η manualintegrate ακολουθεί τις «σχολικές» τεχνικές, ενώ η
#    integral_steps επιστρέφει το δέντρο των κανόνων που χρησιμοποίησε.
# ============================================================
print("\n=== Β. ΚΑΤΑ ΠΑΡΑΓΟΝΤΕΣ (manualintegrate) ===")

kata_meri = [
    (x**2 * sp.exp(x),      "διπλή εφαρμογή: u=x^2, dv=e^x dx"),
    (sp.atan(x),            "u = arctan x, dv = dx"),
    (sp.exp(x) * sp.sin(x), "κυκλική: το ολοκλήρωμα επανεμφανίζεται"),
]

for f, dikh_mas in kata_meri:
    F = sp.simplify(manualintegrate(f, x))
    kanonas = type(integral_steps(f, x)).__name__
    print(f"\n  ∫ {f} dx = {F} + C")
    print(f"      κανόνας SymPy : {kanonas}")
    print(f"      δική μας επιλογή: {dikh_mas}")
    print(f"      check(F, f)   : {check(F, f)}")

# Η κυκλική περίπτωση σε «σχολική» μορφή e^x*(sin x - cos x)/2
F_kyk = sp.simplify(manualintegrate(sp.exp(x) * sp.sin(x), x))
sxolikh = sp.exp(x) * (sp.sin(x) - sp.cos(x)) / 2
print(f"\n  Σχολική μορφή e^x*(sin x - cos x)/2 ίδια με SymPy: "
      f"{sp.simplify(F_kyk - sxolikh) == 0}")


# ============================================================
# Γ. ΜΕΡΙΚΑ ΚΛΑΣΜΑΤΑ — apart και ολοκλήρωση όρο προς όρο
# ============================================================
print("\n=== Γ. ΜΕΡΙΚΑ ΚΛΑΣΜΑΤΑ (apart) ===")

ritles = [
    (2*x + 3) / (x**2 - x - 2),
    (2*x**2 - 1) / (x * (x + 1)**2),
    (x**3 - 1) / (x**2 + x + 1),
]

for f in ritles:
    pf = sp.apart(f, x)
    oroi = sp.Add.make_args(pf)
    athroisma = sp.Add(*[sp.integrate(t, x) for t in oroi])
    ameso = sp.integrate(f, x)
    print(f"\n  f(x) = {f}")
    print(f"      apart        : {pf}")
    # expand_log: εμφανίζει τους λογαρίθμους αναλυτικά, όχι συμπτυγμένους
    print(f"      όρο προς όρο : {sp.expand_log(sp.expand(athroisma), force=True)} + C")
    print(f"      άμεσο        : {sp.expand_log(sp.expand(ameso), force=True)} + C")
    # Δύο αντιπαράγωγοι ταυτίζονται αν διαφέρουν το πολύ κατά σταθερά,
    # δηλαδή αν η παράγωγος της διαφοράς τους είναι μηδέν.
    print(f"      διαφέρουν το πολύ κατά σταθερά: "
          f"{sp.simplify(sp.diff(athroisma - ameso, x)) == 0}")
    print(f"      πανομοιότυπες εκφράσεις       : "
          f"{sp.simplify(athroisma - ameso) == 0}")
    print(f"      check(άμεσο, f)               : {check(ameso, f)}")

# Η τρίτη ρητή έχει βαθμό αριθμητή >= βαθμό παρονομαστή:
# χρειάζεται ΠΡΩΤΑ διαίρεση πολυωνύμων.
phliko, ypoloipo = sp.div(x**3 - 1, x**2 + x + 1, x)
print(f"\n  Διαίρεση πολυωνύμων: (x^3-1) : (x^2+x+1) -> "
      f"πηλίκο {phliko}, υπόλοιπο {ypoloipo}")
print(f"  Παραγοντοποίηση     : x^3-1 = {sp.factor(x**3 - 1)}")


# ============================================================
# Δ. ΤΡΙΓΩΝΟΜΕΤΡΙΚΗ ΑΝΤΙΚΑΤΑΣΤΑΣΗ — συμβολικά vs αριθμητικά
#    ∫_0^1 sqrt(x^2+1) dx   (με x = tan θ)
# ============================================================
print("\n=== Δ. ΤΡΙΓΩΝΟΜΕΤΡΙΚΗ ΑΝΤΙΚΑΤΑΣΤΑΣΗ ===")

f_d = sp.sqrt(x**2 + 1)
F_d = sp.integrate(f_d, x)
print(f"  ∫ sqrt(x^2+1) dx = {F_d} + C")
print(f"      check(F, f) = {check(F_d, f_d)}")

symvolika = sp.integrate(f_d, (x, 0, 1))
kleisto = sp.sqrt(2)/2 + sp.log(1 + sp.sqrt(2))/2      # sqrt2/2 + ln(1+sqrt2)/2
print(f"\n  ∫_0^1 sqrt(x^2+1) dx = {symvolika}")
print(f"      κλειστή μορφή sqrt(2)/2 + ln(1+sqrt(2))/2 : "
      f"{sp.simplify(symvolika - kleisto) == 0}")

sym_val = float(symvolika)
num_val, sfalma = quad(lambda t: np.sqrt(t**2 + 1), 0, 1)
apoklisi = abs(sym_val - num_val)
print(f"      συμβολικά          = {sym_val:.12f}   (αναμ. 1.147793574696)")
print(f"      scipy quad         = {num_val:.12f}   (εκτ. σφάλμα {sfalma:.1e})")
print(f"      |απόκλιση|         = {apoklisi:.2e}")
print(f"      απόκλιση < 1e-8    : {apoklisi < 1e-8}")

# --- Σκιασμένη περιοχή ---------------------------------------
xv = np.linspace(-0.3, 1.3, 400)
yv = np.sqrt(xv**2 + 1)
xs = np.linspace(0, 1, 200)
ys = np.sqrt(xs**2 + 1)

plt.figure(figsize=(7, 4.2))
plt.plot(xv, yv, 'b-', lw=2, label=r'$y=\sqrt{x^2+1}$')
plt.fill_between(xs, 0, ys, alpha=0.30, color='tab:green',
                 label=f'εμβαδόν = {sym_val:.6f}')
plt.axhline(0, color='k', lw=0.6)
plt.axvline(0, color='k', lw=0.6)
plt.xlim(-0.3, 1.3)
plt.ylim(0, 1.7)
plt.xlabel('x')
plt.ylabel('y')
plt.title(r'$\int_0^1\sqrt{x^2+1}\,dx$  (τριγωνομετρική αντικατάσταση $x=\tan\theta$)')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

print("\nΟλοκληρώθηκε.")
plt.show()
