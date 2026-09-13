# ============================================================
# python_aoristo.py
# Κεφάλαιο 13 — Αόριστο Ολοκλήρωμα (SymPy & Matplotlib)
# Ματζάκος, Ν. (2026). Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
# ============================================================
#
# ΒΙΒΛΙΟΘΗΚΕΣ:
#   sympy       → συμβολική ολοκλήρωση και επαλήθευση
#   numpy       → αριθμητικά πλέγματα για τα γραφήματα
#   matplotlib  → γραφικές παραστάσεις
#
# ΒΑΣΙΚΕΣ ΕΝΤΟΛΕΣ:
#   sympy.integrate(f, x)        → αόριστο ολοκλήρωμα ∫f(x)dx
#   sympy.diff(F, x)             → παραγώγιση (έλεγχος)
#   sympy.simplify(expr)         → απλοποίηση· simplify(F'-f)==0 ⇔ σωστό
#   sympy.lambdify(x, expr)      → μετατροπή σε αριθμητική συνάρτηση
#   matplotlib.pyplot            → plot, subplots
#
# Δραστηριότητα βιβλίου:
#   (α) πίνακας βασικών ολοκληρωμάτων
#   (β) συνάρτηση check(f) — αυτόματη επαλήθευση
#   (γ) γραμμική αντικατάσταση ∫f(ax+b)dx = (1/a)F(ax+b) + C
#   (δ) αντικατάσταση u — f και F σε κοινό διάγραμμα
#   (ε) πρόβλημα αρχικών τιμών + κίνηση a(t)=6t-4, v(0)=2, x(0)=1
# ============================================================

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

x, t, u, a, b, C = sp.symbols('x t u a b C')

print("=" * 60)
print(" Κεφάλαιο 13: Αόριστο Ολοκλήρωμα — Python")
print("=" * 60)

# ── Α. (α) Πίνακας βασικών ολοκληρωμάτων ──────────────────
print("\n──── Α. (α) Πίνακας βασικών ολοκληρωμάτων ────")

basics = [
    ("x^4",           x**4),
    ("sqrt(x)",       sp.sqrt(x)),
    ("e^x",           sp.exp(x)),
    ("1/x",           1/x),
    ("sin(x)",        sp.sin(x)),
    ("sec^2(x)",      sp.sec(x)**2),
    ("1/(1+x^2)",     1/(1 + x**2)),
    ("1/sqrt(1-x^2)", 1/sp.sqrt(1 - x**2)),
]

print(f"  {'f(x)':<16s} → F(x) = ∫f(x)dx  (+ C)")
table = []
for name, f in basics:
    F = sp.trigsimp(sp.integrate(f, x))
    table.append((name, f, F))
    print(f"  {name:<16s} → {F}")

print("\n  [Αναμένεται: x^5/5, 2x^(3/2)/3, exp(x), log(x), -cos(x),")
print("               tan(x), atan(x), asin(x)]")

# ── Β. (β) Αυτόματη επαλήθευση: η συνάρτηση check(f) ──────
print("\n──── Β. (β) Αυτόματη επαλήθευση με check(f) ────")


def check(f, var=x):
    """Ολοκληρώνει την f, παραγωγίζει το αποτέλεσμα και ελέγχει ότι
    simplify(diff(F, var) - f) == 0. Επιστρέφει (F, True/False)."""
    F = sp.integrate(f, var)
    residual = sp.simplify(sp.diff(F, var) - f)
    return F, residual == 0


all_ok = True
for name, f, _ in table:
    F, ok = check(f)
    all_ok = all_ok and ok
    print(f"  check({name:<16s}) → F = {str(F):<18s}  d/dx F - f = 0 ? {ok}")

print(f"\n  Πέρασαν όλοι οι έλεγχοι του (α); {all_ok}")

# ── Γ. (γ) Γραμμική αντικατάσταση ─────────────────────────
print("\n──── Γ. (γ) Γραμμική αντικατάσταση ────")

for name, f, aval in [("e^(2x)",   sp.exp(2*x),  2),
                      ("cos(3x)",  sp.cos(3*x),  3),
                      ("(4x+1)^5", (4*x + 1)**5, 4)]:
    F, ok = check(f)
    print(f"  ∫{name:<10s} dx = {sp.factor(F)}")
    print(f"      a = {aval}  →  εμφανίζεται ο παράγοντας 1/a = 1/{aval}"
          f"   [έλεγχος: {ok}]")

# Το (4x+1)^5 σε κλειστή μορφή: διαφέρει από το (4x+1)^6/24 κατά σταθερά
d = sp.simplify((4*x + 1)**6/24 - sp.integrate((4*x + 1)**5, x))
print(f"\n  (4x+1)^6/24 - F(x) = {d}  (σταθερά — απορροφάται στο C)")

# Συμβολική επιβεβαίωση του κανόνα ∫f(ax+b)dx = (1/a)F(ax+b) + C
print("\n  Συμβολική επιβεβαίωση του κανόνα ∫f(ax+b)dx = (1/a)·F(ax+b) + C:")
aa, bb = sp.symbols('a b', nonzero=True, real=True)
for name, fu in [("e^u", sp.exp(u)), ("cos(u)", sp.cos(u)),
                 ("1/u", 1/u), ("u^5", u**5)]:
    Fu = sp.integrate(fu, u)                      # F: αντιπαράγωγος της f
    cand = Fu.subs(u, aa*x + bb)/aa               # υποψήφια αντιπαράγωγος
    res = sp.simplify(sp.diff(cand, x) - fu.subs(u, aa*x + bb))
    print(f"    f(u) = {name:<8s}  d/dx[(1/a)F(ax+b)] - f(ax+b) = {res}")
print("    [Όλα 0 ⇒ ο κανόνας επαληθεύεται συμβολικά.]")

# ── Δ. (δ) Αντικατάσταση u = g(x) ─────────────────────────
print("\n──── Δ. (δ) Αντικατάσταση u = g(x) ────")

subs_cases = [
    ("2x·cos(x^2)",  2*x*sp.cos(x**2),      (-2.2, 2.2)),
    ("x·e^(x^2)",    x*sp.exp(x**2),        (-1.5, 1.5)),
    ("sin^3(x)cos x", sp.sin(x)**3*sp.cos(x), (0.0, 2*np.pi)),
    ("x·sqrt(x+1)",  x*sp.sqrt(x + 1),      (-1.0, 3.0)),
]

results = []
for name, f, rng in subs_cases:
    F, ok = check(f)
    results.append((name, f, F, rng))
    print(f"  ∫{name:<15s} dx = {sp.simplify(F)}     [έλεγχος F'-f=0: {ok}]")

print("\n  [Αναμένεται: sin(x^2), exp(x^2)/2, sin^4(x)/4,")
print("               2(x+1)^(3/2)(3x-2)/15]")

# f και F στο ΙΔΙΟ διάγραμμα — πρόσημο της f ↔ μονοτονία της F
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
fig.suptitle("Αντικατάσταση u: η f (πρόσημο) και η αντιπαράγωγός της F "
             "(μονοτονία)", fontsize=12, fontweight='bold')

for ax, (name, f, F, (lo, hi)) in zip(axes.ravel(), results):
    xs = np.linspace(lo + 1e-6, hi, 500)
    fn = sp.lambdify(x, f, 'numpy')
    Fn = sp.lambdify(x, F, 'numpy')
    yf = np.real(np.asarray(fn(xs), dtype=complex))
    yF = np.real(np.asarray(Fn(xs), dtype=complex))
    ax.plot(xs, yf, 'b-', lw=1.8, label=f"f(x) = {name}")
    ax.plot(xs, yF, 'r-', lw=1.8, label="F(x) = ∫f dx")
    ax.fill_between(xs, 0, yf, where=(yf > 0), color='green', alpha=.15)
    ax.fill_between(xs, 0, yf, where=(yf < 0), color='red',   alpha=.15)
    ax.axhline(0, color='k', lw=.6)
    ax.legend(fontsize=8); ax.grid(alpha=.3)

plt.tight_layout()

# Απάντηση στο ερώτημα του βιβλίου — αριθμητικά, όχι με ισχυρισμό
print("\n  Πρόσημο της f ↔ μονοτονία της F (δειγματοληψία σε 400 σημεία):")
for name, f, F, (lo, hi) in results:
    xs = np.linspace(lo + 1e-3, hi - 1e-3, 400)
    fn = sp.lambdify(x, f, 'numpy')
    Fn = sp.lambdify(x, F, 'numpy')
    yf = np.real(np.asarray(fn(xs), dtype=complex))
    yF = np.real(np.asarray(Fn(xs), dtype=complex))
    dF = np.gradient(yF, xs)                 # αριθμητική παράγωγος της F
    same = np.sign(dF[3:-3]) == np.sign(yf[3:-3])
    print(f"    {name:<15s}: sign(F') == sign(f) στο "
          f"{100*np.mean(same):.1f}% των σημείων")
print("    ⇒ όπου f > 0 η F αύξουσα, όπου f < 0 η F φθίνουσα (F' = f).")

# ── Ε. (ε) Πρόβλημα αρχικών τιμών ─────────────────────────
print("\n──── Ε. (ε) Πρόβλημα αρχικών τιμών ────")

# Ε1. y' = cos x  →  y = sin x + C,  με y(0) = 1
F_cos = sp.integrate(sp.cos(x), x)
print(f"  y' = cos x  →  y = {F_cos} + C")
C_sol = sp.solve(sp.Eq(F_cos.subs(x, 0) + C, 1), C)
print(f"  Συνθήκη y(0) = 1:  sin(0) + C = 1  →  C = {C_sol[0]}"
      f"   [Αναμένεται C = 1]")
y_part = F_cos + C_sol[0]
print(f"  Μερική λύση: y = {y_part}   (έλεγχος y(0) = {y_part.subs(x, 0)})")

xs = np.linspace(-2*np.pi, 2*np.pi, 600)
plt.figure(figsize=(8, 4.6))
for c in (-2, -1, 0, 1, 2):
    if c == 1:
        plt.plot(xs, np.sin(xs) + c, 'r-', lw=2.6,
                 label="C = 1  (μερική λύση, y(0)=1)")
    else:
        plt.plot(xs, np.sin(xs) + c, lw=1.2, alpha=.8, label=f"C = {c}")
plt.plot([0], [1], 'ro', ms=8, zorder=5)
plt.annotate("(0, 1)", (0, 1), textcoords="offset points", xytext=(10, 8))
plt.axhline(0, color='k', lw=.6); plt.axvline(0, color='k', lw=.6)
plt.title("y' = cos x: η οικογένεια y = sin x + C  (C = -2,…,2)")
plt.xlabel("x"); plt.ylabel("y"); plt.legend(fontsize=8); plt.grid(alpha=.3)
plt.tight_layout()

# Ε2. Κίνηση: a(t) = 6t - 4,  v(0) = 2,  x(0) = 1
print("\n  Κίνηση: a(t) = 6t - 4,  v(0) = 2,  x(0) = 1")
a_t = 6*t - 4
C1, C2 = sp.symbols('C1 C2')

v_gen = sp.integrate(a_t, t) + C1
C1_val = sp.solve(sp.Eq(v_gen.subs(t, 0), 2), C1)[0]
v_t = sp.expand(v_gen.subs(C1, C1_val))
print(f"    v(t) = ∫a dt = {v_t}      (C1 = {C1_val}, v(0) = {v_t.subs(t, 0)})")

x_gen = sp.integrate(v_t, t) + C2
C2_val = sp.solve(sp.Eq(x_gen.subs(t, 0), 1), C2)[0]
x_t = sp.expand(x_gen.subs(C2, C2_val))
print(f"    x(t) = ∫v dt = {x_t}   (C2 = {C2_val}, x(0) = {x_t.subs(t, 0)})")
print(f"    [Αναμένεται v(t) = 3t^2 - 4t + 2,  x(t) = t^3 - 2t^2 + 2t + 1]")

print(f"    Έλεγχοι: v'(t) - a(t) = {sp.simplify(sp.diff(v_t, t) - a_t)},"
      f"  x'(t) - v(t) = {sp.simplify(sp.diff(x_t, t) - v_t)}")

ts = np.linspace(0, 2.5, 400)
a_n = sp.lambdify(t, a_t, 'numpy')
v_n = sp.lambdify(t, v_t, 'numpy')
x_n = sp.lambdify(t, x_t, 'numpy')

fig, (axa, axv, axx) = plt.subplots(3, 1, figsize=(7.5, 8), sharex=True)
fig.suptitle("Κίνηση: a(t) = 6t - 4,  v(0) = 2,  x(0) = 1",
             fontsize=12, fontweight='bold')

axa.plot(ts, a_n(ts), 'g-', lw=2)
axa.axhline(0, color='k', lw=.6)
axa.set_ylabel("a(t)"); axa.set_title("επιτάχυνση  a(t) = 6t - 4", fontsize=10)
axa.grid(alpha=.3)

axv.plot(ts, v_n(ts), 'b-', lw=2)
axv.axhline(0, color='k', lw=.6)
axv.set_ylabel("v(t)")
axv.set_title(f"ταχύτητα  v(t) = ∫a dt = {v_t}", fontsize=10)
axv.plot([0], [2], 'bo', ms=7)
axv.grid(alpha=.3)

axx.plot(ts, x_n(ts), 'r-', lw=2)
axx.axhline(0, color='k', lw=.6)
axx.set_ylabel("x(t)"); axx.set_xlabel("t")
axx.set_title(f"θέση  x(t) = ∫v dt = {x_t}", fontsize=10)
axx.plot([0], [1], 'ro', ms=7)
axx.grid(alpha=.3)

plt.tight_layout()

# Ελάχιστο της ταχύτητας: εκεί όπου a(t) = 0
t_star = sp.solve(sp.Eq(a_t, 0), t)[0]
print(f"    a(t) = 0 στο t = {t_star}  →  εκεί η v έχει ακρότατο: "
      f"v({t_star}) = {v_t.subs(t, t_star)}")

print("\n✓ Ολοκληρώθηκε το Κεφάλαιο 13.")
plt.show()
