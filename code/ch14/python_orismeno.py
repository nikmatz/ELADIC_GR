# ============================================================
# python_orismeno.py
# Κεφάλαιο 14 — Ορισμένο Ολοκλήρωμα: Αθροίσματα Riemann
# Ματζάκος, Ν. (2026). Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
# ============================================================
#
# ΒΙΒΛΙΟΘΗΚΕΣ:
#   numpy       → διανυσματικά αθροίσματα Riemann (γρήγορα, χωρίς βρόχους)
#   sympy       → συμβολικό άθροισμα και όριο (από τον ορισμό)
#   scipy       → αριθμητική ολοκλήρωση αναφοράς
#   matplotlib  → bar, fill_between, loglog
#
# ΒΑΣΙΚΕΣ ΕΝΤΟΛΕΣ:
#   numpy.linspace(a, b, n+1)     → διαμέριση του [a,b]
#   numpy.sum(f(x)) * dx          → άθροισμα Riemann
#   sympy.summation(expr,(i,1,n)) → κλειστός τύπος αθροίσματος
#   sympy.limit(expr, n, oo)      → όριο καθώς n → ∞
#   matplotlib.bar()              → ορθογώνια Riemann
#   matplotlib.fill_between()     → σκίαση χωρίου
#
# Δραστηριότητα βιβλίου:
#   (α) riemann(f,a,b,n,rule) για f(x)=x² στο [0,2], n=4,10,10²,10⁴
#   (β) οπτικοποίηση ορθογωνίων για n=4 και n=20
#   (γ) ταχύτητα σύγκλισης σε διάγραμμα log-log
#   (δ) συμβολικά από τον ορισμό (sympy.summation, sympy.limit)
#   (ε) ιδιότητες: γραμμικότητα, πρόσθεση, αντιστροφή, συμμετρίες
#   (ζ) φράγματα για e^(x²) στο [0,1] και στένεμά τους
#   (η) μέση τιμή της x² στο [0,3] με fill_between
# ============================================================

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad

print("=" * 62)
print(" Κεφάλαιο 14: Ορισμένο Ολοκλήρωμα — Αθροίσματα Riemann")
print("=" * 62)

EXACT = 8/3          # η ακριβής τιμή του ∫₀² x² dx


# ── Α. (α) Η συνάρτηση riemann(f, a, b, n, rule) ──────────
def riemann(f, a, b, n, rule='left'):
    """Άθροισμα Riemann της f στο [a,b] με n υποδιαστήματα.

    rule = 'left'  → αριστερά άκρα   x_(i-1)
    rule = 'mid'   → μέσα            (x_(i-1)+x_i)/2
    rule = 'right' → δεξιά άκρα      x_i
    Υλοποίηση με numpy (διανυσματική) — δουλεύει άνετα και για n = 10^4.
    """
    edges = np.linspace(a, b, n + 1)
    dx = (b - a)/n
    if rule == 'left':
        pts = edges[:-1]
    elif rule == 'right':
        pts = edges[1:]
    elif rule == 'mid':
        pts = (edges[:-1] + edges[1:])/2
    else:
        raise ValueError("rule ∈ {'left', 'mid', 'right'}")
    return float(np.sum(f(pts)) * dx)


f = lambda z: z**2

print("\n──── Α. (α) riemann(f, a, b, n, rule) για f(x)=x² στο [0,2] ────")
print(f"  Ακριβής τιμή: ∫₀²x²dx = 8/3 = {EXACT:.10f}")
print(f"\n  {'n':>7} {'left':>14} {'mid':>14} {'right':>14}"
      f" {'|mid-8/3|':>12}")
for n in (4, 10, 10**2, 10**4):
    L = riemann(f, 0, 2, n, 'left')
    M = riemann(f, 0, 2, n, 'mid')
    R = riemann(f, 0, 2, n, 'right')
    print(f"  {n:>7d} {L:>14.8f} {M:>14.8f} {R:>14.8f} {abs(M-EXACT):>12.2e}")

print("\n  Έλεγχος L_n < 8/3 < R_n για κάθε n:")
for n in (4, 10, 10**2, 10**4):
    L = riemann(f, 0, 2, n, 'left')
    R = riemann(f, 0, 2, n, 'right')
    print(f"    n = {n:>6d}: {L:.8f} < {EXACT:.8f} < {R:.8f}  →  "
          f"{L < EXACT < R}")

# ── Β. (β) Οπτικοποίηση των ορθογωνίων Riemann ────────────
print("\n──── Β. (β) Τα ορθογώνια Riemann για n = 4 και n = 20 ────")

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
fig.suptitle("Αθροίσματα Riemann για f(x) = x² στο [0,2]  —  "
             "όσο μεγαλώνει το n, τα ορθογώνια γεμίζουν το χωρίο",
             fontsize=11, fontweight='bold')

for ax, n in zip(axes, (4, 20)):
    edges = np.linspace(0, 2, n + 1)
    dx = 2/n
    ax.bar(edges[:-1], f(edges[1:]), width=dx, align='edge',
           color='steelblue', edgecolor='white', alpha=.55,
           label=f'δεξιά άκρα (n={n})')
    ax.bar(edges[:-1], f(edges[:-1]), width=dx, align='edge',
           color='none', edgecolor='darkorange', lw=1.2,
           label=f'αριστερά άκρα (n={n})')
    zz = np.linspace(0, 2, 400)
    ax.plot(zz, f(zz), 'r-', lw=2.2, label='f(x) = x²')
    L = riemann(f, 0, 2, n, 'left'); R = riemann(f, 0, 2, n, 'right')
    ax.set_title(f"n = {n}:  L_n = {L:.4f},  R_n = {R:.4f}", fontsize=10)
    ax.set_xlabel("x"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    print(f"  n = {n:2d}: L_n = {L:.6f},  R_n = {R:.6f},  "
          f"πλάτος φραγμάτων R_n - L_n = {R-L:.6f}")

plt.tight_layout()

# ── Γ. (γ) Ταχύτητα σύγκλισης — διάγραμμα log-log ─────────
print("\n──── Γ. (γ) Ταχύτητα σύγκλισης (log-log) ────")

ns = np.array([2**k for k in range(2, 15)])          # 4 … 16384
errors = {}
for rule in ('left', 'mid', 'right'):
    errors[rule] = np.array([abs(riemann(f, 0, 2, int(n), rule) - EXACT)
                             for n in ns])

plt.figure(figsize=(7.2, 5))
styles = {'left': ('o-', 'darkorange'), 'mid': ('s-', 'seagreen'),
          'right': ('^-', 'steelblue')}
for rule in ('left', 'mid', 'right'):
    mk, col = styles[rule]
    plt.loglog(ns, errors[rule], mk, color=col, ms=4, lw=1.4,
               label=f"{rule}")
# Γραμμές αναφοράς κλίσης -1 και -2
plt.loglog(ns, 4.0/ns, 'k--', lw=1, alpha=.6, label='αναφορά ~ 1/n')
plt.loglog(ns, 4.0/ns**2, 'k:', lw=1.2, alpha=.8, label='αναφορά ~ 1/n²')
plt.xlabel("n (πλήθος υποδιαστημάτων)")
plt.ylabel("|S_n - 8/3|")
plt.title("Ταχύτητα σύγκλισης των τριών κανόνων (log-log)")
plt.legend(fontsize=9); plt.grid(alpha=.3, which='both')
plt.tight_layout()

print("  Κλίση της ευθείας στο log-log (= -p, όπου σφάλμα ~ n^(-p)):")
for rule in ('left', 'mid', 'right'):
    slope = np.polyfit(np.log(ns), np.log(errors[rule]), 1)[0]
    print(f"    {rule:>5s}: κλίση = {slope:+.4f}  →  σφάλμα ~ n^({slope:.2f})"
          f"   [τάξη p ≈ {abs(slope):.0f}]")
print("  ⇒ οι left/right συγκλίνουν σαν 1/n, ο mid σαν 1/n².")

# ── Δ. (δ) Συμβολικά, από τον ορισμό ──────────────────────
print("\n──── Δ. (δ) Συμβολικά από τον ορισμό (χωρίς αντιπαράγωγο) ────")

i = sp.symbols('i', integer=True, positive=True)
n = sp.symbols('n', integer=True, positive=True)

# Δx = 2/n, δεξιά άκρα x_i = 2i/n
R_n = sp.factor(sp.simplify(sp.summation((2*i/n)**2 * (2/n), (i, 1, n))))
L_n = sp.factor(sp.simplify(sp.summation((2*(i-1)/n)**2 * (2/n), (i, 1, n))))
M_n = sp.factor(sp.simplify(sp.summation(((2*i-1)/n)**2 * (2/n), (i, 1, n))))

print(f"  Σ i² = {sp.factor(sp.summation(i**2, (i, 1, n)))}")
print(f"  R_n = {R_n}")
print(f"  L_n = {L_n}")
print(f"  M_n = {M_n}")
print(f"  lim R_n = {sp.limit(R_n, n, sp.oo)}   [ΑΝΑΜΕΝΟΜΕΝΟ 8/3]")
print(f"  lim L_n = {sp.limit(L_n, n, sp.oo)}")
print(f"  lim M_n = {sp.limit(M_n, n, sp.oo)}")

xs_ = sp.symbols('x')
print(f"  Επαλήθευση με sympy.integrate: "
      f"∫₀²x²dx = {sp.integrate(xs_**2, (xs_, 0, 2))}")

# ── Ε. (ε) Ιδιότητες — αριθμητική επαλήθευση ──────────────
print("\n──── Ε. (ε) Ιδιότητες του ορισμένου ολοκληρώματος ────")

Q = lambda g, a, b: quad(g, a, b)[0]      # αριθμητικό ∫_a^b g

# Γραμμικότητα
lhs = Q(lambda z: 3*z**2 + 5*np.sin(z), 0, 1)
rhs = 3*Q(lambda z: z**2, 0, 1) + 5*Q(np.sin, 0, 1)
print(f"  Γραμμικότητα : ∫₀¹(3x²+5sin x) = {lhs:.12f}")
print(f"                 3∫₀¹x² + 5∫₀¹sin x = {rhs:.12f}"
      f"   |διαφορά| = {abs(lhs-rhs):.2e}")

# Πρόσθεση διαστημάτων
whole = Q(lambda z: z**2, 0, 3)
split = Q(lambda z: z**2, 0, 1) + Q(lambda z: z**2, 1, 3)
print(f"  Πρόσθεση     : ∫₀³x² = {whole:.12f} ,  ∫₀¹+∫₁³ = {split:.12f}"
      f"   |διαφορά| = {abs(whole-split):.2e}")

# Αντιστροφή ορίων
fwd = Q(lambda z: z**2, 1, 4)
bwd = Q(lambda z: z**2, 4, 1)
print(f"  Αντιστροφή   : ∫₁⁴x² = {fwd:.12f} ,  ∫₄¹x² = {bwd:.12f}"
      f"   άθροισμα = {fwd+bwd:.2e}")

# Συμμετρίες
odd = Q(lambda z: z**3, -2, 2)
even_full = Q(lambda z: z**2, -2, 2)
even_half = 2*Q(lambda z: z**2, 0, 2)
print(f"  Περιττή      : ∫₋₂²x³ = {odd:.2e}   [ΑΝΑΜΕΝΟΜΕΝΟ 0]")
print(f"  Άρτια        : ∫₋₂²x² = {even_full:.12f} ,  "
      f"2∫₀²x² = {even_half:.12f}   |διαφορά| = {abs(even_full-even_half):.2e}")

# ── Ζ. (ζ) Φράγματα για f(x) = e^(x²) στο [0,1] ───────────
print("\n──── Ζ. (ζ) Φράγματα για f(x) = e^(x²) στο [0,1] ────")

g = lambda z: np.exp(z**2)
I_exact = Q(g, 0, 1)
print(f"  ∫₀¹ e^(x²) dx = {I_exact:.12f}   (quad)")
print(f"  Η f είναι αύξουσα στο [0,1] (f' = 2x·e^(x²) ≥ 0),")
print(f"  άρα m = f(0) = {g(0.0):.6f} και M = f(1) = e = {g(1.0):.6f}.")
print(f"  Χονδρικό φράγμα: {g(0.0):.6f} ≤ ∫ ≤ {g(1.0):.6f}   →  "
      f"{g(0.0) <= I_exact <= g(1.0)}")

print("\n  Στένεμα με διάσπαση του [0,1] σε k ίσα υποδιαστήματα")
print("  (σε καθένα m_j = f(αριστερό άκρο), M_j = f(δεξιό άκρο)):")
print(f"  {'k':>3} {'κάτω φράγμα':>16} {'άνω φράγμα':>16} {'πλάτος':>12}"
      f" {'περιέχει ∫;':>12}")
for k in (1, 2, 4, 8):
    e_ = np.linspace(0, 1, k + 1)
    h = 1/k
    lower = float(np.sum(g(e_[:-1])) * h)     # f αύξουσα ⇒ min στο αριστερό
    upper = float(np.sum(g(e_[1:])) * h)      #             max στο δεξιό
    print(f"  {k:>3d} {lower:>16.10f} {upper:>16.10f} {upper-lower:>12.6f}"
          f" {str(lower <= I_exact <= upper):>12s}")

print("\n  Σύγκριση: στο [0,1] ισχύει x² ≤ x, άρα e^(x²) ≤ e^x.")
I_ex = Q(np.exp, 0, 1)
print(f"    ∫₀¹e^(x²)dx = {I_exact:.12f}   ≤   ∫₀¹e^x dx = {I_ex:.12f}"
      f"   →  {I_exact <= I_ex}")

# ── Η. (η) Μέση τιμή της f(x) = x² στο [0,3] ──────────────
print("\n──── Η. (η) Μέση τιμή της f(x) = x² στο [0,3] ────")

N_big = 10**4
I_riem = riemann(f, 0, 3, N_big, 'mid')
fbar = I_riem / (3 - 0)
print(f"  ∫₀³x²dx ≈ {I_riem:.10f} (μεσαίο άθροισμα Riemann, n = {N_big})"
      f"   [ακριβές: 9]")
print(f"  Μέση τιμή f̄ = ∫₀³f/(b-a) ≈ {fbar:.10f}   [ΑΝΑΜΕΝΟΜΕΝΟ 3]")

xi = np.sqrt(fbar)
print(f"  ΘΜΤ: f(ξ) = f̄  ⇒  ξ = √f̄ ≈ {xi:.10f}   [ΑΝΑΜΕΝΟΜΕΝΟ √3 ≈ "
      f"{np.sqrt(3):.10f}]")
print(f"  Εμβαδόν ορθογωνίου f̄·(b-a) = {fbar*3:.10f}  έναντι  "
      f"∫₀³f = {I_riem:.10f}   |διαφορά| = {abs(fbar*3 - I_riem):.2e}")

zz = np.linspace(0, 3, 400)
plt.figure(figsize=(7.5, 5))
plt.fill_between(zz, 0, f(zz), color='steelblue', alpha=.35,
                 label="χωρίο κάτω από την f(x)=x²   (εμβαδόν = 9)")
plt.fill_between([0, 3], 0, [fbar, fbar], color='crimson', alpha=.18,
                 label=f"ορθογώνιο ύψους f̄ = {fbar:.4f}  "
                       f"(εμβαδόν = {fbar*3:.4f})")
plt.plot(zz, f(zz), 'b-', lw=2.2)
plt.axhline(fbar, color='crimson', lw=2, ls='--')
plt.plot([xi], [fbar], 'ko', ms=8, zorder=5)
plt.annotate(f"ξ ≈ {xi:.4f}", (xi, fbar), textcoords="offset points",
             xytext=(10, -18), fontsize=10)
plt.vlines(xi, 0, fbar, color='k', ls=':', lw=1.2)
plt.xlabel("x"); plt.ylabel("y"); plt.xlim(0, 3); plt.ylim(0, 9.5)
plt.title("Μέση τιμή: τα δύο εμβαδά είναι ίσα (Θεώρημα Μέσης Τιμής)")
plt.legend(fontsize=9, loc='upper left'); plt.grid(alpha=.3)
plt.tight_layout()

print("\n✓ Ολοκληρώθηκε το Κεφάλαιο 14.")
plt.show()
