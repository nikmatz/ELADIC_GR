# ============================================================
# python_ode.py
# Κεφάλαιο 15 — Διαφορικές Εξισώσεις
# Βιβλίο: Μαθηματικά Ι · ΑΣΠΕΤΕ
# Συγγραφέας: Ν. Ματζάκος
# ============================================================
#
# ΒΙΒΛΙΟΘΗΚΕΣ:
#   sympy              → ode, dsolve, classify_ode
#   scipy.integrate    → solve_ivp (αριθμητική επίλυση)
#   numpy              → αριθμητικές λύσεις
#   matplotlib         → λύσεις και φασικά πορτρέτα
#
# ΒΑΣΙΚΕΣ ΕΝΤΟΛΕΣ:
#   sympy.dsolve(eq, f(x))          → γενική λύση ΔΕ
#   sympy.classify_ode(eq)          → ταξινόμηση ΔΕ
#   scipy.integrate.solve_ivp       → αριθμητική επίλυση
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sympy import (symbols, Function, dsolve, classify_ode,
                   Eq, diff, exp, sin, cos, tan, sqrt,
                   simplify, lambdify, pprint, pi, E,
                   solve, Rational, log)
from scipy.integrate import solve_ivp

x, t = symbols('x t')
y    = Function('y')
P    = Function('P')

print("=" * 55)
print(" Κεφάλαιο 15: Διαφορικές Εξισώσεις — Python")
print("=" * 55)

# ── Α. Χωριζόμενες Μεταβλητές ────────────────────────────
print("\n── Α. Χωριζόμενες Μεταβλητές ──")

sep_cases = [
    (Eq(diff(y(x),x), x*y(x)),            "dy/dx = x·y"),
    (Eq(diff(y(x),x), y(x)**2),           "dy/dx = y²"),
    (Eq(diff(y(x),x), (x**2+1)/(y(x)+1)),"dy/dx = (x²+1)/(y+1)"),
]

for eq, label in sep_cases:
    sol = dsolve(eq, y(x))
    ode_type = classify_ode(eq, y(x))[0]
    print(f"  {label}  [{ode_type}]")
    print(f"    {sol}")

# Ειδική λύση: dy/dx = x·y, y(0)=2
eq_sp = Eq(diff(y(x),x), x*y(x))
sol_sp = dsolve(eq_sp, y(x), ics={y(0): 2})
print(f"\n  Ειδική λύση dy/dx=x·y, y(0)=2:")
print(f"    {sol_sp}")

# ── Β. Γραμμικές ΔΕ 1ης Τάξης ────────────────────────────
print("\n── Β. Γραμμικές ΔΕ 1ης Τάξης: y' + P(x)y = Q(x) ──")

linear_cases = [
    (Eq(diff(y(x),x) + 2*y(x), 4*x),          "y' + 2y = 4x"),
    (Eq(diff(y(x),x) - y(x)/x, x**2),          "y' - y/x = x²"),
    (Eq(x*diff(y(x),x) + y(x), x*sin(x)),      "x·y' + y = x·sinx"),
]

for eq, label in linear_cases:
    sol = dsolve(eq, y(x))
    print(f"  {label}")
    print(f"    {sol}")

# Με αρχική συνθήκη
eq_lin = Eq(diff(y(x),x) + 2*y(x), 4*x)
sol_lin = dsolve(eq_lin, y(x), ics={y(0): 1})
print(f"\n  Ειδική λύση y'+2y=4x, y(0)=1:")
print(f"    {sol_lin}")

# ── Γ. ΔΕ 2ης Τάξης — Ομογενείς ─────────────────────────
print("\n── Γ. ΔΕ 2ης Τάξης: ay'' + by' + cy = 0 ──")

hom_cases = [
    (Eq(diff(y(x),x,2) - 5*diff(y(x),x) + 6*y(x), 0),
     "y'' - 5y' + 6y = 0", "r=2,3 (πραγματικές)"),
    (Eq(diff(y(x),x,2) + 4*diff(y(x),x) + 4*y(x), 0),
     "y'' + 4y' + 4y = 0", "r=-2 (διπλή)"),
    (Eq(diff(y(x),x,2) + 4*y(x), 0),
     "y'' + 4y = 0",        "r=±2i (μιγαδικές)"),
    (Eq(diff(y(x),x,2) + 2*diff(y(x),x) + 5*y(x), 0),
     "y'' + 2y' + 5y = 0",  "r=-1±2i (αποσβ.)"),
]

for eq, label, note in hom_cases:
    sol = dsolve(eq, y(x))
    print(f"  {label}  [{note}]")
    print(f"    {sol}")

# Ειδική λύση y''+4y=0, y(0)=1, y'(0)=0
eq_shm = Eq(diff(y(x),x,2) + 4*y(x), 0)
sol_shm = dsolve(eq_shm, y(x), ics={y(0): 1, diff(y(x),x).subs(x,0): 0})
print(f"\n  Ειδική y''+4y=0, y(0)=1, y'(0)=0:")
print(f"    {sol_shm}")

# ── Δ. Μη Ομογενείς ΔΕ 2ης Τάξης ─────────────────────────
print("\n── Δ. Μη Ομογενείς: ay'' + by' + cy = f(x) ──")

inhom_cases = [
    (Eq(diff(y(x),x,2) - 3*diff(y(x),x) + 2*y(x), exp(x)),
     "y'' - 3y' + 2y = eˣ"),
    (Eq(diff(y(x),x,2) + y(x), sin(x)),
     "y'' + y = sinx  (συντονισμός)"),
    (Eq(diff(y(x),x,2) + 2*diff(y(x),x) + 5*y(x), 10*cos(2*x)),
     "y'' + 2y' + 5y = 10cos(2x)"),
]

for eq, label in inhom_cases:
    sol = dsolve(eq, y(x))
    print(f"  {label}")
    print(f"    {sol}")

# ── Ε. Εφαρμογές ─────────────────────────────────────────
print("\n── Ε. Εφαρμογές ──")

# E1. Εκθετική αύξηση/φθορά: dP/dt = k·P
print("  Μοντέλο Malthus: dP/dt = k·P")
k_growth = 0.03
P_func   = lambda t_val: 1000 * np.exp(k_growth * t_val)
for yr in [0, 10, 25, 50]:
    print(f"    P({yr}) = {P_func(yr):.0f}")

# E2. Logistic growth: dP/dt = r·P·(1 - P/K)
print("\n  Λογιστικό μοντέλο: dP/dt = 0.1·P·(1 - P/1000)")
r_log, K_log = 0.1, 1000.0

def logistic(t_val, P_val):
    return r_log * P_val[0] * (1 - P_val[0]/K_log)

sol_log = solve_ivp(logistic, [0, 100], [50], dense_output=True)
t_log   = np.linspace(0, 100, 300)
P_log   = sol_log.sol(t_log)[0]
print(f"    P(0)=50, P(50)≈{P_log[150]:.0f}, P(100)≈{P_log[-1]:.0f}")

# E3. Μηχανική ταλάντωση: my'' + cy' + ky = 0
print("\n  Αποσβενόμενη Ταλάντωση: y'' + 2y' + 5y = 0")
def damped(t_val, yv):
    return [yv[1], -2*yv[1] - 5*yv[0]]

sol_damp = solve_ivp(damped, [0, 10], [1, 0], dense_output=True)
t_damp   = np.linspace(0, 10, 400)
y_damp   = sol_damp.sol(t_damp)

# E4. Κύκλωμα RC: RC·dQ/dt + Q = E·C, RC=0.1, E=10V
print("  Κύκλωμα RC: 0.1·dQ/dt + Q = 1  (Q(0)=0)")
RC_val = 0.1

def rc_circuit(t_val, Q):
    return [(1 - Q[0]) / RC_val]

sol_rc = solve_ivp(rc_circuit, [0, 1], [0], dense_output=True)
t_rc   = np.linspace(0, 1, 300)
Q_rc   = sol_rc.sol(t_rc)[0]
print(f"    Q(0.1s)≈{Q_rc[30]:.4f},  Q(∞)→1.0 (=E·C)")

# ── Στ. Αριθμητική Επίλυση & Φασικά Πορτρέτα ─────────────
print("\n── Στ. Γραφικές Παραστάσεις ──")

fig = plt.figure(figsize=(14, 9))
fig.suptitle("Διαφορικές Εξισώσεις", fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(2, 3, fig, hspace=0.45, wspace=0.35)

# 1. Χωριζόμενες — λύσεις dy/dx = x·y για διάφορες C
ax1 = fig.add_subplot(gs[0, 0])
xv1 = np.linspace(-2, 2, 400)
for C_val, col in zip([-2, -1, 1, 2], ['royalblue','tomato','forestgreen','orange']):
    ax1.plot(xv1, C_val*np.exp(xv1**2/2), lw=1.8, color=col, label=f'C={C_val}')
ax1.set_ylim(-6, 6); ax1.set_xlim(-2, 2)
ax1.axhline(0, color='k', lw=0.5); ax1.axvline(0, color='k', lw=0.5)
ax1.set_title(r"$dy/dx = xy$: οικογένεια λύσεων", fontsize=10)
ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

# 2. Γραμμική ΔΕ: y' + 2y = 4x — ειδική λύση
ax2 = fig.add_subplot(gs[0, 1])
xv2 = np.linspace(0, 3, 300)
# Γενική: y = 2x - 1 + C·e^(-2x)
for C_val, col in zip([-2, 0, 1, 3], ['royalblue','tomato','forestgreen','orange']):
    ax2.plot(xv2, 2*xv2 - 1 + C_val*np.exp(-2*xv2), lw=1.8, color=col,
             label=f'C={C_val}')
ax2.plot(xv2, 2*xv2 - 1 + 2*np.exp(-2*xv2), 'k', lw=2.5, ls='--', label='y(0)=1')
ax2.set_title(r"$y'+2y=4x$: οικογένεια λύσεων", fontsize=10)
ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

# 3. Αποσβενόμενη ταλάντωση
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(t_damp, y_damp[0], 'royalblue', lw=2.5, label='y(t) — απόσβεση')
ax3.plot(t_damp, y_damp[1], 'tomato', lw=1.5, ls='--', label="y'(t)")
envelope = np.exp(-t_damp)
ax3.plot(t_damp,  envelope, 'gray', lw=1, ls=':')
ax3.plot(t_damp, -envelope, 'gray', lw=1, ls=':')
ax3.axhline(0, color='k', lw=0.5)
ax3.set_title(r"$y''+2y'+5y=0$: αποσβ. ταλάντωση", fontsize=10)
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

# 4. Λογιστικό μοντέλο vs εκθετικό
ax4 = fig.add_subplot(gs[1, 0])
t4  = np.linspace(0, 100, 300)
P_exp = 50 * np.exp(r_log * t4)
ax4.plot(t4, P_log, 'royalblue', lw=2.5, label='Λογιστικό')
ax4.plot(t4, np.clip(P_exp, 0, 1200), 'tomato', lw=1.8, ls='--', label='Εκθετικό')
ax4.axhline(K_log, color='gray', ls=':', lw=1.5, label=f'K={K_log:.0f}')
ax4.set_xlabel('t (χρόνια)'); ax4.set_ylabel('P(t)')
ax4.set_title("Λογιστικό vs Εκθετικό μοντέλο", fontsize=10)
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

# 5. Κύκλωμα RC
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(t_rc, Q_rc, 'royalblue', lw=2.5, label='Q(t)')
ax5.plot(t_rc, 1 - Q_rc, 'tomato', lw=1.8, ls='--', label='V_C(t)=1-Q(t)')
ax5.axhline(1, color='gray', ls=':', lw=1)
ax5.set_xlabel('t (s)'); ax5.set_ylabel('Q (C)')
ax5.set_title("Κύκλωμα RC: φόρτιση", fontsize=10)
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

# 6. Φασικό πορτρέτο y'' + 2y' + 5y = 0
ax6 = fig.add_subplot(gs[1, 2])
for y0, v0, col in zip([1, 2, -1, 0.5], [0, 0, 0, 2],
                        ['royalblue','tomato','forestgreen','orange']):
    sol_ph = solve_ivp(damped, [0, 12], [y0, v0], dense_output=True)
    t_ph   = np.linspace(0, 12, 500)
    yph    = sol_ph.sol(t_ph)
    ax6.plot(yph[0], yph[1], lw=1.8, color=col,
             label=f'({y0},{v0})')
ax6.axhline(0, color='k', lw=0.5); ax6.axvline(0, color='k', lw=0.5)
ax6.set_xlabel('y'); ax6.set_ylabel("y'")
ax6.set_title("Φασικό Πορτρέτο: αποσβ. ταλάντωση", fontsize=10)
ax6.legend(fontsize=7); ax6.grid(True, alpha=0.3)

plt.savefig("ode_solutions.png", dpi=120, bbox_inches='tight')
print("Το διάγραμμα αποθηκεύτηκε: ode_solutions.png")
plt.show()
