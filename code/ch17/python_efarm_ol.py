# ============================================================
# python_efarm_ol.py
# Κεφάλαιο 17 — Εφαρμογές Ολοκληρωτικού Λογισμού
#   Εμβαδόν, Όγκος Περιστροφής, Μήκος Τόξου, Φυσικές Εφαρμογές
# Ματζάκος, Ν. (2026). Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
# ------------------------------------------------------------
# ΒΑΣΙΚΕΣ ΕΝΤΟΛΕΣ:
#   sympy.integrate(f, (x,a,b))   -> ορισμένο ολοκλήρωμα
#   sympy.diff(f, x)              -> παράγωγος
#   sympy.solve(f-g, x)           -> σημεία τομής
#   scipy.integrate.quad(f, a, b) -> αριθμητικό ολοκλήρωμα
#   matplotlib.fill_between()     -> σκιασμένη περιοχή
#   mpl_toolkits.mplot3d          -> 3D στερεό εκ περιστροφής
# ============================================================
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, cumulative_trapezoid
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (ενεργοποιεί το 3D)

x = sp.symbols('x', real=True)
y = sp.symbols('y', nonnegative=True)

print("=" * 62)
print(" Κεφάλαιο 17: Εφαρμογές Ολοκληρωτικού Λογισμού")
print("=" * 62)


# ============================================================
# Α. ΕΜΒΑΔΟΝ ΜΕΤΑΞΥ ΚΑΜΠΥΛΩΝ
# ============================================================
print("\n=== Α. ΕΜΒΑΔΟΝ ===")

# Α1.  int_{-1}^{1} |x^3 - x| dx
f_abs = sp.Abs(x**3 - x)
A1 = sp.integrate(f_abs, (x, -1, 1))
A1_num, A1_err = quad(lambda t: abs(t**3 - t), -1, 1)
print(f"\nΑ1. ∫_(-1)^1 |x^3-x| dx = {A1} = {float(A1)}   (αναμ. 1/2)")
print(f"    scipy quad            = {A1_num:.12f}  (εκτ. σφάλμα {A1_err:.1e})")
print(f"    |διαφορά| < 1e-8      : {abs(float(A1) - A1_num) < 1e-8}")

# Α2.  Εμβαδόν μεταξύ y = -x^2+4 και y = x^2-2x
f = -x**2 + 4
g = x**2 - 2*x
# ΠΑΝΤΑ ταξινομούμε τις ρίζες, αλλιώς βγαίνει αρνητικό εμβαδόν.
tomes = sorted(sp.solve(sp.Eq(f, g), x))
a, b = tomes[0], tomes[1]
A2 = sp.integrate(f - g, (x, a, b))
print(f"\nΑ2. f(x) = {f},  g(x) = {g}")
print(f"    τομές (ταξινομημένες) : x = {a}, x = {b}   (αναμ. -1, 2)")
print(f"    Εμβαδόν = ∫_({a})^({b}) (f-g) dx = {A2}   (αναμ. 9)")
print(f"    θετικό εμβαδόν        : {A2 > 0}")

xv = np.linspace(float(a) - 0.8, float(b) + 0.8, 400)
fv = -xv**2 + 4
gv = xv**2 - 2*xv
xs = np.linspace(float(a), float(b), 300)
fs_ = -xs**2 + 4
gs_ = xs**2 - 2*xs

plt.figure(figsize=(7, 4.4))
plt.plot(xv, fv, 'b-', lw=2, label=r'$f(x)=-x^2+4$')
plt.plot(xv, gv, 'r-', lw=2, label=r'$g(x)=x^2-2x$')
plt.fill_between(xs, gs_, fs_, alpha=0.30, color='tab:green',
                 label=f'Εμβαδόν = {A2}')
plt.plot([float(a), float(b)], [float(f.subs(x, a)), float(f.subs(x, b))],
         'ko', ms=5)
plt.axhline(0, color='k', lw=0.6)
plt.xlabel('x'); plt.ylabel('y')
plt.title('Α. Εμβαδόν μεταξύ δύο καμπυλών')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()


# ============================================================
# Β. ΟΓΚΟΣ ΠΕΡΙΣΤΡΟΦΗΣ — και οι τρεις μέθοδοι στο ΙΔΙΟ στερεό
#    Χωρίο ανάμεσα στις y = sqrt(x) (έξω) και y = x (μέσα),
#    περιστροφή γύρω από τον x-άξονα. Τομές: x = 0, 1.
# ============================================================
print("\n=== Β. ΟΓΚΟΣ ΠΕΡΙΣΤΡΟΦΗΣ ===")

R_ex = sp.sqrt(x)      # εξωτερική ακτίνα
r_es = x               # εσωτερική ακτίνα
tomesB = sorted(sp.solve(sp.Eq(x, x**2), x))     # sqrt(x)=x <=> x=x^2, x>=0
print(f"\nΧωρίο: y = sqrt(x) και y = x,  τομές x = {tomesB}   (αναμ. [0, 1])")

# (i) ΔΙΣΚΟΙ: ο όγκος ως διαφορά δύο συμπαγών στερεών
V_disk_ex = sp.pi * sp.integrate(R_ex**2, (x, 0, 1))
V_disk_es = sp.pi * sp.integrate(r_es**2, (x, 0, 1))
V_disks = sp.simplify(V_disk_ex - V_disk_es)
# (ii) ΔΑΚΤΥΛΙΟΙ: V = pi*∫ (R^2 - r^2) dx
V_wash = sp.simplify(sp.pi * sp.integrate(R_ex**2 - r_es**2, (x, 0, 1)))
# (iii) ΚΕΛΥΦΗ ως προς y: ακτίνα y, ύψος (x_δεξιά - x_αριστερά) = y - y^2
V_shell = sp.simplify(2*sp.pi * sp.integrate(y*(y - y**2), (y, 0, 1)))

print(f"  (i)   δίσκοι    : π∫x dx - π∫x² dx = {V_disk_ex} - {V_disk_es} = {V_disks}")
print(f"  (ii)  δακτύλιοι : π∫(R²-r²) dx                     = {V_wash}")
print(f"  (iii) κελύφη    : 2π∫y(y-y²) dy                     = {V_shell}")
print(f"  αριθμητικά: {float(V_wash):.12f}   (αναμ. π/6 = {float(sp.pi/6):.12f})")
print(f"  δίσκοι == δακτύλιοι : {sp.simplify(V_disks - V_wash) == 0}")
print(f"  δακτύλιοι == κελύφη : {sp.simplify(V_wash - V_shell) == 0}")
print(f"  και οι τρεις ίσες με π/6 : "
      f"{sp.simplify(V_disks - sp.pi/6) == 0 and sp.simplify(V_shell - sp.pi/6) == 0}")

# Οι υπόλοιπες περιπτώσεις της δραστηριότητας Maxima
V_sqrt = sp.pi * sp.integrate(x, (x, 0, 4))                 # δίσκοι, y=sqrt(x), [0,4]
V_sin = sp.pi * sp.integrate(sp.sin(x)**2, (x, 0, sp.pi))   # δίσκοι, y=sin x, [0,π]
V_x3 = 2*sp.pi * sp.integrate(x*x**3, (x, 0, 2))            # κελύφη, y=x^3, [0,2]
print(f"\n  δίσκοι  y=sqrt(x), [0,4] : V = {V_sqrt}  (αναμ. 8π)")
print(f"  δίσκοι  y=sin(x), [0,π]  : V = {V_sin}  (αναμ. π²/2)")
print(f"  κελύφη  y=x³ γύρω από y, [0,2]: V = {V_x3}  (αναμ. 64π/5)")
print(f"  έλεγχοι: {sp.simplify(V_sqrt-8*sp.pi)==0}, "
      f"{sp.simplify(V_sin-sp.pi**2/2)==0}, {sp.simplify(V_x3-sp.Rational(64,5)*sp.pi)==0}")

# --- 3D οπτικοποίηση του στερεού (δακτυλιοειδές) ---
tt = np.linspace(0, 1, 60)
th = np.linspace(0, 2*np.pi, 60)
T, TH = np.meshgrid(tt, th)
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection='3d')
# εξωτερική επιφάνεια: ακτίνα sqrt(x)
ax.plot_surface(T, np.sqrt(T)*np.cos(TH), np.sqrt(T)*np.sin(TH),
                color='tab:blue', alpha=0.45, linewidth=0)
# εσωτερική επιφάνεια: ακτίνα x
ax.plot_surface(T, T*np.cos(TH), T*np.sin(TH),
                color='tab:red', alpha=0.75, linewidth=0)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title('Β. Στερεό εκ περιστροφής: y=√x (έξω) και y=x (μέσα), V = π/6')
plt.tight_layout()


# ============================================================
# Γ. ΜΗΚΟΣ ΤΟΞΟΥ:  L = ∫_a^b sqrt(1 + f'(x)^2) dx
# ============================================================
print("\n=== Γ. ΜΗΚΟΣ ΤΟΞΟΥ ===")

# Γ1.  y = x^(3/2) στο [0,4]
y1 = x**sp.Rational(3, 2)
ds1 = sp.sqrt(1 + sp.diff(y1, x)**2)
L1 = sp.simplify(sp.integrate(ds1, (x, 0, 4)))
L1_num, L1_err = quad(sp.lambdify(x, ds1, 'numpy'), 0, 4)
L1_kleisto = (80*sp.sqrt(10) - 8) / 27
print(f"\nΓ1. y = x^(3/2), [0,4]")
print(f"    f'(x) = {sp.diff(y1, x)},  sqrt(1+f'^2) = {sp.simplify(ds1)}")
print(f"    L (sympy) = {L1} = {float(L1):.12f}   (αναμ. (80√10-8)/27 = 9.073415289388)")
print(f"    L (quad)  = {L1_num:.12f}  (εκτ. σφάλμα {L1_err:.1e})")
print(f"    κλειστή μορφή σωστή : {sp.simplify(L1 - L1_kleisto) == 0}")
print(f"    |sympy - quad| < 1e-8: {abs(float(L1) - L1_num) < 1e-8}")

# Γ2.  y = ln(cos x) στο [0, π/3]  ->  sqrt(1+tan²x) = sec x
y2 = sp.log(sp.cos(x))
d2 = sp.diff(y2, x)
ds2 = sp.sqrt(sp.trigsimp(1 + d2**2))
L2 = sp.simplify(sp.integrate(sp.sec(x), (x, 0, sp.pi/3)))
L2_num, L2_err = quad(lambda t: 1/np.cos(t), 0, np.pi/3)
print(f"\nΓ2. y = ln(cos x), [0, π/3]")
print(f"    f'(x) = {sp.simplify(d2)},  trigsimp(1+f'^2) = {sp.trigsimp(1 + d2**2)}")
print(f"    L (sympy) = {float(L2):.12f}   (αναμ. ln(2+√3) = 1.316957896925)")
print(f"    L (quad)  = {L2_num:.12f}  (εκτ. σφάλμα {L2_err:.1e})")
# Η SymPy επιστρέφει τον λογάριθμο σε άλλη (ισοδύναμη) μορφή, γι' αυτό
# ελέγχουμε ισοδύναμα ότι  e^L = 2+√3.
print(f"    e^L == 2+√3          : "
      f"{sp.simplify(sp.exp(L2) - (2 + sp.sqrt(3))) == 0}")
print(f"    |sympy - quad| < 1e-8: {abs(float(L2) - L2_num) < 1e-8}")

# --- Σωρευτικό μήκος τόξου L(t) δίπλα στην καμπύλη ---
tgrid = np.linspace(0, 4, 400)
ds1_num = sp.lambdify(x, ds1, 'numpy')
Lcum = cumulative_trapezoid(ds1_num(tgrid), tgrid, initial=0.0)
print(f"\n    Σωρευτικό μήκος: L(0) = {Lcum[0]:.6f}, L(4) = {Lcum[-1]:.9f}")
print(f"    L(4) ≈ L (sympy)     : {abs(Lcum[-1] - float(L1)) < 1e-4}")

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(tgrid, tgrid**1.5, 'b-', lw=2)
axs[0].set_title(r'Καμπύλη $y=x^{3/2}$ στο $[0,4]$')
axs[0].set_xlabel('x'); axs[0].set_ylabel('y'); axs[0].grid(True, alpha=0.3)
axs[1].plot(tgrid, Lcum, 'g-', lw=2,
            label=r"$L(t)=\int_0^t\sqrt{1+[f'(u)]^2}\,du$")
axs[1].axhline(float(L1), color='r', ls='--', lw=1,
               label=f'ολικό μήκος = {float(L1):.6f}')
axs[1].set_title('Γ. Σωρευτικό μήκος τόξου')
axs[1].set_xlabel('t'); axs[1].set_ylabel('L(t)')
axs[1].legend(); axs[1].grid(True, alpha=0.3)
plt.tight_layout()


# ============================================================
# Δ. ΦΥΣΙΚΕΣ ΕΦΑΡΜΟΓΕΣ
# ============================================================
print("\n=== Δ. ΦΥΣΙΚΕΣ ΕΦΑΡΜΟΓΕΣ ===")

# Δ1. Έργο ελατηρίου: F(x) = 200x N, x από 0 έως 0,1 m
F_elat = 200*x
W = sp.integrate(F_elat, (x, 0, sp.Rational(1, 10)))
print(f"\nΔ1. Έργο ελατηρίου, F(x) = 200x N, x ∈ [0, 0.1] m")
print(f"    W = ∫_0^0.1 200x dx = {W} J = {float(W)} J   (αναμ. 1 J)")
print(f"    τύπος k·x²/2 = {200*sp.Rational(1,10)**2/2} J, ταυτίζεται: "
      f"{sp.simplify(W - 200*sp.Rational(1,10)**2/2) == 0}")

# Δ2. Κέντρο βάρους του χωρίου κάτω από την y = x^2, [0,3]
fy = x**2
Ar = sp.integrate(fy, (x, 0, 3))
xbar = sp.integrate(x*fy, (x, 0, 3)) / Ar
ybar = sp.integrate(fy**2/2, (x, 0, 3)) / Ar
print(f"\nΔ2. Κέντρο βάρους χωρίου κάτω από y = x², [0,3]")
print(f"    A = {Ar}   (αναμ. 9)")
print(f"    x̄ = {xbar} = {float(xbar)}   (αναμ. 9/4 = 2.25)")
print(f"    ȳ = {ybar} = {float(ybar)}   (αναμ. 27/10 = 2.7)")
print(f"    έλεγχοι: {Ar == 9}, {xbar == sp.Rational(9,4)}, {ybar == sp.Rational(27,10)}")

# Δ3. Υδροστατική δύναμη σε κατακόρυφο τοίχωμα ορθογωνικής δεξαμενής
#     πλάτος w = 4 m, βάθος d = 3 m, ρg = 9810 N/m^3
#     F = ρg * ∫_0^d h * w dh   (h = βάθος από την επιφάνεια)
h = sp.symbols('h', nonnegative=True)
rho_g, w_pl, d_bath = 9810, 4, 3
F_ydro = rho_g * sp.integrate(h * w_pl, (h, 0, d_bath))
print(f"\nΔ3. Υδροστατική δύναμη σε κατακόρυφο τοίχωμα {w_pl} m × {d_bath} m")
print(f"    ρg = {rho_g} N/m³")
print(f"    F = ρg·∫_0^{d_bath} h·{w_pl} dh = {F_ydro} N = {float(F_ydro)/1000:.3f} kN")
print(f"    (αναμ. 9810·4·3²/2 = 176580 N)")
F_typos = sp.Rational(rho_g * w_pl * d_bath**2, 2)
print(f"    τύπος F = ρg·w·d²/2 = {F_typos} N, ταυτίζεται: "
      f"{sp.simplify(F_ydro - F_typos) == 0}")
# Το κέντρο πίεσης βρίσκεται σε βάθος (∫h·h·w dh)/(∫h·w dh)
h_cp = sp.integrate(h*h*w_pl, (h, 0, d_bath)) / sp.integrate(h*w_pl, (h, 0, d_bath))
print(f"    κέντρο πίεσης σε βάθος {h_cp} m = {float(h_cp)} m   (= 2d/3 = 2.0 m)")

# --- Γράφημα κατανομής πίεσης ---
hh = np.linspace(0, d_bath, 200)
plt.figure(figsize=(6.5, 4.2))
plt.fill_betweenx(hh, 0, rho_g*hh/1000, alpha=0.35, color='tab:cyan',
                  label=f'F = {float(F_ydro)/1000:.2f} kN (πλάτος {w_pl} m)')
plt.plot(rho_g*hh/1000, hh, 'b-', lw=2, label='p(h) = ρgh')
plt.axhline(float(h_cp), color='r', ls='--', lw=1,
            label=f'κέντρο πίεσης h = {float(h_cp)} m')
plt.gca().invert_yaxis()
plt.xlabel('πίεση p (kPa)'); plt.ylabel('βάθος h (m)')
plt.title('Δ. Υδροστατική πίεση σε τοίχωμα δεξαμενής 4 m × 3 m')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()

print("\nΟλοκληρώθηκε.")
plt.show()
