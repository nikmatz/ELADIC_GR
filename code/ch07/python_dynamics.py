# ============================================================
# python_dynamics.py
# Κεφάλαιο 7 — Δυναμικά Συστήματα και Αλυσίδες Markov
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

# ── Α. Εξέλιξη x_{k+1} = A x_k ──────────────────────────────
A  = np.array([[0.9, 0.1], [0.1, 0.9]])
x0 = np.array([1.0, 0.0])

evals = np.linalg.eigvals(A)
print("Ιδιοτιμές:", np.round(evals, 4))
print("ρ(A) =", round(max(abs(evals)), 4), "< 1 → ασυμπτωτικά σταθερό")

traj = [x0]
for k in range(50):
    traj.append(A @ traj[-1])
traj = np.array(traj)
for k in range(1, 6):
    print(f"x_{k} =", np.round(traj[k], 4))

# ── Β. Αλυσίδα Markov: σταθερή κατανομή ─────────────────────
P = np.array([[0.9, 0.2], [0.1, 0.8]])
vals, vecs = np.linalg.eig(P)
i = np.argmin(abs(vals - 1))
pi = np.real(vecs[:, i]); pi = pi / pi.sum()
print("Σταθερή κατανομή π =", np.round(pi, 4))
print("Έλεγχος Pπ = π:", np.allclose(P @ pi, pi))

# ── Γ. Φασικό πορτρέτο / τροχιές ────────────────────────────
plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
plt.plot(traj[:, 0], traj[:, 1], 'o-', ms=3)
plt.xlabel('$x_1$'); plt.ylabel('$x_2$'); plt.title('Τροχιά στον χώρο κατάστασης')
plt.grid(alpha=.3)
plt.subplot(1, 2, 2)
plt.plot(traj[:, 0], label='$x_1(k)$'); plt.plot(traj[:, 1], label='$x_2(k)$')
plt.xlabel('$k$'); plt.title('Συνιστώσες ανά βήμα'); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.show()
