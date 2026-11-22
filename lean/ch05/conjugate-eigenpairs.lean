-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 5 · Συζυγή Ζεύγη Ιδιοτιμών · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ5.SyzygeisIdiotimes
open Matrix

variable {n : ℕ}

-- ο πραγματικός πίνακας A, ιδωμένος ως μιγαδικός
def toC (A : Matrix (Fin n) (Fin n) ℝ) : Matrix (Fin n) (Fin n) ℂ := A.map Complex.ofReal

theorem star_toC_mulVec (A : Matrix (Fin n) (Fin n) ℝ) (v : Fin n → ℂ) :
    star (toC A *ᵥ v) = toC A *ᵥ star v := by
  have hA : (toC A).map (starRingEnd ℂ) = toC A := by
    ext i j
    simp [toC]
  funext i
  rw [Pi.star_apply, Complex.star_def, RingHom.map_mulVec, hA]
  rfl

theorem conj_eigenpair (A : Matrix (Fin n) (Fin n) ℝ) (μ : ℂ) (v : Fin n → ℂ)
    (hv : v ≠ 0) (h : toC A *ᵥ v = μ • v) :
    star v ≠ 0 ∧ toC A *ᵥ star v = star μ • star v := by
  constructor
  · rw [star_ne_zero]
    exact hv
  · rw [← star_toC_mulVec]
    rw [h]
    rw [star_smul]

theorem real_eigenvalue (A : Matrix (Fin n) (Fin n) ℝ) (μ : ℂ) (v : Fin n → ℂ)
    (hμ : star μ = μ) (h : toC A *ᵥ v = μ • v) : toC A *ᵥ star v = μ • star v := by
  rw [← star_toC_mulVec, h, star_smul, hμ]

end Κεφ5.SyzygeisIdiotimes
