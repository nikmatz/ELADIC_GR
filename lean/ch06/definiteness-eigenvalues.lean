-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 6 · Χαρακτηρισμός Οριστικότητας μέσω Ιδιοτιμών · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ6.Oristikotita
open Matrix

variable {n : ℕ}

theorem pos_def_quadratic (A : Matrix (Fin n) (Fin n) ℝ) :
    A.PosDef ↔ A.IsHermitian ∧ ∀ ⦃x : Fin n → ℝ⦄, x ≠ 0 → 0 < x ⬝ᵥ (A *ᵥ x) := by
  rw [posDef_iff_dotProduct_mulVec]
  simp only [star_trivial]

theorem pos_def_iff (A : Matrix (Fin n) (Fin n) ℝ) (hA : A.IsHermitian) :
    A.PosDef ↔ ∀ i, 0 < hA.eigenvalues i :=
  hA.posDef_iff_eigenvalues_pos

theorem pos_semidef_iff (A : Matrix (Fin n) (Fin n) ℝ) (hA : A.IsHermitian) :
    A.PosSemidef ↔ ∀ i, 0 ≤ hA.eigenvalues i := by
  rw [hA.posSemidef_iff_eigenvalues_nonneg, Pi.le_def]
  simp only [Pi.zero_apply]

-- οι ιδιοτιμές του -A είναι οι αντίθετες των ιδιοτιμών του A
theorem neg_eig_exists (A : Matrix (Fin n) (Fin n) ℝ) (hA : A.IsHermitian) (i : Fin n) :
    ∃ j, hA.neg.eigenvalues j = -hA.eigenvalues i := by
  have h : -hA.eigenvalues i ∈ spectrum ℝ (-A) := by
    rw [← spectrum.neg_eq, Set.neg_mem_neg]
    exact hA.eigenvalues_mem_spectrum_real i
  rwa [hA.neg.spectrum_real_eq_range_eigenvalues, Set.mem_range] at h

theorem eig_exists_neg (A : Matrix (Fin n) (Fin n) ℝ) (hA : A.IsHermitian) (i : Fin n) :
    ∃ j, hA.eigenvalues j = -hA.neg.eigenvalues i := by
  have h : hA.neg.eigenvalues i ∈ spectrum ℝ (-A) := hA.neg.eigenvalues_mem_spectrum_real i
  rwa [← spectrum.neg_eq, Set.mem_neg, hA.spectrum_real_eq_range_eigenvalues, Set.mem_range] at h

theorem neg_def_iff (A : Matrix (Fin n) (Fin n) ℝ) (hA : A.IsHermitian) :
    (-A).PosDef ↔ ∀ i, hA.eigenvalues i < 0 := by
  rw [pos_def_iff (-A) hA.neg]
  constructor
  · intro h i
    obtain ⟨j, hj⟩ := neg_eig_exists A hA i
    have := h j
    linarith
  · intro h i
    obtain ⟨j, hj⟩ := eig_exists_neg A hA i
    have := h j
    linarith

theorem neg_semidef_iff (A : Matrix (Fin n) (Fin n) ℝ) (hA : A.IsHermitian) :
    (-A).PosSemidef ↔ ∀ i, hA.eigenvalues i ≤ 0 := by
  rw [pos_semidef_iff (-A) hA.neg]
  constructor
  · intro h i
    obtain ⟨j, hj⟩ := neg_eig_exists A hA i
    have := h j
    linarith
  · intro h i
    obtain ⟨j, hj⟩ := eig_exists_neg A hA i
    have := h j
    linarith

theorem indefinite_iff (A : Matrix (Fin n) (Fin n) ℝ) (hA : A.IsHermitian) :
    (¬ A.PosSemidef ∧ ¬ (-A).PosSemidef) ↔
      (∃ i, hA.eigenvalues i < 0) ∧ (∃ i, 0 < hA.eigenvalues i) := by
  rw [pos_semidef_iff A hA, neg_semidef_iff A hA]
  simp only [not_forall, not_le]

end Κεφ6.Oristikotita
