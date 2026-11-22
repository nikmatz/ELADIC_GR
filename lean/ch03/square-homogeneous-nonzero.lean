-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 3 · Μη Μηδενικές Λύσεις Τετραγωνικού Ομογενούς · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ3.TetragonikoOmogenes
open Matrix Module

variable {m n : ℕ}

theorem nonzero_iff_det (A : Matrix (Fin n) (Fin n) ℝ) :
    (∃ x : Fin n → ℝ, x ≠ 0 ∧ A *ᵥ x = 0) ↔ A.det = 0 :=
  exists_mulVec_eq_zero_iff

theorem more_unknowns (A : Matrix (Fin m) (Fin n) ℝ) (h : m < n) :
    ∃ x : Fin n → ℝ, x ≠ 0 ∧ A *ᵥ x = 0 := by
  have hr : A.rank ≤ m := by simpa using A.rank_le_card_height
  have hrn : A.rank + finrank ℝ (LinearMap.ker A.mulVecLin) = n := by
    rw [← finrank_range_mulVecLin, LinearMap.finrank_range_add_finrank_ker, finrank_fin_fun]
  have hk : LinearMap.ker A.mulVecLin ≠ ⊥ := by
    intro hbot
    rw [hbot, finrank_bot] at hrn
    omega
  obtain ⟨x, hx, hne⟩ := Submodule.exists_mem_ne_zero_of_ne_bot hk
  exact ⟨x, hne, LinearMap.mem_ker.mp hx⟩

end Κεφ3.TetragonikoOmogenes
