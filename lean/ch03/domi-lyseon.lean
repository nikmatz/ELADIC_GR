-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 3 · Δομή Λύσεων Μη Ομογενούς Συστήματος · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ3.DomiLyseon
open Matrix

variable {m n : ℕ}

theorem general_solution (A : Matrix (Fin m) (Fin n) ℝ) (b : Fin m → ℝ) (xp : Fin n → ℝ)
    (hp : A *ᵥ xp = b) (x : Fin n → ℝ) :
    A *ᵥ x = b ↔ ∃ xh, A *ᵥ xh = 0 ∧ x = xp + xh := by
  constructor
  · intro hx
    refine ⟨x - xp, ?_, ?_⟩
    · rw [mulVec_sub, hx, hp, sub_self]
    · abel
  · rintro ⟨xh, hh, rfl⟩
    rw [mulVec_add, hp, hh, add_zero]

theorem difference (A : Matrix (Fin m) (Fin n) ℝ) (b : Fin m → ℝ) (x y : Fin n → ℝ)
    (hx : A *ᵥ x = b) (hy : A *ᵥ y = b) : A *ᵥ (x - y) = 0 := by
  rw [mulVec_sub, hx, hy, sub_self]

end Κεφ3.DomiLyseon
