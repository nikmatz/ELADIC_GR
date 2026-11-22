-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 0 · Λύση Εξίσωσης 1ου Βαθμού · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ0.Exisosi

theorem is_solution (a b : ℝ) (ha : a ≠ 0) : a * (-b / a) + b = 0 := by
  rw [← mul_div_assoc, mul_div_cancel_left₀ _ ha]
  ring

theorem unique (a b x : ℝ) (ha : a ≠ 0) (hx : a * x + b = 0) : x = -b / a := by
  rw [eq_div_iff ha]
  linarith

theorem exists_unique (a b : ℝ) (ha : a ≠ 0) : ∃! x : ℝ, a * x + b = 0 :=
  ⟨-b / a, is_solution a b ha, fun x hx => unique a b x ha hx⟩

end Κεφ0.Exisosi
