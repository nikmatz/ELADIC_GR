-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 0 · Βασικές Αλγεβρικές Ταυτότητες · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ0.Tautotites

theorem sq_add (a b : ℝ) : (a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2 := by
  rw [sq]
  rw [add_mul, mul_add, mul_add]
  ring

theorem sq_sub (a b : ℝ) : (a - b) ^ 2 = a ^ 2 - 2 * a * b + b ^ 2 := by
  rw [sq]
  rw [sub_mul, mul_sub, mul_sub]
  ring

theorem diff_sq (a b : ℝ) : (a - b) * (a + b) = a ^ 2 - b ^ 2 := by
  ring

theorem cube_add (a b : ℝ) : (a + b) ^ 3 = a ^ 3 + 3 * a ^ 2 * b + 3 * a * b ^ 2 + b ^ 3 := by
  ring

theorem cube_sub (a b : ℝ) : (a - b) ^ 3 = a ^ 3 - 3 * a ^ 2 * b + 3 * a * b ^ 2 - b ^ 3 := by
  ring

theorem sum_cubes (a b : ℝ) : a ^ 3 + b ^ 3 = (a + b) * (a ^ 2 - a * b + b ^ 2) := by
  ring

theorem diff_cubes (a b : ℝ) : a ^ 3 - b ^ 3 = (a - b) * (a ^ 2 + a * b + b ^ 2) := by
  ring

theorem binomial (a b : ℝ) (n : ℕ) :
    (a + b) ^ n = ∑ k ∈ Finset.range (n + 1), (n.choose k : ℝ) * a ^ (n - k) * b ^ k := by
  rw [add_comm a b, add_pow]
  apply Finset.sum_congr rfl
  intro k _
  ring

end Κεφ0.Tautotites
