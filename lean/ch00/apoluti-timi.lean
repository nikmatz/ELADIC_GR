-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 0 · Ιδιότητες Απόλυτης Τιμής · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ0.ApolutiTimi

theorem nonneg (a : ℝ) : |a| ≥ 0 := abs_nonneg a

theorem eq_zero_iff (a : ℝ) : |a| = 0 ↔ a = 0 := abs_eq_zero

theorem abs_product (a b : ℝ) : |a * b| = |a| * |b| := abs_mul a b

theorem abs_quotient (a b : ℝ) : |a / b| = |a| / |b| := abs_div a b

theorem lt_iff (a b : ℝ) : |a| < b ↔ -b < a ∧ a < b := abs_lt

theorem triangle (a b : ℝ) : |a + b| ≤ |a| + |b| := by
  have ha₁ : -|a| ≤ a := neg_abs_le a
  have ha₂ : a ≤ |a| := le_abs_self a
  have hb₁ : -|b| ≤ b := neg_abs_le b
  have hb₂ : b ≤ |b| := le_abs_self b
  rw [abs_le]
  constructor
  · linarith
  · linarith

theorem reverse_triangle (a b : ℝ) : |(|a| - |b|)| ≤ |a - b| := by
  have h₁ : |a| ≤ |a - b| + |b| := by
    calc |a| = |a - b + b| := by rw [sub_add_cancel]
      _ ≤ |a - b| + |b| := triangle (a - b) b
  have h₂ : |b| ≤ |a - b| + |a| := by
    calc |b| = |b - a + a| := by rw [sub_add_cancel]
      _ ≤ |b - a| + |a| := triangle (b - a) a
      _ = |a - b| + |a| := by rw [abs_sub_comm]
  rw [abs_le]
  constructor
  · linarith
  · linarith

theorem distance (a b : ℝ) : dist a b = |a - b| := Real.dist_eq a b

end Κεφ0.ApolutiTimi
