-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 16 · Επαναλαμβανόμενος Γραμμικός Παράγοντας · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ16.EpanalambanomenosParagontas

-- ο όρος A/(x - a): ∫ A/(x - a) dx = A ln|x - a| + C
theorem int_first (A a x : ℝ) (hx : x ≠ a) :
    HasDerivAt (fun x => A * Real.log |x - a|) (A / (x - a)) x := by
  simp only [Real.log_abs]
  have h : HasDerivAt (fun x => Real.log (x - a)) (1 / (x - a)) x :=
    ((hasDerivAt_id' x).sub_const a).log (sub_ne_zero.mpr hx)
  convert h.const_mul A using 1
  ring

-- οι όροι A/(x - a)^j με j ≥ 2, γραμμένο j = m + 2:
-- ∫ A/(x - a)^(m+2) dx = -A/((m + 1)(x - a)^(m+1)) + C
theorem int_power (A a x : ℝ) (m : ℕ) (hx : x ≠ a) :
    HasDerivAt (fun x => -A / ((m + 1) * (x - a) ^ (m + 1))) (A / (x - a) ^ (m + 2)) x := by
  have hxa : x - a ≠ 0 := sub_ne_zero.mpr hx
  -- ((x - a)^(m+1))' = (m + 1)(x - a)^m
  have h1 : HasDerivAt (fun x => (x - a) ^ (m + 1)) ((m + 1) * (x - a) ^ m) x := by
    simpa using ((hasDerivAt_id' x).sub_const a).fun_pow (m + 1)
  -- ((m + 1)(x - a)^(m+1))' = (m + 1)(m + 1)(x - a)^m
  have h2 := h1.const_mul ((m : ℝ) + 1)
  -- ο παρονομαστής δεν μηδενίζεται
  have hden : ((m : ℝ) + 1) * (x - a) ^ (m + 1) ≠ 0 :=
    mul_ne_zero (by positivity) (pow_ne_zero _ hxa)
  -- κανόνας του πηλίκου
  have h3 := (hasDerivAt_const x (-A)).fun_div h2 hden
  convert h3 using 1
  field_simp
  ring

theorem int_example (x : ℝ) (hx : x ≠ 1) :
    HasDerivAt (fun x => -3 / (x - 1)) (3 / (x - 1) ^ 2) x := by
  have h := int_power 3 1 x 0 hx
  simp only [Nat.cast_zero, zero_add, one_mul, pow_one] at h
  exact h

end Κεφ16.EpanalambanomenosParagontas
