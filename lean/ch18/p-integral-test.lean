-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 18 · Κριτήριο $p$-Ολοκληρώματος (1ου και 2ου Είδους) · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (23/11/2026).

import Mathlib

namespace Κεφ18.KritirioP
open MeasureTheory

-- (α) για a > 0, το ολοκλήρωμα του 1/x^p από a έως +∞ συγκλίνει ακριβώς όταν p > 1
theorem p_test_infinity (a p : ℝ) (ha : 0 < a) :
    IntegrableOn (fun x : ℝ => 1 / x ^ p) (Set.Ioi a) ↔ 1 < p := by
  have h : Set.EqOn (fun x : ℝ => 1 / x ^ p) (fun x => x ^ (-p)) (Set.Ioi a) := fun x hx => by
    simp only [Real.rpow_neg (ha.trans (Set.mem_Ioi.mp hx)).le, one_div]
  rw [integrableOn_congr_fun h measurableSet_Ioi, integrableOn_Ioi_rpow_iff ha]
  constructor <;> intro h <;> linarith

theorem p_value_infinity (p : ℝ) (hp : 1 < p) : ∫ x in Set.Ioi 1, 1 / x ^ p = 1 / (p - 1) := by
  have h : Set.EqOn (fun x : ℝ => 1 / x ^ p) (fun x => x ^ (-p)) (Set.Ioi 1) := fun x hx => by
    simp only [Real.rpow_neg (zero_le_one.trans (Set.mem_Ioi.mp hx).le), one_div]
  rw [setIntegral_congr_fun measurableSet_Ioi h, integral_Ioi_rpow_of_lt (a := -p) (by linarith) one_pos,
    Real.one_rpow]
  have h1 : -p + 1 ≠ 0 := ne_of_lt (by linarith)
  have h2 : p - 1 ≠ 0 := ne_of_gt (by linarith)
  field_simp
  ring

-- (β) το ολοκλήρωμα του 1/x^p από 0 έως 1 συγκλίνει ακριβώς όταν p < 1
theorem p_test_zero (p : ℝ) : IntegrableOn (fun x : ℝ => 1 / x ^ p) (Set.Ioo 0 1) ↔ p < 1 := by
  have h : Set.EqOn (fun x : ℝ => 1 / x ^ p) (fun x => x ^ (-p)) (Set.Ioo 0 1) := fun x hx => by
    simp only [Real.rpow_neg hx.1.le, one_div]
  rw [integrableOn_congr_fun h measurableSet_Ioo, intervalIntegral.integrableOn_Ioo_rpow_iff one_pos]
  constructor <;> intro h <;> linarith

theorem p_value_zero (p : ℝ) (hp : p < 1) : ∫ x in (0 : ℝ)..1, 1 / x ^ p = 1 / (1 - p) := by
  have h : Set.EqOn (fun x : ℝ => 1 / x ^ p) (fun x => x ^ (-p)) (Set.uIcc 0 1) := fun x hx => by
    rw [Set.uIcc_of_le zero_le_one] at hx
    simp only [Real.rpow_neg hx.1, one_div]
  have h1 : -p + 1 ≠ 0 := ne_of_gt (by linarith)
  rw [intervalIntegral.integral_congr h, integral_rpow (r := -p) (Or.inl (by linarith)), Real.one_rpow,
    Real.zero_rpow h1]
  ring

end Κεφ18.KritirioP
