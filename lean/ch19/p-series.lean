-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 19 · p-Σειρά · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (23/11/2026).

import Mathlib

namespace Κεφ19.PSeira
open Filter Topology

-- ο όρος για n = 0 είναι 1/0^p, ένας αριθμός που δεν επηρεάζει τη σύγκλιση
theorem p_series (p : ℝ) : Summable (fun n : ℕ => 1 / (n : ℝ) ^ p) ↔ 1 < p :=
  Real.summable_one_div_nat_rpow

theorem harmonic_diverges :
    Tendsto (fun N => ∑ n ∈ Finset.range N, 1 / ((n : ℝ) + 1)) atTop atTop :=
  Real.tendsto_sum_range_one_div_nat_succ_atTop

theorem basel : HasSum (fun n : ℕ => 1 / (n : ℝ) ^ 2) (Real.pi ^ 2 / 6) :=
  hasSum_zeta_two

theorem half_diverges : ¬ Summable (fun n : ℕ => 1 / (n : ℝ) ^ (1 / 2 : ℝ)) := by
  rw [Real.summable_one_div_nat_rpow]
  norm_num

end Κεφ19.PSeira
