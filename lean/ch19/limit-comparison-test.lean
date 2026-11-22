-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 19 · Κριτήριο Σύγκρισης Ορίου · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (23/11/2026).

import Mathlib

namespace Κεφ19.OriakoKritirio
open Filter Topology Asymptotics

theorem limit_comparison (a b : ℕ → ℝ) (L : ℝ) (hL : 0 < L) (hb : ∀ n, 0 < b n)
    (hlim : Tendsto (fun n => a n / b n) atTop (𝓝 L)) : Summable a ↔ Summable b := by
  -- a = O(b), γιατί a/b → L
  have hab : a =O[atTop] b :=
    isBigO_of_div_tendsto_nhds (Eventually.of_forall fun n h0 => absurd h0 (hb n).ne') L hlim
  -- b = O(a), γιατί b/a → 1/L
  have hba : b =O[atTop] a := by
    have hinv : Tendsto (fun n => b n / a n) atTop (𝓝 L⁻¹) := by
      refine (hlim.inv₀ hL.ne').congr' ?_
      filter_upwards with n
      exact inv_div (a n) (b n)
    refine isBigO_of_div_tendsto_nhds ?_ L⁻¹ hinv
    filter_upwards [hlim.eventually (lt_mem_nhds hL)] with n hn h0
    rw [h0, zero_div] at hn
    exact absurd hn (lt_irrefl 0)
  exact ⟨fun ha => summable_of_isBigO_nat' ha hba, fun hb' => summable_of_isBigO_nat' hb' hab⟩

end Κεφ19.OriakoKritirio
