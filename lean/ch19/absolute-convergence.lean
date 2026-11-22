-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 19 · Απόλυτη Σύγκλιση · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (23/11/2026).

import Mathlib

namespace Κεφ19.ApolytiSygklisi
open Filter Topology

-- η σειρά συγκλίνει: τα μερικά αθροίσματα S_N = a₀ + … + a_{N-1} έχουν όριο
theorem abs_conv (a : ℕ → ℝ) (T : ℝ)
    (h : Tendsto (fun N => ∑ n ∈ Finset.range N, |a n|) atTop (𝓝 T)) :
    ∃ S, Tendsto (fun N => ∑ n ∈ Finset.range N, a n) atTop (𝓝 S) := by
  -- η σειρά των |a n| έχει μη αρνητικούς όρους και συγκλίνει: είναι αθροίσιμη
  have hs : Summable (fun n => |a n|) :=
    ((hasSum_iff_tendsto_nat_of_nonneg (fun n => abs_nonneg (a n)) T).mpr h).summable
  -- άρα και η σειρά των a n είναι αθροίσιμη, και τα μερικά της αθροίσματα συγκλίνουν
  exact ⟨_, (summable_abs_iff.mp hs).hasSum.tendsto_sum_nat⟩

end Κεφ19.ApolytiSygklisi
