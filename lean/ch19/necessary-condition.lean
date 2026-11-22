-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 19 · Αναγκαία Συνθήκη Σύγκλισης · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (23/11/2026).

import Mathlib

namespace Κεφ19.AnagkaiaSynthiki
open Filter Topology

-- αν τα μερικά αθροίσματα S_N συγκλίνουν, τότε a n → 0
theorem necessary (a : ℕ → ℝ) (S : ℝ)
    (h : Tendsto (fun N => ∑ n ∈ Finset.range N, a n) atTop (𝓝 S)) :
    Tendsto a atTop (𝓝 0) := by
  -- και η ακολουθία S_{N+1} τείνει στο S
  have h1 : Tendsto (fun N => ∑ n ∈ Finset.range (N + 1), a n) atTop (𝓝 S) :=
    h.comp (tendsto_add_atTop_nat 1)
  -- a_N = S_{N+1} - S_N
  have h2 : (fun N => ∑ n ∈ Finset.range (N + 1), a n - ∑ n ∈ Finset.range N, a n) = a := by
    funext N
    rw [Finset.sum_range_succ]
    ring
  -- άρα a_N → S - S = 0
  have h3 := h1.sub h
  rw [h2, sub_self] at h3
  exact h3

-- το αντίστροφο δεν ισχύει: 1/(n+1) → 0, αλλά η αρμονική σειρά αποκλίνει
theorem harmonic :
    Tendsto (fun n : ℕ => 1 / ((n : ℝ) + 1)) atTop (𝓝 0) ∧
      Tendsto (fun N => ∑ n ∈ Finset.range N, 1 / ((n : ℝ) + 1)) atTop atTop :=
  ⟨tendsto_one_div_add_atTop_nhds_zero_nat, Real.tendsto_sum_range_one_div_nat_succ_atTop⟩

theorem diverges_of_not_tendsto (a : ℕ → ℝ) (h : ¬ Tendsto a atTop (𝓝 0)) :
    ¬ ∃ S, Tendsto (fun N => ∑ n ∈ Finset.range N, a n) atTop (𝓝 S) := by
  rintro ⟨S, hS⟩
  exact h (necessary a S hS)

end Κεφ19.AnagkaiaSynthiki
