-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 19 · Κριτήριο Ολοκληρώματος · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (23/11/2026).

import Mathlib

namespace Κεφ19.KritirioOloklirwmatos
open MeasureTheory Filter Topology

-- f φθίνουσα και μη αρνητική στο [1, +∞): η σειρά f(1) + f(2) + … γράφεται Σ f(n + 1)
theorem integral_test (f : ℝ → ℝ) (hf : AntitoneOn f (Set.Ici 1))
    (h0 : ∀ x ∈ Set.Ici (1 : ℝ), 0 ≤ f x) :
    Summable (fun n : ℕ => f (n + 1)) ↔ IntegrableOn f (Set.Ioi 1) := by
  have hf' : AntitoneOn f (Set.Ici ((1 : ℕ) : ℝ)) := by simpa using hf
  have h0' : ∀ t ∈ Set.Ioi ((1 : ℕ) : ℝ), 0 ≤ f t := fun t ht => by
    simp only [Nat.cast_one, Set.mem_Ioi] at ht
    exact h0 t (Set.mem_Ici.mpr ht.le)
  constructor
  · intro hs
    have h := hf'.integrableOn_Ioi_of_summable_comp_add (by simpa using hs) h0'
    simpa using h
  · intro hi
    have h := hf'.summable_of_integrableOn_Ioi (by simpa using hi) h0'
    have h2 := (summable_nat_add_iff 1).mpr h
    simpa using h2

-- φράγματα με τα μερικά αθροίσματα: ∫ από 1 έως 1 + N ≤ f(1) + … + f(N)
theorem integral_le_partial_sum (f : ℝ → ℝ) (hf : AntitoneOn f (Set.Ici 1)) (N : ℕ) :
    ∫ x in (1 : ℝ)..1 + N, f x ≤ ∑ i ∈ Finset.range N, f (1 + i) :=
  (hf.mono Set.Icc_subset_Ici_self).integral_le_sum

-- και f(2) + … + f(N + 1) ≤ ∫ από 1 έως 1 + N
theorem partial_sum_le_integral (f : ℝ → ℝ) (hf : AntitoneOn f (Set.Ici 1)) (N : ℕ) :
    ∑ i ∈ Finset.range N, f (1 + (i + 1 : ℕ)) ≤ ∫ x in (1 : ℝ)..1 + N, f x :=
  (hf.mono Set.Icc_subset_Ici_self).sum_le_integral

-- οι ανισότητες του βιβλίου, όταν η σειρά συγκλίνει
theorem integral_test_bounds (f : ℝ → ℝ) (hf : AntitoneOn f (Set.Ici 1))
    (h0 : ∀ x ∈ Set.Ici (1 : ℝ), 0 ≤ f x) (hs : Summable (fun n : ℕ => f (n + 1))) :
    ∫ x in Set.Ioi (1 : ℝ), f x ≤ ∑' n : ℕ, f (n + 1) ∧
      ∑' n : ℕ, f (n + 1) ≤ f 1 + ∫ x in Set.Ioi (1 : ℝ), f x := by
  have hi : IntegrableOn f (Set.Ioi 1) := (integral_test f hf h0).mp hs
  -- το ∫ από 1 έως 1 + N τείνει στο ∫ από 1 έως +∞
  have hI : Tendsto (fun N : ℕ => ∫ x in (1 : ℝ)..1 + N, f x) atTop
      (𝓝 (∫ x in Set.Ioi (1 : ℝ), f x)) :=
    intervalIntegral_tendsto_integral_Ioi 1 hi
      (tendsto_atTop_add_const_left atTop 1 tendsto_natCast_atTop_atTop)
  -- τα μερικά αθροίσματα f(1) + … + f(N) τείνουν στο άθροισμα της σειράς
  have hS : Tendsto (fun N => ∑ i ∈ Finset.range N, f (1 + i)) atTop
      (𝓝 (∑' n : ℕ, f (n + 1))) := by
    have h := hs.hasSum.tendsto_sum_nat
    simp only [add_comm (1 : ℝ)]
    exact h
  have hS1 : Tendsto (fun N : ℕ => ∑ i ∈ Finset.range (N + 1), f (1 + i)) atTop
      (𝓝 (∑' n : ℕ, f (n + 1))) :=
    hS.comp (tendsto_add_atTop_nat 1)
  -- f(1) + f(2) + … + f(N + 1) ≤ f(1) + ∫ από 1 έως 1 + N
  have h2 : ∀ N : ℕ, ∑ i ∈ Finset.range (N + 1), f (1 + i) ≤ f 1 + ∫ x in (1 : ℝ)..1 + N, f x := by
    intro N
    rw [Finset.sum_range_succ', Nat.cast_zero, add_zero]
    linarith [partial_sum_le_integral f hf N]
  exact ⟨le_of_tendsto_of_tendsto' hI hS (integral_le_partial_sum f hf),
    le_of_tendsto_of_tendsto' hS1 (hI.const_add (f 1)) h2⟩

end Κεφ19.KritirioOloklirwmatos
