-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 18 · Οριακό Κριτήριο Σύγκρισης (γενικευμένα ολοκληρώματα) · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (23/11/2026).

import Mathlib

namespace Κεφ18.OriakoKritirio
open MeasureTheory Filter Topology Asymptotics

theorem limit_comparison (f g : ℝ → ℝ) (a L : ℝ) (hL : 0 < L)
    (hf : ContinuousOn f (Set.Ici a)) (hg : ContinuousOn g (Set.Ici a))
    (hg0 : ∀ x ∈ Set.Ici a, 0 < g x)
    (hlim : Tendsto (fun x => f x / g x) atTop (𝓝 L)) :
    IntegrableOn f (Set.Ici a) ↔ IntegrableOn g (Set.Ici a) := by
  -- f = O(g) για x → +∞, γιατί f/g → L
  have hfg : f =O[atTop] g := by
    refine isBigO_of_div_tendsto_nhds ?_ L hlim
    filter_upwards [eventually_ge_atTop a] with x hx h0
    exact absurd h0 (hg0 x hx).ne'
  -- g = O(f) για x → +∞, γιατί g/f → 1/L
  have hgf : g =O[atTop] f := by
    have hinv : Tendsto (fun x => g x / f x) atTop (𝓝 L⁻¹) := by
      refine (hlim.inv₀ hL.ne').congr' ?_
      filter_upwards with x
      exact inv_div (f x) (g x)
    refine isBigO_of_div_tendsto_nhds ?_ L⁻¹ hinv
    filter_upwards [hlim.eventually (lt_mem_nhds hL)] with x hx h0
    rw [h0, zero_div] at hx
    exact absurd hx (lt_irrefl 0)
  -- σύγκριση στο άπειρο: οι συνεχείς f, g είναι ολοκληρώσιμες σε κάθε [a, R]
  constructor
  · intro hfi
    exact (hg.locallyIntegrableOn measurableSet_Ici).integrableOn_of_isBigO_atTop hgf
      (integrableAtFilter_atTop_iff.mpr ⟨a, hfi⟩)
  · intro hgi
    exact (hf.locallyIntegrableOn measurableSet_Ici).integrableOn_of_isBigO_atTop hfg
      (integrableAtFilter_atTop_iff.mpr ⟨a, hgi⟩)

end Κεφ18.OriakoKritirio
