-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 19 · Θεώρημα Μονότονης Σύγκλισης · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (23/11/2026).

import Mathlib

namespace Κεφ19.MonotoniSygklisi
open Filter Topology

-- αύξουσα και άνω φραγμένη από M: συγκλίνει στο sup, και sup ≤ M
theorem monotone_convergence (a : ℕ → ℝ) (M : ℝ) (hmono : Monotone a) (hM : ∀ n, a n ≤ M) :
    Tendsto a atTop (𝓝 (⨆ n, a n)) ∧ (⨆ n, a n) ≤ M := by
  -- το σύνολο των όρων είναι άνω φραγμένο
  have hbdd : BddAbove (Set.range a) := ⟨M, by
    rintro _ ⟨n, rfl⟩
    exact hM n⟩
  refine ⟨?_, ciSup_le hM⟩
  rw [Metric.tendsto_atTop]
  intro ε hε
  -- το sup - ε δεν είναι άνω φράγμα: υπάρχει N με a N > sup - ε
  obtain ⟨N, hN⟩ := exists_lt_of_lt_ciSup (show (⨆ n, a n) - ε < ⨆ n, a n by linarith)
  refine ⟨N, fun n hn => ?_⟩
  -- για n ≥ N: sup - ε < a N ≤ a n ≤ sup
  have h1 : a N ≤ a n := hmono hn
  have h2 : a n ≤ ⨆ n, a n := le_ciSup hbdd n
  rw [Real.dist_eq, abs_lt]
  constructor <;> linarith

theorem antitone_convergence (a : ℕ → ℝ) (m : ℝ) (hanti : Antitone a) (hm : ∀ n, m ≤ a n) :
    Tendsto a atTop (𝓝 (⨅ n, a n)) ∧ m ≤ ⨅ n, a n :=
  ⟨tendsto_atTop_ciInf hanti ⟨m, by
    rintro _ ⟨n, rfl⟩
    exact hm n⟩, le_ciInf hm⟩

end Κεφ19.MonotoniSygklisi
