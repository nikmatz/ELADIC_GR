-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 19 · Κριτήριο Cauchy για Ακολουθίες · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (23/11/2026).

import Mathlib

namespace Κεφ19.KritirioCauchy
open Filter Topology

theorem cauchy_criterion (a : ℕ → ℝ) :
    (∃ L, Tendsto a atTop (𝓝 L)) ↔
      ∀ ε > 0, ∃ N, ∀ m ≥ N, ∀ n ≥ N, |a m - a n| < ε := by
  have h : CauchySeq a ↔ ∀ ε > 0, ∃ N, ∀ m ≥ N, ∀ n ≥ N, |a m - a n| < ε := by
    simp only [Metric.cauchySeq_iff, Real.dist_eq]
  rw [← h]
  exact ⟨fun ⟨L, hL⟩ => hL.cauchySeq, cauchySeq_tendsto_of_complete⟩

end Κεφ19.KritirioCauchy
