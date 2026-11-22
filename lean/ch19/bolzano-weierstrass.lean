-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 19 · Θεώρημα Bolzano–Weierstrass · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (23/11/2026).

import Mathlib

namespace Κεφ19.BolzanoWeierstrass
open Filter Topology

-- φραγμένη ακολουθία: |a n| ≤ M για κάθε n
theorem bolzano_weierstrass (a : ℕ → ℝ) (M : ℝ) (h : ∀ n, |a n| ≤ M) :
    ∃ L, ∃ φ : ℕ → ℕ, StrictMono φ ∧ Tendsto (fun k => a (φ k)) atTop (𝓝 L) := by
  obtain ⟨L, -, φ, hφ, hlim⟩ := tendsto_subseq_of_bounded (x := a) (Metric.isBounded_Icc (-M) M)
    (fun n => abs_le.mp (h n))
  exact ⟨L, φ, hφ, hlim⟩

end Κεφ19.BolzanoWeierstrass
