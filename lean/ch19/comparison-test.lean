-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 19 · Κριτήριο Σύγκρισης · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (23/11/2026).

import Mathlib

namespace Κεφ19.KritirioSygkrisis

theorem comparison (a b : ℕ → ℝ) (ha : ∀ n, 0 ≤ a n) (hab : ∀ n, a n ≤ b n)
    (hb : Summable b) : Summable a := by
  -- τα μερικά αθροίσματα της a είναι φραγμένα από το άθροισμα της b
  have hbound : ∀ N, ∑ n ∈ Finset.range N, a n ≤ ∑' n, b n := by
    intro N
    -- όρο προς όρο: a n ≤ b n
    have h1 : ∑ n ∈ Finset.range N, a n ≤ ∑ n ∈ Finset.range N, b n :=
      Finset.sum_le_sum fun n _ => hab n
    -- κάθε μερικό άθροισμα της b είναι ≤ του αθροίσματος της σειράς, γιατί b n ≥ 0
    have h2 : ∑ n ∈ Finset.range N, b n ≤ ∑' n, b n :=
      hb.sum_le_tsum (Finset.range N) fun n _ => (ha n).trans (hab n)
    exact h1.trans h2
  -- μη αρνητικοί όροι με φραγμένα μερικά αθροίσματα: η σειρά συγκλίνει
  exact summable_of_sum_range_le ha hbound

-- το ίδιο, όταν 0 ≤ a n ≤ b n ισχύει μόνο για n ≥ N
theorem comparison_from (a b : ℕ → ℝ) (N : ℕ) (ha : ∀ n ≥ N, 0 ≤ a n)
    (hab : ∀ n ≥ N, a n ≤ b n) (hb : Summable b) : Summable a := by
  -- εφαρμόζουμε το κριτήριο στις ουρές a (n + N) και b (n + N)
  have h := comparison (fun n => a (n + N)) (fun n => b (n + N))
    (fun n => ha (n + N) (Nat.le_add_left N n)) (fun n => hab (n + N) (Nat.le_add_left N n))
    ((summable_nat_add_iff N).mpr hb)
  exact (summable_nat_add_iff N).mp h

theorem comparison_diverges (a b : ℕ → ℝ) (ha : ∀ n, 0 ≤ a n) (hab : ∀ n, a n ≤ b n)
    (hdiv : ¬ Summable a) : ¬ Summable b :=
  fun hb => hdiv (comparison a b ha hab hb)

end Κεφ19.KritirioSygkrisis
