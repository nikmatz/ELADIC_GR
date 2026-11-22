-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 9 · Θεώρημα Ενδιάμεσης Τιμής (ΘΕΤ) · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ9.EndiamesiTimi

theorem intermediate_value (f : ℝ → ℝ) (a b k : ℝ) (hab : a ≤ b)
    (hf : ContinuousOn f (Set.Icc a b)) (hk : k ∈ Set.uIcc (f a) (f b)) :
    ∃ c ∈ Set.Icc a b, f c = k := by
  rw [← Set.uIcc_of_le hab] at hf ⊢
  exact intermediate_value_uIcc hf hk

theorem bolzano (f : ℝ → ℝ) (a b : ℝ) (hab : a < b) (hf : ContinuousOn f (Set.Icc a b))
    (hsign : f a * f b < 0) : ∃ c ∈ Set.Ioo a b, f c = 0 := by
  rcases mul_neg_iff.mp hsign with ⟨ha, hb⟩ | ⟨ha, hb⟩
  · exact intermediate_value_Ioo' hab.le hf ⟨hb, ha⟩
  · exact intermediate_value_Ioo hab.le hf ⟨ha, hb⟩

end Κεφ9.EndiamesiTimi
