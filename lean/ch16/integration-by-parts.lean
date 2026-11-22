-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 16 · Ολοκλήρωση κατά Παράγοντες · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ16.KataParagontes

-- αν η G είναι αντιπαράγωγος της v u' (δηλαδή G = ∫ v du), τότε η u v - G είναι αντιπαράγωγος της u v'
theorem parts (u v u' v' G : ℝ → ℝ) (hu : ∀ x, HasDerivAt u (u' x) x)
    (hv : ∀ x, HasDerivAt v (v' x) x) (hG : ∀ x, HasDerivAt G (v x * u' x) x) (x : ℝ) :
    HasDerivAt (fun x => u x * v x - G x) (u x * v' x) x := by
  -- κανόνας γινομένου: (u v)' = u' v + u v'
  have h1 : HasDerivAt (fun x => u x * v x) (u' x * v x + u x * v' x) x := (hu x).mul (hv x)
  -- αφαιρούμε την G, που έχει παράγωγο v u'
  have h2 : HasDerivAt (fun x => u x * v x - G x) (u' x * v x + u x * v' x - v x * u' x) x :=
    h1.sub (hG x)
  convert h2 using 1
  ring

theorem parts_example (x : ℝ) :
    HasDerivAt (fun x => x * Real.exp x - Real.exp x) (x * Real.exp x) x :=
  parts (fun x => x) Real.exp (fun _ => 1) Real.exp Real.exp hasDerivAt_id'
    Real.hasDerivAt_exp (fun x => by simpa using Real.hasDerivAt_exp x) x

end Κεφ16.KataParagontes
