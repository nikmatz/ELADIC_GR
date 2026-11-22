-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 16 · Ολοκλήρωση με Αντικατάσταση · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ16.Antikatastasi

-- αν F' = f, τότε η F(g(x)) είναι αντιπαράγωγος της f(g(x)) g'(x)
theorem substitution (F f g g' : ℝ → ℝ) (hF : ∀ u, HasDerivAt F (f u) u)
    (hg : ∀ x, HasDerivAt g (g' x) x) (x : ℝ) :
    HasDerivAt (fun x => F (g x)) (f (g x) * g' x) x := by
  -- η F έχει παράγωγο f(u) στο σημείο u = g(x)
  have h1 := hF (g x)
  -- κανόνας αλυσίδας
  have h2 := h1.comp x (hg x)
  exact h2

theorem substitution_example (x : ℝ) :
    HasDerivAt (fun x => Real.sin (x ^ 2)) (Real.cos (x ^ 2) * (2 * x)) x :=
  substitution Real.sin Real.cos (fun x => x ^ 2) (fun x => 2 * x) Real.hasDerivAt_sin
    (fun x => by simpa using hasDerivAt_pow 2 x) x

end Κεφ16.Antikatastasi
