-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 0 · Πολλαπλασιασμός σε Πολική Μορφή · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ0.Poliki
open Complex

theorem mul_polar (r₁ r₂ θ₁ θ₂ : ℝ) :
    (r₁ * exp (θ₁ * I)) * (r₂ * exp (θ₂ * I)) = (r₁ * r₂) * exp ((θ₁ + θ₂) * I) := by
  rw [add_mul]
  rw [exp_add]
  ring

theorem div_polar (r₁ r₂ θ₁ θ₂ : ℝ) :
    (r₁ * exp (θ₁ * I)) / (r₂ * exp (θ₂ * I)) = (r₁ / r₂) * exp ((θ₁ - θ₂) * I) := by
  rw [sub_mul]
  rw [exp_sub]
  ring

end Κεφ0.Poliki
