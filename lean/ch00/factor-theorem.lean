-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 0 · Θεώρημα Παράγοντα · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ0.Paragontas
open Polynomial

theorem division (P : ℝ[X]) (r : ℝ) :
    P = (X - C r) * (P /ₘ (X - C r)) + C (P.eval r) := by
  have h := modByMonic_add_div P (X - C r)
  rw [modByMonic_X_sub_C_eq_C_eval] at h
  rw [add_comm]
  exact h.symm

theorem factor (P : ℝ[X]) (r : ℝ) : P.eval r = 0 ↔ (X - C r) ∣ P := by
  constructor
  · intro h
    rw [division P r, h, C_0, add_zero]
    exact dvd_mul_right (X - C r) _
  · intro h
    obtain ⟨Q, hQ⟩ := h
    rw [hQ, eval_mul, eval_sub, eval_X, eval_C, sub_self, zero_mul]

theorem quotient_degree (P : ℝ[X]) (r : ℝ) :
    (P /ₘ (X - C r)).natDegree = P.natDegree - 1 := by
  rw [natDegree_divByMonic P (monic_X_sub_C r), natDegree_X_sub_C]

end Κεφ0.Paragontas
