-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 1 · Θεώρημα Βάσης — Μοναδικότητα Πλήθους · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ1.BasiMonadikotita
open Module

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem card_eq_dim {ι : Type*} [Fintype ι] (b : Basis ι ℝ V) :
    Fintype.card ι = finrank ℝ V :=
  (finrank_eq_card_basis b).symm

theorem same_card {ι κ : Type*} [Fintype ι] [Fintype κ] (b₁ : Basis ι ℝ V) (b₂ : Basis κ ℝ V) :
    Fintype.card ι = Fintype.card κ := by
  rw [card_eq_dim b₁, card_eq_dim b₂]

end Κεφ1.BasiMonadikotita
