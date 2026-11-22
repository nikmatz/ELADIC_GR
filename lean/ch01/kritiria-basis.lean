-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 1 · Κριτήρια Βάσης · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ1.KritiriaBasis
open Module Submodule

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem independent_spans [FiniteDimensional ℝ V] {n : ℕ} (v : Fin n → V)
    (hv : LinearIndependent ℝ v) (hn : n = finrank ℝ V) : span ℝ (Set.range v) = ⊤ :=
  hv.span_eq_top_of_card_eq_finrank' (by simpa using hn)

theorem spanning_independent {n : ℕ} (v : Fin n → V) (hv : span ℝ (Set.range v) = ⊤)
    (hn : n = finrank ℝ V) : LinearIndependent ℝ v :=
  linearIndependent_of_top_le_span_of_card_eq_finrank hv.ge (by simpa using hn)

noncomputable def basisOfIndependent [FiniteDimensional ℝ V] {n : ℕ} (v : Fin n → V)
    (hv : LinearIndependent ℝ v) (hn : n = finrank ℝ V) : Basis (Fin n) ℝ V :=
  Basis.mk hv (independent_spans v hv hn).ge

end Κεφ1.KritiriaBasis
