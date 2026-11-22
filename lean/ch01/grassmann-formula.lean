-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 1 · Τύπος του Grassmann · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ1.Grassmann
open Module Submodule

variable {V : Type*} [AddCommGroup V] [Module ℝ V] [FiniteDimensional ℝ V]

theorem sum_form (U W : Submodule ℝ V) :
    finrank ℝ ↥(U ⊔ W) + finrank ℝ ↥(U ⊓ W) = finrank ℝ U + finrank ℝ W :=
  finrank_sup_add_finrank_inf_eq U W

theorem grassmann (U W : Submodule ℝ V) :
    finrank ℝ ↥(U ⊔ W) = finrank ℝ U + finrank ℝ W - finrank ℝ ↥(U ⊓ W) := by
  have := sum_form U W
  omega

end Κεφ1.Grassmann
