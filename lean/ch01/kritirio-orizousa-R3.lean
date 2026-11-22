-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 1 · Κριτήριο Ορίζουσας για Τρία Διανύσματα στο ℝ³ · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ1.KritirioOrizousas
open Matrix

theorem independent_iff_det (A : Matrix (Fin 3) (Fin 3) ℝ) :
    LinearIndependent ℝ A.col ↔ A.det ≠ 0 :=
  linearIndependent_col_iff.trans nonsingular_iff_det_ne_zero

end Κεφ1.KritirioOrizousas
