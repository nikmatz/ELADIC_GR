-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 0 · Θεμελιώδες Θεώρημα της Άλγεβρας · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ0.ThemeliodesAlgebras
open Polynomial

theorem has_root (f : ℂ[X]) (hf : 0 < f.natDegree) : ∃ z : ℂ, f.IsRoot z :=
  Complex.exists_root (natDegree_pos_iff_degree_pos.mp hf)

theorem n_roots (f : ℂ[X]) : Multiset.card f.roots = f.natDegree :=
  IsAlgClosed.card_roots_eq_natDegree

end Κεφ0.ThemeliodesAlgebras
