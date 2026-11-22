-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 1 · Αλγόριθμος Gram–Schmidt · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ1.GramSchmidt
open scoped InnerProductSpace
open InnerProductSpace Submodule

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem orthogonal {k : ℕ} (v : Fin k → V) {i j : Fin k} (h : i ≠ j) :
    ⟪gramSchmidt ℝ v i, gramSchmidt ℝ v j⟫_ℝ = 0 :=
  gramSchmidt_orthogonal ℝ v h

theorem same_span {k : ℕ} (v : Fin k → V) :
    span ℝ (Set.range (gramSchmidt ℝ v)) = span ℝ (Set.range v) :=
  span_gramSchmidt ℝ v

theorem orthonormal {k : ℕ} (v : Fin k → V) (h : LinearIndependent ℝ v) :
    Orthonormal ℝ (gramSchmidtNormed ℝ v) :=
  gramSchmidtNormed_orthonormal h

end Κεφ1.GramSchmidt
