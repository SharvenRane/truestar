/-!
# T8: CRT reconstruction exactness (Ozaki scheme II)

Ozaki II computes an integer matrix product `x` (one entry here) through one integer GEMM per
modulus `m_i`, getting the residues `x mod m_i`, and reconstructs `x` from them. The
reconstruction is exact when `x` lies in a range shorter than the product `M` of the moduli: any
method that returns an integer `y` in the same range with the same residues returns `x` itself.

Proved here (Lean core only): for pairwise coprime moduli, two integers with the same residues
modulo every `m_i` differ by a multiple of `M = prod m_i` (`crt_dvd`); if both lie in a window of
length below `M` (for example the symmetric range `2|x| < M`) they are equal (`crt_unique`).
The existence half (that the usual reconstruction formula hits the residues) is not formalized.
-/
namespace Truestar.Crt

def prod : List Nat → Nat
  | [] => 1
  | m :: ms => m * prod ms

def Pairwise : List Nat → Prop
  | [] => True
  | m :: ms => (∀ k ∈ ms, Nat.Coprime m k) ∧ Pairwise ms

theorem coprime_prod (m : Nat) : ∀ ms : List Nat, (∀ k ∈ ms, Nat.Coprime m k) →
    Nat.Coprime m (prod ms) := by
  intro ms
  induction ms with
  | nil => intro _; simp [prod]
  | cons k ms ih =>
    intro h
    exact Nat.Coprime.mul_right (h k (by simp)) (ih (fun j hj => h j (by simp [hj])))

/-- Same residues modulo pairwise coprime moduli: the difference is a multiple of the product. -/
theorem crt_dvd : ∀ ms : List Nat, Pairwise ms → ∀ n : Nat, (∀ m ∈ ms, m ∣ n) → prod ms ∣ n := by
  intro ms
  induction ms with
  | nil => intro _ n _; simp [prod]
  | cons m ms ih =>
    intro hp n hd
    exact Nat.Coprime.mul_dvd_of_dvd_of_dvd (coprime_prod m ms hp.1) (hd m (by simp))
      (ih hp.2 n (fun k hk => hd k (by simp [hk])))

/-- **T8.** Two integers with equal residues modulo every `m_i` (pairwise coprime) whose
difference is below `M = prod m_i` in magnitude are equal. -/
theorem crt_unique (ms : List Nat) (hp : Pairwise ms) (x y : Int)
    (hres : ∀ m ∈ ms, (m : Int) ∣ x - y) (hwin : (x - y).natAbs < prod ms) : x = y := by
  have hd : prod ms ∣ (x - y).natAbs :=
    crt_dvd ms hp _ (fun m hm => Int.ofNat_dvd_left.mp (hres m hm))
  have := Nat.eq_zero_of_dvd_of_lt hd hwin
  omega

/-- The symmetric range form: `2|x| < M` and `2|y| < M` with equal residues give `x = y`. -/
theorem crt_unique_symm (ms : List Nat) (hp : Pairwise ms) (x y : Int)
    (hres : ∀ m ∈ ms, (m : Int) ∣ x - y)
    (hx : 2 * x.natAbs < prod ms) (hy : 2 * y.natAbs < prod ms) : x = y :=
  crt_unique ms hp x y hres (by omega)

end Truestar.Crt
