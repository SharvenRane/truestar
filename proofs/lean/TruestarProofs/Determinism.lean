/-!
# T2: determinism under a fixed reduction order

A reduction tree has inputs at its leaves and one binary operation at every inner node. The
operation `op` is arbitrary: no associativity, commutativity or any other law is assumed (it can
be float32 addition with any rounding).

An execution is a trace: a list of node positions, processed one after another, each step
reading the stored values of the node's two children and storing `op` of them (a leaf stores
its input). The trace order models any schedule: warps, pipeline stages, SMs, split K partial
results; the only requirement is that a node runs after both its children ("valid trace").
Steps whose children are not yet stored do nothing, and running a node twice is allowed.

Results:
* `run_sound`: whatever the trace, any value stored at a position is the tree value of that
  subtree; in particular a stored root value is `eval T x`.
* `run_complete`: a valid trace that contains the root stores the root.
* `determinism`: every valid trace returns `eval T x`, so two schedules of the same tree give the
  same bits.
-/
namespace Truestar.Det

inductive Tree where
  | leaf : Nat → Tree
  | node : Tree → Tree → Tree
deriving Repr, DecidableEq

def eval {α : Type} (op : α → α → α) (x : Nat → α) : Tree → α
  | .leaf i => x i
  | .node l r => op (eval op x l) (eval op x r)

/-- Position of a node: the path from the root (`false` = left child). -/
abbrev Path := List Bool

/-- The subtree at a position, if the position exists. -/
def at? : Tree → Path → Option Tree
  | t, [] => some t
  | .leaf _, _ :: _ => none
  | .node l _, false :: p => at? l p
  | .node _ r, true :: p => at? r p

theorem at?_append (t : Tree) (p : Path) (b : Bool) :
    at? t (p ++ [b]) = (at? t p).bind (fun s => match s with
      | .leaf _ => none
      | .node l r => if b then some r else some l) := by
  induction p generalizing t with
  | nil => cases t <;> cases b <;> simp [at?]
  | cons c p ih =>
    cases t with
    | leaf _ => rfl
    | node l r => cases c <;> simp [at?, ih]


variable {α : Type} (T : Tree) (op : α → α → α) (x : Nat → α)

abbrev Mem (α : Type) := Path → Option α

/-- The value a step at `p` computes, if the node exists and its children are stored. -/
def newVal (m : Mem α) (p : Path) : Option α :=
  match at? T p with
  | none => none
  | some (.leaf i) => some (x i)
  | some (.node _ _) =>
    match m (p ++ [false]), m (p ++ [true]) with
    | some a, some b => some (op a b)
    | _, _ => none

/-- One step of a trace: evaluate the node at `p` if its children are stored (else no change). -/
def step (m : Mem α) (p : Path) : Mem α :=
  fun q => if q = p then (newVal T op x m p).or (m q) else m q

def run (ps : List Path) : Mem α := ps.foldl (step T op x) (fun _ => none)

def Sound (m : Mem α) : Prop := ∀ p v, m p = some v → ∃ s, at? T p = some s ∧ v = eval op x s

theorem newVal_sound (m : Mem α) (p : Path) (h : Sound T op x m) (v : α)
    (hv : newVal T op x m p = some v) : ∃ s, at? T p = some s ∧ v = eval op x s := by
  unfold newVal at hv
  cases hp : at? T p with
  | none => simp [hp] at hv
  | some s =>
    cases s with
    | leaf i => simp [hp] at hv; subst hv; exact ⟨_, rfl, rfl⟩
    | node l r =>
      cases ha : m (p ++ [false]) with
      | none => simp [hp, ha] at hv
      | some a =>
        cases hb : m (p ++ [true]) with
        | none => simp [hp, ha, hb] at hv
        | some b =>
          simp [hp, ha, hb] at hv; subst hv
          obtain ⟨sl, hsl, rfl⟩ := h _ _ ha
          obtain ⟨sr, hsr, rfl⟩ := h _ _ hb
          rw [at?_append, hp] at hsl hsr
          simp at hsl hsr
          subst hsl; subst hsr
          exact ⟨_, rfl, rfl⟩

theorem step_sound (m : Mem α) (p : Path) (h : Sound T op x m) :
    Sound T op x (step T op x m p) := by
  intro q v hq
  unfold step at hq
  by_cases hqp : q = p
  · subst hqp
    simp only [ite_true] at hq
    cases hn : newVal T op x m q with
    | none => simp [hn] at hq; exact h q v hq
    | some w => simp [hn] at hq; subst hq; exact newVal_sound T op x m q h w hn
  · simp only [hqp, ite_false] at hq; exact h q v hq

theorem foldl_sound (ps : List Path) (m : Mem α) (h : Sound T op x m) :
    Sound T op x (ps.foldl (step T op x) m) := by
  induction ps generalizing m with
  | nil => exact h
  | cons p ps ih => exact ih _ (step_sound T op x m p h)

/-- Soundness for any trace: a stored value is always the tree value of its subtree. -/
theorem run_sound (ps : List Path) : Sound T op x (run T op x ps) :=
  foldl_sound T op x ps _ (fun _ _ h => by simp at h)

theorem step_mono (m : Mem α) (p q : Path) (h : (m q).isSome) : (step T op x m p q).isSome := by
  unfold step
  by_cases hqp : q = p
  · simp only [hqp, ite_true]
    subst hqp
    cases newVal T op x m q <;> simp_all
  · simp only [hqp, ite_false]; exact h

theorem foldl_mono (ps : List Path) (m : Mem α) (q : Path) (h : (m q).isSome) :
    ((ps.foldl (step T op x) m) q).isSome := by
  induction ps generalizing m with
  | nil => exact h
  | cons p ps ih => exact ih _ (step_mono T op x m p q h)

/-- A step at an existing node whose children are stored stores the node. -/
theorem step_stores (m : Mem α) (p : Path) (hpos : at? T p ≠ none)
    (hch : ∀ l r, at? T p = some (.node l r) → (m (p ++ [false])).isSome ∧ (m (p ++ [true])).isSome) :
    (step T op x m p p).isSome := by
  unfold step
  simp only [ite_true]
  cases hp : at? T p with
  | none => exact absurd hp hpos
  | some s =>
    cases s with
    | leaf i => simp [newVal, hp]
    | node l r =>
      obtain ⟨hl, hr⟩ := hch l r hp
      cases ha : m (p ++ [false]) with
      | none => rw [ha] at hl; simp at hl
      | some a =>
        cases hb : m (p ++ [true]) with
        | none => rw [hb] at hr; simp at hr
        | some b => simp [newVal, hp, ha, hb]

/-- A valid trace: every node runs after both of its children. -/
def Valid (ps : List Path) : Prop :=
  ∀ i (hi : i < ps.length) l r, at? T ps[i] = some (.node l r) →
    ps[i] ++ [false] ∈ ps.take i ∧ ps[i] ++ [true] ∈ ps.take i

theorem run_take (ps : List Path) (hv : Valid T ps) :
    ∀ n, ∀ q ∈ ps.take n, at? T q ≠ none → (run T op x (ps.take n) q).isSome := by
  intro n
  induction n with
  | zero => intro q hq; simp at hq
  | succ n ih =>
    intro q hq hpos
    by_cases hn : n < ps.length
    · rw [List.take_succ_eq_append_getElem hn] at hq ⊢
      unfold run at ih ⊢
      rw [List.foldl_append]
      simp only [List.foldl_cons, List.foldl_nil]
      rcases List.mem_append.mp hq with hq' | hq'
      · exact step_mono T op x _ _ _ (ih q hq' hpos)
      · simp at hq'; subst hq'
        apply step_stores T op x _ _ hpos
        intro l r hlr
        obtain ⟨hl, hr⟩ := hv n hn l r hlr
        exact ⟨ih _ hl (by rw [at?_append, hlr]; simp), ih _ hr (by rw [at?_append, hlr]; simp)⟩
    · have : ps.take (n + 1) = ps.take n := by
        rw [List.take_of_length_le (by omega), List.take_of_length_le (by omega)]
      rw [this] at hq ⊢
      exact ih q hq hpos

/-- Completeness: a valid trace that runs the root stores a value at the root. -/
theorem run_complete (ps : List Path) (hv : Valid T ps) (hroot : [] ∈ ps) :
    (run T op x ps []).isSome := by
  have := run_take T op x ps hv ps.length [] (by rw [List.take_length]; exact hroot)
    (by simp [at?])
  rwa [List.take_length] at this

/-- **T2.** Every valid trace of a fixed tree returns `eval T x`: the result is a function of the
inputs only, independent of the schedule. -/
theorem determinism (ps : List Path) (hv : Valid T ps) (hroot : [] ∈ ps) :
    run T op x ps [] = some (eval op x T) := by
  have hs := run_complete T op x ps hv hroot
  cases h : run T op x ps [] with
  | none => rw [h] at hs; simp at hs
  | some v =>
    obtain ⟨s, hs', rfl⟩ := run_sound T op x ps [] v h
    simp [at?] at hs'; subst hs'; rfl

/-- Two valid schedules of the same tree give the same result. -/
theorem schedule_independent (ps qs : List Path) (hp : Valid T ps) (hq : Valid T qs)
    (hrp : [] ∈ ps) (hrq : [] ∈ qs) : run T op x ps [] = run T op x qs [] := by
  rw [determinism T op x ps hp hrp, determinism T op x qs hq hrq]

end Truestar.Det
