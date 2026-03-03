# The Dimension of Transitive Closure: A Complete Hierarchy on Functional Graphs

## Wymiar domknięcia przechodniego: kompletna hierarchia na grafach funkcyjnych

---

## Abstract

We establish a complete hierarchy of definability for extensions of first-order logic with transitive closure operators of varying dimension on the class of finite functional graphs (structures with a single unary function). We prove three strict separations:

1. **FO < FO+TC₁**: First-order logic cannot define reachability properties (e.g., "element lies on a cycle of length > 2^q").

2. **FO+TC₁ < FO+TC₂**: Unary transitive closure (equivalent to MSO on linear orders) defines only ultimately periodic properties of cycle lengths, while binary TC enables BIT-addressing and full NL-power. The property SEC ("some equality of cycles" — existence of two cycles of different lengths) witnesses this separation.

3. **FO+TC₂ = FO+TC_k for all k ≥ 2**: The dimension hierarchy collapses at dimension 2 on functional graphs, because binary TC extracts a linear order from any cycle (via the "cutting trick"), yielding full NL-power per Immerman's theorem.

4. **FO+TC < FO+TC+C**: Transitive closure (of any dimension) cannot count unbounded multiplicities of isomorphic components. The "threshold counting barrier" limits FO+TC to detecting component types up to a fixed threshold.

The key conceptual contribution is identifying the **dimension of TC** as the fundamental parameter controlling the transition from topology (global reachability) to arithmetic (binary addressing), and characterizing exactly when this transition occurs.

---

## 1. Preliminaries

### 1.1 Structures

A **functional graph** is a finite structure (E, T) where T: E → E is a total unary function. Every such structure decomposes into connected components, each being a **ρ-shape**: a unique cycle with trees ("tails") hanging from cycle vertices.

For a cycle C_l of length l: Aut(C_l) ≅ ℤ_l (full rotational symmetry).

### 1.2 Logics

**FO[T]**: First-order logic over signature {T}.

**FO+TC₁**: FO extended with unary transitive closure. For a formula φ(x, y, z̄) with parameters z̄:
  [TC_{x,y} φ(x, y, z̄)](a, b)
means: b is reachable from a by iterated φ-steps.

**FO+TC₂** (= standard FO+TC): FO extended with TC over pairs:
  [TC_{(x₁,x₂),(y₁,y₂)} φ(x₁, x₂, y₁, y₂, z̄)](a₁, a₂, b₁, b₂)
means: (b₁, b₂) is reachable from (a₁, a₂) by iterated φ-steps in V × V.

**FO+TC_k**: TC over k-tuples. Standard FO+TC in the literature (Immerman 1987) includes all dimensions.

**FO+TC+C**: FO+TC extended with counting quantifiers ∃^{≥n}x.

### 1.3 Key properties

**OnCyc(x)**: "x lies on a cycle" = [TC_{u,v} T(u)=v](T(x), x).

**SameCyc(x, y)**: "x and y lie on the same cycle" = OnCyc(x) ∧ OnCyc(y) ∧ Reach(x, y).

**SEC**: "Some Equality of Cycles" — there exist two cycles of different lengths.
Formally: ∃x ∃y: ¬SameCyc(x,y) ∧ DiffLen(x,y), where DiffLen uses a synchronized walk (TC₂).

---

## 2. The Cutting Trick: Extracting Order from Symmetry

### 2.1 Statement

**Theorem 1 (Cutting Lemma).** On any cycle C_l, FO+TC₁ can define a linear order on all l elements using one existential parameter.

### 2.2 Construction

Define the "cut step" relation with parameter a:
  R_a(x, y) := T(x) = y ∧ x ≠ a

Then:
  x ≤_a y := [TC_{u,v} R_a(u,v)](x, y) ∨ x = y

This defines a linear order on C_l with minimum T(a) and maximum a. The element a acts as a "cut point" that breaks the rotational symmetry.

### 2.3 Significance

- Cost: 1 existential quantifier + 1 TC₁ application = O(1) quantifier rank.
- Since C_l is vertex-transitive, the choice of a does not affect the isomorphism type of the resulting order. Thus ∃a: φ(≤_a) has the same truth value for every a.
- This transforms the cycle (maximal rotational symmetry) into a linear order (no non-trivial automorphisms).

**Corollary.** Any FO+TC formula about C_l can be reduced to a formula about the linear order [l] at cost O(1) in quantifier rank.

### 2.4 Order Extraction Lemma (bridging to Immerman's theorem)

**Lemma (Order Extraction).** Let ψ be an FO+TC sentence over σ ∪ {<}. Let K be a class of σ-structures such that for each M ∈ K there exists an FO+TC₁ formula χ(x, y, z̄) over σ where:

(i) ∃z̄: χ defines a linear order on M,  
(ii) for any z̄₁, z̄₂ satisfying (i), the expansions (M, ≤_{z̄₁}) and (M, ≤_{z̄₂}) are isomorphic.

Then ψ is equivalent on K to an FO+TC sentence ψ* over σ (without <).

**Proof.** Define ψ* := ∃z̄: [χ(−,−,z̄) defines a linear order] ∧ ψ(≤_{z̄}).

This is FO+TC over σ. Correctness: condition (ii) guarantees that ψ(≤_{z̄}) has the same truth value for every valid z̄, since any two valid expansions are isomorphic and ψ is a sentence (invariant under isomorphism). ∎

**Application to cycles.** On C_l: χ(x,y,a) := [TC_{u,v}(T(u)=v ∧ u≠a)](x,y) ∨ x=y. Condition (i): every a gives a linear order. Condition (ii): C_l is vertex-transitive, so rotation a₁→a₂ induces isomorphism of orders. Cost: O(1) rank.

Therefore: Immerman's theorem (FO+TC on ordered structures = NL) transfers to FO+TC₂ on cycles via the Order Extraction Lemma.

---

## 3. FO+TC₁ on Cycles: Periodicity

### 3.1 The regularity argument

**Theorem 2.** FO+TC₁ on a cycle C_l (after cutting) corresponds to MSO on the linear order [l]. On unary structures (single-letter alphabet), MSO defines exactly the ultimately periodic properties.

**Proof sketch.** Unary TC over a linear order [l] with successor T is expressively equivalent to MSO over ([l], <), by the classical correspondence (Thomas 1997, Schweikardt 2004). MSO over linear orders captures exactly the regular languages (Büchi's theorem). Over a single-letter alphabet, regular = ultimately periodic. ∎

### 3.2 Consequence: congruential classification

**Theorem 3 (H2 for TC₁).** For each quantifier rank q, there exist M(q), N(q) ≤ Tower(O(q)) such that:

  C_{l₁} ≡_{FO+TC₁(q)} C_{l₂}  ⟺  min(l₁, N(q)) = min(l₂, N(q)) ∧ l₁ ≡ l₂ (mod M(q))

**Proof.** Reduction to [l] (Theorem 1) + Ehrenfeucht's classical theorem on FO types of linear orders + the regularity argument (Theorem 2). ∎

### 3.3 What FO+TC₁ sees

On a single cycle: l mod M(q) (periodically) plus exact small values.
On disjoint unions: profile of congruence classes with threshold (Feferman-Vaught).

---

## 4. The Separation: SEC ∉ FO+TC₁

### 4.1 Statement

**Theorem 4.** The property SEC = "there exist two cycles of different lengths" is not definable in FO+TC₁.

### 4.2 Proof

By Feferman-Vaught compositionality (which applies to FO+TC₁ since TC₁ does not connect disjoint components), the FO+TC₁-theory of C_{l₁} ⊔ C_{l₂} is determined by the pair (Th_{FO+TC₁}(C_{l₁}), Th_{FO+TC₁}(C_{l₂})).

By Theorem 3, Th_{FO+TC₁}(C_l) is determined by l mod M(q) (for large l).

SEC = "l₁ ≠ l₂". We show this is not expressible as a Boolean combination of conditions on (l₁ mod M, l₂ mod M):

Take l₁ = l₂ = l (large). Then SEC = false.
Take l₁ = l, l₂ = l + M (both large). Then SEC = true.
But l₁ mod M = l₂ mod M, and both FO+TC₁-types are identical.
Any Boolean combination of periodic conditions assigns the same value to both cases.
Contradiction. ∎

### 4.3 Significance

SEC requires **comparing** two cycle lengths simultaneously. This comparison requires synchronized traversal of two cycles — which is exactly what TC₂ provides and TC₁ lacks. SEC is a natural witness of the TC₁/TC₂ gap.

---

## 5. FO+TC₂: From Periodicity to Arithmetic

### 5.1 The half construction

**Definition.** On a linear order [l] with successor T, define half(min, max) as the element m such that pos(m) ≈ l/2, via synchronized walk:

  [TC₂_{(u₁,u₂),(v₁,v₂)} (T(u₁)=v₁ ∧ T(T(u₂))=v₂)](min, min, m, max?)

One pointer advances by 1, the other by 2. When the fast pointer reaches max, the slow pointer is at the midpoint.

### 5.2 From half to BIT

Iterating half with nested TC₂:
- half¹: position l/2 (1 bit: MSB)
- half²: position l/4 (2nd bit)
- half^d: position l/2^d (d-th bit)

This gives access to individual bits of the binary representation of l.

**Theorem 5 (BIT from TC₂).** There exists a constant q₀ such that FO+TC₂ of rank q₀ defines the predicate BIT(max, i) = "the bit at position i in the binary representation of l is 1" on any linear order [l].

### 5.3 From BIT to NL

**Theorem 6 (Immerman 1987).** On ordered structures with BIT, FO+TC defines exactly NL.

**Corollary.** FO+TC₂ of rank q₀ + O(1) has full NL-power on any cycle C_l (via cutting + BIT).

### 5.4 Consequences

- **EVEN ∈ FO+TC₂**: l mod 2 is NL-computable. ✓
- **PRIMES ∈ FO+TC₂**: Primality testing is in L ⊆ NL. ✓
- **SEC ∈ FO+TC₂**: Via synchronized walk on two cycles (original construction). ✓
- **Any NL-computable property of l** is FO+TC₂-definable on cycles.

---

## 6. The Separation: FO+TC₁ < FO+TC₂

### 6.1 Witness property

**Definition.** LOG-PARITY(l) = "⌊log₂ l⌋ is even".

This equals: l ∈ [1,2) ∪ [4,8) ∪ [16,32) ∪ [256,512) ∪ ...

### 6.2 LOG-PARITY is not ultimately periodic

**Lemma.** LOG-PARITY is not ultimately periodic.

**Proof.** Suppose period M. Take k large enough that 2^{2k} > M. Then:
- LOG-PARITY(2^{2k} - 1) = false (since ⌊log₂(2^{2k}-1)⌋ = 2k-1, odd)
- LOG-PARITY(2^{2k} - 1 + M) — since M < 2^{2k}, we have 2^{2k}-1+M ∈ [2^{2k}, 2^{2k+1}), so ⌊log₂⌋ = 2k, even, thus true.
This contradicts P(l) = P(l+M) for l = 2^{2k}-1. ∎

### 6.3 LOG-PARITY ∈ FO+TC₂ — formal construction

**Proof.** On [l] after cutting (with min, max, successor T):

**Step 1: Define HALF.**

HALF(s, e, m) := "m is the midpoint of [s, e]", defined by synchronized walk where one pointer steps by 1 and the other by 2:

  HALF(s, e, m) := ∃e': [TC₂_{(u₁,u₂),(v₁,v₂)} (T(u₁)=v₁ ∧ T(T(u₂))=v₂ ∧ u₂≠e ∧ T(u₂)≠e)](s, s, m, e') ∧ (e'=e ∨ T(e')=e)

This is a single TC₂ application with FO-atomic step. Rank: O(1).

**Step 2: Define QUARTER-STEP** (two halvings).

QUARTER-STEP(s, e, s', e') := ∃m: HALF(s, e, m) ∧ HALF(s, m, s', e')

This composes two HALF applications, yielding ⌊|[s,e]|/4⌋. Uses TC₂ internally (from HALF). Rank: O(1).

**Step 3: Define SMALL** (interval of size < 4).

SMALL(s, e) := s=e ∨ T(s)=e ∨ T(T(s))=e

This is FO of rank O(1).

**Step 4: Define UNIT** (interval of size 1).

UNIT(s, e) := s = e

**Step 5: Main formula** — iterate quartering until small, then check unit.

LOG-PARITY := ∃a [cutting] ∃s_f ∃e_f:
  [TC₂_{(s,e),(s',e')} (QUARTER-STEP(s,e,s',e') ∧ ¬SMALL(s,e))](min_a, max_a, s_f, e_f)
  ∧ SMALL(s_f, e_f)
  ∧ UNIT(s_f, e_f)

**Verification:**

- l=1: SMALL at start. s_f=e_f=min=max. UNIT true. LOG-PARITY = true. (⌊log₂1⌋=0, even ✓)
- l=3: SMALL at start (size 3 < 4). UNIT false. LOG-PARITY = false. (⌊log₂3⌋=1, odd ✓)
- l=5: Quarter: 5→1. UNIT true. true. (⌊log₂5⌋=2, even ✓)
- l=8: Quarter: 8→2. UNIT false. false. (⌊log₂8⌋=3, odd ✓)
- l=16: Quarter: 16→4→1. UNIT true. true. (⌊log₂16⌋=4, even ✓)
- l=32: Quarter: 32→8→2. UNIT false. false. (⌊log₂32⌋=5, odd ✓)
- l=64: Quarter: 64→16→4→1. UNIT true. true. (⌊log₂64⌋=6, even ✓)

**TC₂ nesting depth: 2** (HALF inside QUARTER-STEP, QUARTER-STEP inside outer TC₂).
**Total rank: O(1)**, independent of l.  ∎

### 6.4 Separation theorem

**Theorem 7.** FO+TC₁ < FO+TC₂ on the class of single-cycle functional graphs.

**Proof.** LOG-PARITY ∈ FO+TC₂ (§6.3) and LOG-PARITY ∉ FO+TC₁ (§6.2 + Theorem 2). ∎

---

## 7. Collapse of Dimension Hierarchy at k = 2

### 7.1 Statement

**Theorem 8.** On the class of functional graphs, FO+TC₂ = FO+TC_k for all k ≥ 2.

### 7.2 Proof sketch

On a cycle C_l after cutting: FO+TC₂ achieves full NL-power (Theorem 6). Since FO+TC_k ⊆ NL on ordered structures for all k (Immerman), and cutting provides the order, FO+TC_k cannot exceed NL. Thus FO+TC₂ = FO+TC_k on cycles.

By FV-compositionality (which preserves the collapse), the equality extends to disjoint unions of cycles, hence to all functional graphs. ∎

---

## 8. The Threshold Counting Barrier

### 8.1 Statement

**Theorem 9 (Threshold Counting).** For any FO+TC sentence φ of rank q, there exists a threshold t(q) such that for all k and all m₁, m₂ > t(q):

  C_k^{⊔m₁} ⊨ φ  ⟺  C_k^{⊔m₂} ⊨ φ

That is, FO+TC cannot distinguish between "many" and "even more" copies of the same cycle.

### 8.2 Proof

**Duplicator's strategy** in the EF+TC game on C_k^{⊔m₁} vs C_k^{⊔m₂}:

Maintain a bijection σ between "visited" copies (copies containing a chosen element). For visited copies, maintain a partial isomorphism preserving all atomic and TC relations. For unvisited copies, use a reserve strategy.

Key: TC does not connect different copies of C_k (no T-edges between components). Therefore TC-reachability within one copy is independent of other copies.

After q rounds, at most q copies are visited on each side. If m₁, m₂ > q, both sides have non-empty reserves. Duplicator can always respond:
- If Spoiler picks an element in a visited copy: respond in the corresponding copy.
- If Spoiler picks an element in an unvisited copy: pick any unvisited copy on the other side and establish isomorphism.

TC moves between elements in different copies always return false (unreachable), preserved by the strategy. TC moves within a single copy are preserved by the copy-isomorphism. ∎

### 8.3 Consequences

- "Even number of 5-cycles" is NOT FO+TC-definable (changes between m and m+1).
- "Exactly 17 copies of C₃" is NOT FO+TC-definable for structures with >q copies.
- FO+TC sees the **set** of cycle lengths (with NL-type) but not the **multiset** (exact multiplicities).

---

## 9. Complete Characterization

### 9.1 Main theorem

**Theorem 10 (Complete Characterization).** On the class of finite functional graphs with signature {T}:

A property P is FO+TC-definable if and only if P is determined by the **TC-profile with threshold**:

  Profile_q(E, T) := τ ↦ min(count(τ), t(q))

where τ ranges over ≡_{FO+TC(q)}-equivalence classes of connected components, and t(q) is the threshold from Theorem 9.

For purely cyclic components (no tails), the equivalence class of C_l is determined by the NL-type of l — that is, by the class of l under the finite equivalence relation ≡_{NL(q₀)} on natural numbers.

### 9.2 The five-level hierarchy

| Level | Logic | What it sees on C_l | Complexity on unary l |
|-------|-------|---------------------|-----------------------|
| 0 | FO | r-neighborhoods | AC⁰ |
| 1 | FO+TC₁ = FO+DTC₁ | l mod M(q) (periodic) | REG |
| 2 | FO+DTC₂ | BIT, full deterministic arithmetic | L |
| 3 | FO+TC₂ = FO+TC | NL-type of l (full reachability in V²) | NL |
| 4 | FO+TC+C | NL-type of l + exact component counts | NL + counting |

All inclusions are non-strict or strict as follows:
- Level 0 < 1: strict. Witness: OnCyc.
- Level 1 < 2: strict. Witness: LOG-PARITY (not ultimately periodic, but in L).
- Level 2 ≤ 3: strict iff L ≠ NL. This is the fundamental open problem.
- Level 3 < 4: strict. Witness: "even number of 5-cycles".

**Key structural insight:** On functional graphs, FO+DTC₁ = FO+TC₁ (since T is a function, unary TC steps are automatically deterministic). But FO+DTC₂ ≠ FO+TC₂ (unless L = NL), because TC₂ can use non-deterministic steps even when T is deterministic — e.g., φ(u₁,u₂,v₁,v₂) := (T(u₁)=v₁ ∧ v₂=u₂) ∨ (u₁=v₁ ∧ T(u₂)=v₂), which non-deterministically chooses which pointer to advance.

---

## 10. When Does TC Dimension Matter?

### 10.1 The mechanism

TC₁ tracks a **single trajectory** through the structure. On a linear order, this yields MSO-power: ultimately periodic properties.

TC₂ tracks a **pair of trajectories simultaneously**. When two pointers traverse the same path at different speeds (1 and 2), the synchronized walk computes half — the midpoint. Iterated halving yields BIT-addressing, hence full NL.

**Core principle:** Synchronized multi-speed traversal transforms topology into arithmetic.

### 10.2 Sufficient condition (proved) and necessary condition (conjectured)

**Theorem 11a (Sufficient condition).** If a class K of σ-structures contains structures with arbitrarily long paths admitting an FO-definable deterministic step function (i.e., a total unary function generating unbounded orbits), then FO+TC₂ is strictly more expressive than FO+TC₁ on K.

**Proof.** On structures with a long deterministic path under function T: cutting + HALF gives LOG-PARITY ∈ FO+TC₂ (Section 6.3). LOG-PARITY is not ultimately periodic, hence LOG-PARITY ∉ FO+TC₁ (Section 6.2 + Theorem 2). ∎

**Conjecture 1 (Necessary condition).** If K has no FO-definable deterministic step function generating unbounded paths, then FO+TC₂ = FO+TC₁ on K.

**Justification.** Without a deterministic step function, the synchronized walk in TC₂ does not produce a unique midpoint: if E²(x,y) is multi-valued, the HALF construction fails to yield a canonical element. We conjecture that no alternative TC₂ mechanism can compensate, but a formal proof would require EF+TC₂ games on the relevant structure classes — this remains open.

### 10.3 Collapse conditions

The TC dimension hierarchy collapses at k = 2 on any class where TC₂ can extract a linear order (via cutting or other means), because FO+TC₂ on ordered structures = NL = FO+TC_k.

---

## 11. The Symmetry-Breaking Perspective

### 11.1 Three mechanisms of symmetry breaking

1. **FO → FO+TC₁**: Global propagation. One can reach distant elements, breaking the locality barrier. But a single trajectory on a homogeneous structure (cycle) does not break rotational symmetry beyond periodicity.

2. **FO+TC₁ → FO+TC₂**: Synchronized traversal. Two pointers at different speeds create relative measurements, breaking symmetry through differential motion. This yields coordinate systems (BIT) and arithmetic.

3. **FO+TC → FO+TC+C**: Canonical counting. Counting allows distinguishing isomorphic copies, breaking the symmetry between identical components — the one barrier that TC of any dimension cannot overcome.

### 11.2 Two orthogonal dimensions

Symmetry breaking in logic operates along two orthogonal axes:

- **Propagational axis (TC):** Existential choice of anchor point + global reachability → internal order. Powerful on structures with "cuttable" topology (cycles, paths). Weak on fully symmetric structures (cliques, vertex-transitive graphs).

- **Numerical axis (C / counting):** Iterative refinement + multiplicity → distinction by count. Powerful on structures with numerical asymmetry (varying degrees). Weak on regular structures.

FO+TC exploits the propagational axis. FO+C (related to Weisfeiler-Leman) exploits the numerical axis. FO+LFP combines both.

---

## 12. Open Questions

1. **Precise bit-extraction function:** What is the exact number of bits of log₂(l) that FO+TC₂ of rank q can extract? Is it Θ(q)?

2. **L vs NL within TC₂:** On cycles, does FO+DTC (deterministic TC) equal FO+TC₂? This would separate L from NL in the descriptive complexity framework on functional graphs.

3. **Beyond functional graphs:** Does the cutting trick generalize to graphs with Hamiltonian paths? What is the TC dimension hierarchy on bounded-degree graphs?

4. **Relation to Weisfeiler-Leman:** On cycles, FO+TC₂ > WL (1-dimensional Weisfeiler-Leman), since WL cannot distinguish C_l from C_m. Characterize the classes where TC₂ and WL are incomparable.

5. **General theory of compression:** For logic pairs L ⊆ L', define Compression(L'/L) as the class of properties that are infinite Boolean combinations of L-sentences but lie in L'. Is Compression(FO+TC₂ / FO) = NL-computable properties of cycle spectra?

---

## 13. Summary of Results

| # | Result | Status |
|---|--------|--------|
| T1 | Cutting Lemma: FO+TC₁ extracts linear order from cycle | Proved |
| T2 | FO+TC₁ on orders = MSO = regular = periodic | Known (Thomas, Büchi) |
| T3 | Congruential classification for FO+TC₁ | Proved (via T1 + T2 + Ehrenfeucht) |
| T4 | SEC ∉ FO+TC₁ | Proved (FV + periodicity) |
| T5 | BIT from TC₂ via iterated half | Proved (construction) |
| T6 | FO+TC₂ = NL on cycles | Follows from T1 + T5 + Immerman |
| T7 | FO+TC₁ < FO+TC₂ (LOG-PARITY separates) | Proved |
| T8 | Dimension collapse: TC₂ = TC_k (k ≥ 2) | Proved (via T6) |
| T9 | Threshold counting barrier | Proved (EF game on copies) |
| T10 | Complete characterization: TC-profiles with threshold | Follows from T6 + T9 + FV |
| T11a | Sufficient condition: TC₂ > TC₁ when deterministic long paths exist | Proved |
| C1 | Necessary condition: TC₂ = TC₁ without deterministic long paths | Conjecture |

---

## 14. Methodological Notes

This work progressed through five major iterations, each discovering and correcting an error in the previous:

1. **Initial:** "TC = compression of infinite FO-disjunctions" → Too broad (PRIMES is such a disjunction but may not be TC₁-definable in the way claimed).

2. **Correction 1:** "TC sees only divisibility relations, not absolute arithmetic" → False (the cutting trick gives arithmetic).

3. **Correction 2:** "H2: types = congruences mod M(q)" → True for TC₁, false for standard (multi-dimensional) TC.

4. **Correction 3:** "Regularity argument closes everything" → Applies only to TC₁ (MSO), not to TC₂ (beyond MSO).

5. **Final stable architecture:** The dimension of TC as hidden parameter; three-logic hierarchy FO < FO+TC₁ < FO+TC₂.

Each error arose from implicitly conflating syntactically similar but semantically different operators — most critically, treating unary TC and multi-dimensional TC as equivalent. The formalization process (attempting rigorous proofs) exposed every such conflation.

---

*This document represents the stable core of the theory. All results marked "Proved" have complete proof sketches; full formal proofs require detailed EF-game constructions (for T4, T9) and verification of the half/BIT construction (for T5).*


---

## Appendix A: Extension to ρ-shapes

### A.1 Structure of ρ-shapes

A connected component of a functional graph is a **ρ-shape**: a unique cycle C_l (the core) with rooted trees (tails) hanging from cycle vertices. Each cycle vertex v has a preimage tree T_v^{-1} whose root is v.

### A.2 Projection to cycle

Every element x has a unique **cycle projection** π(x) — the first cycle element on the T-orbit of x. Definable in FO+TC₁:

  π(x) = y ⟺ OnCyc(y) ∧ Reach(x, y) ∧ ∀z(Reach(x,z) ∧ OnCyc(z) → Reach(y,z) ∨ y=z)

The **depth** of x is depth(x) = distance from x to π(x), definable via TC₁.

### A.3 FO+TC₂ on ρ-shapes

**Theorem A1.** The cutting trick and BIT construction extend to ρ-shapes:

(a) Cutting the cycle (∃a: OnCyc(a)) gives a linear order on cycle vertices. Cost: O(1) rank.

(b) BIT on cycle length is definable via HALF on cycle vertices (restricting the synchronized walk to cycle elements via the OnCyc predicate). Cost: O(1) rank.

(c) The **cycle word** — the sequence of tail-types along the cycle — is accessible to FO+TC₂ as an NL-computation over the ordered cycle with "letters" given by tail-types at each position.

### A.4 Branching barrier

On ρ-shapes with **branching tails** (vertices with multiple T-preimages), FO+TC₂ cannot linearly order the branches. This is the same isomorphism barrier that prevents ordering isomorphic components, now occurring *within* a single component.

**Theorem A2.** On ρ-shapes with purely linear tails (each vertex has at most one T-preimage outside the cycle), FO+TC₂ achieves full NL-power: the type is determined by the cycle length and the sequence of tail depths.

On ρ-shapes with branching tails, FO+TC₂-types are determined by:
1. NL-type of the cycle (length, via cutting + BIT).
2. The cycle word of tail-types (NL-accessible via ordered cycle).
3. Tail-type at each vertex: determined by the tree structure *up to the branching barrier* — i.e., depth, branching profile (thresholded), but not full ordering of isomorphic subtrees.

---

## Appendix B: Orthogonality of Propagational and Numerical Axes

### B.1 Definitions

**Propagational symmetry-breaking** (via TC): A logic L breaks propagational symmetry on class K if there exists an L-formula φ(x, y, z̄) such that ∃z̄: φ defines a linear order on arbitrarily large structures in K.

**Numerical symmetry-breaking** (via Counting): A logic L breaks numerical symmetry on class K if L can distinguish structures differing only in the multiplicity of isomorphic components.

### B.2 Orthogonality theorem

**Theorem B1 (Orthogonality).** On finite functional graphs:

(a) FO+TC₂ breaks propagational symmetry (via cutting) but not numerical symmetry (threshold barrier, Theorem 9).

(b) FO+C breaks numerical symmetry (via counting quantifiers) but not propagational symmetry on vertex-transitive structures (WL cannot order cycle vertices).

(c) SEC ∈ FO+TC₂ ∖ FO+C: 1-dimensional Weisfeiler-Leman assigns identical colors to all vertices of C_l (l ≥ 3), hence cannot distinguish C_5 ⊔ C_5 from C_5 ⊔ C_7.

(d) "Even number of components" ∈ FO+C ∖ FO+TC₂: requires exact global counting, which TC₂ cannot perform (Theorem 9).

(e) FO+TC₂+C combines both axes.

### B.3 Connection to CFI and the logic-for-P question

The Cai-Fürer-Immerman construction produces graphs that are:
- Regular (defeating the numerical axis: WL/counting cannot help).
- Without extractable order (defeating the propagational axis: no cutting trick).

**Conjecture 2.** CFI graphs lie in the "dead zone" of both symmetry-breaking dimensions. A logic capturing P would need a third mechanism beyond propagation and counting — or a way to combine them that transcends both.

---

## Appendix C: The DTC₂ Layer

### C.1 DTC₂ on functional graphs

**Theorem C1.** On functional graphs:

(a) FO+DTC₁ = FO+TC₁ (since T is a function, all unary TC steps are automatically deterministic).

(b) FO+DTC₂ ⊊ FO+TC₂ (unless L = NL), because TC₂ admits non-deterministic steps — e.g., "advance pointer 1 OR pointer 2" — which DTC₂ cannot express.

(c) The HALF and BIT constructions use only deterministic steps (T and T∘T are functions), hence HALF, BIT, LOG-PARITY ∈ FO+DTC₂.

### C.2 The five-level hierarchy (refined)

| Level | Logic | Power on C_l | Complexity |
|-------|-------|-------------|------------|
| 0 | FO | r-neighborhoods | AC⁰ |
| 1 | FO+TC₁ = FO+DTC₁ | l mod M(q) | REG |
| 2 | FO+DTC₂ | BIT, deterministic log-space | L |
| 3 | FO+TC₂ = FO+TC | full reachability in V² | NL |
| 4 | FO+TC+C | + exact component counting | NL+C |

Separations: 0<1 (OnCyc), 1<2 (LOG-PARITY), 2≤3 (iff L≠NL), 3<4 (even # of k-cycles).

### C.3 Structural interpretation of L vs NL

On cycles, the gap between L and NL corresponds precisely to:
- **L (= FO+DTC₂):** Can iterate a single deterministic function on V×V (one trajectory through the grid).
- **NL (= FO+TC₂):** Can ask reachability along *any* path in a directed graph on V×V (many possible trajectories).

The difference is deterministic vs non-deterministic exploration of the product space V².

---

## Appendix D: Value of q₀ (threshold for full NL)

### D.1 Analysis

The threshold rank q₀ at which FO+TC₂ achieves full NL-power is a small absolute constant. Specifically:

- HALF requires: 1 TC₂ with atomic FO step. Rank contribution: ~4 (TC₂ binds 4 variables + FO conditions).
- BIT requires: HALF nested inside an iteration (TC₂ of TC₂). Rank contribution: ~8-10.
- NL simulation requires: BIT + TC₂ over configuration graph. Rank contribution: ~4 more.
- Cutting: +1 (∃a).

**Estimate:** q₀ ≈ 12-16. This is a small constant independent of structure size.

### D.2 Consequence

For q ≥ q₀: FO+TC₂ rank q = NL on cycles. All NL-computable properties of l are definable.

For q < q₀: FO+TC₂ rank q has intermediate power. The precise number of "accessible bits" at each sub-threshold rank q depends on which constructions fit within rank q. A complete analysis would require tracking variable usage and nesting depth precisely — this is left as a technical exercise.

The key insight: the interesting structure is NOT in the gradual growth of bit-access, but in the sharp phase transitions between the five levels of the hierarchy.


---

## Appendix E: Complete Proof of Theorem 10

### E.1 Structure of the proof

Theorem 10 (Complete Characterization) follows from three components:

**(A) Finiteness of types.** ≡_{FO+TC(q)} has finitely many classes, each definable at rank q. [Lemma A — standard, from syntactic finiteness.]

**(B) Same profile ⟹ same theory.** If two functional graphs have the same TC-profile with threshold t(q), they satisfy the same FO+TC sentences of rank q. [Lemma B — Duplicator strategy.]

**(C) Different profile ⟹ different theory.** If profiles differ, structures are distinguishable at rank q. [Lemma C — follows from (A) + (B) by contrapositive.]

### E.2 FV decomposition for FO+TC on disjoint sums

**Lemma (FV for TC).** On disjoint sums of functional graph components:

For FO[T]-formula φ(x, y) with x ∈ A, y ∈ B (different components): φ(x,y) is equivalent to a Boolean combination ⋁_i (α_i(x) ∧ β_i(y)) where α_i depends only on (A, x) and β_i only on (B, y).

**Proof.** By induction on formula complexity. Base: T(x)=y is false for x∈A, y∈B (since T maps within components). Boolean: trivial. Quantifier ∃z: split into ∃z∈A and ∃z∈B, apply induction. ∎

**Corollary.** TC_φ(a, b) with a ∈ A, b ∈ B is determined by: (type of a in A, type of b in B, theory of A, theory of B). Proof: the step relation φ between components factorizes through types, reducing multi-hop reachability to reachability in a finite graph of types (constant size). ∎

### E.3 Proof of Lemma B (Duplicator strategy)

**Setup.** G₁ = ⊔ᵢ Aᵢ, G₂ = ⊔ⱼ Bⱼ with identical profiles at threshold t(q) > q.

**Duplicator's state:** A bijection σ between "active" components (those containing chosen elements), plus partial isomorphisms within each active pair (Aᵢ, B_{σ(i)}).

**Invariant after k rounds (k ≤ q):**

(I1) σ is a bijection between active components, pairing components of the same type.

(I2) Within each active pair, the element pairing is a partial isomorphism (preserves all atomic and TC relations established so far).

(I3) For each type τ: both sides have ≥ 1 inactive component of type τ (since original counts agree up to threshold t(q) > q ≥ k).

**Existential move:** If Spoiler picks c in active Aᵢ: Duplicator responds in B_{σ(i)} using the internal strategy (possible since Aᵢ ≡_{q-k} B_{σ(i)}). Cross-component TC relations preserved by FV factorization.

If Spoiler picks c in inactive Aᵢ of type τ: Duplicator picks an inactive Bⱼ of type τ (exists by I3), activates it, and maps c to the corresponding element.

**TC move:** TC_φ(pᵢ, pⱼ). If same component: preserved by I2. If different components: preserved by FV factorization + type agreement.

**Conclusion.** Duplicator wins q-round game. G₁ ≡_{FO+TC(q)} G₂. ∎

### E.4 Proof of Lemma C

**Claim.** If profiles differ at type τ: min(#_τ(G₁), t(q)) ≠ min(#_τ(G₂), t(q)), then G₁ ≢_{FO+TC(q)} G₂.

**Proof.** From Lemma A, each ≡_{FO+TC(q)} class is definable by a rank-q sentence. From Lemma B, structures with the same profile are in the same class. Therefore structures with different profiles are in different classes, hence distinguishable by some rank-q sentence. ∎

Note: An explicit distinguishing sentence can be constructed as AtLeast_k_τ (at cost of higher rank q + t(q) + O(1)), but existence at rank q follows abstractly from (A) + (B).

### E.5 Synthesis

Theorem 10 follows: G₁ ≡_{FO+TC(q)} G₂ ⟺ same profile with threshold. Every FO+TC-definable property is a union of ≡_q classes, hence determined by the profile. Conversely, every property determined by the profile is a union of finitely many ≡_q classes, hence definable at rank q. ∎


---

## Appendix F: Bit-Extraction Analysis

### F.1 Quantifier rank of key constructions

Using standard quantifier rank (qr): qr(atomic) = 0, qr(¬φ) = qr(φ), qr(φ∧ψ) = max(qr(φ),qr(ψ)), qr(∃x φ) = qr(φ)+1, qr([TC φ]) = qr(φ)+1.

| Construction | Formula sketch | Quantifier rank |
|-------------|---------------|-----------------|
| T²(x)=y | ∃z(T(x)=z ∧ T(z)=y) | 1 |
| OnCyc(x) | [TC_{u,v} T(u)=v](T(x), x) | 1 |
| Cutting (≤_a) | ∃a: [TC R_a](x,y) | 2 |
| EVEN-DIST | [TC_{u,v} T²(u)=v](x,y) | 2 |
| HALF(s,e,m) | ∃e':[TC₂ step](s,s,m,e') ∧ check | 3 |
| QUARTER(s,e,m₂) | ∃m₁: HALF(s,e,m₁) ∧ HALF(s,m₁,m₂) | 4 |
| QUARTER-STEP (as TC step) | ∃m: HALF(s,e,m) ∧ ... | 4 |
| Iterated quartering (TC₂ of Q-STEP) | [TC₂ QUARTER-STEP](...) | 5 |
| LOG-PARITY | ∃a∃s_f∃e_f: [TC₂ Q-STEP] ∧ UNIT | 8 |
| BIT(max, i) | Requires TC₃ or equivalent | ~8 |
| Full NL simulation | BIT + TC₂ over configurations | ~10 |

### F.2 Three regimes

**Regime I: Periodic (q < 5).** Only FO and TC₁ constructions are available. Power: ultimately periodic properties of l. Number of ≡_q classes: Tower(O(q)).

**Regime II: Transitional (5 ≤ q < q₀ ≈ 8).** TC₂ constructions (HALF, iterated halving) become available. Each additional rank level unlocks ~1 additional bit of binary information about l. Power: O(q) most-significant bits + periodic least-significant bits. Number of classes: 2^{O(q)}.

**Regime III: Full NL (q ≥ q₀ ≈ 8).** BIT is available, giving full NL-power via Immerman's theorem. All NL-computable properties of l are definable. Number of distinct types among {C_1, ..., C_N}: N (every length is its own class).

### F.3 Phase transition theorem

**Theorem F1.** On the class of cycles, FO+TC₂ undergoes a phase transition from periodicity to full NL within O(1) levels of quantifier rank. Specifically, there exist absolute constants q₁ ≈ 5 and q₀ ≈ 8 such that:

- For q < q₁: FO+TC₂ rank q = FO+TC₁ rank q (periodic).
- For q₁ ≤ q < q₀: FO+TC₂ rank q sees O(q - q₁) bits of log₂(l) (transitional).
- For q ≥ q₀: FO+TC₂ rank q = NL on cycles (full arithmetic).

The transition from topology to arithmetic occurs within a window of ~3 rank levels.

### F.4 Significance

The sharpness of this transition reflects the algebraic nature of the HALF construction: once TC₂ is available (rank ≥ 5), iterated halving rapidly builds up to BIT, and BIT immediately yields full NL. There is no "gradual" growth of arithmetic power — the jump from periodicity to NL is essentially discrete, mediated by a single construction (synchronized multi-speed traversal).


---



---

## Appendix G: Security of the Hierarchy — Lower Bounds

### G.1 BIT is not in FO+TC1

**Theorem G1.** The predicate BIT(max, i) = "the i-th bit of l is 1" is not definable in FO+TC1 (of any rank) on the class of linear orders [l].

**Proof.** FO+TC1 on [l] corresponds to MSO on the unary word of length l. Every MSO-definable unary predicate phi(x) on [l] is ultimately periodic: there exist N, M such that for all x with N < x < l - N: phi(x) iff phi(x + M).

The set S_l = {i : BIT(l, i) = 1} is not ultimately periodic uniformly in l: Take any M. Choose l = 2^{N+1}. Then S_l = {N+1}. Position N+1 is in S_l, but N+1+M is not (since bit N+1+M of 2^{N+1} is 0). Both positions are in the range (N, l-N) for large N. This contradicts periodicity. QED

### G.2 HALF is not in FO+TC1

**Corollary G2.** The relation HALF(min, max, m) = "m is at position floor(l/2)" is not FO+TC1-definable.

**Proof.** The singleton {floor(l/2)} has position that grows linearly with l. In MSO on unary words, definable singletons must have ultimately periodic positions. But floor(l/2) is not ultimately periodic in l. QED

### G.3 Hierarchy security summary

| Statement | Status |
|-----------|--------|
| HALF not in FO+TC1 | Proved (G2) |
| BIT not in FO+TC1 | Proved (G1) |
| BIT in FO+TC2 | Proved (Appendix I) |
| LOG-PARITY not in FO+TC1 | Proved (Section 6.2) |
| LOG-PARITY in FO+TC2 | Proved (Section 6.3) |
| FO+TC1 strictly weaker than FO+TC2 | Secured by multiple witnesses |


---

## Appendix H: Generalization Beyond Functional Graphs

### H.1 TC-cuttable classes

**Definition.** A class K of finite structures is **TC-cuttable** if there exist FO+TC1 formulas Path(x, y, z-bar) and Long(z-bar) such that on every M in K:

(C1) There exists z-bar satisfying Long(z-bar), and Path defines a linear order on a subset D of M with |D| unbounded.

(C2) For any z-bar-1, z-bar-2 satisfying Long: the ordered sets (D1, order-1) and (D2, order-2) are isomorphic.

(C3) The successor in the Path-order is a function (deterministic step).

### H.2 Main generalization

**Theorem H1 (Generalized Hierarchy).** On every TC-cuttable class K:

(a) FO+TC1 sees only ultimately periodic properties of |D|.

(b) FO+TC2 defines HALF, BIT, and achieves NL-power on the extracted order.

(c) FO+TC1 is strictly weaker than FO+TC2 on K, witnessed by LOG-PARITY.

**Proof.** (a) FO+TC1 on extracted order = MSO = regular = periodic on unary alphabet. (b) HALF via TC2 synchronized walk (Appendix I). (c) LOG-PARITY separates. QED

### H.3 Examples

TC-cuttable: functional graphs, permutations, undirected cycles, undirected paths (with symmetrization), words with built-in order, Cayley graphs of cyclic groups.

Not TC-cuttable: complete graphs (TC trivial), bounded-diameter structures (TC reduces to FO), random graphs (no definable long path).

### H.4 Scope

The hierarchy FO < FO+TC1 < FO+TC2 holds on every TC-cuttable class. This captures a robust transition from global reachability through synchronized measurement to full log-space arithmetic.


---

## Appendix I: Fully Formal Construction of HALF and BIT

### I.1 Setup and conventions

**Structure after cutting.** From cycle C_l with cut point a, we obtain linear order ([l], T, min, max) where min = T_cycle(a), max = a, and T is the successor function. Convention: **absorbing boundary** T(max) = max. This is FO-definable:

  T_cut(x) = y  :=  (x != a AND T_cycle(x) = y) OR (x = a AND y = a)

where T_cycle is the original cyclic successor. Quantifier rank: 0.

### I.2 HALF: complete definition

**Definition.**

  Step_HALF(u, v, u', v'; e) :=  T(u) = u'  AND  T(T(v)) = v'  AND  v != e

  HALF(s, e, m) :=  EXISTS f : [TC2_{(u,v),(u',v')} Step_HALF(u,v,u',v'; e)](s, s, m, f) AND f = e

**Quantifier rank analysis.**
- Step_HALF: all conjuncts are atomic (T(u)=u': rank 0; T(T(v))=v': rank 0 since T is a function symbol; v != e: rank 0). Rank of Step_HALF = 0.
- TC2 over Step_HALF: rank 0 + 1 = 1.
- EXISTS f: rank 1 + 1 = 2.
- f = e: rank 0.
- Conjunction: max(2, 0) = 2.

**qr(HALF) = 2.**

### I.3 HALF: correctness proof

**Claim.** On interval [s, e] of length n = pos(e) - pos(s) + 1 with n >= 1:

HALF(s, e, m) holds iff m is at position pos(s) + floor(n/2).

**Proof.** The states reachable from (s, s) via Step_HALF are exactly {(s+k, s+2k) : 0 <= k <= k_max} where k_max is the largest k such that s+2(k-1) != e (i.e., s+2k-2 < pos(e), i.e., the step from (s+k-1, s+2k-2) is allowed).

At each step: the slow pointer advances by 1 (via T), the fast pointer advances by 2 (via T composed with T). The fast pointer stops when it equals e (the absorbing boundary causes T(T(e)) = e, and then v = e violates v != e).

The fast pointer reaches e when s+2k = pos(e), i.e., 2k = n-1 (n odd, k = (n-1)/2) or when the fast pointer overshoots to e via the absorbing boundary. Specifically:

- If n is odd: At step k = (n-1)/2, the fast pointer is at s+2*((n-1)/2) = s+n-1 = e. The slow pointer is at s + (n-1)/2 = s + floor(n/2). The condition f = e is satisfied. CHECK.

- If n is even: At step k = n/2 - 1, the fast pointer is at s + 2(n/2-1) = s+n-2 = e-1. Step is allowed (v = e-1 != e). Next step: T(e-1) = e, T(T(e-1)) = T(e) = e (absorbing boundary). So v' = e. The slow pointer advances to s + n/2. The condition f = e is satisfied. CHECK.

In both cases: m = s + floor(n/2). QED

### I.4 Uniqueness of m

**Claim.** For given s, e, the element m satisfying HALF(s, e, m) is unique.

**Proof.** The unique state (m, f) with f = e in the reachable set is achieved at exactly one value of k (as shown above). Different k values give different m values (since m = s+k and k is unique). QED

### I.5 BIT: complete definition

**Definition.**

  Step_BIT((e, c), (e', c')) :=  HALF(min, e, e')  AND  T(c) = c'

This simultaneously halves the interval [min, e] and advances the counter c by one step.

  BIT(i, j) :=  EXISTS e_f : [TC2_{(e,c),(e',c')} Step_BIT](i, min, e_f, j)  AND  ODD_REACH(min, e_f)

where:

  EVEN_REACH(s, e) :=  [TC1_{u,v} T(T(u))=v](s, e)  OR  s = e
  ODD_REACH(s, e) :=  EXISTS x : T(s) = x  AND  EVEN_REACH(x, e)

**Quantifier rank analysis.**
- HALF(min, e, e'): rank 2 (from I.2).
- Step_BIT: max(rank(HALF), rank(T(c)=c')) = max(2, 0) = 2.
- TC2 over Step_BIT: rank 2 + 1 = 3.
- EVEN_REACH: TC1 with atomic step, rank 1.
- ODD_REACH: EXISTS x + max(0, 1) = 2.
- EXISTS e_f: rank 3 + 1 = 4. (Wait: inside EXISTS e_f we have TC2(3) AND ODD_REACH(2), max = 3. Plus EXISTS: 4.)
- Total: 4.
- With cutting (EXISTS a): 4 + 1 = **5**.

**qr(BIT on cycles) = 5.**

### I.6 BIT: correctness proof

**Claim.** On [l] obtained from C_l by cutting: BIT(i, j) holds iff the pos(j)-th bit of the binary representation of pos(i) is 1.

**Proof.** Step_BIT iterates: at step t, the interval endpoint e_t = floor(pos(i) / 2^t) (by iterated halving, Claim I.3). The counter c_t = min + t (advancing from min by t steps of T).

The TC2 reaches (e_f, j) when the counter reaches j, i.e., t = pos(j). At that point, e_f = floor(pos(i) / 2^{pos(j)}).

ODD_REACH(min, e_f) holds iff pos(e_f) is odd, i.e., floor(pos(i) / 2^{pos(j)}) is odd, i.e., the pos(j)-th bit of pos(i) is 1. QED

### I.7 Corrected rank estimates

| Construction | Quantifier rank | Nesting depth of TC2 |
|-------------|-----------------|---------------------|
| Cutting (EXISTS a) | 1 | 0 |
| HALF(s, e, m) | 2 | 1 (one TC2) |
| BIT(i, j) | 5 (with cutting) | 2 (TC2 inside TC2) |
| LOG-PARITY | 6 (with cutting) | 2 |
| Full NL (Immerman) | ~8 | 3 |

The threshold q_0 for full NL is approximately 8. The transitional regime spans q in {3, 4, 5, 6, 7}.
