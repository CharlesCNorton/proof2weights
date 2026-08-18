(** Llama-architecture float primitives, on top of the proof2weights binary32
    development: RMSNorm, SiLU, and the trigonometric functions RoPE needs.

    RMSNorm and SiLU compose existing verified f32 operations. f32_sin and
    f32_cos are defined here by argument reduction modulo 2*pi (nearest-integer
    rounding via the add-then-subtract magic-constant trick) followed by a
    Taylor polynomial; this is the specification. The native extraction
    (Llama_native.v) replaces f32_sin and f32_cos with the host's libm, in the
    same trusted manner as the basic arithmetic. The remaining primitives are
    bit-identical between the two builds. *)

From Stdlib Require Import ZArith.
From Stdlib Require Import List.
From Stdlib Require Import Lia.
From Flocq Require Import IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete.

Import ListNotations.
Open Scope Z_scope.

(** SiLU: x * sigmoid(x). *)
Definition f32_silu (x : binary32) : binary32 := f32_mult x (f32_sigmoid x).
Definition f32_silu_vec (v : list binary32) : list binary32 := List.map f32_silu v.

(** RMSNorm: out_i = w_i * (x_i * rsqrt(mean(x^2) + eps)). *)
Definition f32_rmsnorm (w : list binary32) (eps : binary32) (x : list binary32) : list binary32 :=
  let n := f32_of_Z (Z.of_nat (List.length x)) in
  let ss := f32_sum (List.map (fun xi => f32_mult xi xi) x) in
  let ms := f32_div ss n in
  let rs := f32_div f32_one (f32_sqrt (f32_plus ms eps)) in
  List.map (fun '(wi, xi) => f32_mult wi (f32_mult xi rs)) (List.combine w x).

Lemma f32_rmsnorm_length : forall w eps x,
  List.length w = List.length x ->
  List.length (f32_rmsnorm w eps x) = List.length x.
Proof.
  intros w eps x H. unfold f32_rmsnorm.
  rewrite List.length_map, List.length_combine, H, Nat.min_id. reflexivity.
Qed.

(** pi and derived constants as binary32 values. *)
Definition f32_pi : binary32 := f32_div (f32_of_Z 31415927) (f32_of_Z 10000000).
Definition f32_2pi : binary32 := f32_mult f32_two f32_pi.
Definition f32_inv2pi : binary32 := f32_div f32_one f32_2pi.
Definition f32_magic : binary32 := f32_of_Z 8388608.

(** Nearest integer of y, as a binary32, valid for |y| < 2^22. *)
Definition f32_round_int (y : binary32) : binary32 :=
  f32_minus (f32_plus y f32_magic) f32_magic.

(** Reduce x into [-pi, pi]. *)
Definition f32_reduce_2pi (x : binary32) : binary32 :=
  let k := f32_round_int (f32_mult x f32_inv2pi) in
  f32_minus x (f32_mult k f32_2pi).

Definition f32_fac2 : binary32 := f32_of_Z 2.
Definition f32_fac3 : binary32 := f32_of_Z 6.
Definition f32_fac4 : binary32 := f32_of_Z 24.
Definition f32_fac5 : binary32 := f32_of_Z 120.
Definition f32_fac6 : binary32 := f32_of_Z 720.
Definition f32_fac7 : binary32 := f32_of_Z 5040.
Definition f32_fac8 : binary32 := f32_of_Z 40320.
Definition f32_fac9 : binary32 := f32_of_Z 362880.
Definition f32_fac10 : binary32 := f32_of_Z 3628800.
Definition f32_fac11 : binary32 := f32_of_Z 39916800.

(** sin via Taylor on the reduced argument:
    r - r^3/6 + r^5/120 - r^7/5040 + r^9/362880 - r^11/39916800. *)
Definition f32_sin (x : binary32) : binary32 :=
  let r := f32_reduce_2pi x in
  let r2 := f32_mult r r in
  let r3 := f32_mult r2 r in
  let r5 := f32_mult r3 r2 in
  let r7 := f32_mult r5 r2 in
  let r9 := f32_mult r7 r2 in
  let r11 := f32_mult r9 r2 in
  f32_minus
    (f32_plus
      (f32_minus
        (f32_plus
          (f32_minus r (f32_div r3 f32_fac3))
          (f32_div r5 f32_fac5))
        (f32_div r7 f32_fac7))
      (f32_div r9 f32_fac9))
    (f32_div r11 f32_fac11).

(** cos via Taylor on the reduced argument:
    1 - r^2/2 + r^4/24 - r^6/720 + r^8/40320 - r^10/3628800. *)
Definition f32_cos (x : binary32) : binary32 :=
  let r := f32_reduce_2pi x in
  let r2 := f32_mult r r in
  let r4 := f32_mult r2 r2 in
  let r6 := f32_mult r4 r2 in
  let r8 := f32_mult r6 r2 in
  let r10 := f32_mult r8 r2 in
  f32_minus
    (f32_plus
      (f32_minus
        (f32_plus
          (f32_minus f32_one (f32_div r2 f32_fac2))
          (f32_div r4 f32_fac4))
        (f32_div r6 f32_fac6))
      (f32_div r8 f32_fac8))
    (f32_div r10 f32_fac10).

(** * Shared layer primitives

    Slicing a head out of a fused projection, rotary embedding over a prefix of
    a head, and the SwiGLU feed-forward. Llama rotates the whole head, which is
    the [rd = head_dim] case; Qwen3.5 rotates a quarter of it. *)

Definition f32_slice (off len : nat) (r : list binary32) : list binary32 :=
  List.firstn len (List.skipn off r).

Lemma f32_slice_length : forall off len r,
  (off + len <= List.length r)%nat -> List.length (f32_slice off len r) = len.
Proof.
  intros off len r H. unfold f32_slice.
  rewrite List.length_firstn, List.length_skipn. lia.
Qed.

Definition f32_partial_rope (rd : nat) (cosv sinv : list binary32)
                            (x : list binary32) : list binary32 :=
  let half := Nat.div rd 2 in
  let rot := List.firstn rd x in
  let pass := List.skipn rd x in
  let lo := List.firstn half rot in
  let hi := List.skipn half rot in
  let rotated := List.map (fun z => f32_neg z) hi ++ lo in
  let out := List.map (fun '(xi, (ri, (ci, si))) =>
                 f32_plus (f32_mult xi ci) (f32_mult ri si))
               (List.combine rot (List.combine rotated (List.combine cosv sinv))) in
  out ++ pass.

Lemma f32_partial_rope_length : forall rd cosv sinv x,
  (rd <= List.length x)%nat ->
  List.length cosv = rd -> List.length sinv = rd ->
  List.length (f32_partial_rope rd cosv sinv x) = List.length x.
Proof.
  intros rd cosv sinv x Hrd Hc Hs. unfold f32_partial_rope.
  rewrite List.length_app, List.length_map, List.length_skipn.
  rewrite !List.length_combine, List.length_firstn.
  rewrite List.length_app, List.length_map, !List.length_skipn,
          !List.length_firstn, Hc, Hs.
  lia.
Qed.

(** SwiGLU feed-forward: [down (silu (gate x) * up x)]. *)
Definition f32_swiglu (wg wu wd : list (list binary32)) (x : list binary32)
                      : list binary32 :=
  let g := f32_silu_vec (f32_mat_vec_mul wg x) in
  let u := f32_mat_vec_mul wu x in
  f32_mat_vec_mul wd (f32_vec_mult g u).

Lemma f32_swiglu_length : forall wg wu wd x,
  List.length (f32_swiglu wg wu wd x) = List.length wd.
Proof.
  intros. unfold f32_swiglu, f32_mat_vec_mul. apply List.length_map.
Qed.

(** * The Llama layer, assembled

    RMSNorm, the query/key/value projections, rotary embedding on the per-head
    query and key, grouped-query causal attention, the output projection, and
    the SwiGLU block, each inside its residual. This is the composition the
    runner performs; naming it in Rocq is what lets the forward be stated and
    executed as one function. *)

Record llama_attn_weights := mk_llama_attn_weights {
  la_q : list (list binary32);
  la_k : list (list binary32);
  la_v : list (list binary32);
  la_o : list (list binary32)
}.

Record llama_mlp_weights := mk_llama_mlp_weights {
  lm_gate : list (list binary32);
  lm_up : list (list binary32);
  lm_down : list (list binary32)
}.

(** One head's stream, sliced out of a projection and rotated at its position. *)
Definition f32_llama_rope_rows (hd : nat) (cosv sinv : list (list binary32))
    (c : nat) (m : list (list binary32)) : list (list binary32) :=
  List.map (fun p => let '(pos, r) := p in
              f32_partial_rope hd (List.nth pos cosv []) (List.nth pos sinv [])
                (f32_slice (c * hd) hd r))
           (List.combine (List.seq 0 (List.length m)) m).

Definition f32_llama_heads (nh nkv hd : nat) (w : llama_attn_weights)
    (cosv sinv : list (list binary32)) (hn : list (list binary32))
    : list (list (list binary32)) :=
  let q := List.map (fun h => f32_mat_vec_mul (la_q w) h) hn in
  let k := List.map (fun h => f32_mat_vec_mul (la_k w) h) hn in
  let v := List.map (fun h => f32_mat_vec_mul (la_v w) h) hn in
  let group := Nat.div nh nkv in
  let qs := List.map (fun hh => f32_llama_rope_rows hd cosv sinv hh q)
                     (List.seq 0 nh) in
  let ks := List.map (fun c => f32_llama_rope_rows hd cosv sinv c k)
                     (List.seq 0 nkv) in
  let vs := List.map (fun c => List.map (fun r => f32_slice (c * hd) hd r) v)
                     (List.seq 0 nkv) in
  List.map (fun hh =>
      f32_causal_attention (List.nth hh qs [])
        (List.nth (Nat.div hh group) ks []) (List.nth (Nat.div hh group) vs []) hd)
    (List.seq 0 nh).

Definition f32_llama_attn (nh nkv hd : nat) (w : llama_attn_weights)
    (cosv sinv : list (list binary32)) (hn : list (list binary32))
    : list (list binary32) :=
  List.map (fun r => f32_mat_vec_mul (la_o w) r)
           (f32_concat_heads (f32_llama_heads nh nkv hd w cosv sinv hn)).

(** The residual pair the mixer sits in, as on the Qwen path. *)
Definition f32_llama_wrap (eps : binary32) (ln1 ln2 : list binary32)
    (mw : llama_mlp_weights)
    (mix : list (list binary32) -> list (list binary32))
    (h : list (list binary32)) : list (list binary32) :=
  let hn := List.map (fun row => f32_rmsnorm ln1 eps row) h in
  let hidden2 := List.map (fun p => let '(a, b) := p in f32_vec_add a b)
                          (List.combine h (mix hn)) in
  let h2 := List.map (fun row => f32_rmsnorm ln2 eps row) hidden2 in
  List.map (fun p => let '(a, b) := p in f32_vec_add a b)
    (List.combine hidden2
       (List.map (fun row => f32_swiglu (lm_gate mw) (lm_up mw) (lm_down mw) row)
                 h2)).

Definition f32_llama_layer (nh nkv hd : nat) (eps : binary32)
    (ln1 ln2 : list binary32) (aw : llama_attn_weights) (mw : llama_mlp_weights)
    (cosv sinv : list (list binary32)) (h : list (list binary32))
    : list (list binary32) :=
  f32_llama_wrap eps ln1 ln2 mw (f32_llama_attn nh nkv hd aw cosv sinv) h.

Fixpoint f32_llama_stack (fs : list (list (list binary32) -> list (list binary32)))
                         (h : list (list binary32)) : list (list binary32) :=
  match fs with
  | [] => h
  | f :: rest => f32_llama_stack rest (f h)
  end.

Definition f32_llama_forward (eps : binary32) (normw : list binary32)
    (fs : list (list (list binary32) -> list (list binary32)))
    (emb : list (list binary32)) (ids : list nat) : list (list binary32) :=
  List.map (fun row => f32_rmsnorm normw eps row)
           (f32_llama_stack fs (f32_embed_tokens emb ids)).

Definition f32_llama_logits (emb : list (list binary32))
    (h : list (list binary32)) : list (list binary32) :=
  List.map (fun hrow => List.map (fun wrow => f32_dot hrow wrow) emb) h.

Lemma f32_llama_logits_rows : forall emb h,
  List.length (f32_llama_logits emb h) = List.length h.
Proof. intros. unfold f32_llama_logits. apply List.length_map. Qed.

Lemma f32_llama_logits_row_width : forall emb h row,
  In row (f32_llama_logits emb h) -> List.length row = List.length emb.
Proof.
  intros emb h row Hin. unfold f32_llama_logits in Hin.
  apply List.in_map_iff in Hin. destruct Hin as [hr [Heq _]]. subst row.
  apply List.length_map.
Qed.

Lemma f32_llama_forward_rows : forall eps normw fs emb ids,
  List.length (f32_llama_forward eps normw fs emb ids)
  = List.length (f32_llama_stack fs (f32_embed_tokens emb ids)).
Proof. intros. unfold f32_llama_forward. apply List.length_map. Qed.
