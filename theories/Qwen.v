(** * Qwen3.5 primitives: gated attention and gated DeltaNet

    Qwen3.5 does not follow the Llama decoder pattern. Its layers alternate
    three linear-attention blocks with one full-attention block, and both carry
    gates that the Llama path has no counterpart for. This file adds the
    primitives that difference requires, on top of the binary32 development and
    Llama.v.

    What is new, and why:

    - [f32_log_unit] and [f32_softplus]. The DeltaNet decay term is
      [-exp(A_log) * softplus(a + dt_bias)], so a logarithm is needed. Written
      in the stable form [softplus x = max x 0 + log (1 + exp (-|x|))], the
      logarithm is only ever taken on (1, 2], where the arctanh series
      [log m = 2 * artanh ((m-1)/(m+1))] converges quickly with no range
      reduction. That keeps the new transcendental branch-free.

    - [f32_l2norm]. The DeltaNet kernel normalises query and key by their
      Euclidean length before the recurrence.

    - [f32_rmsnorm_zc] and [f32_rmsnorm_gated]. Qwen3.5 uses two RMSNorm
      variants that Llama does not: one whose weight is stored zero-centred and
      applied as [1 + w], and one that multiplies by [silu] of a separate gate
      vector.

    - [f32_causal_conv1d]. The DeltaNet input passes through a depthwise causal
      convolution before the recurrence.

    - [f32_delta_step] and [f32_delta_scan]. The gated delta rule itself: a
      recurrence over a per-head state matrix, decayed by [exp g], corrected
      towards the value by [beta], and read out by the query.

    - [f32_partial_rope]. Qwen3.5 rotates only a prefix of each head and passes
      the rest through unchanged. For text-only input the interleaved
      multimodal RoPE collapses to ordinary RoPE, because the three positional
      axes carry identical indices, so ordinary rotation on the prefix is the
      whole story.

    Everything here is built from the verified f32 operations. *)

From Stdlib Require Import ZArith.
From Stdlib Require Import List.
From Stdlib Require Import Lia.
From Flocq Require Import IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete.
Require Import Llama.

Import ListNotations.
Open Scope Z_scope.

(** * Small helpers *)

Definition f32_max2 (x y : binary32) : binary32 :=
  if f32_lt x y then y else x.

Definition f32_zeros (n : nat) : list binary32 := List.repeat f32_zero n.

Definition f32_three : binary32 := f32_of_Z 3.
Definition f32_five : binary32 := f32_of_Z 5.
Definition f32_seven : binary32 := f32_of_Z 7.
Definition f32_nine : binary32 := f32_of_Z 9.
Definition f32_eleven : binary32 := f32_of_Z 11.
Definition f32_thirteen : binary32 := f32_of_Z 13.

(** * Logarithm on the unit interval

    For [m] in (1, 2], with [u = (m-1)/(m+1)] in [0, 1/3), the series
    [log m = 2 * (u + u^3/3 + u^5/5 + ...)] converges quickly. Seven terms hold
    the absolute error below 2e-7 across the interval, which is where softplus
    always evaluates it. *)

Definition f32_log_unit (m : binary32) : binary32 :=
  let u := f32_div (f32_minus m f32_one) (f32_plus m f32_one) in
  let u2 := f32_mult u u in
  let u3 := f32_mult u2 u in
  let u5 := f32_mult u3 u2 in
  let u7 := f32_mult u5 u2 in
  let u9 := f32_mult u7 u2 in
  let u11 := f32_mult u9 u2 in
  let u13 := f32_mult u11 u2 in
  let s := f32_plus u
            (f32_plus (f32_div u3 f32_three)
              (f32_plus (f32_div u5 f32_five)
                (f32_plus (f32_div u7 f32_seven)
                  (f32_plus (f32_div u9 f32_nine)
                    (f32_plus (f32_div u11 f32_eleven)
                              (f32_div u13 f32_thirteen)))))) in
  f32_mult f32_two s.

(** Softplus in the form that keeps the logarithm on (1, 2]. *)
Definition f32_softplus (x : binary32) : binary32 :=
  let ax := f32_abs x in
  let e := f32_exp_approx (f32_neg ax) in
  f32_plus (f32_max2 x f32_zero) (f32_log_unit (f32_plus f32_one e)).

(** * Euclidean normalisation

    [l2norm x = x * rsqrt (sum x^2 + eps)], the form the DeltaNet kernel uses
    on the query and the key. *)

Definition f32_l2norm (eps : binary32) (v : list binary32) : list binary32 :=
  let ss := f32_dot v v in
  let inv := f32_div f32_one (f32_sqrt (f32_plus ss eps)) in
  List.map (fun x => f32_mult x inv) v.

Lemma f32_l2norm_length : forall eps v,
  List.length (f32_l2norm eps v) = List.length v.
Proof. intros eps v. unfold f32_l2norm. apply List.length_map. Qed.

(** * The two RMSNorm variants

    [f32_rmsnorm_zc] stores the weight zero-centred and applies [1 + w].
    [f32_rmsnorm_gated] applies the weight directly and then multiplies by
    [silu] of a gate vector. *)

Definition f32_rmsnorm_zc (w : list binary32) (eps : binary32) (x : list binary32)
                          : list binary32 :=
  let n := f32_of_Z (Z.of_nat (List.length x)) in
  let ss := f32_sum (List.map (fun xi => f32_mult xi xi) x) in
  let ms := f32_div ss n in
  let rs := f32_div f32_one (f32_sqrt (f32_plus ms eps)) in
  List.map (fun '(wi, xi) => f32_mult (f32_plus f32_one wi) (f32_mult xi rs))
           (List.combine w x).

Lemma f32_rmsnorm_zc_length : forall w eps x,
  List.length w = List.length x ->
  List.length (f32_rmsnorm_zc w eps x) = List.length x.
Proof.
  intros w eps x H. unfold f32_rmsnorm_zc.
  rewrite List.length_map, List.length_combine, H, Nat.min_id. reflexivity.
Qed.

Definition f32_rmsnorm_gated (w : list binary32) (eps : binary32)
                             (gate x : list binary32) : list binary32 :=
  let n := f32_of_Z (Z.of_nat (List.length x)) in
  let ss := f32_sum (List.map (fun xi => f32_mult xi xi) x) in
  let ms := f32_div ss n in
  let rs := f32_div f32_one (f32_sqrt (f32_plus ms eps)) in
  let normed := List.map (fun '(wi, xi) => f32_mult wi (f32_mult xi rs))
                         (List.combine w x) in
  List.map (fun '(ni, gi) => f32_mult ni (f32_silu gi)) (List.combine normed gate).

Lemma f32_rmsnorm_gated_length : forall w eps gate x,
  List.length w = List.length x -> List.length gate = List.length x ->
  List.length (f32_rmsnorm_gated w eps gate x) = List.length x.
Proof.
  intros w eps gate x Hw Hg. unfold f32_rmsnorm_gated.
  rewrite List.length_map, List.length_combine, List.length_map,
          List.length_combine, Hw, Hg, !Nat.min_id. reflexivity.
Qed.

(** * Depthwise causal convolution

    Channel [c] at step [t] sees the [k] most recent values of channel [c],
    zero-padded at the start of the sequence, and the result passes through
    [silu]. *)

Definition f32_conv_window (chans k : nat) (xs : list (list binary32)) (t : nat)
                           : list (list binary32) :=
  let pre := List.firstn (S t) xs in
  let m := List.length pre in
  if Nat.leb k m then List.skipn (m - k) pre
  else List.repeat (f32_zeros chans) (k - m) ++ pre.

Lemma f32_conv_window_length : forall chans k xs t,
  (S t <= List.length xs)%nat ->
  List.length (f32_conv_window chans k xs t) = k.
Proof.
  intros chans k xs t Ht. unfold f32_conv_window.
  rewrite List.length_firstn.
  destruct (Nat.leb k (Nat.min (S t) (List.length xs))) eqn:E.
  - apply Nat.leb_le in E. rewrite List.length_skipn, List.length_firstn. lia.
  - apply Nat.leb_gt in E.
    rewrite List.length_app, List.repeat_length, List.length_firstn. lia.
Qed.

(** One convolution step: [w] holds a length-[k] kernel per channel. *)
Definition f32_conv_step (w : list (list binary32)) (b : list binary32)
                         (win : list (list binary32)) : list binary32 :=
  List.map (fun '(ci, wc) =>
      let acc := f32_sum (List.map (fun '(wj, row) => f32_mult wj (List.nth ci row f32_zero))
                                   (List.combine wc win)) in
      f32_silu (f32_plus acc (List.nth ci b f32_zero)))
    (List.combine (List.seq 0 (List.length w)) w).

Lemma f32_conv_step_length : forall w b win,
  List.length (f32_conv_step w b win) = List.length w.
Proof.
  intros w b win. unfold f32_conv_step.
  rewrite List.length_map, List.length_combine, List.length_seq, Nat.min_id.
  reflexivity.
Qed.

Definition f32_causal_conv1d (chans k : nat) (w : list (list binary32))
                             (b : list binary32) (xs : list (list binary32))
                             : list (list binary32) :=
  List.map (fun t => f32_conv_step w b (f32_conv_window chans k xs t))
           (List.seq 0 (List.length xs)).

Lemma f32_causal_conv1d_length : forall chans k w b xs,
  List.length (f32_causal_conv1d chans k w b xs) = List.length xs.
Proof.
  intros. unfold f32_causal_conv1d.
  rewrite List.length_map, List.length_seq. reflexivity.
Qed.

(** * The gated delta rule

    The state is a [d_k] by [d_v] matrix. Each step decays it by [exp g],
    reads the memory the current key addresses, corrects towards the value in
    proportion to [beta], and reads the result out with the query. *)

Definition f32_delta_step (beta g : binary32) (q k v : list binary32)
                          (st : list (list binary32))
                          : list (list binary32) * list binary32 :=
  let eg := f32_exp_approx g in
  let st1 := List.map (fun row => List.map (fun x => f32_mult x eg) row) st in
  let kv := f32_mat_vec_mul (f32_mat_transpose st1) k in
  let d := List.map (fun '(vj, kvj) => f32_mult (f32_minus vj kvj) beta)
                    (List.combine v kv) in
  let st2 := List.map (fun '(ki, row) =>
                 List.map (fun '(sij, dj) => f32_plus sij (f32_mult ki dj))
                          (List.combine row d))
               (List.combine k st1) in
  let o := f32_mat_vec_mul (f32_mat_transpose st2) q in
  (st2, o).

Fixpoint f32_delta_scan (betas gs : list binary32)
                        (qs ks vs : list (list binary32))
                        (st : list (list binary32)) : list (list binary32) :=
  match betas, gs, qs, ks, vs with
  | b :: bs, g :: gs', q :: qs', k :: ks', v :: vs' =>
      let (st', o) := f32_delta_step b g q k v st in
      o :: f32_delta_scan bs gs' qs' ks' vs' st'
  | _, _, _, _, _ => []
  end.

Lemma f32_delta_scan_length : forall betas gs qs ks vs st,
  List.length (f32_delta_scan betas gs qs ks vs st)
  = Nat.min (List.length betas)
      (Nat.min (List.length gs)
        (Nat.min (List.length qs) (Nat.min (List.length ks) (List.length vs)))).
Proof.
  induction betas as [|b bs IH]; intros gs qs ks vs st; [reflexivity|].
  destruct gs as [|g gs]; [reflexivity|].
  destruct qs as [|q qs]; [reflexivity|].
  destruct ks as [|k ks]; [reflexivity|].
  destruct vs as [|v vs]; [reflexivity|].
  simpl. destruct (f32_delta_step b g q k v st) as [st' o] eqn:E.
  simpl. f_equal. apply IH.
Qed.

(** The zero state. *)
Definition f32_delta_state0 (dk dv : nat) : list (list binary32) :=
  List.repeat (f32_zeros dv) dk.

(** * Partial rotary embedding

    Only the first [rd] entries of a head are rotated; the rest pass through.
    [cos] and [sin] are supplied with [rd] entries, as the caller computes
    them. *)

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

(** * Layer pieces

    The runner drives the per-head and per-token loops, as it does for the
    Llama path; these are the pieces it composes. *)

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

(** The output gate the full-attention layers apply: multiply elementwise by
    [sigmoid] of a gate vector. *)
Definition f32_gate_sigmoid (gate v : list binary32) : list binary32 :=
  List.map (fun '(vi, gi) => f32_mult vi (f32_sigmoid gi)) (List.combine v gate).

Lemma f32_gate_sigmoid_length : forall gate v,
  List.length gate = List.length v ->
  List.length (f32_gate_sigmoid gate v) = List.length v.
Proof.
  intros gate v H. unfold f32_gate_sigmoid.
  rewrite List.length_map, List.length_combine, H, Nat.min_id. reflexivity.
Qed.

(** The DeltaNet decay term: [g = -exp(A_log) * softplus(a + dt_bias)]. *)
Definition f32_delta_decay (a_log dt_bias a : binary32) : binary32 :=
  f32_neg (f32_mult (f32_exp_approx a_log) (f32_softplus (f32_plus a dt_bias))).

(** Per-head query and key preparation: normalise, then scale the query by
    [1/sqrt d_k]. *)
Definition f32_delta_prep_q (eps : binary32) (dk : nat) (q : list binary32)
                            : list binary32 :=
  let s := f32_div f32_one (f32_sqrt (f32_of_Z (Z.of_nat dk))) in
  List.map (fun x => f32_mult x s) (f32_l2norm eps q).

Lemma f32_delta_prep_q_length : forall eps dk q,
  List.length (f32_delta_prep_q eps dk q) = List.length q.
Proof.
  intros. unfold f32_delta_prep_q.
  rewrite List.length_map. apply f32_l2norm_length.
Qed.

(** * The layers, assembled

    The primitives above are what one Qwen3.5 layer is built from; these are the
    layers themselves, in the order the runner composes them. A DeltaNet layer
    projects the input to a fused query/key/value triple, passes it through the
    depthwise causal convolution, runs the gated delta recurrence per head,
    normalises each head against its own gate, and projects back. A gated
    attention layer normalises the per-head query and key, rotates a prefix of
    each, attends causally, gates the output and projects back. Both sit inside
    the same residual pair with a SwiGLU feed-forward. *)

Definition f32_slice (off len : nat) (r : list binary32) : list binary32 :=
  List.firstn len (List.skipn off r).

Lemma f32_slice_length : forall off len r,
  (off + len <= List.length r)%nat -> List.length (f32_slice off len r) = len.
Proof.
  intros off len r H. unfold f32_slice.
  rewrite List.length_firstn, List.length_skipn. lia.
Qed.

Record qwen_delta_weights := mk_qwen_delta_weights {
  qd_in_qkv : list (list binary32);
  qd_in_z : list (list binary32);
  qd_in_a : list (list binary32);
  qd_in_b : list (list binary32);
  qd_conv_w : list (list binary32);
  qd_a_log : list binary32;
  qd_dt_bias : list binary32;
  qd_norm_w : list binary32;
  qd_out : list (list binary32)
}.

Record qwen_attn_weights := mk_qwen_attn_weights {
  qa_q : list (list binary32);
  qa_k : list (list binary32);
  qa_v : list (list binary32);
  qa_o : list (list binary32);
  qa_q_norm : list binary32;
  qa_k_norm : list binary32
}.

Record qwen_mlp_weights := mk_qwen_mlp_weights {
  qm_gate : list (list binary32);
  qm_up : list (list binary32);
  qm_down : list (list binary32)
}.

(** One DeltaNet head over the whole sequence: prepare the query and key, read
    the value, form the decay and the correction rate, and scan. *)
Definition f32_qwen_delta_head (ldim lhd : nat) (eps : binary32)
    (alog dtb : list binary32) (hh : nat)
    (c : list (list binary32)) (av bv : list (list binary32))
    : list (list binary32) :=
  let qs := List.map (fun r => f32_delta_prep_q eps lhd (f32_slice (hh * lhd) lhd r)) c in
  let ks := List.map (fun r => f32_l2norm eps (f32_slice (ldim + hh * lhd) lhd r)) c in
  let vs := List.map (fun r => f32_slice (2 * ldim + hh * lhd) lhd r) c in
  let betas := List.map (fun r => f32_sigmoid (List.nth hh r f32_zero)) bv in
  let gs := List.map (fun r => f32_delta_decay (List.nth hh alog f32_zero)
                                 (List.nth hh dtb f32_zero)
                                 (List.nth hh r f32_zero)) av in
  f32_delta_scan betas gs qs ks vs (f32_delta_state0 lhd lhd).

(** One head's contribution, after the scan: each token's state read-out is
    normalised against its own slice of the gate stream. *)
Definition f32_qwen_delta_head_out (ldim lhd : nat) (eps : binary32)
    (nw alog dtb : list binary32) (hh : nat)
    (c av bv zs : list (list binary32)) : list (list binary32) :=
  List.map (fun p => let '(zr, o) := p in
              f32_rmsnorm_gated nw eps (f32_slice (hh * lhd) lhd zr) o)
           (List.combine zs (f32_qwen_delta_head ldim lhd eps alog dtb hh c av bv)).

Definition f32_qwen_delta_mix (lnh lhd ck : nat) (eps : binary32)
    (w : qwen_delta_weights) (hn : list (list binary32)) : list (list binary32) :=
  let ldim := (lnh * lhd)%nat in
  let qkv := List.map (fun h => f32_mat_vec_mul (qd_in_qkv w) h) hn in
  let c := f32_causal_conv1d (3 * ldim) ck (qd_conv_w w) [] qkv in
  let zs := List.map (fun h => f32_mat_vec_mul (qd_in_z w) h) hn in
  let av := List.map (fun h => f32_mat_vec_mul (qd_in_a w) h) hn in
  let bv := List.map (fun h => f32_mat_vec_mul (qd_in_b w) h) hn in
  let heads := List.map (fun hh =>
      f32_qwen_delta_head_out ldim lhd eps (qd_norm_w w) (qd_a_log w) (qd_dt_bias w)
                              hh c av bv zs)
      (List.seq 0 lnh) in
  List.map (fun t => f32_mat_vec_mul (qd_out w)
              (List.concat (List.map (fun ho => List.nth t ho []) heads)))
           (List.seq 0 (List.length hn)).

Lemma f32_qwen_delta_mix_rows : forall lnh lhd ck eps w hn,
  List.length (f32_qwen_delta_mix lnh lhd ck eps w hn) = List.length hn.
Proof.
  intros. unfold f32_qwen_delta_mix. cbv zeta.
  rewrite List.length_map, List.length_seq. reflexivity.
Qed.

(** One gated full-attention layer over the whole sequence. The query stream
    carries its gate interleaved with the query itself, which is why the query
    projection is twice as wide as the head dimension. *)
Definition f32_qwen_attn_mix (nh nkv hd rd : nat) (eps : binary32)
    (w : qwen_attn_weights) (cosv sinv : list (list binary32))
    (hn : list (list binary32)) : list (list binary32) :=
  let qg := List.map (fun h => f32_mat_vec_mul (qa_q w) h) hn in
  let kraw := List.map (fun h => f32_mat_vec_mul (qa_k w) h) hn in
  let vraw := List.map (fun h => f32_mat_vec_mul (qa_v w) h) hn in
  let group := Nat.div nh nkv in
  let ks := List.map (fun c =>
      List.map (fun p => let '(pos, r) := p in
                  f32_partial_rope rd (List.nth pos cosv []) (List.nth pos sinv [])
                    (f32_rmsnorm_zc (qa_k_norm w) eps (f32_slice (c * hd) hd r)))
               (List.combine (List.seq 0 (List.length kraw)) kraw))
      (List.seq 0 nkv) in
  let vs := List.map (fun c => List.map (fun r => f32_slice (c * hd) hd r) vraw)
                     (List.seq 0 nkv) in
  let qs := List.map (fun hh =>
      List.map (fun p => let '(pos, r) := p in
                  f32_partial_rope rd (List.nth pos cosv []) (List.nth pos sinv [])
                    (f32_rmsnorm_zc (qa_q_norm w) eps (f32_slice (hh * hd * 2) hd r)))
               (List.combine (List.seq 0 (List.length qg)) qg))
      (List.seq 0 nh) in
  let heads := List.map (fun hh =>
      f32_causal_attention (List.nth hh qs []) (List.nth (Nat.div hh group) ks [])
                           (List.nth (Nat.div hh group) vs []) hd)
      (List.seq 0 nh) in
  let gates := List.map (fun r =>
      List.concat (List.map (fun hh => f32_slice (hh * hd * 2 + hd) hd r)
                            (List.seq 0 nh))) qg in
  List.map (fun p => let '(g, o) := p in f32_mat_vec_mul (qa_o w) (f32_gate_sigmoid g o))
           (List.combine gates (f32_concat_heads heads)).

(** The residual pair both layer kinds sit in. *)
Definition f32_qwen_wrap (eps : binary32) (ln1 ln2 : list binary32)
    (mlp : qwen_mlp_weights)
    (mix : list (list binary32) -> list (list binary32))
    (h : list (list binary32)) : list (list binary32) :=
  let hn := List.map (fun row => f32_rmsnorm_zc ln1 eps row) h in
  let hidden2 := List.map (fun p => let '(a, b) := p in f32_vec_add a b)
                          (List.combine h (mix hn)) in
  let h2 := List.map (fun row => f32_rmsnorm_zc ln2 eps row) hidden2 in
  List.map (fun p => let '(a, b) := p in f32_vec_add a b)
    (List.combine hidden2
       (List.map (fun row => f32_swiglu (qm_gate mlp) (qm_up mlp) (qm_down mlp) row)
                 h2)).

(** The final norm and the tied-embedding logit projection. *)
Definition f32_qwen_final (eps : binary32) (normw : list binary32)
    (h : list (list binary32)) : list (list binary32) :=
  List.map (fun row => f32_rmsnorm_zc normw eps row) h.

Definition f32_qwen_logits (emb : list (list binary32))
    (h : list (list binary32)) : list (list binary32) :=
  List.map (fun hrow => List.map (fun wrow => f32_dot hrow wrow) emb) h.

Lemma f32_qwen_logits_rows : forall emb h,
  List.length (f32_qwen_logits emb h) = List.length h.
Proof. intros. unfold f32_qwen_logits. apply List.length_map. Qed.

Lemma f32_qwen_logits_row_width : forall emb h row,
  In row (f32_qwen_logits emb h) -> List.length row = List.length emb.
Proof.
  intros emb h row Hin. unfold f32_qwen_logits in Hin.
  apply List.in_map_iff in Hin. destruct Hin as [hr [Heq _]]. subst row.
  apply List.length_map.
Qed.

(** * The stack and the forward pass

    Qwen3.5 alternates layer kinds, so the stack is a list of layer forwards
    rather than a list of uniform weight records; the runner picks the kind by
    [i mod 4 = 3]. Composing them left to right is the whole of the forward,
    followed by the final norm and the tied-embedding projection. *)

Fixpoint f32_qwen_stack (fs : list (list (list binary32) -> list (list binary32)))
                        (h : list (list binary32)) : list (list binary32) :=
  match fs with
  | [] => h
  | f :: rest => f32_qwen_stack rest (f h)
  end.

Definition f32_qwen_forward (eps : binary32) (normw : list binary32)
    (fs : list (list (list binary32) -> list (list binary32)))
    (emb : list (list binary32)) (ids : list nat) : list (list binary32) :=
  f32_qwen_final eps normw (f32_qwen_stack fs (f32_embed_tokens emb ids)).

Definition f32_qwen_next_token_logits (eps : binary32) (normw : list binary32)
    (fs : list (list (list binary32) -> list (list binary32)))
    (emb : list (list binary32)) (ids : list nat) : list binary32 :=
  match List.rev (f32_qwen_logits emb (f32_qwen_forward eps normw fs emb ids)) with
  | [] => []
  | last :: _ => last
  end.

Lemma f32_qwen_final_rows : forall eps normw h,
  List.length (f32_qwen_final eps normw h) = List.length h.
Proof. intros. unfold f32_qwen_final. apply List.length_map. Qed.
