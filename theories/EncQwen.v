(** * The Qwen3.5 pass over any arithmetic, and the enclosure of its logits

    The Qwen3.5 layers of Qwen.v written once over the arithmetic of RunErr.v,
    with two operations beyond it, the logarithm and the absolute value, which
    softplus takes. At binary32, with [f32_log_unit] and [f32_abs], the pass is
    [f32_qwen_logits_of] by conversion. At exact real arithmetic, with [ln] and
    [Rabs] and the exponential taken on its argument saturated to [[-88, 88]],
    it is the reference, and in the enclosure arithmetic it encloses every
    logit of that reference. The rotary tables are those the binary32 pass
    forms. *)

From Stdlib Require Import ZArith Reals List.
From Flocq Require Import Core IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete Float_error RunErr Llama Qwen Loadpre Enclose EncExt EncLlama.

Import ListNotations.

(** * Weights over any carrier *)

Record gqdelta (T : Type) := mk_gqdelta {
  gqd_in_qkv : list (list T); gqd_in_z : list (list T);
  gqd_in_a : list (list T); gqd_in_b : list (list T);
  gqd_conv_w : list (list T);
  gqd_a_log : list T; gqd_dt_bias : list T; gqd_norm_w : list T;
  gqd_out : list (list T)
}.

Record gqattn (T : Type) := mk_gqattn {
  gqa_q : list (list T); gqa_k : list (list T); gqa_v : list (list T); gqa_o : list (list T);
  gqa_q_norm : list T; gqa_k_norm : list T
}.

Record gqmlp (T : Type) := mk_gqmlp {
  gqp_gate : list (list T); gqp_up : list (list T); gqp_down : list (list T)
}.

Inductive gqlayer (T : Type) :=
  | GQAttn : list T -> list T -> gqmlp T -> gqattn T -> gqlayer T
  | GQDelta : list T -> list T -> gqmlp T -> gqdelta T -> gqlayer T.

Record gqmodel (T : Type) := mk_gqmodel {
  gqw_emb : list (list T); gqw_norm : list T; gqw_layers : list (gqlayer T)
}.

Arguments mk_gqdelta {T}.
Arguments gqd_in_qkv {T}. Arguments gqd_in_z {T}. Arguments gqd_in_a {T}.
Arguments gqd_in_b {T}. Arguments gqd_conv_w {T}. Arguments gqd_a_log {T}.
Arguments gqd_dt_bias {T}. Arguments gqd_norm_w {T}. Arguments gqd_out {T}.
Arguments mk_gqattn {T}.
Arguments gqa_q {T}. Arguments gqa_k {T}. Arguments gqa_v {T}. Arguments gqa_o {T}.
Arguments gqa_q_norm {T}. Arguments gqa_k_norm {T}.
Arguments mk_gqmlp {T}.
Arguments gqp_gate {T}. Arguments gqp_up {T}. Arguments gqp_down {T}.
Arguments GQAttn {T}. Arguments GQDelta {T}.
Arguments mk_gqmodel {T}.
Arguments gqw_emb {T}. Arguments gqw_norm {T}. Arguments gqw_layers {T}.

Definition to_gqmlp (w : qwen_mlp_weights) : gqmlp binary32 :=
  mk_gqmlp (qm_gate w) (qm_up w) (qm_down w).

Definition to_gqattn (w : qwen_attn_weights) : gqattn binary32 :=
  mk_gqattn (qa_q w) (qa_k w) (qa_v w) (qa_o w) (qa_q_norm w) (qa_k_norm w).

Definition to_gqdelta (w : qwen_delta_weights) : gqdelta binary32 :=
  mk_gqdelta (qd_in_qkv w) (qd_in_z w) (qd_in_a w) (qd_in_b w) (qd_conv_w w)
    (qd_a_log w) (qd_dt_bias w) (qd_norm_w w) (qd_out w).

Definition to_gqlayer (L : qwen_layer_weights) : gqlayer binary32 :=
  match L with
  | QAttn ln1 ln2 mlp aw => GQAttn ln1 ln2 (to_gqmlp mlp) (to_gqattn aw)
  | QDelta ln1 ln2 mlp dw => GQDelta ln1 ln2 (to_gqmlp mlp) (to_gqdelta dw)
  end.

Definition to_gqmodel (m : qwen_model_weights) : gqmodel binary32 :=
  mk_gqmodel (qw_emb m) (qw_norm m) (List.map to_gqlayer (qw_layers m)).

Section Maps.

Context {T1 T2 : Type} (f : T1 -> T2).

Local Notation mv := (List.map f).
Local Notation mm := (List.map (List.map f)).

Definition gqmlp_map (w : gqmlp T1) : gqmlp T2 :=
  mk_gqmlp (mm (gqp_gate w)) (mm (gqp_up w)) (mm (gqp_down w)).

Definition gqattn_map (w : gqattn T1) : gqattn T2 :=
  mk_gqattn (mm (gqa_q w)) (mm (gqa_k w)) (mm (gqa_v w)) (mm (gqa_o w))
    (mv (gqa_q_norm w)) (mv (gqa_k_norm w)).

Definition gqdelta_map (w : gqdelta T1) : gqdelta T2 :=
  mk_gqdelta (mm (gqd_in_qkv w)) (mm (gqd_in_z w)) (mm (gqd_in_a w)) (mm (gqd_in_b w))
    (mm (gqd_conv_w w)) (mv (gqd_a_log w)) (mv (gqd_dt_bias w)) (mv (gqd_norm_w w))
    (mm (gqd_out w)).

Definition gqlayer_map (L : gqlayer T1) : gqlayer T2 :=
  match L with
  | GQAttn ln1 ln2 mlp aw => GQAttn (mv ln1) (mv ln2) (gqmlp_map mlp) (gqattn_map aw)
  | GQDelta ln1 ln2 mlp dw => GQDelta (mv ln1) (mv ln2) (gqmlp_map mlp) (gqdelta_map dw)
  end.

Definition gqmodel_map (m : gqmodel T1) : gqmodel T2 :=
  mk_gqmodel (mm (gqw_emb m)) (mv (gqw_norm m)) (List.map gqlayer_map (gqw_layers m)).

End Maps.

(** * The pass *)

Section GQwen.

Context {T : Type} (A : ops T) (lnT absT : T -> T).

Definition g_softplus (x : T) : T :=
  let ax := absT x in
  let e := op_exp A (op_neg A ax) in
  op_plus A (op_max A x (op_const A f32_zero)) (lnT (op_plus A (op_const A f32_one) e)).

Definition g_l2norm (eps : T) (v : list T) : list T :=
  let ss := op_dot A v v in
  let inv := op_div A (op_const A f32_one) (op_sqrt A (op_plus A ss eps)) in
  List.map (fun x => op_mult A x inv) v.

Definition g_rmsnorm_zc (w : list T) (eps : T) (x : list T) : list T :=
  let n := op_const A (f32_of_Z (Z.of_nat (List.length x))) in
  let ss := g_sum A (List.map (fun xi => op_mult A xi xi) x) in
  let ms := op_div A ss n in
  let rs := op_div A (op_const A f32_one) (op_sqrt A (op_plus A ms eps)) in
  List.map (fun '(wi, xi) => op_mult A (op_plus A (op_const A f32_one) wi) (op_mult A xi rs))
           (List.combine w x).

Definition g_rmsnorm_gated (w : list T) (eps : T) (gate x : list T) : list T :=
  let n := op_const A (f32_of_Z (Z.of_nat (List.length x))) in
  let ss := g_sum A (List.map (fun xi => op_mult A xi xi) x) in
  let ms := op_div A ss n in
  let rs := op_div A (op_const A f32_one) (op_sqrt A (op_plus A ms eps)) in
  let normed := List.map (fun '(wi, xi) => op_mult A wi (op_mult A xi rs))
                         (List.combine w x) in
  List.map (fun '(ni, gi) => op_mult A ni (g_silu A gi)) (List.combine normed gate).

Definition g_zeros (n : nat) : list T := List.repeat (op_const A f32_zero) n.

Definition g_conv_window (chans k : nat) (xs : list (list T)) (t : nat) : list (list T) :=
  let pre := List.firstn (S t) xs in
  let m := List.length pre in
  if Nat.leb k m then List.skipn (m - k) pre
  else List.repeat (g_zeros chans) (k - m) ++ pre.

Definition g_conv_step (w : list (list T)) (b : list T) (win : list (list T)) : list T :=
  List.map (fun '(ci, wc) =>
      let acc := g_sum A (List.map (fun '(wj, row) =>
                             op_mult A wj (List.nth ci row (op_const A f32_zero)))
                                   (List.combine wc win)) in
      g_silu A (op_plus A acc (List.nth ci b (op_const A f32_zero))))
    (List.combine (List.seq 0 (List.length w)) w).

Definition g_causal_conv1d (chans k : nat) (w : list (list T)) (b : list T)
    (xs : list (list T)) : list (list T) :=
  List.map (fun t => g_conv_step w b (g_conv_window chans k xs t))
           (List.seq 0 (List.length xs)).

Definition g_delta_step (beta g : T) (q k v : list T) (st : list (list T))
    : list (list T) * list T :=
  let eg := op_exp A g in
  let st1 := List.map (fun row => List.map (fun x => op_mult A x eg) row) st in
  let kv := g_mat_vec_mul A (g_mat_transpose A st1) k in
  let d := List.map (fun '(vj, kvj) => op_mult A (g_minus A vj kvj) beta)
                    (List.combine v kv) in
  let st2 := List.map (fun '(ki, row) =>
                 List.map (fun '(sij, dj) => op_plus A sij (op_mult A ki dj))
                          (List.combine row d))
               (List.combine k st1) in
  let o := g_mat_vec_mul A (g_mat_transpose A st2) q in
  (st2, o).

Fixpoint g_delta_scan (betas gs : list T) (qs ks vs : list (list T))
    (st : list (list T)) : list (list T) :=
  match betas, gs, qs, ks, vs with
  | b :: bs, g :: gs', q :: qs', k :: ks', v :: vs' =>
      let (st', o) := g_delta_step b g q k v st in
      o :: g_delta_scan bs gs' qs' ks' vs' st'
  | _, _, _, _, _ => []
  end.

Definition g_delta_state0 (dk dv : nat) : list (list T) := List.repeat (g_zeros dv) dk.

Definition g_gate_sigmoid (gate v : list T) : list T :=
  List.map (fun '(vi, gi) => op_mult A vi (g_sigmoid A gi)) (List.combine v gate).

Definition g_delta_decay (a_log dt_bias a : T) : T :=
  op_neg A (op_mult A (op_exp A a_log) (g_softplus (op_plus A a dt_bias))).

Definition g_delta_prep_q (eps : T) (dk : nat) (q : list T) : list T :=
  let s := op_div A (op_const A f32_one) (op_sqrt A (op_const A (f32_of_Z (Z.of_nat dk)))) in
  List.map (fun x => op_mult A x s) (g_l2norm eps q).

Definition g_qwen_delta_head (ldim lhd : nat) (eps : T) (alog dtb : list T) (hh : nat)
    (c av bv : list (list T)) : list (list T) :=
  let qs := List.map (fun r => g_delta_prep_q eps lhd (g_slice (hh * lhd) lhd r)) c in
  let ks := List.map (fun r => g_l2norm eps (g_slice (ldim + hh * lhd) lhd r)) c in
  let vs := List.map (fun r => g_slice (2 * ldim + hh * lhd) lhd r) c in
  let betas := List.map (fun r => g_sigmoid A (List.nth hh r (op_const A f32_zero))) bv in
  let gs := List.map (fun r => g_delta_decay (List.nth hh alog (op_const A f32_zero))
                                 (List.nth hh dtb (op_const A f32_zero))
                                 (List.nth hh r (op_const A f32_zero))) av in
  g_delta_scan betas gs qs ks vs (g_delta_state0 lhd lhd).

Definition g_qwen_delta_head_out (ldim lhd : nat) (eps : T) (nw alog dtb : list T) (hh : nat)
    (c av bv zs : list (list T)) : list (list T) :=
  List.map (fun p => let '(zr, o) := p in
              g_rmsnorm_gated nw eps (g_slice (hh * lhd) lhd zr) o)
           (List.combine zs (g_qwen_delta_head ldim lhd eps alog dtb hh c av bv)).

Definition g_qwen_delta_mix (lnh lhd ck : nat) (eps : T) (w : gqdelta T)
    (hn : list (list T)) : list (list T) :=
  let ldim := (lnh * lhd)%nat in
  let qkv := List.map (fun h => g_mat_vec_mul A (gqd_in_qkv w) h) hn in
  let c := g_causal_conv1d (3 * ldim) ck (gqd_conv_w w) [] qkv in
  let zs := List.map (fun h => g_mat_vec_mul A (gqd_in_z w) h) hn in
  let av := List.map (fun h => g_mat_vec_mul A (gqd_in_a w) h) hn in
  let bv := List.map (fun h => g_mat_vec_mul A (gqd_in_b w) h) hn in
  let heads := List.map (fun hh =>
      g_qwen_delta_head_out ldim lhd eps (gqd_norm_w w) (gqd_a_log w) (gqd_dt_bias w)
                            hh c av bv zs)
      (List.seq 0 lnh) in
  List.map (fun t => g_mat_vec_mul A (gqd_out w)
              (List.concat (List.map (fun ho => List.nth t ho []) heads)))
           (List.seq 0 (List.length hn)).

Definition g_qwen_attn_mix (nh nkv hd rd : nat) (eps : T) (w : gqattn T)
    (cosv sinv : list (list T)) (hn : list (list T)) : list (list T) :=
  let qg := List.map (fun h => g_mat_vec_mul A (gqa_q w) h) hn in
  let kraw := List.map (fun h => g_mat_vec_mul A (gqa_k w) h) hn in
  let vraw := List.map (fun h => g_mat_vec_mul A (gqa_v w) h) hn in
  let group := Nat.div nh nkv in
  let ks := List.map (fun c =>
      List.map (fun p => let '(pos, r) := p in
                  g_partial_rope A rd (List.nth pos cosv []) (List.nth pos sinv [])
                    (g_rmsnorm_zc (gqa_k_norm w) eps (g_slice (c * hd) hd r)))
               (List.combine (List.seq 0 (List.length kraw)) kraw))
      (List.seq 0 nkv) in
  let vs := List.map (fun c => List.map (fun r => g_slice (c * hd) hd r) vraw)
                     (List.seq 0 nkv) in
  let qs := List.map (fun hh =>
      List.map (fun p => let '(pos, r) := p in
                  g_partial_rope A rd (List.nth pos cosv []) (List.nth pos sinv [])
                    (g_rmsnorm_zc (gqa_q_norm w) eps (g_slice (hh * hd * 2) hd r)))
               (List.combine (List.seq 0 (List.length qg)) qg))
      (List.seq 0 nh) in
  let heads := List.map (fun hh =>
      g_causal_attention A (List.nth hh qs []) (List.nth (Nat.div hh group) ks [])
                           (List.nth (Nat.div hh group) vs []) hd)
      (List.seq 0 nh) in
  let gates := List.map (fun r =>
      List.concat (List.map (fun hh => g_slice (hh * hd * 2 + hd) hd r)
                            (List.seq 0 nh))) qg in
  List.map (fun p => let '(g, o) := p in g_mat_vec_mul A (gqa_o w) (g_gate_sigmoid g o))
           (List.combine gates (g_concat_heads heads)).

Definition g_qwen_wrap (eps : T) (ln1 ln2 : list T) (mlp : gqmlp T)
    (mix : list (list T) -> list (list T)) (h : list (list T)) : list (list T) :=
  let hn := List.map (fun row => g_rmsnorm_zc ln1 eps row) h in
  let hidden2 := List.map (fun p => let '(a, b) := p in g_vec_add A a b)
                          (List.combine h (mix hn)) in
  let h2 := List.map (fun row => g_rmsnorm_zc ln2 eps row) hidden2 in
  List.map (fun p => let '(a, b) := p in g_vec_add A a b)
    (List.combine hidden2
       (List.map (fun row => g_swiglu A (gqp_gate mlp) (gqp_up mlp) (gqp_down mlp) row) h2)).

Definition g_qwen_layer (nh nkv hd rd lnh lhd ck : nat) (eps : T)
    (cosv sinv : list (list T)) (L : gqlayer T) (h : list (list T)) : list (list T) :=
  match L with
  | GQAttn ln1 ln2 mlp aw =>
      g_qwen_wrap eps ln1 ln2 mlp (g_qwen_attn_mix nh nkv hd rd eps aw cosv sinv) h
  | GQDelta ln1 ln2 mlp dw =>
      g_qwen_wrap eps ln1 ln2 mlp (g_qwen_delta_mix lnh lhd ck eps dw) h
  end.

Fixpoint g_qwen_stack (nh nkv hd rd lnh lhd ck : nat) (eps : T)
    (cosv sinv : list (list T)) (Ls : list (gqlayer T)) (h : list (list T)) : list (list T) :=
  match Ls with
  | [] => h
  | L :: rest => g_qwen_stack nh nkv hd rd lnh lhd ck eps cosv sinv rest
                   (g_qwen_layer nh nkv hd rd lnh lhd ck eps cosv sinv L h)
  end.

Definition g_qwen_logits (nh nkv hd rd lnh lhd ck : nat) (eps : T) (m : gqmodel T)
    (cosv sinv : list (list T)) (ids : list nat) : list (list T) :=
  let h := g_qwen_stack nh nkv hd rd lnh lhd ck eps cosv sinv (gqw_layers m)
             (g_embed_tokens (gqw_emb m) ids) in
  let hn := List.map (fun row => g_rmsnorm_zc (gqw_norm m) eps row) h in
  List.map (fun hrow => List.map (fun wrow => op_dot A hrow wrow) (gqw_emb m)) hn.

End GQwen.

(** * At binary32 the pass is the model's *)

Ltac unfold_qwen :=
  cbv [g_qwen_layer g_qwen_wrap g_qwen_attn_mix g_qwen_delta_mix g_qwen_delta_head_out
       g_qwen_delta_head g_delta_prep_q g_delta_decay g_gate_sigmoid g_delta_state0
       g_delta_scan g_delta_step g_causal_conv1d g_conv_step g_conv_window g_zeros
       g_rmsnorm_gated g_rmsnorm_zc g_l2norm g_softplus
       g_partial_rope g_slice g_swiglu g_silu g_vec_mult g_sigmoid g_sum g_vec_add
       g_mat_vec_mul g_concat_heads g_causal_attention g_attend g_attn_scale g_softmax
       g_max_vec g_minus g_mat_transpose
       op_const op_plus op_neg op_mult op_div op_sqrt op_max op_exp op_dot f32_ops
       to_gqlayer to_gqmlp to_gqattn to_gqdelta
       gqp_gate gqp_up gqp_down gqa_q gqa_k gqa_v gqa_o gqa_q_norm gqa_k_norm
       gqd_in_qkv gqd_in_z gqd_in_a gqd_in_b gqd_conv_w gqd_a_log gqd_dt_bias gqd_norm_w gqd_out
       f32_qwen_wrap f32_qwen_attn_mix f32_qwen_delta_mix f32_qwen_delta_head_out
       f32_qwen_delta_head f32_delta_prep_q f32_delta_decay f32_gate_sigmoid f32_delta_state0
       f32_delta_scan f32_delta_step f32_causal_conv1d f32_conv_step f32_conv_window f32_zeros
       f32_rmsnorm_gated f32_rmsnorm_zc f32_l2norm f32_softplus
       f32_partial_rope f32_slice f32_swiglu f32_silu_vec f32_silu f32_vec_mult f32_sigmoid
       f32_sum f32_vec_add f32_mat_vec_mul f32_concat_heads f32_causal_attention f32_attend
       f32_attn_scale f32_softmax f32_max_vec f32_exp_vec f32_minus f32_mat_transpose f32_max2
       qm_gate qm_up qm_down qa_q qa_k qa_v qa_o qa_q_norm qa_k_norm
       qd_in_qkv qd_in_z qd_in_a qd_in_b qd_conv_w qd_a_log qd_dt_bias qd_norm_w qd_out].

Definition f32_qwen_layer_fn (nh nkv hd rd lnh lhd ck : nat) (eps : binary32)
    (cosv sinv : list (list binary32)) (L : qwen_layer_weights)
    : list (list binary32) -> list (list binary32) :=
  match L with
  | QAttn ln1 ln2 mlp aw =>
      f32_qwen_wrap eps ln1 ln2 mlp (f32_qwen_attn_mix nh nkv hd rd eps aw cosv sinv)
  | QDelta ln1 ln2 mlp dw =>
      f32_qwen_wrap eps ln1 ln2 mlp (f32_qwen_delta_mix lnh lhd ck eps dw)
  end.

Lemma g_qwen_layer_f32 : forall nh nkv hd rd lnh lhd ck eps cosv sinv L h,
  g_qwen_layer f32_ops f32_log_unit f32_abs nh nkv hd rd lnh lhd ck eps cosv sinv
    (to_gqlayer L) h
  = f32_qwen_layer_fn nh nkv hd rd lnh lhd ck eps cosv sinv L h.
Proof.
  intros nh nkv hd rd lnh lhd ck eps cosv sinv L h.
  destruct L as [ln1 ln2 [g u dn] [q k v o qn kn] | ln1 ln2 [g u dn] [iq iz ia ib cw al dt nw ou]];
    unfold f32_qwen_layer_fn; unfold_qwen; reflexivity.
Qed.

Lemma g_qwen_stack_f32 : forall nh nkv hd rd lnh lhd ck eps cosv sinv Ls h,
  g_qwen_stack f32_ops f32_log_unit f32_abs nh nkv hd rd lnh lhd ck eps cosv sinv
    (List.map to_gqlayer Ls) h
  = f32_qwen_stack (List.map (f32_qwen_layer_fn nh nkv hd rd lnh lhd ck eps cosv sinv) Ls) h.
Proof.
  intros nh nkv hd rd lnh lhd ck eps cosv sinv Ls.
  induction Ls as [|L Ls IH]; intros h; [reflexivity|].
  cbn [List.map g_qwen_stack f32_qwen_stack]. rewrite g_qwen_layer_f32. apply IH.
Qed.

Theorem g_qwen_logits_f32 : forall nh nkv hd rd lnh lhd ck eps m ids,
  g_qwen_logits f32_ops f32_log_unit f32_abs nh nkv hd rd lnh lhd ck eps (to_gqmodel m)
    (f32_rope_cos (qw_invf m) (List.length ids)) (f32_rope_sin (qw_invf m) (List.length ids)) ids
  = f32_qwen_logits_of nh nkv hd rd lnh lhd ck eps m ids.
Proof.
  intros nh nkv hd rd lnh lhd ck eps m ids.
  unfold g_qwen_logits, f32_qwen_logits_of, f32_qwen_logits, f32_qwen_forward, f32_qwen_final.
  cbv zeta. cbn [to_gqmodel gqw_layers gqw_emb gqw_norm].
  rewrite g_qwen_stack_f32.
  replace (List.map (fun L => match L with
                              | QAttn ln1 ln2 mlp aw =>
                                  f32_qwen_wrap eps ln1 ln2 mlp
                                    (f32_qwen_attn_mix nh nkv hd rd eps aw
                                       (f32_rope_cos (qw_invf m) (List.length ids))
                                       (f32_rope_sin (qw_invf m) (List.length ids)))
                              | QDelta ln1 ln2 mlp dw =>
                                  f32_qwen_wrap eps ln1 ln2 mlp (f32_qwen_delta_mix lnh lhd ck eps dw)
                              end) (qw_layers m))
    with (List.map (f32_qwen_layer_fn nh nkv hd rd lnh lhd ck eps
                      (f32_rope_cos (qw_invf m) (List.length ids))
                      (f32_rope_sin (qw_invf m) (List.length ids))) (qw_layers m))
    by reflexivity.
  cbv [g_rmsnorm_zc g_sum g_embed_tokens f32_rmsnorm_zc f32_sum f32_embed_tokens
       f32_lookup_embedding op_const op_plus op_mult op_div op_sqrt op_dot f32_ops].
  reflexivity.
Qed.

(** * The pass preserves any relation the arithmetic preserves *)

Section RQwen.

Context {T1 T2 : Type} (A1 : ops T1) (A2 : ops T2) (rel : T1 -> T2 -> Prop).
Context (HR : ops_rel A1 A2 rel).
Context (ln1 : T1 -> T1) (ln2 : T2 -> T2) (abs1 : T1 -> T1) (abs2 : T2 -> T2).
Context (Hln : forall x1 x2, rel x1 x2 -> rel (ln1 x1) (ln2 x2)).
Context (Habs : forall x1 x2, rel x1 x2 -> rel (abs1 x1) (abs2 x2)).

Local Notation V := (Forall2 rel).
Local Notation M := (Forall2 (Forall2 rel)).

Definition gqmlp_rel (w1 : gqmlp T1) (w2 : gqmlp T2) : Prop :=
  M (gqp_gate w1) (gqp_gate w2) /\ M (gqp_up w1) (gqp_up w2) /\ M (gqp_down w1) (gqp_down w2).

Definition gqattn_rel (w1 : gqattn T1) (w2 : gqattn T2) : Prop :=
  M (gqa_q w1) (gqa_q w2) /\ M (gqa_k w1) (gqa_k w2) /\ M (gqa_v w1) (gqa_v w2)
  /\ M (gqa_o w1) (gqa_o w2) /\ V (gqa_q_norm w1) (gqa_q_norm w2)
  /\ V (gqa_k_norm w1) (gqa_k_norm w2).

Definition gqdelta_rel (w1 : gqdelta T1) (w2 : gqdelta T2) : Prop :=
  M (gqd_in_qkv w1) (gqd_in_qkv w2) /\ M (gqd_in_z w1) (gqd_in_z w2)
  /\ M (gqd_in_a w1) (gqd_in_a w2) /\ M (gqd_in_b w1) (gqd_in_b w2)
  /\ M (gqd_conv_w w1) (gqd_conv_w w2) /\ V (gqd_a_log w1) (gqd_a_log w2)
  /\ V (gqd_dt_bias w1) (gqd_dt_bias w2) /\ V (gqd_norm_w w1) (gqd_norm_w w2)
  /\ M (gqd_out w1) (gqd_out w2).

Definition gqlayer_rel (L1 : gqlayer T1) (L2 : gqlayer T2) : Prop :=
  match L1, L2 with
  | GQAttn a1 b1 m1 w1, GQAttn a2 b2 m2 w2 =>
      V a1 a2 /\ V b1 b2 /\ gqmlp_rel m1 m2 /\ gqattn_rel w1 w2
  | GQDelta a1 b1 m1 w1, GQDelta a2 b2 m2 w2 =>
      V a1 a2 /\ V b1 b2 /\ gqmlp_rel m1 m2 /\ gqdelta_rel w1 w2
  | _, _ => False
  end.

Definition gqmodel_rel (m1 : gqmodel T1) (m2 : gqmodel T2) : Prop :=
  M (gqw_emb m1) (gqw_emb m2) /\ V (gqw_norm m1) (gqw_norm m2)
  /\ Forall2 gqlayer_rel (gqw_layers m1) (gqw_layers m2).

Lemma r_softplus : forall x1 x2, rel x1 x2 -> rel (g_softplus A1 ln1 abs1 x1) (g_softplus A2 ln2 abs2 x2).
Proof.
  intros x1 x2 H. unfold g_softplus. cbv zeta.
  apply (rel_plus HR).
  - apply (rel_max HR); [exact H | apply (rel_const HR)].
  - apply Hln. apply (rel_plus HR); [apply (rel_const HR)|].
    apply (rel_exp HR), (rel_neg HR), Habs, H.
Qed.

Lemma r_l2norm : forall e1 e2 v1 v2, rel e1 e2 -> V v1 v2 ->
  V (g_l2norm A1 e1 v1) (g_l2norm A2 e2 v2).
Proof.
  intros e1 e2 v1 v2 He Hv. unfold g_l2norm. cbv zeta.
  apply Forall2_map2 with (P := rel); [exact Hv|]. intros a b Hab.
  apply (rel_mult HR); [exact Hab|].
  apply (rel_div HR); [apply (rel_const HR)|]. apply (rel_sqrt HR).
  apply (rel_plus HR); [apply (rel_dot HR); assumption | exact He].
Qed.

Lemma r_inv_rms : forall e1 e2 x1 x2, rel e1 e2 -> V x1 x2 ->
  rel (op_div A1 (op_const A1 f32_one)
         (op_sqrt A1 (op_plus A1 (op_div A1 (g_sum A1 (List.map (fun xi => op_mult A1 xi xi) x1))
                                          (op_const A1 (f32_of_Z (Z.of_nat (List.length x1))))) e1)))
      (op_div A2 (op_const A2 f32_one)
         (op_sqrt A2 (op_plus A2 (op_div A2 (g_sum A2 (List.map (fun xi => op_mult A2 xi xi) x2))
                                          (op_const A2 (f32_of_Z (Z.of_nat (List.length x2))))) e2))).
Proof.
  intros e1 e2 x1 x2 He Hx. rewrite (Forall2_length _ _ _ _ _ Hx).
  apply (rel_div HR); [apply (rel_const HR)|]. apply (rel_sqrt HR).
  apply (rel_plus HR); [|exact He].
  apply (rel_div HR); [|apply (rel_const HR)].
  apply (r_sum A1 A2 rel HR). apply Forall2_map2 with (P := rel); [exact Hx|].
  intros u v Huv. apply (rel_mult HR); assumption.
Qed.

Lemma r_rmsnorm_zc : forall w1 w2 e1 e2 x1 x2, V w1 w2 -> rel e1 e2 -> V x1 x2 ->
  V (g_rmsnorm_zc A1 w1 e1 x1) (g_rmsnorm_zc A2 w2 e2 x2).
Proof.
  intros w1 w2 e1 e2 x1 x2 Hw He Hx. unfold g_rmsnorm_zc. cbv zeta.
  pose proof (r_inv_rms _ _ _ _ He Hx) as Hr.
  apply Forall2_map_combine with (P := rel) (Q := rel); [exact Hw | exact Hx |].
  intros a d b e Had Hbe. cbv beta iota.
  apply (rel_mult HR); [apply (rel_plus HR); [apply (rel_const HR) | exact Had]|].
  apply (rel_mult HR); assumption.
Qed.

Lemma r_rmsnorm_gated : forall w1 w2 e1 e2 g1 g2 x1 x2,
  V w1 w2 -> rel e1 e2 -> V g1 g2 -> V x1 x2 ->
  V (g_rmsnorm_gated A1 w1 e1 g1 x1) (g_rmsnorm_gated A2 w2 e2 g2 x2).
Proof.
  intros w1 w2 e1 e2 g1 g2 x1 x2 Hw He Hg Hx. unfold g_rmsnorm_gated. cbv zeta.
  pose proof (r_inv_rms _ _ _ _ He Hx) as Hr.
  apply Forall2_map_combine with (P := rel) (Q := rel); [| exact Hg |].
  - apply Forall2_map_combine with (P := rel) (Q := rel); [exact Hw | exact Hx |].
    intros a d b e Had Hbe. cbv beta iota.
    apply (rel_mult HR); [exact Had|]. apply (rel_mult HR); assumption.
  - intros a d b e Had Hbe. cbv beta iota.
    apply (rel_mult HR); [exact Had | apply (r_silu A1 A2 rel HR), Hbe].
Qed.

Lemma r_zeros : forall n, V (g_zeros A1 n) (g_zeros A2 n).
Proof. intros n. unfold g_zeros. apply Forall2_repeat, (rel_const HR). Qed.

Lemma r_conv_window : forall chans k xs1 xs2 t, M xs1 xs2 ->
  M (g_conv_window A1 chans k xs1 t) (g_conv_window A2 chans k xs2 t).
Proof.
  intros chans k xs1 xs2 t Hx. unfold g_conv_window. cbv zeta.
  pose proof (Forall2_firstn _ _ V (S t) _ _ Hx) as Hp.
  rewrite (Forall2_length _ _ _ _ _ Hp).
  destruct (Nat.leb k (List.length (List.firstn (S t) xs2))).
  - apply Forall2_skipn, Hp.
  - apply Forall2_app; [|exact Hp]. apply Forall2_repeat, r_zeros.
Qed.

Lemma r_conv_step : forall w1 w2 b1 b2 win1 win2, M w1 w2 -> V b1 b2 -> M win1 win2 ->
  V (g_conv_step A1 w1 b1 win1) (g_conv_step A2 w2 b2 win2).
Proof.
  intros w1 w2 b1 b2 win1 win2 Hw Hb Hwin. unfold g_conv_step.
  pose proof (Forall2_combine_seq _ _ V w1 w2 0 Hw) as Hc.
  apply Forall2_map2 with (P := fun p q => fst p = fst q /\ V (snd p) (snd q)); [exact Hc|].
  intros [i1 a1] [i2 a2] [Hi Ha]. cbn [fst snd] in Hi, Ha. subst i2. cbv beta iota zeta.
  apply (r_silu A1 A2 rel HR). apply (rel_plus HR).
  - apply (r_sum A1 A2 rel HR).
    apply Forall2_map_combine with (P := rel) (Q := V); [exact Ha | exact Hwin |].
    intros a d b e Had Hbe. cbv beta iota.
    apply (rel_mult HR); [exact Had | apply Forall2_nth; [exact Hbe | apply (rel_const HR)]].
  - apply Forall2_nth; [exact Hb | apply (rel_const HR)].
Qed.

Lemma r_causal_conv1d : forall chans k w1 w2 b1 b2 xs1 xs2, M w1 w2 -> V b1 b2 -> M xs1 xs2 ->
  M (g_causal_conv1d A1 chans k w1 b1 xs1) (g_causal_conv1d A2 chans k w2 b2 xs2).
Proof.
  intros chans k w1 w2 b1 b2 xs1 xs2 Hw Hb Hx. unfold g_causal_conv1d.
  rewrite (Forall2_length _ _ _ _ _ Hx).
  apply Forall2_map_same. intros t.
  apply r_conv_step; [exact Hw | exact Hb | apply r_conv_window, Hx].
Qed.

Lemma r_delta_step : forall b1 b2 g1 g2 q1 q2 k1 k2 v1 v2 st1 st2,
  rel b1 b2 -> rel g1 g2 -> V q1 q2 -> V k1 k2 -> V v1 v2 -> M st1 st2 ->
  M (fst (g_delta_step A1 b1 g1 q1 k1 v1 st1)) (fst (g_delta_step A2 b2 g2 q2 k2 v2 st2))
  /\ V (snd (g_delta_step A1 b1 g1 q1 k1 v1 st1)) (snd (g_delta_step A2 b2 g2 q2 k2 v2 st2)).
Proof.
  intros b1 b2 g1 g2 q1 q2 k1 k2 v1 v2 st1 st2 Hb Hg Hq Hk Hv Hst.
  unfold g_delta_step. cbv zeta. cbn [fst snd].
  assert (H1 : M (List.map (fun row => List.map (fun x => op_mult A1 x (op_exp A1 g1)) row) st1)
                 (List.map (fun row => List.map (fun x => op_mult A2 x (op_exp A2 g2)) row) st2)).
  { apply Forall2_map2 with (P := V); [exact Hst|]. intros a b Hab.
    apply Forall2_map2 with (P := rel); [exact Hab|]. intros c d Hcd.
    apply (rel_mult HR); [exact Hcd | apply (rel_exp HR), Hg]. }
  assert (Hd : V (List.map (fun '(vj, kvj) => op_mult A1 (g_minus A1 vj kvj) b1)
                   (List.combine v1 (g_mat_vec_mul A1 (g_mat_transpose A1
                      (List.map (fun row => List.map (fun x => op_mult A1 x (op_exp A1 g1)) row) st1)) k1)))
                 (List.map (fun '(vj, kvj) => op_mult A2 (g_minus A2 vj kvj) b2)
                   (List.combine v2 (g_mat_vec_mul A2 (g_mat_transpose A2
                      (List.map (fun row => List.map (fun x => op_mult A2 x (op_exp A2 g2)) row) st2)) k2)))).
  { apply Forall2_map_combine with (P := rel) (Q := rel); [exact Hv | |].
    - apply (r_mat_vec_mul A1 A2 rel HR); [apply (r_mat_transpose A1 A2 rel HR), H1 | exact Hk].
    - intros a d b e Had Hbe. cbv beta iota.
      apply (rel_mult HR); [apply (r_minus A1 A2 rel HR); assumption | exact Hb]. }
  assert (H2 : M (List.map (fun '(ki, row) =>
                    List.map (fun '(sij, dj) => op_plus A1 sij (op_mult A1 ki dj))
                      (List.combine row (List.map (fun '(vj, kvj) => op_mult A1 (g_minus A1 vj kvj) b1)
                         (List.combine v1 (g_mat_vec_mul A1 (g_mat_transpose A1
                            (List.map (fun row => List.map (fun x => op_mult A1 x (op_exp A1 g1)) row) st1)) k1)))))
                    (List.combine k1 (List.map (fun row => List.map (fun x => op_mult A1 x (op_exp A1 g1)) row) st1)))
                 (List.map (fun '(ki, row) =>
                    List.map (fun '(sij, dj) => op_plus A2 sij (op_mult A2 ki dj))
                      (List.combine row (List.map (fun '(vj, kvj) => op_mult A2 (g_minus A2 vj kvj) b2)
                         (List.combine v2 (g_mat_vec_mul A2 (g_mat_transpose A2
                            (List.map (fun row => List.map (fun x => op_mult A2 x (op_exp A2 g2)) row) st2)) k2)))))
                    (List.combine k2 (List.map (fun row => List.map (fun x => op_mult A2 x (op_exp A2 g2)) row) st2)))).
  { apply Forall2_map_combine with (P := rel) (Q := V); [exact Hk | exact H1 |].
    intros a d b e Had Hbe. cbv beta iota.
    apply Forall2_map_combine with (P := rel) (Q := rel); [exact Hbe | exact Hd |].
    intros a' d' b' e' Had' Hbe'. cbv beta iota.
    apply (rel_plus HR); [exact Had' | apply (rel_mult HR); assumption]. }
  split; [exact H2|].
  apply (r_mat_vec_mul A1 A2 rel HR); [apply (r_mat_transpose A1 A2 rel HR), H2 | exact Hq].
Qed.

Lemma r_delta_scan : forall bs1 bs2 gs1 gs2 qs1 qs2 ks1 ks2 vs1 vs2 st1 st2,
  V bs1 bs2 -> V gs1 gs2 -> M qs1 qs2 -> M ks1 ks2 -> M vs1 vs2 -> M st1 st2 ->
  M (g_delta_scan A1 bs1 gs1 qs1 ks1 vs1 st1) (g_delta_scan A2 bs2 gs2 qs2 ks2 vs2 st2).
Proof.
  intros bs1 bs2 gs1 gs2 qs1 qs2 ks1 ks2 vs1 vs2 st1 st2 Hb.
  revert gs1 gs2 qs1 qs2 ks1 ks2 vs1 vs2 st1 st2.
  induction Hb as [|b1 b2 bs1 bs2 Hb0 Hb IH];
    intros gs1 gs2 qs1 qs2 ks1 ks2 vs1 vs2 st1 st2 Hg Hq Hk Hv Hst; [constructor|].
  destruct Hg as [|g1 g2 gs1 gs2 Hg0 Hg]; [constructor|].
  destruct Hq as [|q1 q2 qs1 qs2 Hq0 Hq]; [constructor|].
  destruct Hk as [|k1 k2 ks1 ks2 Hk0 Hk]; [constructor|].
  destruct Hv as [|v1 v2 vs1 vs2 Hv0 Hv]; [constructor|].
  cbn [g_delta_scan].
  destruct (r_delta_step b1 b2 g1 g2 q1 q2 k1 k2 v1 v2 st1 st2 Hb0 Hg0 Hq0 Hk0 Hv0 Hst) as [Hs Ho].
  destruct (g_delta_step A1 b1 g1 q1 k1 v1 st1) as [s1 o1].
  destruct (g_delta_step A2 b2 g2 q2 k2 v2 st2) as [s2 o2].
  cbn [fst snd] in Hs, Ho.
  constructor; [exact Ho|]. apply IH; assumption.
Qed.

Lemma r_delta_state0 : forall dk dv, M (g_delta_state0 A1 dk dv) (g_delta_state0 A2 dk dv).
Proof. intros dk dv. unfold g_delta_state0. apply Forall2_repeat, r_zeros. Qed.

Lemma r_gate_sigmoid : forall g1 g2 v1 v2, V g1 g2 -> V v1 v2 ->
  V (g_gate_sigmoid A1 g1 v1) (g_gate_sigmoid A2 g2 v2).
Proof.
  intros g1 g2 v1 v2 Hg Hv. unfold g_gate_sigmoid.
  apply Forall2_map_combine with (P := rel) (Q := rel); [exact Hv | exact Hg |].
  intros a d b e Had Hbe. cbv beta iota.
  apply (rel_mult HR); [exact Had | apply (r_sigmoid A1 A2 rel HR), Hbe].
Qed.

Lemma r_delta_decay : forall al1 al2 dt1 dt2 a1 a2, rel al1 al2 -> rel dt1 dt2 -> rel a1 a2 ->
  rel (g_delta_decay A1 ln1 abs1 al1 dt1 a1) (g_delta_decay A2 ln2 abs2 al2 dt2 a2).
Proof.
  intros al1 al2 dt1 dt2 a1 a2 Hal Hdt Ha. unfold g_delta_decay.
  apply (rel_neg HR). apply (rel_mult HR); [apply (rel_exp HR), Hal|].
  apply r_softplus. apply (rel_plus HR); assumption.
Qed.

Lemma r_delta_prep_q : forall e1 e2 dk q1 q2, rel e1 e2 -> V q1 q2 ->
  V (g_delta_prep_q A1 e1 dk q1) (g_delta_prep_q A2 e2 dk q2).
Proof.
  intros e1 e2 dk q1 q2 He Hq. unfold g_delta_prep_q. cbv zeta.
  apply Forall2_map2 with (P := rel); [apply r_l2norm; assumption|].
  intros a b Hab. apply (rel_mult HR); [exact Hab|].
  apply (rel_div HR); [apply (rel_const HR) | apply (rel_sqrt HR), (rel_const HR)].
Qed.

Lemma r_qwen_delta_head : forall ldim lhd e1 e2 al1 al2 dt1 dt2 hh c1 c2 av1 av2 bv1 bv2,
  rel e1 e2 -> V al1 al2 -> V dt1 dt2 -> M c1 c2 -> M av1 av2 -> M bv1 bv2 ->
  M (g_qwen_delta_head A1 ln1 abs1 ldim lhd e1 al1 dt1 hh c1 av1 bv1)
    (g_qwen_delta_head A2 ln2 abs2 ldim lhd e2 al2 dt2 hh c2 av2 bv2).
Proof.
  intros ldim lhd e1 e2 al1 al2 dt1 dt2 hh c1 c2 av1 av2 bv1 bv2 He Hal Hdt Hc Hav Hbv.
  unfold g_qwen_delta_head. cbv zeta.
  apply r_delta_scan.
  - apply Forall2_map2 with (P := V); [exact Hbv|]. intros a b Hab.
    apply (r_sigmoid A1 A2 rel HR). apply Forall2_nth; [exact Hab | apply (rel_const HR)].
  - apply Forall2_map2 with (P := V); [exact Hav|]. intros a b Hab.
    apply r_delta_decay; apply Forall2_nth; try assumption; apply (rel_const HR).
  - apply Forall2_map2 with (P := V); [exact Hc|]. intros a b Hab.
    apply r_delta_prep_q; [exact He | apply (r_slice rel), Hab].
  - apply Forall2_map2 with (P := V); [exact Hc|]. intros a b Hab.
    apply r_l2norm; [exact He | apply (r_slice rel), Hab].
  - apply Forall2_map2 with (P := V); [exact Hc|]. intros a b Hab. apply (r_slice rel), Hab.
  - apply r_delta_state0.
Qed.

Lemma r_qwen_delta_mix : forall lnh lhd ck e1 e2 w1 w2 hn1 hn2,
  rel e1 e2 -> gqdelta_rel w1 w2 -> M hn1 hn2 ->
  M (g_qwen_delta_mix A1 ln1 abs1 lnh lhd ck e1 w1 hn1)
    (g_qwen_delta_mix A2 ln2 abs2 lnh lhd ck e2 w2 hn2).
Proof.
  intros lnh lhd ck e1 e2 w1 w2 hn1 hn2 He Hw Hh.
  destruct Hw as (Hqkv & Hz & Ha & Hb & Hcw & Hal & Hdt & Hnw & Ho).
  unfold g_qwen_delta_mix. cbv zeta.
  pose proof (r_proj_rows A1 A2 rel HR _ _ _ _ Hqkv Hh) as Pqkv.
  pose proof (r_proj_rows A1 A2 rel HR _ _ _ _ Hz Hh) as Pz.
  pose proof (r_proj_rows A1 A2 rel HR _ _ _ _ Ha Hh) as Pa.
  pose proof (r_proj_rows A1 A2 rel HR _ _ _ _ Hb Hh) as Pb.
  assert (Pc : M (g_causal_conv1d A1 (3 * (lnh * lhd)) ck (gqd_conv_w w1) []
                    (List.map (fun h => g_mat_vec_mul A1 (gqd_in_qkv w1) h) hn1))
                 (g_causal_conv1d A2 (3 * (lnh * lhd)) ck (gqd_conv_w w2) []
                    (List.map (fun h => g_mat_vec_mul A2 (gqd_in_qkv w2) h) hn2)))
    by (apply r_causal_conv1d; [exact Hcw | constructor | exact Pqkv]).
  assert (Ph : Forall2 M
     (List.map (fun hh => g_qwen_delta_head_out A1 ln1 abs1 (lnh * lhd) lhd e1 (gqd_norm_w w1)
                            (gqd_a_log w1) (gqd_dt_bias w1) hh
                            (g_causal_conv1d A1 (3 * (lnh * lhd)) ck (gqd_conv_w w1) []
                               (List.map (fun h => g_mat_vec_mul A1 (gqd_in_qkv w1) h) hn1))
                            (List.map (fun h => g_mat_vec_mul A1 (gqd_in_a w1) h) hn1)
                            (List.map (fun h => g_mat_vec_mul A1 (gqd_in_b w1) h) hn1)
                            (List.map (fun h => g_mat_vec_mul A1 (gqd_in_z w1) h) hn1))
               (List.seq 0 lnh))
     (List.map (fun hh => g_qwen_delta_head_out A2 ln2 abs2 (lnh * lhd) lhd e2 (gqd_norm_w w2)
                            (gqd_a_log w2) (gqd_dt_bias w2) hh
                            (g_causal_conv1d A2 (3 * (lnh * lhd)) ck (gqd_conv_w w2) []
                               (List.map (fun h => g_mat_vec_mul A2 (gqd_in_qkv w2) h) hn2))
                            (List.map (fun h => g_mat_vec_mul A2 (gqd_in_a w2) h) hn2)
                            (List.map (fun h => g_mat_vec_mul A2 (gqd_in_b w2) h) hn2)
                            (List.map (fun h => g_mat_vec_mul A2 (gqd_in_z w2) h) hn2))
               (List.seq 0 lnh))).
  { apply Forall2_map_same. intros hh. unfold g_qwen_delta_head_out.
    apply Forall2_map_combine with (P := V) (Q := V); [exact Pz | |].
    - apply r_qwen_delta_head; assumption.
    - intros a d b e Had Hbe. cbv beta iota.
      apply r_rmsnorm_gated; [exact Hnw | exact He | apply (r_slice rel), Had | exact Hbe]. }
  rewrite (Forall2_length _ _ _ _ _ Hh).
  apply Forall2_map_same. intros t.
  apply (r_mat_vec_mul A1 A2 rel HR); [exact Ho|].
  apply Forall2_concat. apply Forall2_map2 with (P := M); [exact Ph|].
  intros a b Hab. apply Forall2_nth; [exact Hab | constructor].
Qed.

Lemma r_qwen_attn_mix : forall nh nkv hd rd e1 e2 w1 w2 cs1 cs2 sn1 sn2 hn1 hn2,
  rel e1 e2 -> gqattn_rel w1 w2 -> M cs1 cs2 -> M sn1 sn2 -> M hn1 hn2 ->
  M (g_qwen_attn_mix A1 nh nkv hd rd e1 w1 cs1 sn1 hn1)
    (g_qwen_attn_mix A2 nh nkv hd rd e2 w2 cs2 sn2 hn2).
Proof.
  intros nh nkv hd rd e1 e2 w1 w2 cs1 cs2 sn1 sn2 hn1 hn2 He Hw Hc Hs Hh.
  destruct Hw as (Hq & Hk & Hv & Ho & Hqn & Hkn).
  unfold g_qwen_attn_mix. cbv zeta.
  pose proof (r_proj_rows A1 A2 rel HR _ _ _ _ Hq Hh) as Pq.
  pose proof (r_proj_rows A1 A2 rel HR _ _ _ _ Hk Hh) as Pk.
  pose proof (r_proj_rows A1 A2 rel HR _ _ _ _ Hv Hh) as Pv.
  assert (Rot : forall nw1 nw2 raw1 raw2 (sel : nat -> nat), V nw1 nw2 -> M raw1 raw2 ->
            forall c, M
              (List.map (fun p => let '(pos, r) := p in
                  g_partial_rope A1 rd (List.nth pos cs1 []) (List.nth pos sn1 [])
                    (g_rmsnorm_zc A1 nw1 e1 (g_slice (sel c) hd r)))
                 (List.combine (List.seq 0 (List.length raw1)) raw1))
              (List.map (fun p => let '(pos, r) := p in
                  g_partial_rope A2 rd (List.nth pos cs2 []) (List.nth pos sn2 [])
                    (g_rmsnorm_zc A2 nw2 e2 (g_slice (sel c) hd r)))
                 (List.combine (List.seq 0 (List.length raw2)) raw2))).
  { intros nw1 nw2 raw1 raw2 sel Hnw Hraw c.
    pose proof (Forall2_combine_seq _ _ V raw1 raw2 0 Hraw) as Hcr.
    apply Forall2_map2 with (P := fun p q => fst p = fst q /\ V (snd p) (snd q)); [exact Hcr|].
    intros [i1 a1] [i2 a2] [Hi Ha]. cbn [fst snd] in Hi, Ha. subst i2. cbv beta iota.
    apply (r_partial_rope A1 A2 rel HR).
    - apply Forall2_nth; [exact Hc | constructor].
    - apply Forall2_nth; [exact Hs | constructor].
    - apply r_rmsnorm_zc; [exact Hnw | exact He | apply (r_slice rel), Ha]. }
  assert (Pks : Forall2 M
     (List.map (fun c => List.map (fun p => let '(pos, r) := p in
         g_partial_rope A1 rd (List.nth pos cs1 []) (List.nth pos sn1 [])
           (g_rmsnorm_zc A1 (gqa_k_norm w1) e1 (g_slice (c * hd) hd r)))
        (List.combine (List.seq 0 (List.length (List.map (fun h => g_mat_vec_mul A1 (gqa_k w1) h) hn1)))
           (List.map (fun h => g_mat_vec_mul A1 (gqa_k w1) h) hn1))) (List.seq 0 nkv))
     (List.map (fun c => List.map (fun p => let '(pos, r) := p in
         g_partial_rope A2 rd (List.nth pos cs2 []) (List.nth pos sn2 [])
           (g_rmsnorm_zc A2 (gqa_k_norm w2) e2 (g_slice (c * hd) hd r)))
        (List.combine (List.seq 0 (List.length (List.map (fun h => g_mat_vec_mul A2 (gqa_k w2) h) hn2)))
           (List.map (fun h => g_mat_vec_mul A2 (gqa_k w2) h) hn2))) (List.seq 0 nkv))).
  { apply Forall2_map_same. intros c. exact (Rot _ _ _ _ (fun c => (c * hd)%nat) Hkn Pk c). }
  assert (Pqs : Forall2 M
     (List.map (fun hh => List.map (fun p => let '(pos, r) := p in
         g_partial_rope A1 rd (List.nth pos cs1 []) (List.nth pos sn1 [])
           (g_rmsnorm_zc A1 (gqa_q_norm w1) e1 (g_slice (hh * hd * 2) hd r)))
        (List.combine (List.seq 0 (List.length (List.map (fun h => g_mat_vec_mul A1 (gqa_q w1) h) hn1)))
           (List.map (fun h => g_mat_vec_mul A1 (gqa_q w1) h) hn1))) (List.seq 0 nh))
     (List.map (fun hh => List.map (fun p => let '(pos, r) := p in
         g_partial_rope A2 rd (List.nth pos cs2 []) (List.nth pos sn2 [])
           (g_rmsnorm_zc A2 (gqa_q_norm w2) e2 (g_slice (hh * hd * 2) hd r)))
        (List.combine (List.seq 0 (List.length (List.map (fun h => g_mat_vec_mul A2 (gqa_q w2) h) hn2)))
           (List.map (fun h => g_mat_vec_mul A2 (gqa_q w2) h) hn2))) (List.seq 0 nh))).
  { apply Forall2_map_same. intros hh. exact (Rot _ _ _ _ (fun hh => (hh * hd * 2)%nat) Hqn Pq hh). }
  assert (Pvs : Forall2 M
     (List.map (fun c => List.map (fun r => g_slice (c * hd) hd r)
        (List.map (fun h => g_mat_vec_mul A1 (gqa_v w1) h) hn1)) (List.seq 0 nkv))
     (List.map (fun c => List.map (fun r => g_slice (c * hd) hd r)
        (List.map (fun h => g_mat_vec_mul A2 (gqa_v w2) h) hn2)) (List.seq 0 nkv))).
  { apply Forall2_map_same. intros c.
    apply Forall2_map2 with (P := V); [exact Pv|]. intros a b Hab. apply (r_slice rel), Hab. }
  apply Forall2_map_combine with (P := V) (Q := V).
  - apply Forall2_map2 with (P := V); [exact Pq|]. intros a b Hab.
    apply Forall2_concat. apply Forall2_map_same. intros hh. apply (r_slice rel), Hab.
  - apply (r_concat_heads rel). apply Forall2_map_same. intros hh.
    apply (r_causal_attention A1 A2 rel HR);
      apply Forall2_nth; first [exact Pqs | exact Pks | exact Pvs | constructor].
  - intros a d b e Had Hbe. cbv beta iota.
    apply (r_mat_vec_mul A1 A2 rel HR); [exact Ho | apply r_gate_sigmoid; assumption].
Qed.

Lemma r_qwen_wrap : forall e1 e2 l11 l12 l21 l22 m1 m2 mix1 mix2 h1 h2,
  rel e1 e2 -> V l11 l12 -> V l21 l22 -> gqmlp_rel m1 m2 ->
  (forall x1 x2, M x1 x2 -> M (mix1 x1) (mix2 x2)) -> M h1 h2 ->
  M (g_qwen_wrap A1 e1 l11 l21 m1 mix1 h1) (g_qwen_wrap A2 e2 l12 l22 m2 mix2 h2).
Proof.
  intros e1 e2 l11 l12 l21 l22 m1 m2 mix1 mix2 h1 h2 He H1 H2 (Hg & Hu & Hd) Hmix Hh.
  unfold g_qwen_wrap. cbv zeta.
  assert (Hn : forall w1 w2 x1 x2, V w1 w2 -> M x1 x2 ->
            M (List.map (fun row => g_rmsnorm_zc A1 w1 e1 row) x1)
              (List.map (fun row => g_rmsnorm_zc A2 w2 e2 row) x2)).
  { intros w1 w2 x1 x2 Hw Hx. apply Forall2_map2 with (P := V); [exact Hx|].
    intros a b Hab. apply r_rmsnorm_zc; assumption. }
  assert (H2' : M (List.map (fun p => let '(a, b) := p in g_vec_add A1 a b)
                    (List.combine h1 (mix1 (List.map (fun row => g_rmsnorm_zc A1 l11 e1 row) h1))))
                  (List.map (fun p => let '(a, b) := p in g_vec_add A2 a b)
                    (List.combine h2 (mix2 (List.map (fun row => g_rmsnorm_zc A2 l12 e2 row) h2))))).
  { apply (r_add_rows A1 A2 rel HR); [exact Hh|]. apply Hmix, Hn; assumption. }
  apply (r_add_rows A1 A2 rel HR); [exact H2'|].
  apply Forall2_map2 with (P := V); [apply Hn; assumption|].
  intros a b Hab. apply (r_swiglu A1 A2 rel HR); assumption.
Qed.

Lemma r_qwen_layer : forall nh nkv hd rd lnh lhd ck e1 e2 cs1 cs2 sn1 sn2 L1 L2 h1 h2,
  rel e1 e2 -> M cs1 cs2 -> M sn1 sn2 -> gqlayer_rel L1 L2 -> M h1 h2 ->
  M (g_qwen_layer A1 ln1 abs1 nh nkv hd rd lnh lhd ck e1 cs1 sn1 L1 h1)
    (g_qwen_layer A2 ln2 abs2 nh nkv hd rd lnh lhd ck e2 cs2 sn2 L2 h2).
Proof.
  intros nh nkv hd rd lnh lhd ck e1 e2 cs1 cs2 sn1 sn2 L1 L2 h1 h2 He Hc Hs HL Hh.
  destruct L1 as [a1 b1 m1 w1 | a1 b1 m1 w1]; destruct L2 as [a2 b2 m2 w2 | a2 b2 m2 w2];
    cbn [gqlayer_rel] in HL; try contradiction;
    destruct HL as (Ha & Hb & Hm & Hw); cbn [g_qwen_layer];
    (apply r_qwen_wrap; [exact He | exact Ha | exact Hb | exact Hm | | exact Hh]);
    intros x1 x2 Hx.
  - apply r_qwen_attn_mix; assumption.
  - apply r_qwen_delta_mix; assumption.
Qed.

Lemma r_qwen_stack : forall nh nkv hd rd lnh lhd ck e1 e2 cs1 cs2 sn1 sn2 Ls1 Ls2 h1 h2,
  rel e1 e2 -> M cs1 cs2 -> M sn1 sn2 -> Forall2 gqlayer_rel Ls1 Ls2 -> M h1 h2 ->
  M (g_qwen_stack A1 ln1 abs1 nh nkv hd rd lnh lhd ck e1 cs1 sn1 Ls1 h1)
    (g_qwen_stack A2 ln2 abs2 nh nkv hd rd lnh lhd ck e2 cs2 sn2 Ls2 h2).
Proof.
  intros nh nkv hd rd lnh lhd ck e1 e2 cs1 cs2 sn1 sn2 Ls1 Ls2 h1 h2 He Hc Hs HLs.
  revert h1 h2.
  induction HLs as [|L1 L2 Ls1 Ls2 HL HLs IH]; intros h1 h2 Hh;
    cbn [g_qwen_stack]; [exact Hh|].
  apply IH. apply r_qwen_layer; assumption.
Qed.

Lemma r_qwen_logits : forall nh nkv hd rd lnh lhd ck e1 e2 m1 m2 cs1 cs2 sn1 sn2 ids,
  rel e1 e2 -> gqmodel_rel m1 m2 -> M cs1 cs2 -> M sn1 sn2 ->
  M (g_qwen_logits A1 ln1 abs1 nh nkv hd rd lnh lhd ck e1 m1 cs1 sn1 ids)
    (g_qwen_logits A2 ln2 abs2 nh nkv hd rd lnh lhd ck e2 m2 cs2 sn2 ids).
Proof.
  intros nh nkv hd rd lnh lhd ck e1 e2 m1 m2 cs1 cs2 sn1 sn2 ids He (Hemb & Hnorm & HLs) Hc Hs.
  unfold g_qwen_logits. cbv zeta.
  apply Forall2_map2 with (P := V).
  - apply Forall2_map2 with (P := V).
    + apply r_qwen_stack; try assumption.
      unfold g_embed_tokens. apply Forall2_map_same. intros t.
      apply Forall2_nth; [exact Hemb | constructor].
    + intros a b Hab. apply r_rmsnorm_zc; assumption.
  - intros a b Hab. apply Forall2_map2 with (P := V); [exact Hemb|].
    intros c d Hcd. apply (rel_dot HR); assumption.
Qed.

End RQwen.

Lemma gqmodel_rel_map2 : forall (T0 T1 T2 : Type) (rel : T1 -> T2 -> Prop)
                                (f : T0 -> T1) (g : T0 -> T2) m,
  (forall x, rel (f x) (g x)) -> gqmodel_rel rel (gqmodel_map f m) (gqmodel_map g m).
Proof.
  intros T0 T1 T2 rel f g m H. unfold gqmodel_rel, gqmodel_map.
  cbn [gqw_emb gqw_norm gqw_layers].
  assert (HV : forall l, Forall2 rel (List.map f l) (List.map g l))
    by (intros; apply Forall2_map_same, H).
  assert (HM : forall l, Forall2 (Forall2 rel) (List.map (List.map f) l) (List.map (List.map g) l))
    by (intros; apply Forall2_map_same; intros; apply HV).
  repeat split; try apply HV; try apply HM.
  apply Forall2_map_same. intros [a b mlp w | a b mlp w]; cbn [gqlayer_map gqlayer_rel];
    unfold gqmlp_rel, gqattn_rel, gqdelta_rel, gqmlp_map, gqattn_map, gqdelta_map;
    cbn [gqp_gate gqp_up gqp_down gqa_q gqa_k gqa_v gqa_o gqa_q_norm gqa_k_norm
         gqd_in_qkv gqd_in_z gqd_in_a gqd_in_b gqd_conv_w gqd_a_log gqd_dt_bias gqd_norm_w gqd_out];
    repeat split; try apply HV; try apply HM.
Qed.

(** * The enclosure of the checkpoint's logits

    Run on the checkpoint's weights and the rotary tables the binary32 pass
    forms, the pass in the enclosure arithmetic returns for every logit an
    interval that contains the logit of the exact real reference: the same
    network on the same weights, tables and tokens, in real arithmetic, with
    every exponential the true exponential of its argument saturated to
    [[-88, 88]] and every logarithm the true logarithm. At binary32 the same
    pass is [f32_qwen_logits_of] ([g_qwen_logits_f32]). *)
Theorem qwen_logits_enclosed : forall FB prec nh nkv hd rd lnh lhd ck eps m ids, (149 <= FB)%Z ->
  let cosv := f32_rope_cos (qw_invf m) (List.length ids) in
  let sinv := f32_rope_sin (qw_invf m) (List.length ids) in
  Forall2 (Forall2 (enc_in FB))
    (g_qwen_logits (enc_ops FB prec) (enc_ln FB prec) enc_abs nh nkv hd rd lnh lhd ck
       (enc_const eps) (gqmodel_map enc_const (to_gqmodel m))
       (List.map (List.map enc_const) cosv) (List.map (List.map enc_const) sinv) ids)
    (g_qwen_logits real_ops ln Rabs nh nkv hd rd lnh lhd ck
       (B2R eps) (gqmodel_map B2R (to_gqmodel m))
       (List.map (List.map B2R) cosv) (List.map (List.map B2R) sinv) ids).
Proof.
  intros FB prec nh nkv hd rd lnh lhd ck eps m ids HFB cosv sinv.
  assert (HT : forall t, Forall2 (Forall2 (enc_in FB))
                 (List.map (List.map enc_const) t) (List.map (List.map B2R) t)).
  { intros t. apply Forall2_map_same. intros r. apply Forall2_map_same. intros x.
    apply enc_const_in. }
  apply (r_qwen_logits (enc_ops FB prec) real_ops (enc_in FB) (enc_ops_rel FB HFB prec)
           (enc_ln FB prec) ln enc_abs Rabs (enc_ln_in FB prec) (enc_abs_in FB)).
  - apply enc_const_in.
  - apply gqmodel_rel_map2. intros x. apply enc_const_in.
  - apply HT.
  - apply HT.
Qed.
