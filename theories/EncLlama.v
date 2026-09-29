(** * The Llama pass over any arithmetic, and the enclosure of its logits

    The Llama layer of Llama.v written once over the arithmetic of RunErr.v.
    At binary32 it is [f32_llama_logits_of] by conversion. At exact real
    arithmetic, with the exponential taken on its argument saturated to
    [[-88, 88]], it is the reference, and in the enclosure arithmetic of
    Enclose.v it encloses every logit of that reference. The rotary tables are
    the ones the binary32 pass forms, and both passes take them as inputs, as
    they take the weights. *)

From Stdlib Require Import ZArith Reals List.
From Flocq Require Import Core IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete Float_error RunErr Llama Qwen Loadpre Enclose.

Import ListNotations.

(** * Weights over any carrier *)

Record gllayer (T : Type) := mk_gllayer {
  gl_ln1 : list T; gl_ln2 : list T;
  gl_q : list (list T); gl_k : list (list T); gl_v : list (list T); gl_o : list (list T);
  gl_gate : list (list T); gl_up : list (list T); gl_down : list (list T)
}.

Record glmodel (T : Type) := mk_glmodel {
  glm_emb : list (list T); glm_norm : list T; glm_layers : list (gllayer T)
}.

Arguments mk_gllayer {T}.
Arguments gl_ln1 {T}. Arguments gl_ln2 {T}.
Arguments gl_q {T}. Arguments gl_k {T}. Arguments gl_v {T}. Arguments gl_o {T}.
Arguments gl_gate {T}. Arguments gl_up {T}. Arguments gl_down {T}.
Arguments mk_glmodel {T}.
Arguments glm_emb {T}. Arguments glm_norm {T}. Arguments glm_layers {T}.

Definition to_gllayer (L : llama_layer_weights) : gllayer binary32 :=
  mk_gllayer (ll_ln1 L) (ll_ln2 L)
    (la_q (ll_attn L)) (la_k (ll_attn L)) (la_v (ll_attn L)) (la_o (ll_attn L))
    (lm_gate (ll_mlp L)) (lm_up (ll_mlp L)) (lm_down (ll_mlp L)).

Definition to_glmodel (m : llama_model_weights) : glmodel binary32 :=
  mk_glmodel (lw_emb m) (lw_norm m) (List.map to_gllayer (lw_layers m)).

Definition gllayer_map {T1 T2 : Type} (f : T1 -> T2) (L : gllayer T1) : gllayer T2 :=
  mk_gllayer (List.map f (gl_ln1 L)) (List.map f (gl_ln2 L))
    (List.map (List.map f) (gl_q L)) (List.map (List.map f) (gl_k L))
    (List.map (List.map f) (gl_v L)) (List.map (List.map f) (gl_o L))
    (List.map (List.map f) (gl_gate L)) (List.map (List.map f) (gl_up L))
    (List.map (List.map f) (gl_down L)).

Definition glmodel_map {T1 T2 : Type} (f : T1 -> T2) (m : glmodel T1) : glmodel T2 :=
  mk_glmodel (List.map (List.map f) (glm_emb m)) (List.map f (glm_norm m))
    (List.map (gllayer_map f) (glm_layers m)).

(** * The pass *)

Section GLlama.

Context {T : Type} (A : ops T).

Definition g_silu (x : T) : T := op_mult A x (g_sigmoid A x).

Definition g_vec_mult (xs ys : list T) : list T :=
  List.map (fun '(x, y) => op_mult A x y) (List.combine xs ys).

Definition g_rmsnorm (w : list T) (eps : T) (x : list T) : list T :=
  let n := op_const A (f32_of_Z (Z.of_nat (List.length x))) in
  let ss := g_sum A (List.map (fun xi => op_mult A xi xi) x) in
  let ms := op_div A ss n in
  let rs := op_div A (op_const A f32_one) (op_sqrt A (op_plus A ms eps)) in
  List.map (fun '(wi, xi) => op_mult A wi (op_mult A xi rs)) (List.combine w x).

Definition g_slice (off len : nat) (r : list T) : list T :=
  List.firstn len (List.skipn off r).

Definition g_partial_rope (rd : nat) (cosv sinv : list T) (x : list T) : list T :=
  let half := Nat.div rd 2 in
  let rot := List.firstn rd x in
  let pass := List.skipn rd x in
  let lo := List.firstn half rot in
  let hi := List.skipn half rot in
  let rotated := List.map (fun z => op_neg A z) hi ++ lo in
  let out := List.map (fun '(xi, (ri, (ci, si))) =>
                 op_plus A (op_mult A xi ci) (op_mult A ri si))
               (List.combine rot (List.combine rotated (List.combine cosv sinv))) in
  out ++ pass.

Definition g_swiglu (wg wu wd : list (list T)) (x : list T) : list T :=
  let g := List.map g_silu (g_mat_vec_mul A wg x) in
  let u := g_mat_vec_mul A wu x in
  g_mat_vec_mul A wd (g_vec_mult g u).

Definition g_rope_rows (hd : nat) (cosv sinv : list (list T))
    (c : nat) (m : list (list T)) : list (list T) :=
  List.map (fun p => let '(pos, r) := p in
              g_partial_rope hd (List.nth pos cosv []) (List.nth pos sinv [])
                (g_slice (c * hd) hd r))
           (List.combine (List.seq 0 (List.length m)) m).

Definition g_llama_heads (nh nkv hd : nat) (L : gllayer T)
    (cosv sinv : list (list T)) (hn : list (list T)) : list (list (list T)) :=
  let q := List.map (fun h => g_mat_vec_mul A (gl_q L) h) hn in
  let k := List.map (fun h => g_mat_vec_mul A (gl_k L) h) hn in
  let v := List.map (fun h => g_mat_vec_mul A (gl_v L) h) hn in
  let group := Nat.div nh nkv in
  let qs := List.map (fun hh => g_rope_rows hd cosv sinv hh q) (List.seq 0 nh) in
  let ks := List.map (fun c => g_rope_rows hd cosv sinv c k) (List.seq 0 nkv) in
  let vs := List.map (fun c => List.map (fun r => g_slice (c * hd) hd r) v)
                     (List.seq 0 nkv) in
  List.map (fun hh =>
      g_causal_attention A (List.nth hh qs [])
        (List.nth (Nat.div hh group) ks []) (List.nth (Nat.div hh group) vs []) hd)
    (List.seq 0 nh).

Definition g_llama_attn (nh nkv hd : nat) (L : gllayer T)
    (cosv sinv : list (list T)) (hn : list (list T)) : list (list T) :=
  List.map (fun r => g_mat_vec_mul A (gl_o L) r)
           (g_concat_heads (g_llama_heads nh nkv hd L cosv sinv hn)).

Definition g_llama_layer (nh nkv hd : nat) (eps : T) (L : gllayer T)
    (cosv sinv : list (list T)) (h : list (list T)) : list (list T) :=
  let hn := List.map (fun row => g_rmsnorm (gl_ln1 L) eps row) h in
  let hidden2 := List.map (fun p => let '(a, b) := p in g_vec_add A a b)
                          (List.combine h (g_llama_attn nh nkv hd L cosv sinv hn)) in
  let h2 := List.map (fun row => g_rmsnorm (gl_ln2 L) eps row) hidden2 in
  List.map (fun p => let '(a, b) := p in g_vec_add A a b)
    (List.combine hidden2
       (List.map (fun row => g_swiglu (gl_gate L) (gl_up L) (gl_down L) row) h2)).

Fixpoint g_llama_stack (nh nkv hd : nat) (eps : T) (Ls : list (gllayer T))
    (cosv sinv : list (list T)) (h : list (list T)) : list (list T) :=
  match Ls with
  | [] => h
  | L :: rest => g_llama_stack nh nkv hd eps rest cosv sinv
                   (g_llama_layer nh nkv hd eps L cosv sinv h)
  end.

Definition g_llama_logits (nh nkv hd : nat) (eps : T) (m : glmodel T)
    (cosv sinv : list (list T)) (ids : list nat) : list (list T) :=
  let h := g_llama_stack nh nkv hd eps (glm_layers m) cosv sinv
             (g_embed_tokens (glm_emb m) ids) in
  let hn := List.map (fun row => g_rmsnorm (glm_norm m) eps row) h in
  List.map (fun hrow => List.map (fun wrow => op_dot A hrow wrow) (glm_emb m)) hn.

End GLlama.

(** * At binary32 the pass is the model's *)

Lemma g_llama_layer_f32 : forall nh nkv hd eps L cosv sinv h,
  g_llama_layer f32_ops nh nkv hd eps (to_gllayer L) cosv sinv h
  = f32_llama_layer nh nkv hd eps (ll_ln1 L) (ll_ln2 L) (ll_attn L) (ll_mlp L) cosv sinv h.
Proof.
  intros nh nkv hd eps [ln1 ln2 [q k v o] [g u dn]] cosv sinv h.
  cbv [g_llama_layer g_llama_attn g_llama_heads g_rope_rows g_partial_rope g_slice g_rmsnorm
       g_swiglu g_silu g_vec_mult g_sigmoid g_sum g_vec_add g_mat_vec_mul g_concat_heads
       g_causal_attention g_attend g_attn_scale g_softmax g_max_vec g_minus g_mat_transpose
       op_const op_plus op_neg op_mult op_div op_sqrt op_max op_exp op_dot f32_ops
       to_gllayer gl_ln1 gl_ln2 gl_q gl_k gl_v gl_o gl_gate gl_up gl_down
       f32_llama_layer f32_llama_wrap f32_llama_attn f32_llama_heads f32_llama_rope_rows
       f32_partial_rope f32_slice f32_rmsnorm f32_swiglu f32_silu_vec f32_silu f32_vec_mult
       f32_sigmoid f32_sum f32_vec_add f32_mat_vec_mul f32_concat_heads f32_causal_attention
       f32_attend f32_attn_scale f32_softmax f32_max_vec f32_exp_vec f32_minus
       f32_mat_transpose f32_max2
       ll_ln1 ll_ln2 ll_attn ll_mlp la_q la_k la_v la_o lm_gate lm_up lm_down].
  reflexivity.
Qed.

Lemma g_llama_stack_f32 : forall nh nkv hd eps Ls cosv sinv h,
  g_llama_stack f32_ops nh nkv hd eps (List.map to_gllayer Ls) cosv sinv h
  = f32_llama_stack
      (List.map (fun L => f32_llama_layer nh nkv hd eps (ll_ln1 L) (ll_ln2 L)
                            (ll_attn L) (ll_mlp L) cosv sinv) Ls) h.
Proof.
  intros nh nkv hd eps Ls cosv sinv. induction Ls as [|L Ls IH]; intros h; [reflexivity|].
  cbn [List.map g_llama_stack f32_llama_stack]. rewrite g_llama_layer_f32. apply IH.
Qed.

Theorem g_llama_logits_f32 : forall nh nkv hd eps m ids,
  g_llama_logits f32_ops nh nkv hd eps (to_glmodel m)
    (f32_rope_cos (lw_invf m) (List.length ids)) (f32_rope_sin (lw_invf m) (List.length ids)) ids
  = f32_llama_logits_of nh nkv hd eps m ids.
Proof.
  intros nh nkv hd eps m ids.
  unfold g_llama_logits, f32_llama_logits_of, f32_llama_logits, f32_llama_forward.
  cbv zeta. cbn [to_glmodel glm_layers glm_emb glm_norm].
  rewrite g_llama_stack_f32.
  cbv [g_rmsnorm g_sum g_embed_tokens f32_rmsnorm f32_sum f32_embed_tokens f32_lookup_embedding
       op_const op_plus op_mult op_div op_sqrt op_dot f32_ops].
  reflexivity.
Qed.

(** * The pass preserves any relation the arithmetic preserves *)

Section RLlama.

Context {T1 T2 : Type} (A1 : ops T1) (A2 : ops T2) (rel : T1 -> T2 -> Prop).
Context (HR : ops_rel A1 A2 rel).

Local Notation V := (Forall2 rel).
Local Notation M := (Forall2 (Forall2 rel)).

Definition gllayer_rel (L1 : gllayer T1) (L2 : gllayer T2) : Prop :=
  V (gl_ln1 L1) (gl_ln1 L2) /\ V (gl_ln2 L1) (gl_ln2 L2)
  /\ M (gl_q L1) (gl_q L2) /\ M (gl_k L1) (gl_k L2)
  /\ M (gl_v L1) (gl_v L2) /\ M (gl_o L1) (gl_o L2)
  /\ M (gl_gate L1) (gl_gate L2) /\ M (gl_up L1) (gl_up L2) /\ M (gl_down L1) (gl_down L2).

Definition glmodel_rel (m1 : glmodel T1) (m2 : glmodel T2) : Prop :=
  M (glm_emb m1) (glm_emb m2) /\ V (glm_norm m1) (glm_norm m2)
  /\ Forall2 gllayer_rel (glm_layers m1) (glm_layers m2).

Lemma r_silu : forall x1 x2, rel x1 x2 -> rel (g_silu A1 x1) (g_silu A2 x2).
Proof.
  intros x1 x2 H. unfold g_silu. apply (rel_mult HR); [exact H|].
  apply (r_sigmoid A1 A2 rel HR), H.
Qed.

Lemma r_vec_mult : forall xs1 xs2 ys1 ys2, V xs1 xs2 -> V ys1 ys2 ->
  V (g_vec_mult A1 xs1 ys1) (g_vec_mult A2 xs2 ys2).
Proof.
  intros xs1 xs2 ys1 ys2 Hx Hy. unfold g_vec_mult.
  apply Forall2_map_combine with (P := rel) (Q := rel); [exact Hx | exact Hy |].
  intros a d b e Had Hbe. cbv beta iota. apply (rel_mult HR); assumption.
Qed.

Lemma r_rmsnorm : forall w1 w2 e1 e2 x1 x2, V w1 w2 -> rel e1 e2 -> V x1 x2 ->
  V (g_rmsnorm A1 w1 e1 x1) (g_rmsnorm A2 w2 e2 x2).
Proof.
  intros w1 w2 e1 e2 x1 x2 Hw He Hx. unfold g_rmsnorm. cbv zeta.
  rewrite (Forall2_length _ _ _ _ _ Hx).
  apply Forall2_map_combine with (P := rel) (Q := rel); [exact Hw | exact Hx |].
  intros a d b e Had Hbe. cbv beta iota.
  apply (rel_mult HR); [exact Had|]. apply (rel_mult HR); [exact Hbe|].
  apply (rel_div HR); [apply (rel_const HR)|]. apply (rel_sqrt HR).
  apply (rel_plus HR); [|exact He].
  apply (rel_div HR); [|apply (rel_const HR)].
  apply (r_sum A1 A2 rel HR). apply Forall2_map2 with (P := rel); [exact Hx|].
  intros u v Huv. apply (rel_mult HR); assumption.
Qed.

Lemma r_slice : forall off len r1 r2, V r1 r2 -> V (g_slice off len r1) (g_slice off len r2).
Proof. intros off len r1 r2 H. unfold g_slice. apply Forall2_firstn, Forall2_skipn, H. Qed.

Lemma r_partial_rope : forall rd c1 c2 s1 s2 x1 x2, V c1 c2 -> V s1 s2 -> V x1 x2 ->
  V (g_partial_rope A1 rd c1 s1 x1) (g_partial_rope A2 rd c2 s2 x2).
Proof.
  intros rd c1 c2 s1 s2 x1 x2 Hc Hs Hx. unfold g_partial_rope. cbv zeta.
  apply Forall2_app; [|apply Forall2_skipn, Hx].
  pose proof (Forall2_firstn _ _ rel rd _ _ Hx) as Hrot.
  assert (Hrt : V (List.map (fun z => op_neg A1 z) (List.skipn (Nat.div rd 2) (List.firstn rd x1))
                   ++ List.firstn (Nat.div rd 2) (List.firstn rd x1))
                  (List.map (fun z => op_neg A2 z) (List.skipn (Nat.div rd 2) (List.firstn rd x2))
                   ++ List.firstn (Nat.div rd 2) (List.firstn rd x2))).
  { apply Forall2_app; [|apply Forall2_firstn, Hrot].
    apply Forall2_map2 with (P := rel); [apply Forall2_skipn, Hrot|].
    intros a b Hab. apply (rel_neg HR), Hab. }
  pose proof (Forall2_combine _ _ _ _ _ _ _ _ _ _ Hc Hs) as Hcs.
  pose proof (Forall2_combine _ _ _ _ _ _ _ _ _ _ Hrt Hcs) as Hrcs.
  apply Forall2_map_combine
    with (P := rel)
         (Q := fun p q => rel (fst p) (fst q)
                          /\ (rel (fst (snd p)) (fst (snd q)) /\ rel (snd (snd p)) (snd (snd q))));
    [exact Hrot | exact Hrcs |].
  intros a d [ri [ci si]] [ri' [ci' si']] Had [Hr [Hci Hsi]]. cbn [fst snd] in *.
  cbv beta iota. apply (rel_plus HR); apply (rel_mult HR); assumption.
Qed.

Lemma r_swiglu : forall g1 g2 u1 u2 d1 d2 x1 x2, M g1 g2 -> M u1 u2 -> M d1 d2 -> V x1 x2 ->
  V (g_swiglu A1 g1 u1 d1 x1) (g_swiglu A2 g2 u2 d2 x2).
Proof.
  intros g1 g2 u1 u2 d1 d2 x1 x2 Hg Hu Hd Hx. unfold g_swiglu. cbv zeta.
  apply (r_mat_vec_mul A1 A2 rel HR); [exact Hd|].
  apply r_vec_mult.
  - apply Forall2_map2 with (P := rel); [apply (r_mat_vec_mul A1 A2 rel HR); assumption|].
    intros a b Hab. apply r_silu, Hab.
  - apply (r_mat_vec_mul A1 A2 rel HR); assumption.
Qed.

Lemma r_rope_rows : forall hd cs1 cs2 sn1 sn2 c m1 m2, M cs1 cs2 -> M sn1 sn2 -> M m1 m2 ->
  M (g_rope_rows A1 hd cs1 sn1 c m1) (g_rope_rows A2 hd cs2 sn2 c m2).
Proof.
  intros hd cs1 cs2 sn1 sn2 c m1 m2 Hc Hs Hm. unfold g_rope_rows.
  pose proof (Forall2_combine_seq _ _ V m1 m2 0 Hm) as Hcm.
  apply Forall2_map2 with (P := fun p q => fst p = fst q /\ V (snd p) (snd q)); [exact Hcm|].
  intros [i1 a1] [i2 a2] [Hi Ha]. cbn [fst snd] in Hi, Ha. subst i2. cbv beta iota.
  apply r_partial_rope.
  - apply Forall2_nth; [exact Hc | constructor].
  - apply Forall2_nth; [exact Hs | constructor].
  - apply r_slice, Ha.
Qed.

Lemma r_proj_rows : forall w1 w2 h1 h2, M w1 w2 -> M h1 h2 ->
  M (List.map (fun h => g_mat_vec_mul A1 w1 h) h1) (List.map (fun h => g_mat_vec_mul A2 w2 h) h2).
Proof.
  intros w1 w2 h1 h2 Hw Hh. apply Forall2_map2 with (P := V); [exact Hh|].
  intros a b Hab. apply (r_mat_vec_mul A1 A2 rel HR); assumption.
Qed.

Lemma r_llama_heads : forall nh nkv hd L1 L2 cs1 cs2 sn1 sn2 h1 h2,
  gllayer_rel L1 L2 -> M cs1 cs2 -> M sn1 sn2 -> M h1 h2 ->
  Forall2 M (g_llama_heads A1 nh nkv hd L1 cs1 sn1 h1) (g_llama_heads A2 nh nkv hd L2 cs2 sn2 h2).
Proof.
  intros nh nkv hd L1 L2 cs1 cs2 sn1 sn2 h1 h2 HL Hc Hs Hh.
  destruct HL as (_ & _ & Hq & Hk & Hv & _).
  unfold g_llama_heads. cbv zeta.
  pose proof (r_proj_rows _ _ _ _ Hq Hh) as Pq.
  pose proof (r_proj_rows _ _ _ _ Hk Hh) as Pk.
  pose proof (r_proj_rows _ _ _ _ Hv Hh) as Pv.
  apply Forall2_map_same. intros hh.
  apply (r_causal_attention A1 A2 rel HR).
  - apply Forall2_nth; [|constructor]. apply Forall2_map_same. intros i.
    apply r_rope_rows; assumption.
  - apply Forall2_nth; [|constructor]. apply Forall2_map_same. intros i.
    apply r_rope_rows; assumption.
  - apply Forall2_nth; [|constructor]. apply Forall2_map_same. intros i.
    apply Forall2_map2 with (P := V); [exact Pv|]. intros a b Hab. apply r_slice, Hab.
Qed.

Lemma r_llama_attn : forall nh nkv hd L1 L2 cs1 cs2 sn1 sn2 h1 h2,
  gllayer_rel L1 L2 -> M cs1 cs2 -> M sn1 sn2 -> M h1 h2 ->
  M (g_llama_attn A1 nh nkv hd L1 cs1 sn1 h1) (g_llama_attn A2 nh nkv hd L2 cs2 sn2 h2).
Proof.
  intros nh nkv hd L1 L2 cs1 cs2 sn1 sn2 h1 h2 HL Hc Hs Hh. unfold g_llama_attn.
  apply r_proj_rows; [apply HL|].
  apply (r_concat_heads rel). apply r_llama_heads; assumption.
Qed.

Lemma r_add_rows : forall a1 a2 b1 b2, M a1 a2 -> M b1 b2 ->
  M (List.map (fun p => let '(a, b) := p in g_vec_add A1 a b) (List.combine a1 b1))
    (List.map (fun p => let '(a, b) := p in g_vec_add A2 a b) (List.combine a2 b2)).
Proof.
  intros a1 a2 b1 b2 Ha Hb.
  apply Forall2_map_combine with (P := V) (Q := V); [exact Ha | exact Hb |].
  intros a d b e Had Hbe. cbv beta iota. apply (r_vec_add A1 A2 rel HR); assumption.
Qed.

Lemma r_llama_layer : forall nh nkv hd e1 e2 L1 L2 cs1 cs2 sn1 sn2 h1 h2,
  rel e1 e2 -> gllayer_rel L1 L2 -> M cs1 cs2 -> M sn1 sn2 -> M h1 h2 ->
  M (g_llama_layer A1 nh nkv hd e1 L1 cs1 sn1 h1) (g_llama_layer A2 nh nkv hd e2 L2 cs2 sn2 h2).
Proof.
  intros nh nkv hd e1 e2 L1 L2 cs1 cs2 sn1 sn2 h1 h2 He HL Hc Hs Hh.
  pose proof HL as (H1 & H2 & _ & _ & _ & _ & Hg & Hu & Hd).
  unfold g_llama_layer. cbv zeta.
  assert (Hn : forall w1 w2 x1 x2, V w1 w2 -> M x1 x2 ->
            M (List.map (fun row => g_rmsnorm A1 w1 e1 row) x1)
              (List.map (fun row => g_rmsnorm A2 w2 e2 row) x2)).
  { intros w1 w2 x1 x2 Hw Hx. apply Forall2_map2 with (P := V); [exact Hx|].
    intros a b Hab. apply r_rmsnorm; assumption. }
  assert (H2' : M (List.map (fun p => let '(a, b) := p in g_vec_add A1 a b)
                    (List.combine h1 (g_llama_attn A1 nh nkv hd L1 cs1 sn1
                       (List.map (fun row => g_rmsnorm A1 (gl_ln1 L1) e1 row) h1))))
                  (List.map (fun p => let '(a, b) := p in g_vec_add A2 a b)
                    (List.combine h2 (g_llama_attn A2 nh nkv hd L2 cs2 sn2
                       (List.map (fun row => g_rmsnorm A2 (gl_ln1 L2) e2 row) h2))))).
  { apply r_add_rows; [exact Hh|]. apply r_llama_attn; try assumption. apply Hn; assumption. }
  apply r_add_rows; [exact H2'|].
  apply Forall2_map2 with (P := V); [apply Hn; assumption|].
  intros a b Hab. apply r_swiglu; assumption.
Qed.

Lemma r_llama_stack : forall nh nkv hd e1 e2 Ls1 Ls2 cs1 cs2 sn1 sn2 h1 h2,
  rel e1 e2 -> Forall2 gllayer_rel Ls1 Ls2 -> M cs1 cs2 -> M sn1 sn2 -> M h1 h2 ->
  M (g_llama_stack A1 nh nkv hd e1 Ls1 cs1 sn1 h1) (g_llama_stack A2 nh nkv hd e2 Ls2 cs2 sn2 h2).
Proof.
  intros nh nkv hd e1 e2 Ls1 Ls2 cs1 cs2 sn1 sn2 h1 h2 He HLs Hc Hs. revert h1 h2.
  induction HLs as [|L1 L2 Ls1 Ls2 HL HLs IH]; intros h1 h2 Hh;
    cbn [g_llama_stack]; [exact Hh|].
  apply IH. apply r_llama_layer; assumption.
Qed.

Lemma r_llama_logits : forall nh nkv hd e1 e2 m1 m2 cs1 cs2 sn1 sn2 ids,
  rel e1 e2 -> glmodel_rel m1 m2 -> M cs1 cs2 -> M sn1 sn2 ->
  M (g_llama_logits A1 nh nkv hd e1 m1 cs1 sn1 ids) (g_llama_logits A2 nh nkv hd e2 m2 cs2 sn2 ids).
Proof.
  intros nh nkv hd e1 e2 m1 m2 cs1 cs2 sn1 sn2 ids He (Hemb & Hnorm & HLs) Hc Hs.
  unfold g_llama_logits. cbv zeta.
  apply Forall2_map2 with (P := V).
  - apply Forall2_map2 with (P := V).
    + apply r_llama_stack; try assumption.
      unfold g_embed_tokens. apply Forall2_map_same. intros t.
      apply Forall2_nth; [exact Hemb | constructor].
    + intros a b Hab. apply r_rmsnorm; assumption.
  - intros a b Hab. apply Forall2_map2 with (P := V); [exact Hemb|].
    intros c d Hcd. apply (rel_dot HR); assumption.
Qed.

End RLlama.

Lemma glmodel_rel_map2 : forall (T0 T1 T2 : Type) (rel : T1 -> T2 -> Prop)
                                (f : T0 -> T1) (g : T0 -> T2) m,
  (forall x, rel (f x) (g x)) -> glmodel_rel rel (glmodel_map f m) (glmodel_map g m).
Proof.
  intros T0 T1 T2 rel f g m H. unfold glmodel_rel, glmodel_map.
  cbn [glm_emb glm_norm glm_layers].
  assert (HV : forall l, Forall2 rel (List.map f l) (List.map g l))
    by (intros; apply Forall2_map_same, H).
  assert (HM : forall l, Forall2 (Forall2 rel) (List.map (List.map f) l) (List.map (List.map g) l))
    by (intros; apply Forall2_map_same; intros; apply HV).
  repeat split; try apply HV; try apply HM.
  apply Forall2_map_same. intros L.
  unfold gllayer_rel, gllayer_map.
  cbn [gl_ln1 gl_ln2 gl_q gl_k gl_v gl_o gl_gate gl_up gl_down].
  repeat split; try apply HV; try apply HM.
Qed.

(** * The enclosure of the checkpoint's logits

    Run on the checkpoint's weights and the rotary tables the binary32 pass
    forms, the pass in the enclosure arithmetic returns for every logit an
    interval that contains the logit of the exact real reference: the same
    network on the same weights, tables and tokens, in real arithmetic, with
    every exponential the true exponential of its argument saturated to
    [[-88, 88]]. At binary32 the same pass is [f32_llama_logits_of]
    ([g_llama_logits_f32]). *)
Theorem llama_logits_enclosed : forall FB prec nh nkv hd eps m ids, (149 <= FB)%Z ->
  let cosv := f32_rope_cos (lw_invf m) (List.length ids) in
  let sinv := f32_rope_sin (lw_invf m) (List.length ids) in
  Forall2 (Forall2 (enc_in FB))
    (g_llama_logits (enc_ops FB prec) nh nkv hd (enc_const eps)
       (glmodel_map enc_const (to_glmodel m))
       (List.map (List.map enc_const) cosv) (List.map (List.map enc_const) sinv) ids)
    (g_llama_logits real_ops nh nkv hd (B2R eps) (glmodel_map B2R (to_glmodel m))
       (List.map (List.map B2R) cosv) (List.map (List.map B2R) sinv) ids).
Proof.
  intros FB prec nh nkv hd eps m ids HFB cosv sinv.
  assert (HT : forall t, Forall2 (Forall2 (enc_in FB))
                 (List.map (List.map enc_const) t) (List.map (List.map B2R) t)).
  { intros t. apply Forall2_map_same. intros r. apply Forall2_map_same. intros x.
    apply enc_const_in. }
  apply (r_llama_logits (enc_ops FB prec) real_ops (enc_in FB) (enc_ops_rel FB HFB prec)).
  - apply enc_const_in.
  - apply glmodel_rel_map2. intros x. apply enc_const_in.
  - apply HT.
  - apply HT.
Qed.
