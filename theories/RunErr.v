(** * A forward pass that carries its own error bound

    The GPT-2 forward pass is written here once, over an abstract arithmetic,
    and instantiated three times: at binary32, where it is [f32_gpt2_logits]
    by conversion; at exact real arithmetic with the exponential evaluated
    exactly on its saturated argument, where it is the reference; and at the
    annotated arithmetic of Annot.v and AnnExp.v, where every value travels
    with a binary64 bound on its distance from the reference. One relational
    theorem,
    proved for any relation the arithmetic operations preserve, gives both
    facts the bound needs: the binary32 components of the annotated pass are
    the logits the model computes, and each is within its bound of the exact
    real logit.

    The bound is computed during the pass itself, from the values that pass
    produces, so it holds for the input the pass was run on. Its conditions are
    checked as the pass proceeds; an operation whose check fails reports an
    infinite bound. *)

From Stdlib Require Import ZArith Reals Lra Lia List Bool.
From Flocq Require Import Core IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete Qwen Float_error Bound64 Annot AnnExp.

Import ListNotations.
Open Scope R_scope.

(** * An arithmetic *)

Record ops (T : Type) := mk_ops {
  op_const : binary32 -> T;
  op_plus : T -> T -> T;
  op_neg : T -> T;
  op_mult : T -> T -> T;
  op_div : T -> T -> T;
  op_sqrt : T -> T;
  op_max : T -> T -> T;
  op_exp : T -> T;
  op_dot : list T -> list T -> T
}.

Arguments op_const {T} _ _.
Arguments op_plus {T} _ _ _.
Arguments op_neg {T} _ _.
Arguments op_mult {T} _ _ _.
Arguments op_div {T} _ _ _.
Arguments op_sqrt {T} _ _.
Arguments op_max {T} _ _ _.
Arguments op_exp {T} _ _.
Arguments op_dot {T} _ _ _.

(** The dot products of the annotated and the real arithmetic accumulate in the
    order [f32_dot] does. *)
Fixpoint ann_dot_aux (xs ys : list ann) (acc : ann) : ann :=
  match xs, ys with
  | x :: xs', y :: ys' => ann_dot_aux xs' ys' (ann_plus acc (ann_mult x y))
  | _, _ => acc
  end.

Definition ann_dot (xs ys : list ann) : ann := ann_dot_aux xs ys (ann_const f32_zero).

Definition f32_ops : ops binary32 :=
  mk_ops binary32 (fun c => c) f32_plus f32_neg f32_mult f32_div f32_sqrt
    f32_max2 f32_exp_approx f32_dot.

Definition ann_ops : ops ann :=
  mk_ops ann ann_const ann_plus ann_neg ann_mult ann_div ann_sqrt
    ann_max ann_exp ann_dot.

Definition real_ops : ops R :=
  mk_ops R (fun c => B2R c) Rplus Ropp Rmult Rdiv sqrt Rmax
    (fun r => exp (Rsat r)) Rdot.

(** * The weights *)

Record gblock (T : Type) := mk_gblock {
  gb_ln1w : list T; gb_ln1b : list T;
  gb_caw : list (list T); gb_cab : list T;
  gb_cpw : list (list T); gb_cpb : list T;
  gb_ln2w : list T; gb_ln2b : list T;
  gb_fcw : list (list T); gb_fcb : list T;
  gb_mpw : list (list T); gb_mpb : list T
}.

Record gmodel (T : Type) := mk_gmodel {
  gm_wte : list (list T); gm_wpe : list (list T);
  gm_blocks : list (gblock T);
  gm_lnfw : list T; gm_lnfb : list T
}.

Arguments mk_gblock {T}.
Arguments gb_ln1w {T}. Arguments gb_ln1b {T}.
Arguments gb_caw {T}. Arguments gb_cab {T}.
Arguments gb_cpw {T}. Arguments gb_cpb {T}.
Arguments gb_ln2w {T}. Arguments gb_ln2b {T}.
Arguments gb_fcw {T}. Arguments gb_fcb {T}.
Arguments gb_mpw {T}. Arguments gb_mpb {T}.
Arguments mk_gmodel {T}.
Arguments gm_wte {T}. Arguments gm_wpe {T}. Arguments gm_blocks {T}.
Arguments gm_lnfw {T}. Arguments gm_lnfb {T}.

Definition to_gblock (b : f32_block_weights) : gblock binary32 :=
  mk_gblock (f32_ln_weight (f32_block_ln_1 b)) (f32_ln_bias (f32_block_ln_1 b))
    (f32_attn_c_attn_weight (f32_block_attn b)) (f32_attn_c_attn_bias (f32_block_attn b))
    (f32_attn_c_proj_weight (f32_block_attn b)) (f32_attn_c_proj_bias (f32_block_attn b))
    (f32_ln_weight (f32_block_ln_2 b)) (f32_ln_bias (f32_block_ln_2 b))
    (f32_mlp_c_fc_weight (f32_block_mlp b)) (f32_mlp_c_fc_bias (f32_block_mlp b))
    (f32_mlp_c_proj_weight (f32_block_mlp b)) (f32_mlp_c_proj_bias (f32_block_mlp b)).

Definition to_gmodel (m : f32_model_weights) : gmodel binary32 :=
  mk_gmodel (f32_wte m) (f32_wpe m) (List.map to_gblock (f32_blocks m))
    (f32_ln_weight (f32_ln_f m)) (f32_ln_bias (f32_ln_f m)).

Definition gblock_map {T1 T2 : Type} (f : T1 -> T2) (b : gblock T1) : gblock T2 :=
  mk_gblock (List.map f (gb_ln1w b)) (List.map f (gb_ln1b b))
    (List.map (List.map f) (gb_caw b)) (List.map f (gb_cab b))
    (List.map (List.map f) (gb_cpw b)) (List.map f (gb_cpb b))
    (List.map f (gb_ln2w b)) (List.map f (gb_ln2b b))
    (List.map (List.map f) (gb_fcw b)) (List.map f (gb_fcb b))
    (List.map (List.map f) (gb_mpw b)) (List.map f (gb_mpb b)).

Definition gmodel_map {T1 T2 : Type} (f : T1 -> T2) (m : gmodel T1) : gmodel T2 :=
  mk_gmodel (List.map (List.map f) (gm_wte m)) (List.map (List.map f) (gm_wpe m))
    (List.map (gblock_map f) (gm_blocks m)) (List.map f (gm_lnfw m)) (List.map f (gm_lnfb m)).

(** The model's weights as annotated values, each exact, and as the reals they
    denote. *)
Definition ann_model (m : f32_model_weights) : gmodel ann := gmodel_map ann_const (to_gmodel m).
Definition real_model (m : f32_model_weights) : gmodel R := gmodel_map B2R (to_gmodel m).

(** * The forward pass, over any arithmetic *)

Section Generic.

Context {T : Type} (A : ops T).

Definition g_minus (x y : T) : T := op_plus A x (op_neg A y).

Definition g_vec_add (xs ys : list T) : list T :=
  List.map (fun '(x, y) => op_plus A x y) (List.combine xs ys).

Definition g_mat_vec_mul (m : list (list T)) (v : list T) : list T :=
  List.map (fun row => op_dot A row v) m.

Definition g_sum (xs : list T) : T := List.fold_left (op_plus A) xs (op_const A f32_zero).

Definition g_sigmoid (x : T) : T :=
  let neg_x := op_neg A x in
  let exp_neg_x := op_exp A neg_x in
  let denom := op_plus A (op_const A f32_one) exp_neg_x in
  op_div A (op_const A f32_one) denom.

Definition g_tanh (x : T) : T :=
  let two_x := op_mult A (op_const A f32_two) x in
  let sig := g_sigmoid two_x in
  g_minus (op_mult A (op_const A f32_two) sig) (op_const A f32_one).

Definition g_gelu (x : T) : T :=
  let x3 := op_mult A x (op_mult A x x) in
  let inner := op_mult A (op_const A f32_gelu_c1)
                 (op_plus A x (op_mult A (op_const A f32_gelu_c2) x3)) in
  op_mult A (op_mult A (op_const A f32_half) x) (op_plus A (op_const A f32_one) (g_tanh inner)).

Definition g_gelu_vec (v : list T) : list T := List.map g_gelu v.

Definition g_max_vec (xs : list T) : T :=
  match xs with
  | [] => op_const A f32_zero
  | x :: rest => List.fold_left (fun acc y => op_max A acc y) rest x
  end.

Definition g_softmax (logits : list T) : list T :=
  let max_val := g_max_vec logits in
  let shifted := List.map (fun x => g_minus x max_val) logits in
  let exps := List.map (op_exp A) shifted in
  let sum_exp := g_sum exps in
  List.map (fun e => op_div A e sum_exp) exps.

Definition g_mean (xs : list T) : T :=
  let n := op_const A (f32_of_Z (Z.of_nat (List.length xs))) in
  op_div A (g_sum xs) n.

Definition g_variance (xs : list T) (mean : T) : T :=
  let n := op_const A (f32_of_Z (Z.of_nat (List.length xs))) in
  let sq := List.map (fun x => let d := g_minus x mean in op_mult A d d) xs in
  op_div A (g_sum sq) n.

Definition g_layer_norm_vec (gamma beta : list T) (eps : T) (x : list T) : list T :=
  let mu := g_mean x in
  let var := g_variance x mu in
  let denom := op_sqrt A (op_plus A var eps) in
  List.map (fun '(idx, xi) =>
    let g := List.nth idx gamma (op_const A f32_one) in
    let b := List.nth idx beta (op_const A f32_zero) in
    let normalized := op_div A (g_minus xi mu) denom in
    op_plus A (op_mult A g normalized) b
  ) (List.combine (List.seq 0 (List.length x)) x).

Definition g_layer_norm_2d (gamma beta : list T) (eps : T) (m : list (list T))
  : list (list T) :=
  List.map (g_layer_norm_vec gamma beta eps) m.

Definition g_mat_transpose (m : list (list T)) : list (list T) :=
  match m with
  | [] => []
  | row :: _ =>
      List.map (fun col_idx =>
        List.map (fun row_data => List.nth col_idx row_data (op_const A f32_zero)) m
      ) (List.seq 0 (List.length row))
  end.

Definition g_linear_forward (w : list (list T)) (b : list T) (x : list T) : list T :=
  g_vec_add (g_mat_vec_mul (g_mat_transpose w) x) b.

Definition g_linear_forward_2d (w : list (list T)) (b : list T) (x : list (list T))
  : list (list T) :=
  List.map (g_linear_forward w b) x.

Definition g_add_matrices (a b : list (list T)) : list (list T) :=
  List.map (fun '(row_a, row_b) => g_vec_add row_a row_b) (List.combine a b).

Definition g_split_row_into_heads (num_heads : nat) (row : list T) : list (list T) :=
  let head_dim := Nat.div (List.length row) num_heads in
  List.map (fun h => List.firstn head_dim (List.skipn (h * head_dim) row))
           (List.seq 0 num_heads).

Definition g_split_into_heads (num_heads : nat) (m : list (list T))
  : list (list (list T)) :=
  let rows_split := List.map (g_split_row_into_heads num_heads) m in
  List.map (fun h => List.map (fun row_heads => List.nth h row_heads []) rows_split)
           (List.seq 0 num_heads).

Definition g_concat_heads (heads : list (list (list T))) : list (list T) :=
  match heads with
  | [] => []
  | first_head :: _ =>
      List.map (fun seq_idx =>
        List.concat (List.map (fun head => List.nth seq_idx head []) heads)
      ) (List.seq 0 (List.length first_head))
  end.

Definition g_attn_scale (d_k : nat) : T :=
  op_div A (op_const A f32_one) (op_sqrt A (op_const A (f32_of_Z (Z.of_nat d_k)))).

Definition g_attend (qrow : list T) (ks vs : list (list T)) (d_k : nat) : list T :=
  let scores := List.map (fun kj => op_mult A (op_dot A qrow kj) (g_attn_scale d_k)) ks in
  let w := g_softmax scores in
  List.map (fun vcol => op_dot A w vcol) (g_mat_transpose vs).

Definition g_causal_attention (q k v : list (list T)) (d_k : nat) : list (list T) :=
  List.map (fun p => let '(i, qrow) := p in
              g_attend qrow (List.firstn (S i) k) (List.firstn (S i) v) d_k)
           (List.combine (List.seq 0 (List.length q)) q).

Definition g_attention_forward (n_embd n_head : nat)
    (c_attn_w : list (list T)) (c_attn_b : list T)
    (c_proj_w : list (list T)) (c_proj_b : list T)
    (hidden : list (list T)) : list (list T) :=
  let qkv := g_linear_forward_2d c_attn_w c_attn_b hidden in
  let d := n_embd in
  let head_dim := Nat.div d n_head in
  let split_qkv row :=
    (List.firstn d row, List.firstn d (List.skipn d row), List.skipn (2 * d) row) in
  let qkv_split := List.map split_qkv qkv in
  let qs := List.map (fun '(q, _, _) => q) qkv_split in
  let ks := List.map (fun '(_, k, _) => k) qkv_split in
  let vs := List.map (fun '(_, _, v) => v) qkv_split in
  let q_heads := g_split_into_heads n_head qs in
  let k_heads := g_split_into_heads n_head ks in
  let v_heads := g_split_into_heads n_head vs in
  let head_outputs := List.map (fun '(qh, (kh, vh)) => g_causal_attention qh kh vh head_dim)
                               (List.combine q_heads (List.combine k_heads v_heads)) in
  let concat := g_concat_heads head_outputs in
  g_linear_forward_2d c_proj_w c_proj_b concat.

Definition g_mlp_forward
    (c_fc_w : list (list T)) (c_fc_b : list T)
    (c_proj_w : list (list T)) (c_proj_b : list T)
    (hidden : list (list T)) : list (list T) :=
  let h := g_linear_forward_2d c_fc_w c_fc_b hidden in
  let h_gelu := List.map g_gelu_vec h in
  g_linear_forward_2d c_proj_w c_proj_b h_gelu.

Definition g_block_forward (cfg : gpt2_inference_config) (eps : T) (bl : gblock T)
    (hidden : list (list T)) : list (list T) :=
  let ln1 := g_layer_norm_2d (gb_ln1w bl) (gb_ln1b bl) eps hidden in
  let attn_out := g_attention_forward (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)
                    (gb_caw bl) (gb_cab bl) (gb_cpw bl) (gb_cpb bl) ln1 in
  let hidden2 := g_add_matrices hidden attn_out in
  let ln2 := g_layer_norm_2d (gb_ln2w bl) (gb_ln2b bl) eps hidden2 in
  let mlp_out := g_mlp_forward (gb_fcw bl) (gb_fcb bl) (gb_mpw bl) (gb_mpb bl) ln2 in
  g_add_matrices hidden2 mlp_out.

Fixpoint g_blocks_forward (cfg : gpt2_inference_config) (eps : T)
    (blocks : list (gblock T)) (hidden : list (list T)) : list (list T) :=
  match blocks with
  | [] => hidden
  | b :: rest => g_blocks_forward cfg eps rest (g_block_forward cfg eps b hidden)
  end.

Definition g_embed_tokens (e : list (list T)) (ids : list nat) : list (list T) :=
  List.map (fun t => List.nth t e []) ids.

Definition g_embed_positions (e : list (list T)) (n : nat) : list (list T) :=
  List.map (fun t => List.nth t e []) (List.seq 0 n).

Definition g_gpt2_forward (cfg : gpt2_inference_config) (eps : T) (m : gmodel T)
    (toks : list nat) : list (list T) :=
  let seq_len := List.length toks in
  let tok := g_embed_tokens (gm_wte m) toks in
  let pos := g_embed_positions (gm_wpe m) seq_len in
  let hidden := g_add_matrices tok pos in
  let transformed := g_blocks_forward cfg eps (gm_blocks m) hidden in
  g_layer_norm_2d (gm_lnfw m) (gm_lnfb m) eps transformed.

Definition g_gpt2_logits (cfg : gpt2_inference_config) (eps : T) (m : gmodel T)
    (toks : list nat) : list (list T) :=
  let hidden := g_gpt2_forward cfg eps m toks in
  List.map (fun h_row => List.map (fun w_row => op_dot A h_row w_row) (gm_wte m)) hidden.

End Generic.

(** * At binary32 the generic pass is the model's

    Both sides are unfolded to the same term before they are compared, so the
    comparison never evaluates a binary32 constant. *)

Ltac unfold_both :=
  cbv [g_block_forward g_layer_norm_2d g_layer_norm_vec g_mean g_variance g_sum g_minus
       g_attention_forward g_linear_forward_2d g_linear_forward g_vec_add g_mat_vec_mul
       g_mat_transpose g_split_into_heads g_split_row_into_heads g_concat_heads
       g_causal_attention g_attend g_attn_scale g_softmax g_max_vec
       g_add_matrices g_mlp_forward g_gelu_vec g_gelu g_tanh g_sigmoid
       g_embed_tokens g_embed_positions
       op_const op_plus op_neg op_mult op_div op_sqrt op_max op_exp op_dot
       f32_ops to_gblock gb_ln1w gb_ln1b gb_caw gb_cab gb_cpw gb_cpb gb_ln2w gb_ln2b
       gb_fcw gb_fcb gb_mpw gb_mpb
       f32_block_forward f32_layer_norm_2d f32_layer_norm_vec f32_mean f32_variance f32_sum
       f32_minus f32_attention_forward f32_linear_forward_2d f32_linear_forward f32_vec_add
       f32_mat_vec_mul f32_mat_transpose f32_split_into_heads f32_split_row_into_heads
       f32_concat_heads f32_causal_attention f32_attend f32_attn_scale f32_softmax
       f32_max_vec f32_exp_vec f32_add_matrices f32_mlp_forward
       f32_gelu_vec f32_gelu f32_tanh f32_sigmoid f32_max2
       f32_add_matrices f32_embed_tokens f32_embed_positions f32_lookup_embedding].

Lemma g_block_forward_f32 : forall cfg eps b hidden,
  g_block_forward f32_ops cfg eps (to_gblock b) hidden = f32_block_forward cfg eps b hidden.
Proof. intros. unfold_both. reflexivity. Qed.

Lemma g_blocks_forward_f32 : forall cfg eps bs hidden,
  g_blocks_forward f32_ops cfg eps (List.map to_gblock bs) hidden
  = f32_blocks_forward cfg eps bs hidden.
Proof.
  intros cfg eps bs. induction bs as [|b bs IH]; intros hidden; [reflexivity|].
  cbn [List.map g_blocks_forward f32_blocks_forward].
  rewrite g_block_forward_f32. apply IH.
Qed.

Theorem g_gpt2_logits_f32 : forall cfg eps m toks,
  g_gpt2_logits f32_ops cfg eps (to_gmodel m) toks = f32_gpt2_logits cfg eps m toks.
Proof.
  intros cfg eps m toks.
  cbv [g_gpt2_logits g_gpt2_forward to_gmodel gm_wte gm_wpe gm_blocks gm_lnfw gm_lnfb].
  rewrite g_blocks_forward_f32.
  cbv [f32_gpt2_logits f32_gpt2_forward].
  unfold_both. reflexivity.
Qed.

(** * Relations the arithmetic preserves *)

Record ops_rel {T1 T2 : Type} (A1 : ops T1) (A2 : ops T2) (rel : T1 -> T2 -> Prop) : Prop := {
  rel_const : forall c, rel (op_const A1 c) (op_const A2 c);
  rel_plus : forall x1 x2 y1 y2, rel x1 x2 -> rel y1 y2 ->
               rel (op_plus A1 x1 y1) (op_plus A2 x2 y2);
  rel_neg : forall x1 x2, rel x1 x2 -> rel (op_neg A1 x1) (op_neg A2 x2);
  rel_mult : forall x1 x2 y1 y2, rel x1 x2 -> rel y1 y2 ->
               rel (op_mult A1 x1 y1) (op_mult A2 x2 y2);
  rel_div : forall x1 x2 y1 y2, rel x1 x2 -> rel y1 y2 ->
              rel (op_div A1 x1 y1) (op_div A2 x2 y2);
  rel_sqrt : forall x1 x2, rel x1 x2 -> rel (op_sqrt A1 x1) (op_sqrt A2 x2);
  rel_max : forall x1 x2 y1 y2, rel x1 x2 -> rel y1 y2 ->
              rel (op_max A1 x1 y1) (op_max A2 x2 y2);
  rel_exp : forall x1 x2, rel x1 x2 -> rel (op_exp A1 x1) (op_exp A2 x2);
  rel_dot : forall xs1 xs2 ys1 ys2, Forall2 rel xs1 xs2 -> Forall2 rel ys1 ys2 ->
              rel (op_dot A1 xs1 ys1) (op_dot A2 xs2 ys2)
}.

Arguments rel_const {T1 T2 A1 A2 rel} _ _.
Arguments rel_plus {T1 T2 A1 A2 rel} _ _ _ _ _ _ _.
Arguments rel_neg {T1 T2 A1 A2 rel} _ _ _ _.
Arguments rel_mult {T1 T2 A1 A2 rel} _ _ _ _ _ _ _.
Arguments rel_div {T1 T2 A1 A2 rel} _ _ _ _ _ _ _.
Arguments rel_sqrt {T1 T2 A1 A2 rel} _ _ _ _.
Arguments rel_max {T1 T2 A1 A2 rel} _ _ _ _ _ _ _.
Arguments rel_exp {T1 T2 A1 A2 rel} _ _ _ _.
Arguments rel_dot {T1 T2 A1 A2 rel} _ _ _ _ _ _ _.

Section Relational.

Context {T1 T2 : Type} (A1 : ops T1) (A2 : ops T2) (rel : T1 -> T2 -> Prop).
Context (HR : ops_rel A1 A2 rel).

Local Notation V := (Forall2 rel).
Local Notation M := (Forall2 (Forall2 rel)).

Lemma r_minus : forall x1 x2 y1 y2, rel x1 x2 -> rel y1 y2 ->
  rel (g_minus A1 x1 y1) (g_minus A2 x2 y2).
Proof.
  intros. unfold g_minus. apply (rel_plus HR); [assumption | apply (rel_neg HR); assumption].
Qed.

Lemma r_vec_add : forall xs1 xs2 ys1 ys2, V xs1 xs2 -> V ys1 ys2 ->
  V (g_vec_add A1 xs1 ys1) (g_vec_add A2 xs2 ys2).
Proof.
  intros xs1 xs2 ys1 ys2 Hx Hy. unfold g_vec_add.
  apply Forall2_map_combine with (P := rel) (Q := rel); [exact Hx | exact Hy |].
  intros a d b e Had Hbe. cbv beta iota. apply (rel_plus HR); assumption.
Qed.

Lemma r_mat_vec_mul : forall m1 m2 v1 v2, M m1 m2 -> V v1 v2 ->
  V (g_mat_vec_mul A1 m1 v1) (g_mat_vec_mul A2 m2 v2).
Proof.
  intros m1 m2 v1 v2 Hm Hv. unfold g_mat_vec_mul.
  apply Forall2_map2 with (P := V); [exact Hm|].
  intros a b Hab. apply (rel_dot HR); assumption.
Qed.

Lemma r_fold_plus : forall xs1 xs2 a1 a2, V xs1 xs2 -> rel a1 a2 ->
  rel (List.fold_left (op_plus A1) xs1 a1) (List.fold_left (op_plus A2) xs2 a2).
Proof.
  intros xs1 xs2 a1 a2 H. revert a1 a2.
  induction H as [|x1 x2 xs1 xs2 Hx H IH]; intros a1 a2 Ha; cbn [List.fold_left];
    [exact Ha|].
  apply IH. apply (rel_plus HR); assumption.
Qed.

Lemma r_sum : forall xs1 xs2, V xs1 xs2 -> rel (g_sum A1 xs1) (g_sum A2 xs2).
Proof.
  intros. unfold g_sum. apply r_fold_plus; [assumption | apply (rel_const HR)].
Qed.

Ltac rel_ops :=
  repeat first
    [ apply r_minus
    | apply (rel_plus HR) | apply (rel_mult HR) | apply (rel_div HR)
    | apply (rel_neg HR) | apply (rel_sqrt HR) | apply (rel_max HR)
    | apply (rel_exp HR) | apply (rel_const HR) | assumption ].

Lemma r_sigmoid : forall x1 x2, rel x1 x2 -> rel (g_sigmoid A1 x1) (g_sigmoid A2 x2).
Proof.
  intros x1 x2 H. unfold g_sigmoid. cbv zeta.
  apply (rel_div HR); [apply (rel_const HR)|].
  apply (rel_plus HR); [apply (rel_const HR)|].
  apply (rel_exp HR). apply (rel_neg HR). exact H.
Qed.

Lemma r_tanh : forall x1 x2, rel x1 x2 -> rel (g_tanh A1 x1) (g_tanh A2 x2).
Proof.
  intros x1 x2 H. unfold g_tanh. cbv zeta.
  apply r_minus; [|apply (rel_const HR)].
  apply (rel_mult HR); [apply (rel_const HR)|].
  apply r_sigmoid. apply (rel_mult HR); [apply (rel_const HR) | exact H].
Qed.

Lemma r_gelu : forall x1 x2, rel x1 x2 -> rel (g_gelu A1 x1) (g_gelu A2 x2).
Proof.
  intros x1 x2 H. unfold g_gelu. cbv zeta.
  apply (rel_mult HR).
  - apply (rel_mult HR); [apply (rel_const HR) | exact H].
  - apply (rel_plus HR); [apply (rel_const HR)|].
    apply r_tanh. rel_ops.
Qed.

Lemma r_gelu_vec : forall v1 v2, V v1 v2 -> V (g_gelu_vec A1 v1) (g_gelu_vec A2 v2).
Proof.
  intros v1 v2 H. unfold g_gelu_vec.
  apply Forall2_map2 with (P := rel); [exact H|]. intros a b Hab. apply r_gelu, Hab.
Qed.

Lemma r_max_vec : forall xs1 xs2, V xs1 xs2 -> rel (g_max_vec A1 xs1) (g_max_vec A2 xs2).
Proof.
  intros xs1 xs2 H. destruct H as [|x1 x2 xs1 xs2 Hx H]; unfold g_max_vec.
  - apply (rel_const HR).
  - revert x1 x2 Hx.
    induction H as [|y1 y2 ys1 ys2 Hy H IH]; intros x1 x2 Hx; cbn [List.fold_left];
      [exact Hx|].
    apply IH. apply (rel_max HR); assumption.
Qed.

Lemma r_softmax : forall xs1 xs2, V xs1 xs2 -> V (g_softmax A1 xs1) (g_softmax A2 xs2).
Proof.
  intros xs1 xs2 H. unfold g_softmax. cbv zeta.
  pose proof (r_max_vec _ _ H) as Hm.
  assert (He : V (List.map (op_exp A1) (List.map (fun x => g_minus A1 x (g_max_vec A1 xs1)) xs1))
                 (List.map (op_exp A2) (List.map (fun x => g_minus A2 x (g_max_vec A2 xs2)) xs2))).
  { apply Forall2_map2 with (P := rel); [| intros a b Hab; apply (rel_exp HR), Hab].
    apply Forall2_map2 with (P := rel); [exact H|].
    intros a b Hab. apply r_minus; assumption. }
  apply Forall2_map2 with (P := rel); [exact He|].
  intros a b Hab. apply (rel_div HR); [exact Hab | apply r_sum, He].
Qed.

Lemma r_mean : forall xs1 xs2, V xs1 xs2 -> rel (g_mean A1 xs1) (g_mean A2 xs2).
Proof.
  intros xs1 xs2 H. unfold g_mean. cbv zeta.
  rewrite (Forall2_length _ _ _ _ _ H).
  apply (rel_div HR); [apply r_sum, H | apply (rel_const HR)].
Qed.

Lemma r_variance : forall xs1 xs2 m1 m2, V xs1 xs2 -> rel m1 m2 ->
  rel (g_variance A1 xs1 m1) (g_variance A2 xs2 m2).
Proof.
  intros xs1 xs2 m1 m2 H Hm. unfold g_variance. cbv zeta.
  rewrite (Forall2_length _ _ _ _ _ H).
  apply (rel_div HR); [|apply (rel_const HR)].
  apply r_sum. apply Forall2_map2 with (P := rel); [exact H|].
  intros a b Hab. apply (rel_mult HR); apply r_minus; assumption.
Qed.

Lemma r_layer_norm_vec : forall g1 g2 b1 b2 e1 e2 x1 x2,
  V g1 g2 -> V b1 b2 -> rel e1 e2 -> V x1 x2 ->
  V (g_layer_norm_vec A1 g1 b1 e1 x1) (g_layer_norm_vec A2 g2 b2 e2 x2).
Proof.
  intros g1 g2 b1 b2 e1 e2 x1 x2 Hg Hb He Hx. unfold g_layer_norm_vec. cbv zeta.
  pose proof (r_mean _ _ Hx) as Hmu.
  pose proof (r_variance _ _ _ _ Hx Hmu) as Hvar.
  pose proof (Forall2_combine_seq _ _ rel x1 x2 0 Hx) as Hc.
  apply Forall2_map2 with (P := fun p q => fst p = fst q /\ rel (snd p) (snd q)); [exact Hc|].
  intros [i1 a1] [i2 a2] [Hi Ha]. cbn [fst snd] in Hi, Ha. subst i2. cbv beta iota.
  apply (rel_plus HR).
  - apply (rel_mult HR).
    + apply Forall2_nth; [exact Hg | apply (rel_const HR)].
    + apply (rel_div HR); [apply r_minus; assumption|].
      apply (rel_sqrt HR). apply (rel_plus HR); assumption.
  - apply Forall2_nth; [exact Hb | apply (rel_const HR)].
Qed.

Lemma r_layer_norm_2d : forall g1 g2 b1 b2 e1 e2 m1 m2,
  V g1 g2 -> V b1 b2 -> rel e1 e2 -> M m1 m2 ->
  M (g_layer_norm_2d A1 g1 b1 e1 m1) (g_layer_norm_2d A2 g2 b2 e2 m2).
Proof.
  intros g1 g2 b1 b2 e1 e2 m1 m2 Hg Hb He Hm. unfold g_layer_norm_2d.
  apply Forall2_map2 with (P := V); [exact Hm|].
  intros a b Hab. apply r_layer_norm_vec; assumption.
Qed.

Lemma r_mat_transpose : forall m1 m2, M m1 m2 -> M (g_mat_transpose A1 m1) (g_mat_transpose A2 m2).
Proof.
  intros m1 m2 H. destruct H as [|row1 row2 m1' m2' Hrow H]; unfold g_mat_transpose;
    [constructor|].
  rewrite (Forall2_length _ _ _ _ _ Hrow).
  apply Forall2_map_seq. intros i.
  constructor.
  - apply Forall2_nth; [exact Hrow | apply (rel_const HR)].
  - apply Forall2_map2 with (P := V); [exact H|].
    intros a b Hab. apply Forall2_nth; [exact Hab | apply (rel_const HR)].
Qed.

Lemma r_linear_forward : forall w1 w2 b1 b2 x1 x2, M w1 w2 -> V b1 b2 -> V x1 x2 ->
  V (g_linear_forward A1 w1 b1 x1) (g_linear_forward A2 w2 b2 x2).
Proof.
  intros w1 w2 b1 b2 x1 x2 Hw Hb Hx. unfold g_linear_forward.
  apply r_vec_add; [apply r_mat_vec_mul; [apply r_mat_transpose, Hw | exact Hx] | exact Hb].
Qed.

Lemma r_linear_forward_2d : forall w1 w2 b1 b2 x1 x2, M w1 w2 -> V b1 b2 -> M x1 x2 ->
  M (g_linear_forward_2d A1 w1 b1 x1) (g_linear_forward_2d A2 w2 b2 x2).
Proof.
  intros w1 w2 b1 b2 x1 x2 Hw Hb Hx. unfold g_linear_forward_2d.
  apply Forall2_map2 with (P := V); [exact Hx|].
  intros a b Hab. apply r_linear_forward; assumption.
Qed.

Lemma r_add_matrices : forall a1 a2 b1 b2, M a1 a2 -> M b1 b2 ->
  M (g_add_matrices A1 a1 b1) (g_add_matrices A2 a2 b2).
Proof.
  intros a1 a2 b1 b2 Ha Hb. unfold g_add_matrices.
  apply Forall2_map_combine with (P := V) (Q := V); [exact Ha | exact Hb |].
  intros a d b e Had Hbe. cbv beta iota. apply r_vec_add; assumption.
Qed.

Lemma r_split_row : forall nh row1 row2, V row1 row2 ->
  Forall2 V (g_split_row_into_heads nh row1) (g_split_row_into_heads nh row2).
Proof.
  intros nh row1 row2 H. unfold g_split_row_into_heads. cbv zeta.
  rewrite (Forall2_length _ _ _ _ _ H).
  apply Forall2_map_seq. intros i. apply Forall2_firstn, Forall2_skipn, H.
Qed.

Lemma r_split_heads : forall nh m1 m2, M m1 m2 ->
  Forall2 M (g_split_into_heads nh m1) (g_split_into_heads nh m2).
Proof.
  intros nh m1 m2 H. unfold g_split_into_heads. cbv zeta.
  apply Forall2_map_seq. intros i.
  apply Forall2_map2 with (P := Forall2 V).
  - apply Forall2_map2 with (P := V); [exact H|]. intros a b Hab. apply r_split_row, Hab.
  - intros a b Hab. apply Forall2_nth; [exact Hab | constructor].
Qed.

Lemma r_concat_heads : forall hs1 hs2, Forall2 M hs1 hs2 ->
  M (g_concat_heads hs1) (g_concat_heads hs2).
Proof.
  intros hs1 hs2 H. unfold g_concat_heads.
  destruct H as [|h1 h2 hs1' hs2' Hh H]; [constructor|].
  rewrite (Forall2_length _ _ _ _ _ Hh).
  apply Forall2_map_seq. intros i.
  apply Forall2_concat.
  constructor.
  - apply Forall2_nth; [exact Hh | constructor].
  - apply Forall2_map2 with (P := M); [exact H|].
    intros a b Hab. apply Forall2_nth; [exact Hab | constructor].
Qed.

Lemma r_attn_scale : forall d, rel (g_attn_scale A1 d) (g_attn_scale A2 d).
Proof.
  intros d. unfold g_attn_scale.
  apply (rel_div HR); [apply (rel_const HR)|].
  apply (rel_sqrt HR), (rel_const HR).
Qed.

Lemma r_attend : forall q1 q2 ks1 ks2 vs1 vs2 d, V q1 q2 -> M ks1 ks2 -> M vs1 vs2 ->
  V (g_attend A1 q1 ks1 vs1 d) (g_attend A2 q2 ks2 vs2 d).
Proof.
  intros q1 q2 ks1 ks2 vs1 vs2 d Hq Hk Hv. unfold g_attend. cbv zeta.
  assert (Hs : V (List.map (fun kj => op_mult A1 (op_dot A1 q1 kj) (g_attn_scale A1 d)) ks1)
                 (List.map (fun kj => op_mult A2 (op_dot A2 q2 kj) (g_attn_scale A2 d)) ks2)).
  { apply Forall2_map2 with (P := V); [exact Hk|]. intros a b Hab.
    apply (rel_mult HR); [apply (rel_dot HR); assumption | apply r_attn_scale]. }
  pose proof (r_softmax _ _ Hs) as Hw.
  apply Forall2_map2 with (P := V); [apply r_mat_transpose, Hv|].
  intros a b Hab. apply (rel_dot HR); assumption.
Qed.

Lemma r_causal_attention : forall q1 q2 k1 k2 v1 v2 d, M q1 q2 -> M k1 k2 -> M v1 v2 ->
  M (g_causal_attention A1 q1 k1 v1 d) (g_causal_attention A2 q2 k2 v2 d).
Proof.
  intros q1 q2 k1 k2 v1 v2 d Hq Hk Hv. unfold g_causal_attention.
  pose proof (Forall2_combine_seq _ _ V q1 q2 0 Hq) as Hc.
  apply Forall2_map2 with (P := fun p q => fst p = fst q /\ V (snd p) (snd q)); [exact Hc|].
  intros [i1 a1] [i2 a2] [Hi Ha]. cbn [fst snd] in Hi, Ha. subst i2. cbv beta iota.
  apply r_attend; [exact Ha | apply Forall2_firstn, Hk | apply Forall2_firstn, Hv].
Qed.

Lemma r_attention_forward : forall ne nh caw1 caw2 cab1 cab2 cpw1 cpw2 cpb1 cpb2 h1 h2,
  M caw1 caw2 -> V cab1 cab2 -> M cpw1 cpw2 -> V cpb1 cpb2 -> M h1 h2 ->
  M (g_attention_forward A1 ne nh caw1 cab1 cpw1 cpb1 h1)
    (g_attention_forward A2 ne nh caw2 cab2 cpw2 cpb2 h2).
Proof.
  intros ne nh caw1 caw2 cab1 cab2 cpw1 cpw2 cpb1 cpb2 h1 h2 Hcaw Hcab Hcpw Hcpb Hh.
  unfold g_attention_forward. cbv zeta.
  pose proof (r_linear_forward_2d _ _ _ _ _ _ Hcaw Hcab Hh) as Hqkv.
  rewrite !List.map_map.
  apply r_linear_forward_2d; [exact Hcpw | exact Hcpb |].
  apply r_concat_heads.
  apply Forall2_map_combine
    with (P := M) (Q := fun p q => M (fst p) (fst q) /\ M (snd p) (snd q)).
  - apply r_split_heads. apply Forall2_map2 with (P := V); [exact Hqkv|].
    intros a b Hab. cbv beta iota. apply Forall2_firstn, Hab.
  - apply Forall2_combine; apply r_split_heads;
      apply Forall2_map2 with (P := V); try exact Hqkv; intros a b Hab; cbv beta iota.
    + apply Forall2_firstn, Forall2_skipn, Hab.
    + apply Forall2_skipn, Hab.
  - intros a d [b1 c1] [b2 c2] Had [Hb Hc]. cbn [fst snd] in Hb, Hc. cbv beta iota.
    apply r_causal_attention; assumption.
Qed.

Lemma r_mlp_forward : forall cfw1 cfw2 cfb1 cfb2 cpw1 cpw2 cpb1 cpb2 h1 h2,
  M cfw1 cfw2 -> V cfb1 cfb2 -> M cpw1 cpw2 -> V cpb1 cpb2 -> M h1 h2 ->
  M (g_mlp_forward A1 cfw1 cfb1 cpw1 cpb1 h1) (g_mlp_forward A2 cfw2 cfb2 cpw2 cpb2 h2).
Proof.
  intros cfw1 cfw2 cfb1 cfb2 cpw1 cpw2 cpb1 cpb2 h1 h2 Hfw Hfb Hpw Hpb Hh.
  unfold g_mlp_forward. cbv zeta.
  apply r_linear_forward_2d; [exact Hpw | exact Hpb |].
  apply Forall2_map2 with (P := V); [apply r_linear_forward_2d; assumption|].
  intros a b Hab. apply r_gelu_vec, Hab.
Qed.

Definition block_rel (b1 : gblock T1) (b2 : gblock T2) : Prop :=
  V (gb_ln1w b1) (gb_ln1w b2) /\ V (gb_ln1b b1) (gb_ln1b b2)
  /\ M (gb_caw b1) (gb_caw b2) /\ V (gb_cab b1) (gb_cab b2)
  /\ M (gb_cpw b1) (gb_cpw b2) /\ V (gb_cpb b1) (gb_cpb b2)
  /\ V (gb_ln2w b1) (gb_ln2w b2) /\ V (gb_ln2b b1) (gb_ln2b b2)
  /\ M (gb_fcw b1) (gb_fcw b2) /\ V (gb_fcb b1) (gb_fcb b2)
  /\ M (gb_mpw b1) (gb_mpw b2) /\ V (gb_mpb b1) (gb_mpb b2).

Definition model_rel (m1 : gmodel T1) (m2 : gmodel T2) : Prop :=
  M (gm_wte m1) (gm_wte m2) /\ M (gm_wpe m1) (gm_wpe m2)
  /\ Forall2 block_rel (gm_blocks m1) (gm_blocks m2)
  /\ V (gm_lnfw m1) (gm_lnfw m2) /\ V (gm_lnfb m1) (gm_lnfb m2).

Lemma r_block_forward : forall cfg e1 e2 b1 b2 h1 h2,
  rel e1 e2 -> block_rel b1 b2 -> M h1 h2 ->
  M (g_block_forward A1 cfg e1 b1 h1) (g_block_forward A2 cfg e2 b2 h2).
Proof.
  intros cfg e1 e2 b1 b2 h1 h2 He Hb Hh.
  destruct Hb as (H1 & H2 & H3 & H4 & H5 & H6 & H7 & H8 & H9 & H10 & H11 & H12).
  unfold g_block_forward. cbv zeta.
  assert (Hat : M (g_add_matrices A1 h1
                     (g_attention_forward A1 (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)
                        (gb_caw b1) (gb_cab b1) (gb_cpw b1) (gb_cpb b1)
                        (g_layer_norm_2d A1 (gb_ln1w b1) (gb_ln1b b1) e1 h1)))
                  (g_add_matrices A2 h2
                     (g_attention_forward A2 (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)
                        (gb_caw b2) (gb_cab b2) (gb_cpw b2) (gb_cpb b2)
                        (g_layer_norm_2d A2 (gb_ln1w b2) (gb_ln1b b2) e2 h2)))).
  { apply r_add_matrices; [exact Hh|].
    apply r_attention_forward; try assumption.
    apply r_layer_norm_2d; assumption. }
  apply r_add_matrices; [exact Hat|].
  apply r_mlp_forward; try assumption.
  apply r_layer_norm_2d; assumption.
Qed.

Lemma r_blocks_forward : forall cfg e1 e2 bs1 bs2 h1 h2,
  rel e1 e2 -> Forall2 block_rel bs1 bs2 -> M h1 h2 ->
  M (g_blocks_forward A1 cfg e1 bs1 h1) (g_blocks_forward A2 cfg e2 bs2 h2).
Proof.
  intros cfg e1 e2 bs1 bs2 h1 h2 He Hbs. revert h1 h2.
  induction Hbs as [|b1 b2 bs1 bs2 Hb Hbs IH]; intros h1 h2 Hh;
    cbn [g_blocks_forward]; [exact Hh|].
  apply IH. apply r_block_forward; assumption.
Qed.

Lemma r_gpt2_logits : forall cfg e1 e2 m1 m2 toks,
  rel e1 e2 -> model_rel m1 m2 ->
  M (g_gpt2_logits A1 cfg e1 m1 toks) (g_gpt2_logits A2 cfg e2 m2 toks).
Proof.
  intros cfg e1 e2 m1 m2 toks He (Hwte & Hwpe & Hbs & Hlw & Hlb).
  unfold g_gpt2_logits, g_gpt2_forward. cbv zeta.
  apply Forall2_map2 with (P := V).
  - apply r_layer_norm_2d; try assumption.
    apply r_blocks_forward; try assumption.
    apply r_add_matrices.
    + unfold g_embed_tokens. apply Forall2_map_same. intros t.
      apply Forall2_nth; [exact Hwte | constructor].
    + unfold g_embed_positions. apply Forall2_map_same. intros t.
      apply Forall2_nth; [exact Hwpe | constructor].
  - intros a b Hab. apply Forall2_map2 with (P := V); [exact Hwte|].
    intros c d Hcd. apply (rel_dot HR); assumption.
Qed.

End Relational.

(** * The two relations the annotated arithmetic preserves *)

Definition fstrel (a : ann) (x : binary32) : Prop := fst a = x.

Lemma ann_dot_aux_fst : forall xs1 xs2 ys1 ys2 a1 a2,
  Forall2 fstrel xs1 xs2 -> Forall2 fstrel ys1 ys2 -> fstrel a1 a2 ->
  fstrel (ann_dot_aux xs1 ys1 a1) (f32_dot_aux xs2 ys2 a2).
Proof.
  intros xs1 xs2 ys1 ys2 a1 a2 Hx. revert ys1 ys2 a1 a2.
  induction Hx as [|x1 x2 xs1 xs2 Hx0 Hx IH]; intros ys1 ys2 a1 a2 Hy Ha.
  - destruct Hy; exact Ha.
  - destruct Hy as [|y1 y2 ys1 ys2 Hy0 Hy]; [exact Ha|].
    cbn [ann_dot_aux f32_dot_aux]. apply IH; [exact Hy|].
    unfold fstrel in *. subst. reflexivity.
Qed.

Lemma fst_ops_rel : ops_rel ann_ops f32_ops fstrel.
Proof.
  constructor; cbn [op_const op_plus op_neg op_mult op_div op_sqrt op_max op_exp
                    op_dot ann_ops f32_ops];
    unfold fstrel; intros; subst.
  1-7: reflexivity.
  - apply ann_exp_fst.
  - unfold ann_dot, f32_dot. apply ann_dot_aux_fst; [assumption | assumption | reflexivity].
Qed.

Lemma ann_dot_aux_aok : forall xs1 xs2 ys1 ys2 a1 a2,
  Forall2 aok xs1 xs2 -> Forall2 aok ys1 ys2 -> aok a1 a2 ->
  aok (ann_dot_aux xs1 ys1 a1) (Rdotr xs2 ys2 a2).
Proof.
  intros xs1 xs2 ys1 ys2 a1 a2 Hx. revert ys1 ys2 a1 a2.
  induction Hx as [|x1 x2 xs1 xs2 Hx0 Hx IH]; intros ys1 ys2 a1 a2 Hy Ha.
  - destruct Hy; exact Ha.
  - destruct Hy as [|y1 y2 ys1 ys2 Hy0 Hy]; [exact Ha|].
    cbn [ann_dot_aux Rdotr]. apply IH; [exact Hy|].
    apply aok_plus; [exact Ha | apply aok_mult; assumption].
Qed.

Lemma aok_ops_rel : ops_rel ann_ops real_ops aok.
Proof.
  constructor; cbn [op_const op_plus op_neg op_mult op_div op_sqrt op_max op_exp
                    op_dot ann_ops real_ops].
  - apply aok_const.
  - intros. apply aok_plus; assumption.
  - intros. apply aok_neg; assumption.
  - intros. apply aok_mult; assumption.
  - intros. apply aok_div; assumption.
  - intros. apply aok_sqrt; assumption.
  - intros. apply aok_max; assumption.
  - intros. apply aok_exp; assumption.
  - intros. unfold ann_dot, Rdot. apply ann_dot_aux_aok; try assumption.
    apply (aok_const f32_zero).
Qed.

(** * The weights, related *)

Lemma Forall2_map_l : forall (A B : Type) (P : A -> B -> Prop) (f : B -> A) l,
  (forall x, P (f x) x) -> Forall2 P (List.map f l) l.
Proof.
  intros A B P f l H. induction l as [|x l IH]; cbn [List.map]; constructor; auto.
Qed.

Lemma model_rel_to : forall (T : Type) (rel : T -> binary32 -> Prop) (f : binary32 -> T) m,
  (forall x, rel (f x) x) -> model_rel rel (gmodel_map f (to_gmodel m)) (to_gmodel m).
Proof.
  intros T rel f m H. unfold model_rel, gmodel_map, to_gmodel.
  cbn [gm_wte gm_wpe gm_blocks gm_lnfw gm_lnfb].
  assert (HV : forall l, Forall2 rel (List.map f l) l) by (intros; apply Forall2_map_l, H).
  assert (HM : forall l, Forall2 (Forall2 rel) (List.map (List.map f) l) l)
    by (intros; apply Forall2_map_l; intros; apply HV).
  repeat split; try apply HV; try apply HM.
  rewrite List.map_map. apply Forall2_map_same. intros b.
  unfold block_rel, gblock_map, to_gblock.
  cbn [gb_ln1w gb_ln1b gb_caw gb_cab gb_cpw gb_cpb gb_ln2w gb_ln2b gb_fcw gb_fcb gb_mpw gb_mpb].
  repeat split; try apply HV; try apply HM.
Qed.

Lemma model_rel_map2 : forall (T0 T1 T2 : Type) (rel : T1 -> T2 -> Prop)
                              (f : T0 -> T1) (g : T0 -> T2) m,
  (forall x, rel (f x) (g x)) -> model_rel rel (gmodel_map f m) (gmodel_map g m).
Proof.
  intros T0 T1 T2 rel f g m H. unfold model_rel, gmodel_map.
  cbn [gm_wte gm_wpe gm_blocks gm_lnfw gm_lnfb].
  assert (HV : forall l, Forall2 rel (List.map f l) (List.map g l))
    by (intros; apply Forall2_map_same, H).
  assert (HM : forall l, Forall2 (Forall2 rel) (List.map (List.map f) l) (List.map (List.map g) l))
    by (intros; apply Forall2_map_same; intros; apply HV).
  repeat split; try apply HV; try apply HM.
  apply Forall2_map_same. intros b.
  unfold block_rel, gblock_map.
  cbn [gb_ln1w gb_ln1b gb_caw gb_cab gb_cpw gb_cpb gb_ln2w gb_ln2b gb_fcw gb_fcb gb_mpw gb_mpb].
  repeat split; try apply HV; try apply HM.
Qed.

(** * The bound *)

(** Running the annotated pass computes the model's logits, and alongside each
    a bound on its distance from the logit exact real arithmetic produces from
    the same weights and tokens, with every exponential the true exponential of
    its argument saturated to [[-88, 88]]. *)
Theorem gpt2_logits_bounded : forall cfg eps model toks,
  Forall2 (Forall2 fstrel)
    (g_gpt2_logits ann_ops cfg (ann_const eps) (ann_model model) toks)
    (f32_gpt2_logits cfg eps model toks)
  /\ Forall2 (Forall2 aok)
    (g_gpt2_logits ann_ops cfg (ann_const eps) (ann_model model) toks)
    (g_gpt2_logits real_ops cfg (B2R eps) (real_model model) toks).
Proof.
  intros cfg eps model toks. split.
  - rewrite <- g_gpt2_logits_f32.
    apply (r_gpt2_logits ann_ops f32_ops fstrel fst_ops_rel).
    + reflexivity.
    + apply model_rel_to. intros x. reflexivity.
  - apply (r_gpt2_logits ann_ops real_ops aok aok_ops_rel).
    + apply aok_const.
    + apply model_rel_map2. intros x. apply aok_const.
Qed.

(** The same, entry by entry. *)
Corollary gpt2_logit_bounded : forall cfg eps model toks i j,
  let a := List.nth j (List.nth i
             (g_gpt2_logits ann_ops cfg (ann_const eps) (ann_model model) toks) [])
             (ann_const f32_zero) in
  is_finite (snd a) = true ->
  List.nth j (List.nth i (f32_gpt2_logits cfg eps model toks) []) f32_zero = fst a
  /\ Rabs (B2R (fst a)
           - List.nth j (List.nth i
               (g_gpt2_logits real_ops cfg (B2R eps) (real_model model) toks) []) 0)
     <= B2R (snd a).
Proof.
  intros cfg eps model toks i j a Hf.
  destruct (gpt2_logits_bounded cfg eps model toks) as [H1 H2].
  pose proof (Forall2_nth _ _ (Forall2 fstrel) _ _ i [] [] H1 (Forall2_nil _)) as R1.
  pose proof (Forall2_nth _ _ fstrel _ _ j (ann_const f32_zero) f32_zero R1 eq_refl) as E1.
  pose proof (Forall2_nth _ _ (Forall2 aok) _ _ i [] [] H2 (Forall2_nil _)) as R2.
  pose proof (Forall2_nth _ _ aok _ _ j (ann_const f32_zero) 0 R2 (aok_const f32_zero)) as E2.
  fold a in E1, E2.
  split; [symmetry; exact E1|].
  destruct (E2 Hf) as [_ Hb]. exact Hb.
Qed.
