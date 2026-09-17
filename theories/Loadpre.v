(** * Loading a checkpoint through a byte reader

    Loader.v characterises [f32_load_named], which takes the data section as a
    [list byte]. The loaders here take a [reader] instead: the byte at an
    offset, which the caller supplies as a fetch against whatever buffer it
    holds, and which therefore runs on a file of any size.

    Every decode below is written over a reader, and the theorems say each
    returns what the corresponding [f32_load_named] returns, so the model
    [f32_load_model_pre] assembles is the model [f32_load_model] defines,
    transposed as Pretransposed.v's forward pass expects. The offset lookup,
    the decode and the record assembly are therefore extracted code, and a
    runner reads the file and answers the fetch. *)

From Stdlib Require Import List.
From Stdlib Require Import ZArith.
From Stdlib Require Import String.
From Stdlib Require Import Lia.
From Stdlib Require Import PeanoNat.
From Flocq Require Import IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete.
Require Import Loader.
Require Import Runner.
Require Import Llama.
Require Import Qwen.
Require Import Pretransposed.

Import ListNotations.

(** * Reading *)

Definition reader := nat -> byte.

(** The reader a data section induces, which is the one the theorems compare
    against. A runner passes a fetch against its own buffer instead. *)
Definition reader_of (data : list byte) : reader := fun o => List.nth o data 0%Z.

(** [List.map f (List.seq 0 n)], built from the end so that the recursion is on
    the accumulator. A token-embedding matrix has a quarter of a million rows,
    and the extracted [List.map] would want a stack frame for each. *)
Fixpoint map_down {B : Type} (f : nat -> B) (k : nat) (acc : list B) : list B :=
  match k with
  | O => acc
  | S k' => map_down f k' (f k' :: acc)
  end.

Definition map_seq {B : Type} (f : nat -> B) (n : nat) : list B :=
  map_down f n [].

Lemma map_down_spec : forall {B : Type} (f : nat -> B) k acc,
  map_down f k acc = List.map f (List.seq 0 k) ++ acc.
Proof.
  intros B f k. induction k as [|k IH]; intros acc; [reflexivity|].
  cbn [map_down]. rewrite IH, List.seq_S, List.map_app.
  cbn [List.map]. rewrite <- List.app_assoc. reflexivity.
Qed.

Lemma map_seq_spec : forall {B : Type} (f : nat -> B) n,
  map_seq f n = List.map f (List.seq 0 n).
Proof.
  intros B f n. unfold map_seq. rewrite map_down_spec.
  apply List.app_nil_r.
Qed.

Definition f32_read1 (rd : reader) (o : nat) : binary32 :=
  f32_bytes_to_binary32 [rd o; rd (S o); rd (S (S o)); rd (S (S (S o)))].

Definition f32_read_vec (rd : reader) (start count : nat) : list binary32 :=
  map_seq (fun i => f32_read1 rd (start + 4 * i)) count.

Definition f32_read_rows (rd : reader) (start rows cols : nat)
                         : list (list binary32) :=
  map_seq (fun r => f32_read_vec rd (start + 4 * (r * cols)) cols) rows.

(** The stored layout of a GPT-2 weight matrix is [in, out] row-major, so this
    reads its transpose without forming it. *)
Definition f32_read_transposed (rd : reader) (start rows cols : nat)
                               : list (list binary32) :=
  map_seq (fun o => map_seq (fun i => f32_read1 rd (start + 4 * (i * cols + o)))
                            rows)
          cols.

Lemma f32_read_vec_map : forall rd start count,
  f32_read_vec rd start count
  = List.map (fun i => f32_read1 rd (start + 4 * i)) (List.seq 0 count).
Proof. intros. apply map_seq_spec. Qed.

Lemma f32_read_rows_map : forall rd start rows cols,
  f32_read_rows rd start rows cols
  = List.map (fun r => f32_read_vec rd (start + 4 * (r * cols)) cols)
             (List.seq 0 rows).
Proof. intros. apply map_seq_spec. Qed.

Lemma f32_read_transposed_map : forall rd start rows cols,
  f32_read_transposed rd start rows cols
  = List.map (fun o => List.map (fun i => f32_read1 rd (start + 4 * (i * cols + o)))
                                (List.seq 0 rows))
             (List.seq 0 cols).
Proof.
  intros. unfold f32_read_transposed. rewrite map_seq_spec.
  apply List.map_ext. intros o. apply map_seq_spec.
Qed.

(** * What a read returns *)

Lemma list4_eq : forall {A : Type} (l : list A) (d : A),
  List.length l = 4%nat ->
  l = [List.nth 0 l d; List.nth 1 l d; List.nth 2 l d; List.nth 3 l d].
Proof.
  intros A l d H.
  destruct l as [|a l]; [discriminate|]. destruct l as [|b l]; [discriminate|].
  destruct l as [|c l]; [discriminate|]. destruct l as [|e l]; [discriminate|].
  destruct l as [|f l]; [reflexivity|discriminate].
Qed.

Lemma f32_read1_spec : forall data o,
  (o + 4 <= List.length data)%nat ->
  f32_read1 (reader_of data) o = f32_bytes_to_binary32 (take 4 (drop o data)).
Proof.
  intros data o Ho. unfold f32_read1, reader_of.
  assert (Hlen : List.length (take 4 (drop o data)) = 4%nat).
  { unfold take, drop. rewrite List.length_firstn, List.length_skipn. lia. }
  rewrite (list4_eq (take 4 (drop o data)) 0%Z Hlen).
  unfold take, drop.
  rewrite !(nth_firstn 0%Z) by lia.
  rewrite !(nth_skipn 0%Z).
  repeat f_equal; lia.
Qed.

Lemma nth_map_seq0 : forall {B : Type} (f : nat -> B) n i (d : B),
  (i < n)%nat -> List.nth i (List.map f (List.seq 0 n)) d = f i.
Proof.
  intros B f n i d Hi.
  rewrite (List.nth_indep _ d (f 0%nat))
    by (rewrite List.length_map, List.length_seq; exact Hi).
  rewrite List.map_nth, List.seq_nth by exact Hi. reflexivity.
Qed.

(** A vector read at a tensor's offset is the named load of that tensor. *)
Theorem f32_read_vec_load_named : forall header data tname a b rest,
  json_tensor_offsets header tname = a :: b :: rest ->
  (a + 4 * ((b - a) / 4) <= List.length data)%nat ->
  f32_read_vec (reader_of data) a ((b - a) / 4)%nat
  = f32_load_named header data tname.
Proof.
  intros header data tname a b rest Hoff Hlen.
  set (n := ((b - a) / 4)%nat) in *.
  rewrite f32_read_vec_map.
  apply (List.nth_ext _ _ f32_zero f32_zero).
  - rewrite List.length_map, List.length_seq.
    symmetry. exact (f32_load_named_length header data tname a b rest Hoff).
  - intros i Hi.
    rewrite List.length_map, List.length_seq in Hi.
    rewrite nth_map_seq0 by exact Hi.
    rewrite (f32_load_named_nth header data tname a b rest i Hoff Hi).
    apply f32_read1_spec. lia.
Qed.

(** * Reading a checkpoint *)

Definition tensor_start (header : string) (tname : string) : nat :=
  match json_tensor_offsets header tname with
  | a :: _ :: _ => a
  | _ => 0%nat
  end.

Definition f32_read_named_vec (header : string) (rd : reader) (tname : string)
                              (count : nat) : list binary32 :=
  f32_read_vec rd (tensor_start header tname) count.

Definition f32_read_ln (header : string) (rd : reader) (wname bname : string)
                       (d : nat) : f32_layernorm_weights :=
  mk_f32_layernorm_weights
    (f32_read_named_vec header rd wname d)
    (f32_read_named_vec header rd bname d).

Definition f32_read_block (header : string) (rd : reader)
                          (cfg : gpt2_inference_config) (i : nat)
                          : f32_block_weights :=
  let d := gpt2_inf_n_embd cfg in
  let ff := gpt2_inf_n_inner cfg in
  let p := gpt2_block_prefix i in
  mk_f32_block_weights
    (f32_read_ln header rd (p ++ "ln_1.weight") (p ++ "ln_1.bias") d)
    (mk_f32_attention_weights
       (f32_read_transposed rd (tensor_start header (p ++ "attn.c_attn.weight"))
                            d (3 * d))
       (f32_read_named_vec header rd (p ++ "attn.c_attn.bias") (3 * d))
       (f32_read_transposed rd (tensor_start header (p ++ "attn.c_proj.weight"))
                            d d)
       (f32_read_named_vec header rd (p ++ "attn.c_proj.bias") d))
    (f32_read_ln header rd (p ++ "ln_2.weight") (p ++ "ln_2.bias") d)
    (mk_f32_mlp_weights
       (f32_read_transposed rd (tensor_start header (p ++ "mlp.c_fc.weight"))
                            d ff)
       (f32_read_named_vec header rd (p ++ "mlp.c_fc.bias") ff)
       (f32_read_transposed rd (tensor_start header (p ++ "mlp.c_proj.weight"))
                            ff d)
       (f32_read_named_vec header rd (p ++ "mlp.c_proj.bias") d)).

Definition f32_load_model_pre (header : string) (rd : reader)
                              (cfg : gpt2_inference_config) : f32_model_weights :=
  let d := gpt2_inf_n_embd cfg in
  mk_f32_model_weights
    (f32_read_rows rd (tensor_start header "wte.weight")
                   (gpt2_inf_vocab_size cfg) d)
    (f32_read_rows rd (tensor_start header "wpe.weight")
                   (gpt2_inf_n_positions cfg) d)
    (map_seq (f32_read_block header rd cfg) (gpt2_inf_n_layer cfg))
    (f32_read_ln header rd "ln_f.weight" "ln_f.bias" d).

(** * What the reader-based loader returns

    A tensor is loadable when the header gives it two offsets, the span they
    bound holds the expected number of binary32 values, and the data section
    reaches the end of that span. *)

Definition tensor_ok (header : string) (data : list byte) (tname : string)
                     (count : nat) : Prop :=
  exists a b rest,
    json_tensor_offsets header tname = a :: b :: rest
    /\ ((b - a) / 4)%nat = count
    /\ (a + 4 * count <= List.length data)%nat.

Lemma f32_read1_nth_load : forall header data tname count k,
  tensor_ok header data tname count -> (k < count)%nat ->
  f32_read1 (reader_of data) (tensor_start header tname + 4 * k)
  = List.nth k (f32_load_named header data tname) f32_zero.
Proof.
  intros header data tname count k Hok Hk.
  destruct Hok as [a [b [rest [Hoff [Hc Hlen]]]]].
  assert (Hk' : (k < (b - a) / 4)%nat) by (rewrite Hc; exact Hk).
  unfold tensor_start. rewrite Hoff.
  rewrite (f32_load_named_nth header data tname a b rest k Hoff Hk').
  apply f32_read1_spec. lia.
Qed.

Lemma read_vec_ok : forall header data tname count,
  tensor_ok header data tname count ->
  f32_read_named_vec header (reader_of data) tname count
  = f32_load_named header data tname.
Proof.
  intros header data tname count Hok.
  pose proof Hok as Hok'.
  destruct Hok' as [a [b [rest [Hoff [Hc Hlen]]]]].
  unfold f32_read_named_vec, tensor_start. rewrite Hoff.
  rewrite <- Hc.
  apply (f32_read_vec_load_named header data tname a b rest Hoff).
  rewrite Hc. exact Hlen.
Qed.

Lemma load_named_length_ok : forall header data tname count,
  tensor_ok header data tname count ->
  List.length (f32_load_named header data tname) = count.
Proof.
  intros header data tname count [a [b [rest [Hoff [Hc Hlen]]]]].
  rewrite (f32_load_named_length header data tname a b rest Hoff). exact Hc.
Qed.

(** A transposed read is the transpose of the reshape of the named load. *)
Lemma read_transposed_ok : forall header data tname rows cols,
  (0 < rows)%nat -> (0 < cols)%nat ->
  tensor_ok header data tname (rows * cols) ->
  f32_read_transposed (reader_of data) (tensor_start header tname) rows cols
  = f32_mat_transpose
      (f32_reshape_1d_to_2d (f32_load_named header data tname) rows cols).
Proof.
  intros header data tname rows cols Hr Hc Hok.
  rewrite <- (decode_transposed_correct _ rows cols Hr Hc)
    by (rewrite (load_named_length_ok header data tname _ Hok); lia).
  rewrite f32_read_transposed_map. unfold f32_decode_transposed.
  apply List.map_ext_in. intros o Ho.
  apply List.in_seq in Ho. destruct Ho as [_ Ho]. cbn in Ho.
  apply List.map_ext_in. intros i Hi.
  apply List.in_seq in Hi. destruct Hi as [_ Hi]. cbn in Hi.
  apply (f32_read1_nth_load header data tname (rows * cols)); [exact Hok | nia].
Qed.

Lemma reshape_nth_row : forall rows cols vs r,
  (r < rows)%nat ->
  List.nth r (f32_reshape_1d_to_2d vs rows cols) []
  = List.firstn cols (List.skipn (r * cols) vs).
Proof.
  induction rows as [|rows IH]; intros cols vs r Hr; [lia|].
  destruct r as [|r].
  - cbn [f32_reshape_1d_to_2d List.nth]. rewrite Nat.mul_0_l.
    cbn [List.skipn]. reflexivity.
  - cbn [f32_reshape_1d_to_2d List.nth].
    rewrite IH by lia. rewrite List.skipn_skipn, Nat.mul_succ_l.
    reflexivity.
Qed.

(** A row-major read is the reshape of the named load. *)
Lemma read_rows_ok : forall header data tname rows cols,
  tensor_ok header data tname (rows * cols) ->
  f32_read_rows (reader_of data) (tensor_start header tname) rows cols
  = f32_reshape_1d_to_2d (f32_load_named header data tname) rows cols.
Proof.
  intros header data tname rows cols Hok.
  assert (Hvs : List.length (f32_load_named header data tname) = (rows * cols)%nat)
    by (apply (load_named_length_ok header data tname _ Hok)).
  rewrite f32_read_rows_map.
  apply (List.nth_ext _ _ [] []).
  - rewrite List.length_map, List.length_seq.
    symmetry. apply f32_reshape_1d_to_2d_rows.
  - intros r Hr.
    rewrite List.length_map, List.length_seq in Hr.
    rewrite nth_map_seq0 by exact Hr.
    rewrite reshape_nth_row by exact Hr.
    assert (Hspan : ((r + 1) * cols <= rows * cols)%nat)
      by (apply Nat.mul_le_mono_r; lia).
    rewrite f32_read_vec_map.
    apply (List.nth_ext _ _ f32_zero f32_zero).
    + rewrite List.length_map, List.length_seq.
      rewrite List.length_firstn, List.length_skipn, Hvs.
      symmetry. apply Nat.min_l. nia.
    + intros j Hj.
      rewrite List.length_map, List.length_seq in Hj.
      rewrite nth_map_seq0 by exact Hj.
      rewrite (nth_firstn f32_zero) by exact Hj.
      rewrite (nth_skipn f32_zero).
      replace (tensor_start header tname + 4 * (r * cols) + 4 * j)%nat
        with (tensor_start header tname + 4 * (r * cols + j))%nat by lia.
      apply (f32_read1_nth_load header data tname (rows * cols)); [exact Hok | nia].
Qed.

(** * The model the runner assembles is the model the loader defines *)

Definition block_loadable (header : string) (data : list byte)
                          (cfg : gpt2_inference_config) (i : nat) : Prop :=
  let d := gpt2_inf_n_embd cfg in
  let ff := gpt2_inf_n_inner cfg in
  let p := gpt2_block_prefix i in
  tensor_ok header data (p ++ "ln_1.weight") d
  /\ tensor_ok header data (p ++ "ln_1.bias") d
  /\ tensor_ok header data (p ++ "attn.c_attn.weight") (d * (3 * d))
  /\ tensor_ok header data (p ++ "attn.c_attn.bias") (3 * d)
  /\ tensor_ok header data (p ++ "attn.c_proj.weight") (d * d)
  /\ tensor_ok header data (p ++ "attn.c_proj.bias") d
  /\ tensor_ok header data (p ++ "ln_2.weight") d
  /\ tensor_ok header data (p ++ "ln_2.bias") d
  /\ tensor_ok header data (p ++ "mlp.c_fc.weight") (d * ff)
  /\ tensor_ok header data (p ++ "mlp.c_fc.bias") ff
  /\ tensor_ok header data (p ++ "mlp.c_proj.weight") (ff * d)
  /\ tensor_ok header data (p ++ "mlp.c_proj.bias") d.

Definition model_loadable (header : string) (data : list byte)
                          (cfg : gpt2_inference_config) : Prop :=
  let d := gpt2_inf_n_embd cfg in
  (0 < d)%nat /\ (0 < gpt2_inf_n_inner cfg)%nat
  /\ tensor_ok header data "wte.weight" (gpt2_inf_vocab_size cfg * d)
  /\ tensor_ok header data "wpe.weight" (gpt2_inf_n_positions cfg * d)
  /\ tensor_ok header data "ln_f.weight" d
  /\ tensor_ok header data "ln_f.bias" d
  /\ (forall i, (i < gpt2_inf_n_layer cfg)%nat -> block_loadable header data cfg i).

Lemma read_block_ok : forall header data cfg i,
  (0 < gpt2_inf_n_embd cfg)%nat -> (0 < gpt2_inf_n_inner cfg)%nat ->
  block_loadable header data cfg i ->
  f32_read_block header (reader_of data) cfg i
  = transpose_block (f32_load_block header data cfg i).
Proof.
  intros header data cfg i Hd Hff Hb.
  destruct Hb as [H1 [H2 [H3 [H4 [H5 [H6 [H7 [H8 [H9 [H10 [H11 H12]]]]]]]]]]].
  unfold f32_read_block, transpose_block, f32_load_block, f32_make_block,
         transpose_attn, transpose_mlp, f32_make_layernorm, f32_make_attention,
         f32_make_mlp, f32_read_ln.
  cbv zeta.
  rewrite (read_vec_ok _ _ _ _ H1), (read_vec_ok _ _ _ _ H2),
          (read_vec_ok _ _ _ _ H4), (read_vec_ok _ _ _ _ H6),
          (read_vec_ok _ _ _ _ H7), (read_vec_ok _ _ _ _ H8),
          (read_vec_ok _ _ _ _ H10), (read_vec_ok _ _ _ _ H12).
  assert (H3d : (0 < 3 * gpt2_inf_n_embd cfg)%nat) by lia.
  rewrite (read_transposed_ok _ _ _ _ _ Hd H3d H3),
          (read_transposed_ok _ _ _ _ _ Hd Hd H5),
          (read_transposed_ok _ _ _ _ _ Hd Hff H9),
          (read_transposed_ok _ _ _ _ _ Hff Hd H11).
  reflexivity.
Qed.

Theorem f32_load_model_pre_correct : forall header data cfg,
  model_loadable header data cfg ->
  f32_load_model_pre header (reader_of data) cfg
  = transpose_model (f32_load_model header data cfg).
Proof.
  intros header data cfg Hm.
  destruct Hm as [Hd [Hff [Hwte [Hwpe [Hlnw [Hlnb Hblocks]]]]]].
  unfold f32_load_model_pre, transpose_model, f32_load_model, f32_load_blocks,
         f32_make_layernorm, f32_read_ln.
  cbv zeta. cbn [f32_wte f32_wpe f32_blocks f32_ln_f].
  rewrite (read_rows_ok _ _ _ _ _ Hwte), (read_rows_ok _ _ _ _ _ Hwpe),
          (read_vec_ok _ _ _ _ Hlnw), (read_vec_ok _ _ _ _ Hlnb).
  rewrite map_seq_spec, List.map_map.
  f_equal. apply List.map_ext_in. intros i Hi.
  apply List.in_seq in Hi. destruct Hi as [_ Hi]. cbn in Hi.
  apply read_block_ok; [exact Hd | exact Hff | apply Hblocks; exact Hi].
Qed.

(** * Llama

    A Llama checkpoint stores each projection as [out, in] row-major, which is
    the order the forward pass reads, so no transpose is involved. The rotary
    tables are the cosines and sines of position times the stored inverse
    frequencies, in the half-split convention [f32_partial_rope] reads; they are
    written here so the runner computes no trigonometry of its own. *)

Definition layer_prefix (i : nat) : string :=
  ("layers." ++ nat_to_string i ++ ".")%string.

Record llama_layer_weights := mk_llama_layer_weights {
  ll_ln1 : list binary32;
  ll_ln2 : list binary32;
  ll_attn : llama_attn_weights;
  ll_mlp : llama_mlp_weights
}.

Record llama_model_weights := mk_llama_model_weights {
  lw_emb : list (list binary32);
  lw_invf : list binary32;
  lw_norm : list binary32;
  lw_layers : list llama_layer_weights
}.

Definition f32_read_named_rows (header : string) (rd : reader) (tname : string)
                               (rows cols : nat) : list (list binary32) :=
  f32_read_rows rd (tensor_start header tname) rows cols.

Definition f32_read_llama_layer (header : string) (rd : reader)
                                (d nkv hd ff : nat) (i : nat)
                                : llama_layer_weights :=
  let p := layer_prefix i in
  mk_llama_layer_weights
    (f32_read_named_vec header rd (p ++ "input_layernorm.weight") d)
    (f32_read_named_vec header rd (p ++ "post_attention_layernorm.weight") d)
    (mk_llama_attn_weights
       (f32_read_named_rows header rd (p ++ "self_attn.q_proj.weight") d d)
       (f32_read_named_rows header rd (p ++ "self_attn.k_proj.weight") (nkv * hd) d)
       (f32_read_named_rows header rd (p ++ "self_attn.v_proj.weight") (nkv * hd) d)
       (f32_read_named_rows header rd (p ++ "self_attn.o_proj.weight") d d))
    (mk_llama_mlp_weights
       (f32_read_named_rows header rd (p ++ "mlp.gate_proj.weight") ff d)
       (f32_read_named_rows header rd (p ++ "mlp.up_proj.weight") ff d)
       (f32_read_named_rows header rd (p ++ "mlp.down_proj.weight") d ff)).

Definition f32_load_llama (header : string) (rd : reader)
                          (d nl nkv hd ff vocab : nat) : llama_model_weights :=
  mk_llama_model_weights
    (f32_read_named_rows header rd "embed_tokens.weight" vocab d)
    (f32_read_named_vec header rd "rope.inv_freq" (hd / 2))
    (f32_read_named_vec header rd "norm.weight" d)
    (map_seq (f32_read_llama_layer header rd d nkv hd ff) nl).

Definition f32_rope_angles (invf : list binary32) (npos : nat)
                           : list (list binary32) :=
  map_seq (fun pos => List.map (fun fj => f32_mult (f32_of_Z (Z.of_nat pos)) fj) invf)
          npos.

Definition f32_rope_cos (invf : list binary32) (npos : nat)
                        : list (list binary32) :=
  List.map (fun ang => List.map f32_cos ang ++ List.map f32_cos ang)
           (f32_rope_angles invf npos).

Definition f32_rope_sin (invf : list binary32) (npos : nat)
                        : list (list binary32) :=
  List.map (fun ang => List.map f32_sin ang ++ List.map f32_sin ang)
           (f32_rope_angles invf npos).

Definition f32_llama_logits_of (nh nkv hd : nat) (eps : binary32)
                               (m : llama_model_weights) (ids : list nat)
                               : list (list binary32) :=
  let npos := List.length ids in
  let cosv := f32_rope_cos (lw_invf m) npos in
  let sinv := f32_rope_sin (lw_invf m) npos in
  let fs := List.map (fun L => f32_llama_layer nh nkv hd eps (ll_ln1 L) (ll_ln2 L)
                                 (ll_attn L) (ll_mlp L) cosv sinv)
                     (lw_layers m) in
  f32_llama_logits (lw_emb m)
                   (f32_llama_forward eps (lw_norm m) fs (lw_emb m) ids).

(** * Qwen3.5

    Three gated-DeltaNet layers to one gated full-attention layer, so a layer
    carries one mixer's weights or the other's. *)

Inductive qwen_layer_weights :=
  | QAttn : list binary32 -> list binary32 -> qwen_mlp_weights ->
            qwen_attn_weights -> qwen_layer_weights
  | QDelta : list binary32 -> list binary32 -> qwen_mlp_weights ->
             qwen_delta_weights -> qwen_layer_weights.

Record qwen_model_weights := mk_qwen_model_weights {
  qw_emb : list (list binary32);
  qw_invf : list binary32;
  qw_norm : list binary32;
  qw_layers : list qwen_layer_weights
}.

Definition f32_read_qwen_layer (header : string) (rd : reader)
                               (d nh nkv hd rdim ff lnh lhd ck : nat) (i : nat)
                               : qwen_layer_weights :=
  let p := layer_prefix i in
  let kvd := (nkv * hd)%nat in
  let ldim := (lnh * lhd)%nat in
  let qkvd := (3 * ldim)%nat in
  let ln1 := f32_read_named_vec header rd (p ++ "input_layernorm.weight") d in
  let ln2 := f32_read_named_vec header rd (p ++ "post_attention_layernorm.weight") d in
  let mlp := mk_qwen_mlp_weights
               (f32_read_named_rows header rd (p ++ "mlp.gate_proj.weight") ff d)
               (f32_read_named_rows header rd (p ++ "mlp.up_proj.weight") ff d)
               (f32_read_named_rows header rd (p ++ "mlp.down_proj.weight") d ff) in
  if Nat.eqb (Nat.modulo i 4) 3 then
    QAttn ln1 ln2 mlp
      (mk_qwen_attn_weights
         (f32_read_named_rows header rd (p ++ "self_attn.q_proj.weight") (nh * hd * 2) d)
         (f32_read_named_rows header rd (p ++ "self_attn.k_proj.weight") kvd d)
         (f32_read_named_rows header rd (p ++ "self_attn.v_proj.weight") kvd d)
         (f32_read_named_rows header rd (p ++ "self_attn.o_proj.weight") d (nh * hd))
         (f32_read_named_vec header rd (p ++ "self_attn.q_norm.weight") hd)
         (f32_read_named_vec header rd (p ++ "self_attn.k_norm.weight") hd))
  else
    QDelta ln1 ln2 mlp
      (mk_qwen_delta_weights
         (f32_read_named_rows header rd (p ++ "linear_attn.in_proj_qkv.weight") qkvd d)
         (f32_read_named_rows header rd (p ++ "linear_attn.in_proj_z.weight") ldim d)
         (f32_read_named_rows header rd (p ++ "linear_attn.in_proj_a.weight") lnh d)
         (f32_read_named_rows header rd (p ++ "linear_attn.in_proj_b.weight") lnh d)
         (f32_read_named_rows header rd (p ++ "linear_attn.conv1d.weight") qkvd ck)
         (f32_read_named_vec header rd (p ++ "linear_attn.A_log") lnh)
         (f32_read_named_vec header rd (p ++ "linear_attn.dt_bias") lnh)
         (f32_read_named_vec header rd (p ++ "linear_attn.norm.weight") lhd)
         (f32_read_named_rows header rd (p ++ "linear_attn.out_proj.weight") d ldim)).

Definition f32_load_qwen (header : string) (rd : reader)
                         (d nl nh nkv hd rdim ff vocab lnh lhd ck : nat)
                         : qwen_model_weights :=
  mk_qwen_model_weights
    (f32_read_named_rows header rd "embed_tokens.weight" vocab d)
    (f32_read_named_vec header rd "rope.inv_freq" (rdim / 2))
    (f32_read_named_vec header rd "norm.weight" d)
    (map_seq (f32_read_qwen_layer header rd d nh nkv hd rdim ff lnh lhd ck) nl).

Definition f32_qwen_logits_of (nh nkv hd rdim lnh lhd ck : nat) (eps : binary32)
                              (m : qwen_model_weights) (ids : list nat)
                              : list (list binary32) :=
  let npos := List.length ids in
  let cosv := f32_rope_cos (qw_invf m) npos in
  let sinv := f32_rope_sin (qw_invf m) npos in
  let fs := List.map (fun L =>
              match L with
              | QAttn ln1 ln2 mlp aw =>
                  f32_qwen_wrap eps ln1 ln2 mlp
                    (f32_qwen_attn_mix nh nkv hd rdim eps aw cosv sinv)
              | QDelta ln1 ln2 mlp dw =>
                  f32_qwen_wrap eps ln1 ln2 mlp
                    (f32_qwen_delta_mix lnh lhd ck eps dw)
              end) (qw_layers m) in
  f32_qwen_logits (qw_emb m) (f32_qwen_forward eps (qw_norm m) fs (qw_emb m) ids).

(** * Each weight is the named load of its tensor

    The Llama and Qwen3.5 weight records are assembled from reads rather than
    from [f32_load_named], so the two theorems below are what say the records
    hold the tensors the header names. They are the same statement as
    [f32_load_model_pre_correct] and rest on the same two lemmas. *)

Definition named_rows (header : string) (data : list byte) (tname : string)
                      (rows cols : nat) : list (list binary32) :=
  f32_reshape_1d_to_2d (f32_load_named header data tname) rows cols.

Definition llama_layer_loadable (header : string) (data : list byte)
                                (d nkv hd ff : nat) (i : nat) : Prop :=
  let p := layer_prefix i in
  tensor_ok header data (p ++ "input_layernorm.weight") d
  /\ tensor_ok header data (p ++ "post_attention_layernorm.weight") d
  /\ tensor_ok header data (p ++ "self_attn.q_proj.weight") (d * d)
  /\ tensor_ok header data (p ++ "self_attn.k_proj.weight") (nkv * hd * d)
  /\ tensor_ok header data (p ++ "self_attn.v_proj.weight") (nkv * hd * d)
  /\ tensor_ok header data (p ++ "self_attn.o_proj.weight") (d * d)
  /\ tensor_ok header data (p ++ "mlp.gate_proj.weight") (ff * d)
  /\ tensor_ok header data (p ++ "mlp.up_proj.weight") (ff * d)
  /\ tensor_ok header data (p ++ "mlp.down_proj.weight") (d * ff).

Definition llama_layer_named (header : string) (data : list byte)
                             (d nkv hd ff : nat) (i : nat) : llama_layer_weights :=
  let p := layer_prefix i in
  mk_llama_layer_weights
    (f32_load_named header data (p ++ "input_layernorm.weight"))
    (f32_load_named header data (p ++ "post_attention_layernorm.weight"))
    (mk_llama_attn_weights
       (named_rows header data (p ++ "self_attn.q_proj.weight") d d)
       (named_rows header data (p ++ "self_attn.k_proj.weight") (nkv * hd) d)
       (named_rows header data (p ++ "self_attn.v_proj.weight") (nkv * hd) d)
       (named_rows header data (p ++ "self_attn.o_proj.weight") d d))
    (mk_llama_mlp_weights
       (named_rows header data (p ++ "mlp.gate_proj.weight") ff d)
       (named_rows header data (p ++ "mlp.up_proj.weight") ff d)
       (named_rows header data (p ++ "mlp.down_proj.weight") d ff)).

Lemma read_llama_layer_ok : forall header data d nkv hd ff i,
  llama_layer_loadable header data d nkv hd ff i ->
  f32_read_llama_layer header (reader_of data) d nkv hd ff i
  = llama_layer_named header data d nkv hd ff i.
Proof.
  unfold llama_layer_named.
  intros header data d nkv hd ff i Hb.
  destruct Hb as [H1 [H2 [H3 [H4 [H5 [H6 [H7 [H8 H9]]]]]]]].
  unfold f32_read_llama_layer, f32_read_named_rows, named_rows. cbv zeta.
  rewrite (read_vec_ok _ _ _ _ H1), (read_vec_ok _ _ _ _ H2),
          (read_rows_ok _ _ _ _ _ H3), (read_rows_ok _ _ _ _ _ H4),
          (read_rows_ok _ _ _ _ _ H5), (read_rows_ok _ _ _ _ _ H6),
          (read_rows_ok _ _ _ _ _ H7), (read_rows_ok _ _ _ _ _ H8),
          (read_rows_ok _ _ _ _ _ H9).
  reflexivity.
Qed.

Theorem f32_load_llama_correct : forall header data d nl nkv hd ff vocab,
  tensor_ok header data "embed_tokens.weight" (vocab * d) ->
  tensor_ok header data "rope.inv_freq" (hd / 2) ->
  tensor_ok header data "norm.weight" d ->
  (forall i, (i < nl)%nat -> llama_layer_loadable header data d nkv hd ff i) ->
  f32_load_llama header (reader_of data) d nl nkv hd ff vocab
  = mk_llama_model_weights
      (named_rows header data "embed_tokens.weight" vocab d)
      (f32_load_named header data "rope.inv_freq")
      (f32_load_named header data "norm.weight")
      (map_seq (llama_layer_named header data d nkv hd ff) nl).
Proof.
  intros header data d nl nkv hd ff vocab Hemb Hinv Hnorm Hl.
  unfold f32_load_llama, f32_read_named_rows, named_rows.
  rewrite (read_rows_ok _ _ _ _ _ Hemb), (read_vec_ok _ _ _ _ Hinv),
          (read_vec_ok _ _ _ _ Hnorm).
  rewrite !map_seq_spec. f_equal.
  apply List.map_ext_in. intros i Hi.
  apply List.in_seq in Hi. destruct Hi as [_ Hi]. cbn in Hi.
  apply read_llama_layer_ok, Hl, Hi.
Qed.

Definition qwen_layer_loadable (header : string) (data : list byte)
                               (d nh nkv hd ff lnh lhd ck : nat) (i : nat) : Prop :=
  let p := layer_prefix i in
  let kvd := (nkv * hd)%nat in
  let ldim := (lnh * lhd)%nat in
  tensor_ok header data (p ++ "input_layernorm.weight") d
  /\ tensor_ok header data (p ++ "post_attention_layernorm.weight") d
  /\ tensor_ok header data (p ++ "mlp.gate_proj.weight") (ff * d)
  /\ tensor_ok header data (p ++ "mlp.up_proj.weight") (ff * d)
  /\ tensor_ok header data (p ++ "mlp.down_proj.weight") (d * ff)
  /\ (Nat.modulo i 4 = 3%nat ->
      tensor_ok header data (p ++ "self_attn.q_proj.weight") (nh * hd * 2 * d)
      /\ tensor_ok header data (p ++ "self_attn.k_proj.weight") (kvd * d)
      /\ tensor_ok header data (p ++ "self_attn.v_proj.weight") (kvd * d)
      /\ tensor_ok header data (p ++ "self_attn.o_proj.weight") (d * (nh * hd))
      /\ tensor_ok header data (p ++ "self_attn.q_norm.weight") hd
      /\ tensor_ok header data (p ++ "self_attn.k_norm.weight") hd)
  /\ (Nat.modulo i 4 <> 3%nat ->
      tensor_ok header data (p ++ "linear_attn.in_proj_qkv.weight") (3 * ldim * d)
      /\ tensor_ok header data (p ++ "linear_attn.in_proj_z.weight") (ldim * d)
      /\ tensor_ok header data (p ++ "linear_attn.in_proj_a.weight") (lnh * d)
      /\ tensor_ok header data (p ++ "linear_attn.in_proj_b.weight") (lnh * d)
      /\ tensor_ok header data (p ++ "linear_attn.conv1d.weight") (3 * ldim * ck)
      /\ tensor_ok header data (p ++ "linear_attn.A_log") lnh
      /\ tensor_ok header data (p ++ "linear_attn.dt_bias") lnh
      /\ tensor_ok header data (p ++ "linear_attn.norm.weight") lhd
      /\ tensor_ok header data (p ++ "linear_attn.out_proj.weight") (d * ldim)).

Definition qwen_layer_named (header : string) (data : list byte)
                            (d nh nkv hd ff lnh lhd ck : nat) (i : nat)
                            : qwen_layer_weights :=
  let p := layer_prefix i in
    let ln1 := f32_load_named header data (p ++ "input_layernorm.weight") in
    let ln2 := f32_load_named header data (p ++ "post_attention_layernorm.weight") in
    let mlp := mk_qwen_mlp_weights
                 (named_rows header data (p ++ "mlp.gate_proj.weight") ff d)
                 (named_rows header data (p ++ "mlp.up_proj.weight") ff d)
                 (named_rows header data (p ++ "mlp.down_proj.weight") d ff) in
    if Nat.eqb (Nat.modulo i 4) 3 then
      QAttn ln1 ln2 mlp
        (mk_qwen_attn_weights
           (named_rows header data (p ++ "self_attn.q_proj.weight") (nh * hd * 2) d)
           (named_rows header data (p ++ "self_attn.k_proj.weight") (nkv * hd) d)
           (named_rows header data (p ++ "self_attn.v_proj.weight") (nkv * hd) d)
           (named_rows header data (p ++ "self_attn.o_proj.weight") d (nh * hd))
           (f32_load_named header data (p ++ "self_attn.q_norm.weight"))
           (f32_load_named header data (p ++ "self_attn.k_norm.weight")))
    else
      QDelta ln1 ln2 mlp
        (mk_qwen_delta_weights
           (named_rows header data (p ++ "linear_attn.in_proj_qkv.weight") (3 * (lnh * lhd)) d)
           (named_rows header data (p ++ "linear_attn.in_proj_z.weight") (lnh * lhd) d)
           (named_rows header data (p ++ "linear_attn.in_proj_a.weight") lnh d)
           (named_rows header data (p ++ "linear_attn.in_proj_b.weight") lnh d)
           (named_rows header data (p ++ "linear_attn.conv1d.weight") (3 * (lnh * lhd)) ck)
           (f32_load_named header data (p ++ "linear_attn.A_log"))
           (f32_load_named header data (p ++ "linear_attn.dt_bias"))
           (f32_load_named header data (p ++ "linear_attn.norm.weight"))
           (named_rows header data (p ++ "linear_attn.out_proj.weight") d (lnh * lhd))).

Lemma read_qwen_layer_ok : forall header data d nh nkv hd rdim ff lnh lhd ck i,
  qwen_layer_loadable header data d nh nkv hd ff lnh lhd ck i ->
  f32_read_qwen_layer header (reader_of data) d nh nkv hd rdim ff lnh lhd ck i
  = qwen_layer_named header data d nh nkv hd ff lnh lhd ck i.
Proof.
  unfold qwen_layer_named.
  intros header data d nh nkv hd rdim ff lnh lhd ck i Hl.
  destruct Hl as [H1 [H2 [H3 [H4 [H5 [Hattn Hdelta]]]]]].
  unfold f32_read_qwen_layer, f32_read_named_rows, named_rows. cbv zeta.
  rewrite (read_vec_ok _ _ _ _ H1), (read_vec_ok _ _ _ _ H2),
          (read_rows_ok _ _ _ _ _ H3), (read_rows_ok _ _ _ _ _ H4),
          (read_rows_ok _ _ _ _ _ H5).
  destruct (Nat.eqb (Nat.modulo i 4) 3) eqn:Hm.
  - apply Nat.eqb_eq in Hm.
    destruct (Hattn Hm) as [Q [K [V [O [QN KN]]]]].
    rewrite (read_rows_ok _ _ _ _ _ Q), (read_rows_ok _ _ _ _ _ K),
            (read_rows_ok _ _ _ _ _ V), (read_rows_ok _ _ _ _ _ O),
            (read_vec_ok _ _ _ _ QN), (read_vec_ok _ _ _ _ KN).
    reflexivity.
  - apply Nat.eqb_neq in Hm.
    destruct (Hdelta Hm) as [QKV [Z [A [B [C [AL [DT [NW OUT]]]]]]]].
    rewrite (read_rows_ok _ _ _ _ _ QKV), (read_rows_ok _ _ _ _ _ Z),
            (read_rows_ok _ _ _ _ _ A), (read_rows_ok _ _ _ _ _ B),
            (read_rows_ok _ _ _ _ _ C), (read_vec_ok _ _ _ _ AL),
            (read_vec_ok _ _ _ _ DT), (read_vec_ok _ _ _ _ NW),
            (read_rows_ok _ _ _ _ _ OUT).
    reflexivity.
Qed.

Theorem f32_load_qwen_correct :
  forall header data d nl nh nkv hd rdim ff vocab lnh lhd ck,
  tensor_ok header data "embed_tokens.weight" (vocab * d) ->
  tensor_ok header data "rope.inv_freq" (rdim / 2) ->
  tensor_ok header data "norm.weight" d ->
  (forall i, (i < nl)%nat ->
     qwen_layer_loadable header data d nh nkv hd ff lnh lhd ck i) ->
  f32_load_qwen header (reader_of data) d nl nh nkv hd rdim ff vocab lnh lhd ck
  = mk_qwen_model_weights
      (named_rows header data "embed_tokens.weight" vocab d)
      (f32_load_named header data "rope.inv_freq")
      (f32_load_named header data "norm.weight")
      (map_seq (qwen_layer_named header data d nh nkv hd ff lnh lhd ck) nl).
Proof.
  intros header data d nl nh nkv hd rdim ff vocab lnh lhd ck Hemb Hinv Hnorm Hl.
  unfold f32_load_qwen, f32_read_named_rows, named_rows.
  rewrite (read_rows_ok _ _ _ _ _ Hemb), (read_vec_ok _ _ _ _ Hinv),
          (read_vec_ok _ _ _ _ Hnorm).
  rewrite !map_seq_spec. f_equal.
  apply List.map_ext_in. intros i Hi.
  apply List.in_seq in Hi. destruct Hi as [_ Hi]. cbn in Hi.
  apply read_qwen_layer_ok, Hl, Hi.
Qed.
