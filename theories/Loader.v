(** * What the loader returns, and what validation guarantees

    The safetensors loader had no specification: [f32_load_named] scoped to
    the first occurrence of a quoted tensor name, read the next
    [data_offsets], sliced, and decoded, with nothing said about the values it
    produced and no check that the tensor's [dtype] was [F32] at all. This
    file supplies both.

    The first half fixes the meaning of a load: the [i]-th value of a named
    tensor is the binary32 that the four little-endian bytes at offset
    [a + 4i] of the data section denote, where [a] is the tensor's start
    offset. Nothing about the header parser is assumed; the offsets it
    returns are taken as given and the slice is what is characterised.

    The second half connects the shape validators, which the loader never
    called, to the hypotheses the forward-pass shape theorems require, so
    those theorems apply to a model that passed validation rather than to an
    abstract well-shaped one. *)

From Stdlib Require Import ZArith.
From Stdlib Require Import List.
From Stdlib Require Import String.
From Stdlib Require Import Bool.
From Stdlib Require Import Lia.
From Stdlib Require Import PeanoNat.
From Flocq Require Import IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete.

Import ListNotations.

(** * Decoding a slice

    [f32_decode_tensor_1d] walks the byte list four at a time. Its [i]-th
    output is therefore the value at byte offset [4i]. *)

Lemma f32_decode_tensor_1d_nth : forall len bs i,
  (i < len)%nat ->
  List.nth i (f32_decode_tensor_1d len bs) f32_zero
  = f32_bytes_to_binary32 (take 4 (drop (4 * i) bs)).
Proof.
  unfold f32_decode_tensor_1d.
  induction len as [|len IH]; intros bs i Hi; [lia|].
  destruct i as [|i]; cbn [f32_decode_tensor_1d_aux List.nth].
  - reflexivity.
  - rewrite IH by lia.
    unfold drop. rewrite List.skipn_skipn.
    replace (4 * i + 4)%nat with (4 * S i)%nat by lia.
    reflexivity.
Qed.

(** * The specification of a named load *)

Theorem f32_load_named_length : forall header data tname a b rest,
  json_tensor_offsets header tname = a :: b :: rest ->
  List.length (f32_load_named header data tname) = ((b - a) / 4)%nat.
Proof.
  intros header data tname a b rest Hoff.
  unfold f32_load_named. rewrite Hoff.
  apply f32_load_tensor_bytes_length.
Qed.

(** The value-level specification: entry [i] is the binary32 denoted by the
    four little-endian bytes at [a + 4i]. *)
Theorem f32_load_named_nth : forall header data tname a b rest i,
  json_tensor_offsets header tname = a :: b :: rest ->
  (i < (b - a) / 4)%nat ->
  List.nth i (f32_load_named header data tname) f32_zero
  = f32_bytes_to_binary32 (take 4 (drop (a + 4 * i) data)).
Proof.
  intros header data tname a b rest i Hoff Hi.
  unfold f32_load_named. rewrite Hoff.
  unfold f32_load_tensor_bytes.
  rewrite f32_decode_tensor_1d_nth by exact Hi.
  unfold drop. rewrite List.skipn_skipn.
  replace (4 * i + a)%nat with (a + 4 * i)%nat by lia.
  reflexivity.
Qed.

(** A load whose name is absent, or whose header entry carries fewer than two
    offsets, is empty rather than silently misaligned. *)
Theorem f32_load_named_absent : forall header data tname,
  (json_tensor_offsets header tname = [] \/
   exists a, json_tensor_offsets header tname = [a]) ->
  f32_load_named header data tname = [].
Proof.
  intros header data tname [H | [a H]]; unfold f32_load_named; rewrite H; reflexivity.
Qed.

(** * The dtype check the loader never performed *)

Definition json_tensor_dtype (header tname : string) : string :=
  match json_find_after (String.length header) (json_quoted tname) header with
  | None => EmptyString
  | Some sub => json_extract_string "dtype" sub
  end.

Definition f32_load_named_checked (header : string) (data : list byte)
                                  (tname : string) : option (list binary32) :=
  if String.eqb (json_tensor_dtype header tname) "F32"
  then Some (f32_load_named header data tname)
  else None.

Theorem f32_load_named_checked_sound : forall header data tname vs,
  f32_load_named_checked header data tname = Some vs ->
  json_tensor_dtype header tname = "F32"%string
  /\ vs = f32_load_named header data tname.
Proof.
  intros header data tname vs H. unfold f32_load_named_checked in H.
  destruct (String.eqb (json_tensor_dtype header tname) "F32") eqn:E;
    [|discriminate].
  apply String.eqb_eq in E. split; [exact E | congruence].
Qed.

Theorem f32_load_named_checked_rejects : forall header data tname,
  json_tensor_dtype header tname <> "F32"%string ->
  f32_load_named_checked header data tname = None.
Proof.
  intros header data tname H. unfold f32_load_named_checked.
  destruct (String.eqb (json_tensor_dtype header tname) "F32") eqn:E;
    [|reflexivity].
  apply String.eqb_eq in E. contradiction.
Qed.

(** * From validation to the shape hypotheses

    [f32_validate_model] was defined but never connected to anything. These
    lemmas discharge, from a successful validation, the facts the forward-pass
    shape theorems need. *)

Lemma validate_model_wte : forall cfg model,
  f32_validate_model cfg model = true ->
  f32_mat_rows (f32_wte model) = gpt2_inf_vocab_size cfg.
Proof.
  intros cfg model H. unfold f32_validate_model in H.
  apply andb_prop in H; destruct H as [H _].
  apply andb_prop in H; destruct H as [H _].
  apply andb_prop in H; destruct H as [H _].
  apply andb_prop in H; destruct H as [H _].
  eapply f32_validate_2d_rows; exact H.
Qed.

Lemma validate_model_wpe : forall cfg model,
  f32_validate_model cfg model = true ->
  f32_mat_rows (f32_wpe model) = gpt2_inf_n_positions cfg.
Proof.
  intros cfg model H. unfold f32_validate_model in H.
  apply andb_prop in H; destruct H as [H _].
  apply andb_prop in H; destruct H as [H _].
  apply andb_prop in H; destruct H as [H _].
  apply andb_prop in H; destruct H as [_ H].
  eapply f32_validate_2d_rows; exact H.
Qed.

Lemma validate_model_blocks : forall cfg model,
  f32_validate_model cfg model = true ->
  List.length (f32_blocks model) = gpt2_inf_n_layer cfg.
Proof.
  intros cfg model H. unfold f32_validate_model in H.
  apply andb_prop in H; destruct H as [H _].
  apply andb_prop in H; destruct H as [H _].
  apply andb_prop in H; destruct H as [_ H].
  apply Nat.eqb_eq. exact H.
Qed.

(** The forward pass of a validated model emits one logit row per input token,
    each of width [vocab_size], with the width now tied to the configuration
    rather than to the model's own embedding matrix. *)
Theorem validated_logits_shape : forall cfg eps model toks,
  (gpt2_inf_n_head cfg > 0)%nat ->
  f32_validate_model cfg model = true ->
  f32_mat_rows (f32_gpt2_logits cfg eps model toks) = List.length toks
  /\ (forall row, In row (f32_gpt2_logits cfg eps model toks) ->
        List.length row = gpt2_inf_vocab_size cfg).
Proof.
  intros cfg eps model toks Hh Hv. split.
  - apply f32_gpt2_logits_rows; exact Hh.
  - intros row Hin.
    rewrite (f32_gpt2_logits_row_width cfg eps model toks row Hin).
    apply (validate_model_wte cfg model Hv).
Qed.

(** A loaded model carries the right number of blocks by construction, so the
    block count never depends on validation. *)
Theorem loaded_model_blocks : forall header data cfg,
  List.length (f32_blocks (f32_load_model header data cfg))
  = gpt2_inf_n_layer cfg.
Proof. intros. apply f32_load_model_num_blocks. Qed.
