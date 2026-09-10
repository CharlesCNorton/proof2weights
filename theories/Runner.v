(** * The pre-transposed decode the checkpoint runners perform

    The verified linear layer is
    [f32_linear_forward w b x = f32_vec_add (f32_mat_vec_mul (f32_mat_transpose w) x) b],
    and the extracted [f32_mat_transpose] is quadratic in the output dimension
    through list indexing, which is why the checkpoint runners never call it.
    They instead decode each weight matrix directly in transposed order,
    reading the flat buffer at stride [cols], and feed that to
    [f32_mat_vec_mul].

    That substitution is the single place where the runners depart from the
    definitions in a way that is not pure plumbing, and it was checked only on
    fixtures. This file proves it: the transposed decode returns exactly
    [f32_mat_transpose] of the reshape, so the runner's linear layer computes
    the same binary32 values [f32_linear_forward] computes, operation for
    operation. *)

From Stdlib Require Import List.
From Stdlib Require Import Lia.
From Stdlib Require Import PeanoNat.
From Flocq Require Import IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete.
Require Import Cache_attn.

Import ListNotations.

(** * Indexing through [firstn] and [skipn] *)

Lemma nth_firstn : forall {A : Type} (d : A) n m (l : list A),
  (n < m)%nat -> List.nth n (List.firstn m l) d = List.nth n l d.
Proof.
  intros A d n. induction n as [|n IH]; intros m l Hm.
  - destruct m; [lia|]. destruct l; reflexivity.
  - destruct m; [lia|]. destruct l; cbn [List.firstn List.nth].
    + reflexivity.
    + apply IH. lia.
Qed.

Lemma nth_skipn : forall {A : Type} (d : A) n m (l : list A),
  List.nth n (List.skipn m l) d = List.nth (m + n) l d.
Proof.
  intros A d n m. revert n. induction m as [|m IH]; intros n l.
  - reflexivity.
  - destruct l; cbn [List.skipn List.nth Nat.add].
    + destruct n; reflexivity.
    + apply IH.
Qed.

(** * The reshape, indexed

    Row-major: entry [(i, j)] of the reshape is flat index [i * cols + j]. *)

Lemma reshape_nth : forall rows cols data i j,
  (i < rows)%nat -> (j < cols)%nat ->
  List.nth j (List.nth i (f32_reshape_1d_to_2d data rows cols) []) f32_zero
  = List.nth (i * cols + j) data f32_zero.
Proof.
  induction rows as [|rows IH]; intros cols data i j Hi Hj; [lia|].
  destruct i as [|i]; cbn [f32_reshape_1d_to_2d List.nth].
  - cbn [Nat.mul Nat.add]. apply nth_firstn. exact Hj.
  - rewrite IH by lia. rewrite nth_skipn. f_equal. lia.
Qed.

Lemma reshape_rows : forall data rows cols,
  List.length (f32_reshape_1d_to_2d data rows cols) = rows.
Proof. intros. apply f32_reshape_1d_to_2d_rows. Qed.

Lemma reshape_rect : forall rows data cols,
  (rows * cols <= List.length data)%nat ->
  rect (f32_reshape_1d_to_2d data rows cols) cols.
Proof.
  intros rows. induction rows as [|rows IH]; intros data cols Hlen.
  - constructor.
  - cbn [f32_reshape_1d_to_2d]. constructor.
    + rewrite List.length_firstn. cbn [Nat.mul] in Hlen. lia.
    + apply IH. rewrite List.length_skipn. cbn [Nat.mul] in Hlen. lia.
Qed.

(** * The transposed decode

    [rows] and [cols] describe the matrix as stored: [rows] rows of [cols]
    entries, flat index [i * cols + j]. The decode walks the output dimension
    outermost, producing the transpose without building it. *)

Definition f32_decode_transposed (data : list binary32) (rows cols : nat)
                                 : list (list binary32) :=
  List.map (fun o => List.map (fun i => List.nth (i * cols + o) data f32_zero)
                              (List.seq 0 rows))
           (List.seq 0 cols).

Theorem decode_transposed_correct : forall data rows cols,
  (0 < rows)%nat -> (0 < cols)%nat ->
  (rows * cols <= List.length data)%nat ->
  f32_decode_transposed data rows cols
  = f32_mat_transpose (f32_reshape_1d_to_2d data rows cols).
Proof.
  intros data rows cols Hr Hc Hlen.
  assert (Hrect : rect (f32_reshape_1d_to_2d data rows cols) cols)
    by (apply reshape_rect; exact Hlen).
  assert (Hne : f32_reshape_1d_to_2d data rows cols <> []).
  { intro H0. apply (f_equal (@List.length _)) in H0.
    rewrite reshape_rows in H0. cbn in H0. lia. }
  rewrite (transpose_unfold _ cols Hne Hrect).
  unfold f32_decode_transposed.
  apply List.map_ext_in. intros o Ho.
  apply List.in_seq in Ho. destruct Ho as [_ Ho]. cbn in Ho.
  transitivity (List.map (fun r => List.nth o r f32_zero)
                  (List.map (fun i => List.nth i (f32_reshape_1d_to_2d data rows cols) [])
                            (List.seq 0 rows))).
  - rewrite List.map_map.
    apply List.map_ext_in. intros i Hi.
    apply List.in_seq in Hi. destruct Hi as [_ Hi]. cbn in Hi.
    symmetry. apply reshape_nth; assumption.
  - f_equal.
    transitivity (List.map (fun i => List.nth i (f32_reshape_1d_to_2d data rows cols) [])
                    (List.seq 0 (List.length (f32_reshape_1d_to_2d data rows cols)))).
    + rewrite reshape_rows. reflexivity.
    + apply map_nth_seq.
Qed.

(** The runner's linear layer is the verified one. Both sides are lists of
    binary32; this is an equality of values, not a bound. *)
Theorem runner_linear_correct : forall data rows cols b x,
  (0 < rows)%nat -> (0 < cols)%nat ->
  (rows * cols <= List.length data)%nat ->
  f32_vec_add (f32_mat_vec_mul (f32_decode_transposed data rows cols) x) b
  = f32_linear_forward (f32_reshape_1d_to_2d data rows cols) b x.
Proof.
  intros data rows cols b x Hr Hc Hlen.
  unfold f32_linear_forward.
  rewrite (decode_transposed_correct data rows cols Hr Hc Hlen).
  reflexivity.
Qed.

(** Applied rowwise, as the runners apply it. *)
Theorem runner_linear_2d_correct : forall data rows cols b xs,
  (0 < rows)%nat -> (0 < cols)%nat ->
  (rows * cols <= List.length data)%nat ->
  List.map (fun row => f32_vec_add
              (f32_mat_vec_mul (f32_decode_transposed data rows cols) row) b) xs
  = f32_linear_forward_2d (f32_reshape_1d_to_2d data rows cols) b xs.
Proof.
  intros data rows cols b xs Hr Hc Hlen.
  unfold f32_linear_forward_2d.
  apply List.map_ext. intros row.
  apply runner_linear_correct; assumption.
Qed.
