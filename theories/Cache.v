(** * Cached decoding equals full recomputation

    The runners generate autoregressively: a prefill over the prompt, then one
    decode step per token, each reusing state the previous steps produced. The
    definitions in Llama.v and Qwen.v instead recompute the whole sequence. The
    two perform the same multiplications in different groupings, and binary32
    addition is not associative, so agreement is not automatic.

    This file proves it, for the recurrent half of the problem: threading the
    gated-DeltaNet state across calls yields exactly the outputs
    [f32_delta_scan] produces over the whole sequence, and the runner's
    incrementally maintained convolution window is exactly the window
    [f32_causal_conv1d] reads. Attention is treated in Cache_attn.v.

    Nothing here is an approximation argument. The statements are equalities of
    binary32 values. *)

From Stdlib Require Import List.
From Stdlib Require Import Lia.
From Stdlib Require Import PeanoNat.
From Flocq Require Import IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete.
Require Import Llama.
Require Import Qwen.

Import ListNotations.

(** * The state the scan threads

    [f32_delta_scan] returns the per-token outputs and discards the state it
    carried. [f32_delta_state] is the same recursion returning the state
    instead, so the two together describe one pass completely. *)

Fixpoint f32_delta_state (betas gs : list binary32)
                         (qs ks vs : list (list binary32))
                         (st : list (list binary32)) : list (list binary32) :=
  match betas, gs, qs, ks, vs with
  | b :: bs, g :: gs', q :: qs', k :: ks', v :: vs' =>
      f32_delta_state bs gs' qs' ks' vs' (fst (f32_delta_step b g q k v st))
  | _, _, _, _, _ => st
  end.

(** One step, in the form the runner uses it: the output is the second
    component and the state carried forward is the first. *)
Lemma delta_scan_cons : forall b bs g gs q qs k ks v vs st,
  f32_delta_scan (b :: bs) (g :: gs) (q :: qs) (k :: ks) (v :: vs) st
  = snd (f32_delta_step b g q k v st)
    :: f32_delta_scan bs gs qs ks vs (fst (f32_delta_step b g q k v st)).
Proof.
  intros. cbn [f32_delta_scan].
  destruct (f32_delta_step b g q k v st) as [st' o]. cbn [fst snd]. reflexivity.
Qed.

Lemma delta_state_cons : forall b bs g gs q qs k ks v vs st,
  f32_delta_state (b :: bs) (g :: gs) (q :: qs) (k :: ks) (v :: vs) st
  = f32_delta_state bs gs qs ks vs (fst (f32_delta_step b g q k v st)).
Proof. intros. reflexivity. Qed.

(** * Prefill then decode

    Splitting the sequence anywhere and running the second part from the state
    the first part produced gives exactly the outputs of one pass over the
    whole sequence. This is the prefill/decode decomposition the runners
    perform, stated as an equality of binary32 outputs. *)

Theorem delta_scan_app :
  forall bs1 gs1 qs1 ks1 vs1 bs2 gs2 qs2 ks2 vs2 st,
  List.length bs1 = List.length gs1 ->
  List.length bs1 = List.length qs1 ->
  List.length bs1 = List.length ks1 ->
  List.length bs1 = List.length vs1 ->
  f32_delta_scan (bs1 ++ bs2) (gs1 ++ gs2) (qs1 ++ qs2) (ks1 ++ ks2)
                 (vs1 ++ vs2) st
  = f32_delta_scan bs1 gs1 qs1 ks1 vs1 st
    ++ f32_delta_scan bs2 gs2 qs2 ks2 vs2
         (f32_delta_state bs1 gs1 qs1 ks1 vs1 st).
Proof.
  induction bs1 as [|b bs1 IH];
    intros gs1 qs1 ks1 vs1 bs2 gs2 qs2 ks2 vs2 st Hg Hq Hk Hv.
  - destruct gs1; [|discriminate]. destruct qs1; [|discriminate].
    destruct ks1; [|discriminate]. destruct vs1; [|discriminate].
    reflexivity.
  - destruct gs1 as [|g gs1]; [discriminate|].
    destruct qs1 as [|q qs1]; [discriminate|].
    destruct ks1 as [|k ks1]; [discriminate|].
    destruct vs1 as [|v vs1]; [discriminate|].
    cbn [app]. rewrite !delta_scan_cons, delta_state_cons.
    cbn [app]. f_equal.
    apply IH; cbn [List.length] in *; lia.
Qed.

(** The single decode step: extending the sequence by one token appends
    exactly the output that step produces from the threaded state, and changes
    nothing already emitted. *)
Corollary delta_scan_snoc :
  forall bs gs qs ks vs b g q k v st,
  List.length bs = List.length gs ->
  List.length bs = List.length qs ->
  List.length bs = List.length ks ->
  List.length bs = List.length vs ->
  f32_delta_scan (bs ++ [b]) (gs ++ [g]) (qs ++ [q]) (ks ++ [k]) (vs ++ [v]) st
  = f32_delta_scan bs gs qs ks vs st
    ++ [snd (f32_delta_step b g q k v (f32_delta_state bs gs qs ks vs st))].
Proof.
  intros bs gs qs ks vs b g q k v st Hg Hq Hk Hv.
  rewrite delta_scan_app by assumption.
  cbn [f32_delta_scan].
  destruct (f32_delta_step b g q k v (f32_delta_state bs gs qs ks vs st))
    as [st' o].
  reflexivity.
Qed.

(** The state is likewise incremental. *)
Theorem delta_state_app :
  forall bs1 gs1 qs1 ks1 vs1 bs2 gs2 qs2 ks2 vs2 st,
  List.length bs1 = List.length gs1 ->
  List.length bs1 = List.length qs1 ->
  List.length bs1 = List.length ks1 ->
  List.length bs1 = List.length vs1 ->
  f32_delta_state (bs1 ++ bs2) (gs1 ++ gs2) (qs1 ++ qs2) (ks1 ++ ks2)
                  (vs1 ++ vs2) st
  = f32_delta_state bs2 gs2 qs2 ks2 vs2
      (f32_delta_state bs1 gs1 qs1 ks1 vs1 st).
Proof.
  induction bs1 as [|b bs1 IH];
    intros gs1 qs1 ks1 vs1 bs2 gs2 qs2 ks2 vs2 st Hg Hq Hk Hv.
  - destruct gs1; [|discriminate]. destruct qs1; [|discriminate].
    destruct ks1; [|discriminate]. destruct vs1; [|discriminate].
    reflexivity.
  - destruct gs1 as [|g gs1]; [discriminate|].
    destruct qs1 as [|q qs1]; [discriminate|].
    destruct ks1 as [|k ks1]; [discriminate|].
    destruct vs1 as [|v vs1]; [discriminate|].
    cbn [app]. rewrite !delta_state_cons.
    apply IH; cbn [List.length] in *; lia.
Qed.

(** The recurrence is Markov in the state: the outputs after a split depend on
    the prefix only through [f32_delta_state]. This is what licenses discarding
    the prompt after prefill, and it is an equality, not a bound. *)
Corollary delta_markov :
  forall bs1 gs1 qs1 ks1 vs1 bs1' gs1' qs1' ks1' vs1'
         bs2 gs2 qs2 ks2 vs2 st,
  List.length bs1 = List.length gs1 ->
  List.length bs1 = List.length qs1 ->
  List.length bs1 = List.length ks1 ->
  List.length bs1 = List.length vs1 ->
  List.length bs1' = List.length gs1' ->
  List.length bs1' = List.length qs1' ->
  List.length bs1' = List.length ks1' ->
  List.length bs1' = List.length vs1' ->
  f32_delta_state bs1 gs1 qs1 ks1 vs1 st
    = f32_delta_state bs1' gs1' qs1' ks1' vs1' st ->
  List.skipn (List.length bs1)
    (f32_delta_scan (bs1 ++ bs2) (gs1 ++ gs2) (qs1 ++ qs2) (ks1 ++ ks2)
                    (vs1 ++ vs2) st)
  = List.skipn (List.length bs1')
      (f32_delta_scan (bs1' ++ bs2) (gs1' ++ gs2) (qs1' ++ qs2) (ks1' ++ ks2)
                      (vs1' ++ vs2) st).
Proof.
  intros bs1 gs1 qs1 ks1 vs1 bs1' gs1' qs1' ks1' vs1' bs2 gs2 qs2 ks2 vs2 st
         H1 H2 H3 H4 H1' H2' H3' H4' Hst.
  rewrite !delta_scan_app by assumption.
  rewrite Hst.
  assert (L : List.length (f32_delta_scan bs1 gs1 qs1 ks1 vs1 st)
              = List.length bs1).
  { rewrite f32_delta_scan_length. lia. }
  assert (L' : List.length (f32_delta_scan bs1' gs1' qs1' ks1' vs1' st)
               = List.length bs1').
  { rewrite f32_delta_scan_length. lia. }
  rewrite <- L, <- L', !List.skipn_app.
  rewrite !Nat.sub_diag, !List.skipn_all. reflexivity.
Qed.

(** * The convolution window

    The runner keeps the trailing [k-1] inputs and forms the window by
    appending the current one. [f32_conv_window] instead slices the whole
    sequence. The two agree at every step. *)

Definition conv_hist (k : nat) (xs : list (list binary32)) (t : nat)
                     : list (list binary32) :=
  List.skipn (t - (k - 1)) (List.firstn t xs).

Definition f32_conv_window_cached (chans k : nat)
    (hist : list (list binary32)) (x : list binary32) : list (list binary32) :=
  let h := hist ++ [x] in
  let m := List.length h in
  if Nat.leb k m then List.skipn (m - k) h
  else List.repeat (f32_zeros chans) (k - m) ++ h.

Lemma firstn_S_split : forall (xs : list (list binary32)) t,
  (t < List.length xs)%nat ->
  List.firstn (S t) xs = List.firstn t xs ++ [List.nth t xs []].
Proof.
  induction xs as [|x xs IH]; intros t Ht.
  - cbn [List.length] in Ht. lia.
  - destruct t as [|t].
    + reflexivity.
    + cbn [List.length] in Ht.
      cbn [List.firstn List.nth app].
      f_equal. apply IH. lia.
Qed.

Theorem conv_window_cached_correct : forall chans k xs t,
  (0 < k)%nat -> (t < List.length xs)%nat ->
  f32_conv_window_cached chans k (conv_hist k xs t) (List.nth t xs [])
  = f32_conv_window chans k xs t.
Proof.
  intros chans k xs t Hk Ht.
  assert (Hlen : List.length (List.firstn (S t) xs) = S t).
  { rewrite List.length_firstn. lia. }
  assert (Hsk : List.skipn (t - (k - 1)) (List.firstn t xs) ++ [List.nth t xs []]
                = List.skipn (t - (k - 1)) (List.firstn (S t) xs)).
  { rewrite firstn_S_split by exact Ht.
    rewrite List.skipn_app, List.length_firstn, Nat.min_l by lia.
    replace (t - (k - 1) - t)%nat with 0%nat by lia.
    reflexivity. }
  unfold f32_conv_window_cached, f32_conv_window, conv_hist. cbv zeta.
  rewrite Hsk, List.length_skipn, Hlen.
  destruct (Nat.leb k (S t)) eqn:E.
  - assert (E' : (k <= S t)%nat) by (apply Nat.leb_le; exact E).
    replace (S t - (t - (k - 1)))%nat with k by lia.
    rewrite Nat.leb_refl, Nat.sub_diag. cbn [List.skipn].
    f_equal. lia.
  - assert (E' : (S t < k)%nat) by (apply Nat.leb_gt; exact E).
    replace (t - (k - 1))%nat with 0%nat by lia.
    cbn [List.skipn].
    replace (S t - 0)%nat with (S t) by lia.
    rewrite E. reflexivity.
Qed.

(** The runner's update rule: keep the trailing [k-1] entries. *)
Definition last_n (n : nat) (l : list (list binary32)) : list (list binary32) :=
  List.skipn (List.length l - n) l.

(** The window the runner holds after step [t] is exactly the one step [S t]
    needs, so the invariant [conv_hist] describes is the one the runner
    maintains. *)
Lemma conv_hist_step : forall k xs t,
  (0 < k)%nat -> (t < List.length xs)%nat ->
  conv_hist k xs (S t)
  = last_n (k - 1) (conv_hist k xs t ++ [List.nth t xs []]).
Proof.
  intros k xs t Hk Ht.
  assert (Hlen : List.length (List.firstn (S t) xs) = S t).
  { rewrite List.length_firstn. lia. }
  assert (Hsk : List.skipn (t - (k - 1)) (List.firstn t xs) ++ [List.nth t xs []]
                = List.skipn (t - (k - 1)) (List.firstn (S t) xs)).
  { rewrite firstn_S_split by exact Ht.
    rewrite List.skipn_app, List.length_firstn, Nat.min_l by lia.
    replace (t - (k - 1) - t)%nat with 0%nat by lia.
    reflexivity. }
  unfold conv_hist, last_n.
  rewrite Hsk, List.length_skipn, Hlen, List.skipn_skipn.
  f_equal. lia.
Qed.
