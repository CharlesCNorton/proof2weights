(** * Decoders that carry state

    A decoder reads a sequence one element at a time and emits one output per
    element, carrying a state from step to step: a key/value cache, a
    convolution window, a recurrent state, a position. [run] is such a decoder
    and [after] the state it holds once it has read a prefix. It realizes a
    sequence function when every step extends the function's output by exactly
    the row the step emits from that state; a decoder that realizes a function
    computes, on every sequence, exactly the rows the function computes
    ([run_realized]), as equalities of binary32 values.

    The combinators below build decoders for row-wise and positional maps,
    composition, pairing, parallel heads, causal attention over a growing
    cache, the depthwise convolution over a sliding window, the gated delta
    recurrence and a stack of layers; Decode.v assembles them into the
    decoders of the three models. *)

From Stdlib Require Import List Lia PeanoNat.
From Flocq Require Import IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete Llama Qwen Cache Cache_attn Causal.

Import ListNotations.

(** * Decoders and what they realize *)

Fixpoint run {S A B : Type} (step : S -> A -> S * B) (s : S) (xs : list A)
    : list B :=
  match xs with
  | [] => []
  | x :: rest => let r := step s x in snd r :: run step (fst r) rest
  end.

Definition after {S A B : Type} (step : S -> A -> S * B) (s : S) (xs : list A) : S :=
  List.fold_left (fun s x => fst (step s x)) xs s.

Lemma after_snoc : forall {S A B : Type} (step : S -> A -> S * B) s xs x,
  after step s (xs ++ [x]) = fst (step (after step s xs) x).
Proof. intros. unfold after. rewrite fold_left_app. reflexivity. Qed.

Definition realized {S A B : Type} (init : S) (step : S -> A -> S * B)
    (F : list A -> list B) : Prop :=
  F [] = [] /\
  forall xs x, F (xs ++ [x]) = F xs ++ [snd (step (after step init xs) x)].

Lemma run_from : forall {S A B : Type} (init : S) (step : S -> A -> S * B) F,
  realized init step F ->
  forall xs ps, F ps ++ run step (after step init ps) xs = F (ps ++ xs).
Proof.
  intros S A B init step F [_ Hs] xs.
  induction xs as [|x xs IH]; intros ps.
  - cbn [run]. now rewrite !app_nil_r.
  - cbn [run].
    replace (ps ++ x :: xs) with ((ps ++ [x]) ++ xs)
      by (rewrite <- app_assoc; reflexivity).
    rewrite <- (IH (ps ++ [x])), after_snoc, Hs, <- app_assoc. reflexivity.
Qed.

(** The decoder computes the function, row for row, on every sequence. *)
Theorem run_realized : forall {S A B : Type} (init : S) (step : S -> A -> S * B) F,
  realized init step F -> forall xs, run step init xs = F xs.
Proof.
  intros S A B init step F H xs.
  pose proof (run_from init step F H xs []) as E.
  destruct H as [H0 _]. rewrite H0 in E. exact E.
Qed.

Lemma run_length : forall {S A B : Type} (step : S -> A -> S * B) s xs,
  List.length (run step s xs) = List.length xs.
Proof.
  intros S A B step s xs. revert s.
  induction xs as [|x xs IH]; intros s; cbn [run List.length]; [reflexivity|].
  now rewrite IH.
Qed.

Lemma realized_length : forall {S A B : Type} (init : S) (step : S -> A -> S * B) F,
  realized init step F -> forall xs, List.length (F xs) = List.length xs.
Proof.
  intros S A B init step F H xs.
  rewrite <- (run_realized init step F H xs). apply run_length.
Qed.

Lemma realized_ext : forall {S A B : Type} (init : S) (step : S -> A -> S * B)
    (F G : list A -> list B),
  (forall xs, F xs = G xs) -> realized init step F -> realized init step G.
Proof.
  intros S A B init step F G E [H0 Hs].
  split; [now rewrite <- E|]. intros xs x. rewrite <- !E. apply Hs.
Qed.

Lemma after_ext : forall {S A B : Type} (st1 st2 : S -> A -> S * B) s xs,
  (forall s x, st1 s x = st2 s x) -> after st1 s xs = after st2 s xs.
Proof.
  intros S A B st1 st2 s xs E. revert s.
  induction xs as [|x xs IH]; intros s; [reflexivity|].
  cbn [after List.fold_left]. rewrite E. apply IH.
Qed.

Lemma realized_step_ext : forall {S A B : Type} (init : S) (st1 st2 : S -> A -> S * B) F,
  (forall s x, st1 s x = st2 s x) -> realized init st1 F -> realized init st2 F.
Proof.
  intros S A B init st1 st2 F E [H0 Hs].
  split; [exact H0|]. intros xs x.
  rewrite <- E, <- (after_ext st1 st2 init xs E). apply Hs.
Qed.

(** * Row-wise and positional maps *)

Lemma realized_map : forall {A B : Type} (g : A -> B),
  realized tt (fun (s : unit) x => (s, g x)) (List.map g).
Proof.
  intros A B g. split; [reflexivity|].
  intros xs x. now rewrite map_app.
Qed.

Definition mapi {A B : Type} (g : nat -> A -> B) (xs : list A) : list B :=
  List.map (fun p => g (fst p) (snd p))
           (List.combine (List.seq 0 (List.length xs)) xs).

Lemma mapi_length : forall {A B : Type} (g : nat -> A -> B) xs,
  List.length (mapi g xs) = List.length xs.
Proof.
  intros A B g xs. unfold mapi.
  rewrite length_map, length_combine, length_seq. apply Nat.min_id.
Qed.

Lemma mapi_snoc : forall {A B : Type} (g : nat -> A -> B) xs x,
  mapi g (xs ++ [x]) = mapi g xs ++ [g (List.length xs) x].
Proof.
  intros A B g xs x. unfold mapi.
  rewrite length_app, seq_app. cbn [List.length].
  rewrite combine_app_eq by (rewrite length_seq; reflexivity).
  rewrite map_app. reflexivity.
Qed.

Lemma after_mapi : forall {A B : Type} (g : nat -> A -> B) xs,
  after (fun t x => (S t, g t x)) 0%nat xs = List.length xs.
Proof.
  intros A B g xs. induction xs as [|x xs IH] using rev_ind; [reflexivity|].
  rewrite after_snoc, IH, length_app. cbn [fst List.length]. lia.
Qed.

Lemma realized_mapi : forall {A B : Type} (g : nat -> A -> B),
  realized 0%nat (fun t x => (S t, g t x)) (mapi g).
Proof.
  intros A B g. split; [reflexivity|].
  intros xs x. rewrite after_mapi. apply mapi_snoc.
Qed.

(** A map is a positional map that ignores the position. *)
Lemma map_mapi : forall {A B : Type} (g : A -> B) xs,
  List.map g xs = mapi (fun _ x => g x) xs.
Proof.
  intros A B g xs. unfold mapi.
  assert (G : forall n, List.map (fun p => g (snd p))
                                 (List.combine (List.seq n (List.length xs)) xs)
                        = List.map g xs).
  { induction xs as [|x xs IH]; intros n; [reflexivity|].
    cbn [List.length List.seq List.combine List.map]. f_equal. apply IH. }
  symmetry. apply G.
Qed.

(** * Composition *)

Definition comp_step {S1 S2 A B C : Type} (st1 : S1 -> A -> S1 * B)
    (st2 : S2 -> B -> S2 * C) (s : S1 * S2) (x : A) : (S1 * S2) * C :=
  let r1 := st1 (fst s) x in
  let r2 := st2 (snd s) (snd r1) in
  ((fst r1, fst r2), snd r2).

Lemma after_comp : forall {S1 S2 A B C : Type} (i1 : S1) st1 (F : list A -> list B)
    (i2 : S2) (st2 : S2 -> B -> S2 * C),
  realized i1 st1 F ->
  forall xs, after (comp_step st1 st2) (i1, i2) xs = (after st1 i1 xs, after st2 i2 (F xs)).
Proof.
  intros S1 S2 A B C i1 st1 F i2 st2 [F0 Fs] xs.
  induction xs as [|x xs IH] using rev_ind; [cbn; now rewrite F0|].
  rewrite !after_snoc, IH, Fs, after_snoc. reflexivity.
Qed.

Lemma realized_comp : forall {S1 S2 A B C : Type} (i1 : S1) st1 (F : list A -> list B)
    (i2 : S2) st2 (G : list B -> list C),
  realized i1 st1 F -> realized i2 st2 G ->
  realized (i1, i2) (comp_step st1 st2) (fun xs => G (F xs)).
Proof.
  intros S1 S2 A B C i1 st1 F i2 st2 G HF HG.
  pose proof HF as [F0 Fs]. pose proof HG as [G0 Gs].
  split; [now rewrite F0, G0|].
  intros xs x. rewrite (after_comp i1 st1 F i2 st2 HF xs).
  unfold comp_step. cbn [fst snd]. rewrite Fs, Gs. reflexivity.
Qed.

(** * Pairing two decoders over the same input *)

Definition zipw {B C D : Type} (h : B -> C -> D) (ys : list B) (zs : list C)
    : list D :=
  List.map (fun p => h (fst p) (snd p)) (List.combine ys zs).

Definition zip_step {S1 S2 A B C D : Type} (st1 : S1 -> A -> S1 * B)
    (st2 : S2 -> A -> S2 * C) (h : B -> C -> D) (s : S1 * S2) (x : A)
    : (S1 * S2) * D :=
  let r1 := st1 (fst s) x in
  let r2 := st2 (snd s) x in
  ((fst r1, fst r2), h (snd r1) (snd r2)).

Lemma after_zip : forall {S1 S2 A B C D : Type} (i1 : S1) (st1 : S1 -> A -> S1 * B)
    (i2 : S2) (st2 : S2 -> A -> S2 * C) (h : B -> C -> D) xs,
  after (zip_step st1 st2 h) (i1, i2) xs = (after st1 i1 xs, after st2 i2 xs).
Proof.
  intros. induction xs as [|x xs IH] using rev_ind; [reflexivity|].
  rewrite !after_snoc, IH. reflexivity.
Qed.

Lemma realized_zip : forall {S1 S2 A B C D : Type} (i1 : S1) st1
    (F1 : list A -> list B) (i2 : S2) st2 (F2 : list A -> list C) (h : B -> C -> D),
  realized i1 st1 F1 -> realized i2 st2 F2 ->
  realized (i1, i2) (zip_step st1 st2 h) (fun xs => zipw h (F1 xs) (F2 xs)).
Proof.
  intros S1 S2 A B C D i1 st1 F1 i2 st2 F2 h H1 H2.
  pose proof (realized_length i1 st1 F1 H1) as L1.
  pose proof (realized_length i2 st2 F2 H2) as L2.
  destruct H1 as [F10 F1s], H2 as [F20 F2s].
  split; [unfold zipw; now rewrite F10, F20|].
  intros xs x. rewrite after_zip. unfold zip_step. cbn [fst snd].
  unfold zipw. rewrite F1s, F2s.
  rewrite combine_app_eq by (rewrite L1, L2; reflexivity).
  rewrite map_app. reflexivity.
Qed.

(** * Parallel heads, concatenated per position *)

Definition heads_step {S A : Type} (nh : nat) (init : nat -> S)
    (st : nat -> S -> A -> S * list binary32) (ss : list S) (x : A)
    : list S * list binary32 :=
  let rs := List.map (fun hh => st hh (List.nth hh ss (init hh)) x) (List.seq 0 nh) in
  (List.map fst rs, List.concat (List.map snd rs)).

Lemma after_heads : forall {S A : Type} nh (init : nat -> S)
    (st : nat -> S -> A -> S * list binary32) xs,
  after (heads_step nh init st) (List.map init (List.seq 0 nh)) xs
  = List.map (fun hh => after (st hh) (init hh) xs) (List.seq 0 nh).
Proof.
  intros S A nh init st xs. induction xs as [|x xs IH] using rev_ind; [reflexivity|].
  rewrite after_snoc, IH. unfold heads_step. cbn [fst].
  rewrite map_map. apply map_ext_in. intros hh Hh. apply in_seq in Hh.
  rewrite after_snoc, nth_map_seq_lt by lia. reflexivity.
Qed.

Lemma realized_heads : forall {S A : Type} nh (init : nat -> S)
    (st : nat -> S -> A -> S * list binary32)
    (H : nat -> list A -> list (list binary32)),
  (0 < nh)%nat ->
  (forall hh, (hh < nh)%nat -> realized (init hh) (st hh) (H hh)) ->
  realized (List.map init (List.seq 0 nh)) (heads_step nh init st)
           (fun xs => f32_concat_heads (List.map (fun hh => H hh xs) (List.seq 0 nh))).
Proof.
  intros S A nh init st H Hn HH.
  assert (Len : forall hh xs, (hh < nh)%nat -> List.length (H hh xs) = List.length xs)
    by (intros hh xs Hh; exact (realized_length _ _ _ (HH hh Hh) xs)).
  assert (Row : forall xs, f32_concat_heads (List.map (fun hh => H hh xs) (List.seq 0 nh))
      = List.map (fun si => List.concat (List.map (fun hh => List.nth si (H hh xs) [])
                                                  (List.seq 0 nh)))
                 (List.seq 0 (List.length xs))).
  { intros xs. rewrite (concat_heads_seq nh (fun hh => H hh xs) Hn).
    rewrite (Len 0%nat xs Hn). reflexivity. }
  split; [rewrite Row; reflexivity|].
  intros xs x. rewrite after_heads. unfold heads_step. cbn [snd].
  rewrite !Row, length_app. cbn [List.length].
  rewrite seq_app, map_app. cbn [List.seq List.map]. rewrite Nat.add_0_l. f_equal.
  - apply map_ext_in. intros si Hsi. apply in_seq in Hsi. f_equal.
    apply map_ext_in. intros hh Hh. apply in_seq in Hh.
    rewrite (proj2 (HH hh ltac:(lia)) xs x).
    rewrite app_nth1 by (rewrite Len by lia; lia). reflexivity.
  - f_equal. rewrite map_map. f_equal.
    apply map_ext_in. intros hh Hh. apply in_seq in Hh.
    rewrite (proj2 (HH hh ltac:(lia)) xs x).
    rewrite app_nth2 by (rewrite Len by lia; lia).
    rewrite Len, Nat.sub_diag by lia. cbn [List.nth].
    rewrite nth_map_seq_lt by lia. reflexivity.
Qed.

(** * Causal attention over a growing cache *)

Definition attn_step {A : Type} (Q K V : nat -> A -> list binary32) (d_k : nat)
    (s : nat * list (list binary32) * list (list binary32)) (x : A)
    : (nat * list (list binary32) * list (list binary32)) * list binary32 :=
  let t := fst (fst s) in
  let kc := snd (fst s) ++ [K t x] in
  let vc := snd s ++ [V t x] in
  ((S t, kc, vc), f32_attend (Q t x) kc vc d_k).

Lemma after_attn : forall {A : Type} (Q K V : nat -> A -> list binary32) d_k xs,
  after (attn_step Q K V d_k) (0%nat, [], []) xs
  = (List.length xs, mapi K xs, mapi V xs).
Proof.
  intros A Q K V d_k xs. induction xs as [|x xs IH] using rev_ind; [reflexivity|].
  rewrite after_snoc, IH. unfold attn_step. cbn [fst snd].
  rewrite length_app, !mapi_snoc. cbn [List.length]. rewrite Nat.add_1_r.
  reflexivity.
Qed.

Lemma realized_attn : forall {A : Type} (Q K V : nat -> A -> list binary32) d_k,
  realized (0%nat, [], []) (attn_step Q K V d_k)
    (fun xs => f32_causal_attention (mapi Q xs) (mapi K xs) (mapi V xs) d_k).
Proof.
  intros A Q K V d_k. split; [reflexivity|].
  intros xs x. rewrite after_attn. unfold attn_step. cbn [fst snd].
  rewrite !mapi_snoc.
  apply causal_attention_snoc; rewrite !mapi_length; reflexivity.
Qed.

(** * The depthwise convolution over a sliding window *)

Definition conv_state_step (chans k : nat) (w : list (list binary32))
    (b : list binary32) (hist : list (list binary32)) (x : list binary32)
    : list (list binary32) * list binary32 :=
  (last_n (k - 1) (hist ++ [x]),
   f32_conv_step w b (f32_conv_window_cached chans k hist x)).

Lemma conv_window_snoc : forall chans k xs x t,
  (t < List.length xs)%nat ->
  f32_conv_window chans k (xs ++ [x]) t = f32_conv_window chans k xs t.
Proof.
  intros chans k xs x t Ht. unfold f32_conv_window.
  rewrite firstn_app.
  replace (S t - List.length xs)%nat with 0%nat by lia.
  now rewrite firstn_0, app_nil_r.
Qed.

Lemma conv_hist_snoc : forall k xs (x : list binary32),
  conv_hist k (xs ++ [x]) (List.length xs) = conv_hist k xs (List.length xs).
Proof.
  intros k xs x. unfold conv_hist.
  rewrite firstn_app, Nat.sub_diag, firstn_0, app_nil_r. reflexivity.
Qed.

Lemma nth_snoc_last : forall (xs : list (list binary32)) x,
  List.nth (List.length xs) (xs ++ [x]) [] = x.
Proof. intros xs x. rewrite app_nth2, Nat.sub_diag by lia. reflexivity. Qed.

Lemma after_conv : forall chans k w b xs, (0 < k)%nat ->
  after (conv_state_step chans k w b) [] xs = conv_hist k xs (List.length xs).
Proof.
  intros chans k w b xs Hk. induction xs as [|x xs IH] using rev_ind; [reflexivity|].
  rewrite after_snoc, IH. unfold conv_state_step. cbn [fst].
  rewrite length_app. cbn [List.length]. rewrite Nat.add_1_r.
  rewrite (conv_hist_step k (xs ++ [x]) (List.length xs) Hk)
    by (rewrite length_app; cbn [List.length]; lia).
  rewrite conv_hist_snoc, nth_snoc_last. reflexivity.
Qed.

Lemma realized_conv : forall chans k w b, (0 < k)%nat ->
  realized [] (conv_state_step chans k w b) (f32_causal_conv1d chans k w b).
Proof.
  intros chans k w b Hk. split; [reflexivity|].
  intros xs x. rewrite after_conv by exact Hk. unfold conv_state_step. cbn [snd].
  unfold f32_causal_conv1d. rewrite length_app. cbn [List.length].
  rewrite seq_app, map_app. cbn [List.seq List.map]. rewrite Nat.add_0_l. f_equal.
  - apply map_ext_in. intros t Ht. apply in_seq in Ht.
    rewrite conv_window_snoc by lia. reflexivity.
  - f_equal. f_equal.
    rewrite <- (conv_window_cached_correct chans k (xs ++ [x]) (List.length xs) Hk)
      by (rewrite length_app; cbn [List.length]; lia).
    rewrite conv_hist_snoc, nth_snoc_last. reflexivity.
Qed.

(** * The gated delta recurrence *)

Definition delta_state_step {A : Type} (Bf Gf : A -> binary32)
    (Qf Kf Vf : A -> list binary32) (st : list (list binary32)) (x : A)
    : list (list binary32) * list binary32 :=
  f32_delta_step (Bf x) (Gf x) (Qf x) (Kf x) (Vf x) st.

Lemma after_delta : forall {A : Type} (Bf Gf : A -> binary32)
    (Qf Kf Vf : A -> list binary32) st0 xs,
  after (delta_state_step Bf Gf Qf Kf Vf) st0 xs
  = f32_delta_state (List.map Bf xs) (List.map Gf xs) (List.map Qf xs)
                    (List.map Kf xs) (List.map Vf xs) st0.
Proof.
  intros A Bf Gf Qf Kf Vf st0 xs. induction xs as [|x xs IH] using rev_ind; [reflexivity|].
  rewrite after_snoc, IH, !map_app.
  rewrite delta_state_app by (rewrite !length_map; reflexivity).
  reflexivity.
Qed.

Lemma realized_delta : forall {A : Type} (Bf Gf : A -> binary32)
    (Qf Kf Vf : A -> list binary32) st0,
  realized st0 (delta_state_step Bf Gf Qf Kf Vf)
    (fun xs => f32_delta_scan (List.map Bf xs) (List.map Gf xs) (List.map Qf xs)
                              (List.map Kf xs) (List.map Vf xs) st0).
Proof.
  intros A Bf Gf Qf Kf Vf st0. split; [reflexivity|].
  intros xs x. rewrite after_delta. unfold delta_state_step. rewrite !map_app.
  apply delta_scan_snoc; rewrite !length_map; reflexivity.
Qed.

(** * A stack of layers *)

Fixpoint stack_step {S : Type} (steps : list (S -> list binary32 -> S * list binary32))
    (ss : list S) (x : list binary32) : list S * list binary32 :=
  match steps, ss with
  | st :: rest, s :: srest =>
      let r := st s x in
      let r' := stack_step rest srest (snd r) in
      (fst r :: fst r', snd r')
  | _, _ => ([], x)
  end.

Fixpoint stack_fun (fs : list (list (list binary32) -> list (list binary32)))
    (h : list (list binary32)) : list (list binary32) :=
  match fs with
  | [] => h
  | f :: rest => stack_fun rest (f h)
  end.

Lemma realized_stack : forall {S : Type} fs
    (ms : list (S * (S -> list binary32 -> S * list binary32))),
  Forall2 (fun f m => realized (fst m) (snd m) f) fs ms ->
  realized (List.map fst ms) (stack_step (List.map snd ms)) (stack_fun fs).
Proof.
  intros S fs ms H. induction H as [|f m fs ms Hf Hrest IH].
  - split; [reflexivity|]. intros xs x. cbn [List.map stack_fun stack_step snd].
    reflexivity.
  - assert (Af : forall xs, after (stack_step (snd m :: List.map snd ms))
                                  (fst m :: List.map fst ms) xs
                 = after (snd m) (fst m) xs
                   :: after (stack_step (List.map snd ms)) (List.map fst ms) (f xs)).
    { destruct Hf as [F0 Fs]. intros xs.
      induction xs as [|x xs IHx] using rev_ind; [cbn; now rewrite F0|].
      rewrite !after_snoc, IHx, Fs, after_snoc. reflexivity. }
    destruct Hf as [F0 Fs]. destruct IH as [R0 Rs].
    split; [cbn [stack_fun]; now rewrite F0, R0|].
    intros xs x. cbn [List.map stack_fun]. rewrite Af.
    cbn [stack_step fst snd]. rewrite Fs, Rs. reflexivity.
Qed.
