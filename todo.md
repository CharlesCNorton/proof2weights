# Outstanding work

## Regularity witnesses, record by record

The machinery is in place and demonstrated. `Qb` reads a binary32 as the
rational it exactly denotes, `Qb_correct` says that reading is faithful, and
`regz_Q`, `abs_le_Q` and `le_abs_Q` turn a side condition into comparisons a
machine checks by computation, since every quantity a concrete witness produces
is built from `B2R` of concrete floats by addition, multiplication and
division. `regz_e_r2_one` and `abs_e_r_one` are worked instances over the
exponential's reduced argument, and `amp_ok_ones` witnesses the amplification
budget.

What remains is applying it: assembling a full satisfying assignment for
`exp_reg`, and then for the records that contain it (`sig_reg`, `gelu_reg`,
`sp_reg`, `sm_reg`, `dstep_reg`, `conv_chan`) and the ones that do not
(`log_reg`, `sin_reg`, `cos_reg`, `rms_reg`, `ln_reg`, `ca_reg`). At
`f32_one` the exponential's reduced argument and its six powers are exact
powers of two, so those conditions check immediately; the coefficient divisions
and the eight squarings are where the rational check earns its keep.
`f32_dot_regular` and `amp_ok` are the two that already have witnesses.

## Backward error above the linear layers

`f32_dot_backward` gives the backward-error form for the dot product, and
`f32_mat_vec_mul_backward` and `logits_backward` carry it to the matrix-vector
product and to the tied-embedding projection of all three models. Layer
normalization, the exponential and softmax have no backward form, so the
statement stops where the network stops being linear. Carrying it through
would mean a perturbation of the weights rather than of the products.
