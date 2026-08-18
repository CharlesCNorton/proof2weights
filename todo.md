# Outstanding work

## Regularity witnesses, record by record

The reading is in place and demonstrated. `Qb` reads a binary32 as the rational
it exactly denotes, `Qb_correct` says that reading is faithful, and `regz_Q`,
`abs_le_Q` and `le_abs_Q` turn a side condition into comparisons a machine
decides, since every quantity a concrete witness produces is built from `B2R`
of concrete floats by addition, multiplication and division. `regz_e_r2_one`
and `abs_e_r_one` are worked instances over the exponential's reduced argument,
and `amp_ok_ones` witnesses the amplification budget.

Assembling a full assignment per record ran into two costs, and the lemmas that
answer them are proved and waiting.

The exponential squares eight times, so the rational it produces carries a
denominator raised to the two hundred and fifty sixth power. At the argument
one the reduced argument is two to the minus eight and its six powers are exact
powers of two, which keeps that within reach; away from a power of two, as at
GELU's coefficient of 1.702, evaluating it exhausts the evaluator. The
iterations have to be bounded instead, which is what `iter_sq_le_one` does for
any argument whose polynomial lands inside the unit disc. The other conditions
that read the exponential's real value, `sgr_lo_r` and `glr_bs`, want the same
treatment: the value is a square plus one, so it is bounded below by one and its
reciprocal above by one, without computing either.

Separately, the conversion is too slow per goal to drive a record with fifty-six
conditions the way it drives one condition. It should convert once for the whole
record rather than once per field.

Records that pass through a square root need a point where the radicand is a
rational square, since roots are irrational in general; `sqrt_of_sq` reads the
root off there. A one-element RMSNorm row with the shift at zero is such a
point, as is a two-element layer-norm row over one and two, whose variance is
exactly a quarter.

## Backward error above the linear layers

`f32_dot_backward` gives the backward-error form for the dot product, and
`f32_mat_vec_mul_backward` and `logits_backward` carry it to the matrix-vector
product and to the tied-embedding projection of all three models. Layer
normalization, the exponential and softmax have no backward form, so the
statement stops where the network stops being linear. Carrying it further would
mean a perturbation of the weights rather than of the products.
