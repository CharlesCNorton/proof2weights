open Phases1_15_complete

let rec pos_of_int n =
  if n <= 1 then XH
  else if n land 1 = 0 then XO (pos_of_int (n / 2)) else XI (pos_of_int (n / 2))
let z_of_int n = if n = 0 then Z0 else if n > 0 then Zpos (pos_of_int n) else Zneg (pos_of_int (-n))
let rec int_of_pos = function XH -> 1 | XO p -> 2 * int_of_pos p | XI p -> 2 * int_of_pos p + 1
let int_of_z = function Z0 -> 0 | Zpos p -> int_of_pos p | Zneg p -> - (int_of_pos p)

let b2f = function
  | B754_zero _ -> 0.0
  | B754_infinity s -> if s then neg_infinity else infinity
  | B754_nan -> nan
  | B754_finite (s, m, e) ->
      (if s then -1.0 else 1.0) *. float_of_int (int_of_pos m) *. (2.0 ** float_of_int (int_of_z e))

let p name x = Printf.printf "%-22s = %.9g\n" name (b2f x)
let zi = z_of_int
let t a b = f32_div (f32_of_Z (zi a)) (f32_of_Z (zi b))
let () =
  let big = f32_of_Z (zi (-1000000000)) in
  p "2 + 3" (f32_plus f32_two (f32_of_Z (zi 3)));
  p "0.001 + 0.002" (f32_plus (t 1 1000) (t 2 1000));
  p "1.0 + (-1e9)" (f32_plus f32_one big);
  p "0.001 + (-1e9)" (f32_plus (t 1 1000) big);
  p "0.0008 + (-1e9)" (f32_plus (t 8 10000) big);
  p "-0.005 + (-1e9)" (f32_plus (t (-5) 1000) big);
  p "1e6 + (-1e9)" (f32_plus (f32_of_Z (zi 1000000)) big)
