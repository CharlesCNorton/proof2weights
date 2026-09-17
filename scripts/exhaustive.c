/* exhaustive.c - every elementary function on every binary32 input.
 *
 * The extracted elementary functions are compositions of five primitives, and
 * the native build performs each primitive as the binary64 operation narrowed
 * to binary32, which Narrow.v proves equal to the binary32 operation Flocq
 * specifies. This file performs the same compositions, in the same order, with
 * each primitive written that way, and evaluates every one of the 2^32 binary32
 * inputs against the mathematical function computed in binary64.
 *
 * Two quantities are reported per function. The largest error in units in the
 * last place of the result, and the number of inputs at which the result is not
 * the correctly rounded one. Because the reference is a binary64 libm value
 * rather than an infinitely precise one, an input whose true value lies within
 * a guard band of a binary32 midpoint cannot be decided from it; those are
 * counted separately, and scripts/settle_undecided.py decides them against a
 * reference at 120 decimal digits.
 *
 *   gcc -O2 -fopenmp -o exhaustive exhaustive.c -lm
 *   ./exhaustive [function ...]
 *   P2W_UNDECIDED=<dir> ./exhaustive        # also record the undecided inputs
 *
 * With no function named, every function is swept.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* When P2W_UNDECIDED names a directory, each function's undecided inputs are
   written there as <name>.u32, two little-endian words each, the input and the
   result this file computed, so a higher precision reference can settle them
   without repeating the composition. */
static const char *undec_dir = NULL;

static inline float fb(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline uint32_t bf(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

/* the five primitives, each the binary64 result narrowed to binary32 */
static inline float ad(float a, float b) { return (float)((double)a + (double)b); }
static inline float sb(float a, float b) { return (float)((double)a - (double)b); }
static inline float ml(float a, float b) { return (float)((double)a * (double)b); }
static inline float dv(float a, float b) { return (float)((double)a / (double)b); }
static inline float sq(float a)          { return (float)sqrt((double)a); }
static inline float oz(double n)         { return (float)n; }

/* constants, built as the definitions build them */
static float C1, C2, CH, Ctwo, Csix, C24, C120, C720, C5040;
static float Cthree, Cfive, Cseven, Cnine, Celeven, Cthirteen;
static float Cln2hi, Cln2lo, Cinvln2, Cmagic, ChiE, CloE;
static float Cg1, Cg2;
static float Cp4, Cp16, Cp256, Cp65536, Cp2_32, Cp2_64;
static float Cf[20], C2pi_hi, C2pi_lo, Cinv2pi;

static void init_consts(void) {
  C1 = oz(1); Ctwo = oz(2); Csix = oz(6); C24 = oz(24); C120 = oz(120);
  C720 = oz(720); C5040 = oz(5040);
  Cthree = oz(3); Cfive = oz(5); Cseven = oz(7); Cnine = oz(9);
  Celeven = oz(11); Cthirteen = oz(13);
  CH = dv(oz(1), oz(2));
  Cln2hi = dv(oz(355), oz(512));
  Cln2lo = dv(oz(14581891), oz(68719476736.0));
  Cinvln2 = dv(oz(12102203), oz(8388608));
  Cmagic = oz(12582912);
  ChiE = oz(88); CloE = oz(-88);
  Cg1 = dv(oz(7978845608.0), oz(10000000000.0));
  Cg2 = dv(oz(44715), oz(1000000));
  Cp4 = ml(Ctwo, Ctwo); Cp16 = ml(Cp4, Cp4); Cp256 = ml(Cp16, Cp16);
  Cp65536 = ml(Cp256, Cp256); Cp2_32 = ml(Cp65536, Cp65536);
  Cp2_64 = ml(Cp2_32, Cp2_32);
  /* the factorials as binary32 stores them */
  static const double fac[20] = {0,0,2,6,24,120,720,5040,40320,362880,3628800,
    39916800,479001600,6227020800.0,87178291200.0,1307674368000.0,
    20922789888000.0,355687428096000.0,6402373705728000.0,
    121645100408832000.0};
  for (int i = 2; i < 20; i++) Cf[i] = oz(fac[i]);
  C2pi_hi = dv(oz(201), oz(32));
  C2pi_lo = dv(oz(19353072), oz(10000000000.0));
  {
    double tp = 2.0 * 3.14159265358979323846;
    C2pi_lo = dv(oz(19353072), oz(10000000000.0));
    Cinv2pi = dv(C1, (float)tp);
  }
}

static inline float round_int(float y) { return sb(ad(y, Cmagic), Cmagic); }

static float pow2_nat(float a) {
  float acc = C1;
  const float c[7] = {oz(64), oz(32), oz(16), oz(8), oz(4), Ctwo, C1};
  const float p[7] = {Cp2_64, Cp2_32, Cp65536, Cp256, Cp16, Cp4, Ctwo};
  for (int i = 0; i < 7; i++)
    if (c[i] <= a) { a = sb(a, c[i]); acc = ml(acc, p[i]); }
  return acc;
}

static float f_pow2(float k) {
  if (k < 0.0f) return dv(C1, pow2_nat(-k));
  return pow2_nat(k);
}

static float f_exp(float x) {
  float xc = (ChiE < x) ? ChiE : ((x < CloE) ? CloE : x);
  float k = round_int(ml(xc, Cinvln2));
  float r = ad(sb(xc, ml(k, Cln2hi)), ml(k, Cln2lo));
  float r2 = ml(r, r), r3 = ml(r2, r), r4 = ml(r3, r), r5 = ml(r4, r);
  float r6 = ml(r5, r), r7 = ml(r6, r);
  float s = ad(C1, ad(r, ad(dv(r2, Ctwo), ad(dv(r3, Csix), ad(dv(r4, C24),
            ad(dv(r5, C120), ad(dv(r6, C720), dv(r7, C5040))))))));
  return ml(s, f_pow2(k));
}

static float f_sigmoid(float x) { return dv(C1, ad(C1, f_exp(-x))); }
static float f_tanh(float x)    { return sb(ml(Ctwo, f_sigmoid(ml(Ctwo, x))), C1); }

static float f_gelu(float x) {
  float x3 = ml(x, ml(x, x));
  float inner = ml(Cg1, ad(x, ml(Cg2, x3)));
  return ml(ml(CH, x), ad(C1, f_tanh(inner)));
}

static float f_log_unit(float m) {
  float u = dv(sb(m, C1), ad(m, C1));
  float u2 = ml(u, u), u3 = ml(u2, u), u5 = ml(u3, u2), u7 = ml(u5, u2);
  float u9 = ml(u7, u2), u11 = ml(u9, u2), u13 = ml(u11, u2);
  float s = ad(u, ad(dv(u3, Cthree), ad(dv(u5, Cfive), ad(dv(u7, Cseven),
            ad(dv(u9, Cnine), ad(dv(u11, Celeven), dv(u13, Cthirteen)))))));
  return ml(Ctwo, s);
}

static float f_softplus(float x) {
  float ax = fabsf(x);
  float e = f_exp(-ax);
  float mx = (x < 0.0f) ? 0.0f : x;
  return ad(mx, f_log_unit(ad(C1, e)));
}

static float reduce_2pi(float x) {
  float k = round_int(ml(x, Cinv2pi));
  float r = sb(x, ml(k, C2pi_hi));
  return sb(r, ml(k, C2pi_lo));
}

static float f_sin(float x) {
  float r = reduce_2pi(x);
  float r2 = ml(r, r);
  float p[20]; p[3] = ml(r2, r);
  for (int n = 5; n <= 19; n += 2) p[n] = ml(p[n - 2], r2);
  float a = sb(dv(p[17], Cf[17]), dv(p[19], Cf[19]));
  a = sb(a, dv(p[15], Cf[15])); a = ad(a, dv(p[13], Cf[13]));
  a = sb(a, dv(p[11], Cf[11])); a = ad(a, dv(p[9], Cf[9]));
  a = sb(a, dv(p[7], Cf[7]));   a = ad(a, dv(p[5], Cf[5]));
  a = sb(a, dv(p[3], Cf[3]));
  return ad(r, a);
}

static float f_cos(float x) {
  float r = reduce_2pi(x);
  float p[20]; p[2] = ml(r, r);
  for (int n = 4; n <= 18; n += 2) p[n] = ml(p[n - 2], p[2]);
  float a = sb(dv(p[16], Cf[16]), dv(p[18], Cf[18]));
  a = sb(a, dv(p[14], Cf[14])); a = ad(a, dv(p[12], Cf[12]));
  a = sb(a, dv(p[10], Cf[10])); a = ad(a, dv(p[8], Cf[8]));
  a = sb(a, dv(p[6], Cf[6]));   a = ad(a, dv(p[4], Cf[4]));
  a = sb(a, dv(p[2], Cf[2]));
  return ad(C1, a);
}

static float f_sqrt(float x) { return sq(x); }

/* references, in binary64 */
static double r_exp(double x)      { double c = x > 88 ? 88 : (x < -88 ? -88 : x); return exp(c); }
static double r_sigmoid(double x)  { double c = -x; c = c > 88 ? 88 : (c < -88 ? -88 : c); return 1.0 / (1.0 + exp(c)); }
static double r_tanh(double x)     { return r_sigmoid(2*x) * 2.0 - 1.0; }
static double r_gelu(double x)     { double t = 0.7978845608028654 * (x + 0.044715*x*x*x); return 0.5*x*(1.0 + tanh(t)); }
static double r_log_unit(double m) { return log(m); }
static double r_softplus(double x) { return (x > 0 ? x : 0) + log1p(exp(-fabs(x))); }
static double r_sin(double x)      { return sin(x); }
static double r_cos(double x)      { return cos(x); }
static double r_sqrt(double x)     { return sqrt(x); }

typedef float  (*ffn)(float);
typedef double (*dfn)(double);

typedef struct { const char *name; ffn f; dfn r; double lo, hi; } Case;

static const Case CASES[] = {
  { "exp",      f_exp,      r_exp,      -88.0,     88.0 },
  { "sigmoid",  f_sigmoid,  r_sigmoid,  -88.0,     88.0 },
  { "tanh",     f_tanh,     r_tanh,     -44.0,     44.0 },
  { "gelu",     f_gelu,     r_gelu,      -8.0,      8.0 },
  { "log",      f_log_unit, r_log_unit,   1.0,      2.0 },
  { "softplus", f_softplus, r_softplus, -88.0,     88.0 },
  { "sin",      f_sin,      r_sin,  -262144.0, 262144.0 },
  { "cos",      f_cos,      r_cos,  -262144.0, 262144.0 },
  { "sqrt",     f_sqrt,     r_sqrt,       0.0,    HUGE_VAL },
};
static const int NCASES = (int)(sizeof CASES / sizeof CASES[0]);

/* ulp of the binary32 neighbourhood of v */
static double ulp32(double v) {
  float f = (float)fabs(v);
  if (!(f > 0.0f)) return ldexp(1.0, -149);
  int e; frexpf(f, &e);
  double u = ldexp(1.0, e - 24);
  double tiny = ldexp(1.0, -149);
  return u < tiny ? tiny : u;
}

int main(int argc, char **argv) {
  init_consts();
  undec_dir = getenv("P2W_UNDECIDED");
  printf("%-9s %13s %12s %11s %11s %12s %10s\n",
         "function", "inputs", "max ulp", "max abs", "max rel",
         "not c.r.", "undecided");
  for (int ci = 0; ci < NCASES; ci++) {
    const Case *c = &CASES[ci];
    if (argc > 1) {
      int want = 0;
      for (int a = 1; a < argc; a++) if (!strcmp(argv[a], c->name)) want = 1;
      if (!want) continue;
    }
    double worst = 0.0; uint32_t worst_in = 0;
    double wabs = 0.0, wrel = 0.0;
    long long n = 0, bad = 0, undec = 0;
#pragma omp parallel reduction(+:n,bad,undec) reduction(max:wabs,wrel)
    {
      double tworst = 0.0; uint32_t tworst_in = 0;
      uint32_t *ulist = NULL; long un = 0, ucap = 0;
      if (undec_dir) { ucap = 1 << 21; ulist = malloc(ucap * sizeof(uint32_t)); }
#pragma omp for schedule(static)
      for (long long i = 0; i <= 0xFFFFFFFFLL; i++) {
        uint32_t u = (uint32_t)i;
        float x = fb(u);
        if (!(x == x) || fabsf(x) == INFINITY) continue;
        if ((double)x < c->lo || (double)x > c->hi) continue;
        float got = c->f(x);
        double ref = c->r((double)x);
        if (!(ref == ref) || fabs(ref) == INFINITY) continue;
        n++;
        double uu = ulp32(ref);
        double aerr = fabs((double)got - ref);
        double err = aerr / uu;
        if (aerr > wabs) wabs = aerr;
        if (ref != 0.0) { double rr = aerr / fabs(ref); if (rr > wrel) wrel = rr; }
        float cr = (float)ref;
        if (bf(got) != bf(cr)) {
          /* within a guard band of a midpoint the binary64 reference cannot
             decide the binary32 rounding */
          double frac = fabs(ref) / uu;
          double d = fabs(frac - floor(frac) - 0.5);
          if (d < 1e-6) {
            undec++;
            if (ulist && un + 1 < ucap) {
              ulist[un++] = u;
              ulist[un++] = bf(got);
            }
          } else bad++;
        }
        if (err > tworst) { tworst = err; tworst_in = u; }
      }
#pragma omp critical
      {
        if (tworst > worst) { worst = tworst; worst_in = tworst_in; }
        if (ulist && un) {
          char path[512];
          snprintf(path, sizeof path, "%s/%s.u32", undec_dir, c->name);
          FILE *fh = fopen(path, "ab");
          if (fh) { fwrite(ulist, sizeof(uint32_t), (size_t)un, fh); fclose(fh); }
        }
      }
      free(ulist);
    }
    printf("%-9s %13lld %12.4f %11.3e %11.3e %12lld %10lld  ulp worst 0x%08x = %.9g\n",
           c->name, n, worst, wabs, wrel, bad, undec, worst_in,
           (double)fb(worst_in));
    fflush(stdout);
  }
  return 0;
}
