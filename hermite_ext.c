/*
 * hermite_ext.c — C99 implementation of _hermite_basis_forward
 *
 * Computes nh odd-order normalized Hermite functions F_1, F_3, …, F_{2nh-1}
 * at N evaluation points using a log-scaled forward 3-term recurrence.
 *
 * Output layout: Fortran-contiguous (column-major), shape (N, nh).
 * Column ih = F_{2ih+1}(x[0..N-1]), stored at out[ih*N .. ih*N+N-1].
 */

#include "hermite_ext.h"
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* 2^n for integer n in [-1022, 1023] via IEEE 754 exponent-field injection.
 * No libm call; single-instruction on any platform; vectorizes inside i-loops. */
static inline double pow2_int(int n)
{
    uint64_t bits = (uint64_t)(n + 1023) << 52;
    double r;
    memcpy(&r, &bits, 8);
    return r;
}

/* Branchless frexp for non-zero x: sets mant in [0.5, 1) and *exp = binary exponent.
 * Assumes x != 0 and |x| is a normal double (no denormals).  Safe here because
 * all inputs are x = Q/qp with Q >= qfitmin > 0, and intermediate mantissas
 * from the recurrence are always well-conditioned normal doubles. */
static inline void fast_frexp(double x, double* mant, int* exp)
{
    uint64_t bits;
    memcpy(&bits, &x, 8);
    *exp  = (int)((bits >> 52) & 0x7FFu) - 1022;
    bits  = (bits & UINT64_C(0x800FFFFFFFFFFFFF)) | UINT64_C(0x3FE0000000000000);
    memcpy(mant, &bits, 8);
}

void hermite_basis_forward_c(const double* x, int N, int nh, double* out)
{
    if (N <= 0 || nh <= 0) return;

    /* Working arrays — integer exponents for O(1) scaling via pow2_int */
    double* log2c = (double*)malloc((size_t)N * sizeof(double));
    double* Gpm   = (double*)malloc((size_t)N * sizeof(double));
    double* Gcm   = (double*)malloc((size_t)N * sizeof(double));
    double* nm    = (double*)malloc((size_t)N * sizeof(double));
    int*    Gpe   = (int*)   malloc((size_t)N * sizeof(int));
    int*    Gce   = (int*)   malloc((size_t)N * sizeof(int));
    int*    ne    = (int*)   malloc((size_t)N * sizeof(int));

    /* log2c[i] = log2(pi^(-1/4) * exp(-x^2/2)) = -0.25*log2(pi) - x^2/(2*ln2)
     * Kept as log2 (not exp2-ed) so we can add Gce[i] before calling exp2
     * once in the store — prevents overflow when Gce[i] is large positive and
     * log2c[i] is large negative (they cancel in log space). */
    static const double LOG2_PI_025 = 0.41287403236807973;  /* 0.25*log2(pi) */
    static const double INV_2LN2    = 0.72134752044448171;  /* 1/(2*ln(2))   */

    for (int i = 0; i < N; i++) {
        log2c[i] = -LOG2_PI_025 - x[i] * x[i] * INV_2LN2;
    }

    /* G_0 = 1 → frexp(1) = (0.5, 1) */
    for (int i = 0; i < N; i++) {
        Gpm[i] = 0.5;
        Gpe[i] = 1;
    }

    /* G_1 = sqrt(2) * x */
    static const double SQRT2 = 1.4142135623730951;
    for (int i = 0; i < N; i++) {
        fast_frexp(SQRT2 * x[i], &Gcm[i], &Gce[i]);
    }

    /* Store F_1 (column 0) */
    {
        double* col = out;
        for (int i = 0; i < N; i++) {
            col[i] = Gcm[i] * exp2((double)Gce[i] + log2c[i]);
        }
    }

    int ih = 1;
    for (int n = 1; n < 2 * nh - 1; n++) {
        double An = sqrt(2.0 / (n + 1));
        double Bn = sqrt((double)n / (n + 1.0));

        /* Recurrence step: fused per-element loop — all scalar, no exp2() calls */
        for (int i = 0; i < N; i++) {
            /* t1 = An * x[i] * G_curr — normalize to get integer exponent */
            double t1m; int t1e;
            fast_frexp(An * x[i] * Gcm[i], &t1m, &t1e);
            t1e += Gce[i];

            /* t2 = Bn * G_prev */
            double t2m = Bn * Gpm[i];
            int    t2e = Gpe[i];

            /* Subtract at common binary scale using pow2_int (no libm) */
            int maxe = t1e > t2e ? t1e : t2e;
            double diff = t1m * pow2_int(t1e - maxe)
                        - t2m * pow2_int(t2e - maxe);

            int fe;
            fast_frexp(diff, &nm[i], &fe);
            ne[i] = maxe + fe;

            /* Shift: prev ← curr, curr ← next */
            Gpm[i] = Gcm[i];
            Gpe[i] = Gce[i];
            Gcm[i] = nm[i];
            Gce[i] = ne[i];
        }

        /* n+1 is odd when n is even → store column ih */
        if (n % 2 == 0 && ih < nh) {
            double* col = out + (size_t)ih * N;
            for (int i = 0; i < N; i++) {
                col[i] = Gcm[i] * exp2((double)Gce[i] + log2c[i]);
            }
            ih++;
        }
    }

    free(log2c); free(Gpm); free(Gcm); free(nm);
    free(Gpe);   free(Gce); free(ne);
}
