#pragma once
#ifdef __cplusplus
extern "C" {
#endif

void hermite_basis_forward_c(const double* x, int N, int nh, double* out);

#ifdef __cplusplus
}
#endif
