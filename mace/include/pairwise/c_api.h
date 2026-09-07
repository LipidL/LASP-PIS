#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
// Backend: 0 Auto, 1 BruteForce, 2 CellList. Handles are single-flight.
void *pw_create_lj(float epsilon, float sigma, float cutoff, int backend);
void *pw_create_mh1(const char *archive, int backend);
void pw_destroy(void *handle);
const char *pw_last_error(void);
int pw_calculate(void *handle, int n, const float *xyz, const int *elements, const float *cell,
                 double *energy, float *per_atom, float *force, float *virial);
int pw_diagnostics(void *handle, char *buffer, size_t capacity);
#ifdef __cplusplus
}
#endif
