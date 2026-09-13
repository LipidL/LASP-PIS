#pragma once
#include <mace/export.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
// Backend: 0 Auto, 1 BruteForce, 2 CellList. Handles are single-flight.
MACE_EXPORT void *mace_create_mh1(const char *archive, int backend);
// Null/empty head uses the checkpoint default; archives have a fixed head.
MACE_EXPORT void *mace_create_mh1_head(const char *model, int backend, const char *head);
MACE_EXPORT void mace_destroy(void *handle);
MACE_EXPORT const char *mace_last_error(void);
MACE_EXPORT int mace_calculate(void *handle, int n, const float *xyz, const int *elements, const float *cell,
                 double *energy, float *per_atom, float *force, float *virial);
MACE_EXPORT int mace_diagnostics(void *handle, char *buffer, size_t capacity);
#ifdef __cplusplus
}
#endif
