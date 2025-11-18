/* Copyright 2017 UFACTORY Inc. All Rights Reserved.
 *
 * Software License Agreement (BSD License)
 *
 * Author: Jimy Zhang <jimy92@163.com>
 ============================================================================*/

#ifndef XARM_CORE_DEBUG_DEBUG_PRINT_H_
#define XARM_CORE_DEBUG_DEBUG_PRINT_H_

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Print a vector of n float values
 * @param name: prefix string to print before the vector
 * @param vect: pointer to float array
 * @param n: number of elements in the vector
 */
static inline void print_nvect(const char *name, const float *vect, int n) {
  printf("%s", name);
  for (int i = 0; i < n; i++) {
    printf("%.6f", vect[i]);
    if (i < n - 1) printf(", ");
  }
  printf("\n");
}

/**
 * Print hexadecimal data
 * @param name: prefix string to print before the data
 * @param data: pointer to unsigned char array
 * @param len: number of bytes to print
 */
static inline void print_hex(const char *name, const unsigned char *data, int len) {
  printf("%s", name);
  for (int i = 0; i < len; i++) {
    printf("%02X ", data[i]);
    if ((i + 1) % 16 == 0) printf("\n");
  }
  if (len % 16 != 0) printf("\n");
}

#ifdef __cplusplus
}

// C++ overloads (outside extern "C" to allow overloading)
/**
 * Print a vector of n int values (C++ overload for int arrays)
 * @param name: prefix string to print before the vector
 * @param vect: pointer to int array
 * @param n: number of elements in the vector
 */
static inline void print_nvect(const char *name, const int *vect, int n) {
  printf("%s", name);
  for (int i = 0; i < n; i++) {
    printf("%d", vect[i]);
    if (i < n - 1) printf(", ");
  }
  printf("\n");
}

#endif

#endif  // XARM_CORE_DEBUG_DEBUG_PRINT_H_

