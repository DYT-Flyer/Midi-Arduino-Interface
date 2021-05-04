/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * _coder_return_c_api.h
 *
 * Code generation for function 'return_c'
 *
 */

#ifndef _CODER_RETURN_C_API_H
#define _CODER_RETURN_C_API_H

/* Include files */
#include "emlrt.h"
#include "tmwtypes.h"
#include <string.h>

/* Variable Declarations */
extern emlrtCTX emlrtRootTLSGlobal;
extern emlrtContext emlrtContextGlobal;

#ifdef __cplusplus

extern "C" {

#endif

  /* Function Declarations */
  real_T return_c(void);
  void return_c_api(const mxArray *plhs[1]);
  void return_c_atexit(void);
  void return_c_initialize(void);
  void return_c_terminate(void);
  void return_c_xil_shutdown(void);
  void return_c_xil_terminate(void);

#ifdef __cplusplus

}
#endif
#endif

/* End of code generation (_coder_return_c_api.h) */
