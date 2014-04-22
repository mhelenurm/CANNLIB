#ifndef MHLIB_ACTIVATION_H
#define MHLIB_ACTIVATION_H

#include "neural_types.h"

typedef decimal (*activationFxn)(void*, decimal);
typedef decimal (*activationDer)(void*, decimal);

typedef struct
{
  void* data;
  activationFxn function;
  activationDer derivative;
} activation;

extern decimal activationEval(activation act, decimal input);
extern decimal activationFuncMax(activation act);
extern decimal activationFuncMin(activation act);
extern decimal activationDerEval(activation act, decimal input);
extern void activation_free(activation act);

extern decimal activationSigmoidFunc(void* data, decimal input);
extern decimal activationSigmoidDeri(void* data, decimal input);
extern activation activation_make_sigmoid(decimal k);

extern decimal activationTanhFunc(void* data, decimal input);
extern decimal activationTanhDeri(void* data, decimal input);
extern activation activation_make_tanh();

extern decimal activationStepFunc(void* data, decimal input);
extern decimal activationStepDeri(void* data, decimal input);
extern activation activation_make_step();

extern decimal activationLinearFunc(void* data, decimal input);
extern decimal activationLinearDeri(void* data, decimal input);
extern activation activation_make_linear(decimal a, decimal b);

extern decimal activationRectifierFunc(void* data, decimal input);
extern decimal activationRectifierDeri(void* data, decimal input);
extern activation activation_make_rectifier();

extern decimal activationSoftplusFunc(void* data, decimal input);
extern decimal activationSoftplusDeri(void* data, decimal input);
extern activation activation_make_softplus();

extern decimal activationInverseAbsFunc(void* data, decimal input);
extern decimal activationInverseAbsDeri(void* data, decimal input);
extern activation activation_make_inverseAbs();

#endif
