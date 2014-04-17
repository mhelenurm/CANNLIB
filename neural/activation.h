#ifndef MHLIB_ACTIVATION_H
#define MHLIB_ACTIVATION_H

typedef double (*activationFxn)(void*, double);
typedef double (*activationDer)(void*, double);

typedef struct
{
  void* data;
  activationFxn function;
  activationDer derivative;
} activation;

extern double activationEval(activation act, double input);
extern double activationDerEval(activation act, double input);
extern void activation_free(activation act);

extern double activationSigmoidFunc(void* data, double input);
extern double activationSigmoidDeri(void* data, double input);
extern activation activation_make_sigmoid(double k);

extern double activationTanhFunc(void* data, double input);
extern double activationTanhDeri(void* data, double input);
extern activation activation_make_tanh();

extern double activationStepFunc(void* data, double input);
extern double activationStepDeri(void* data, double input);
extern activation activation_make_step();

extern double activationLinearFunc(void* data, double input);
extern double activationLinearDeri(void* data, double input);
extern activation activation_make_linear(double a, double b);

extern double activationRectifierFunc(void* data, double input);
extern double activationRectifierDeri(void* data, double input);
extern activation activation_make_rectifier();

extern double activationSoftplusFunc(void* data, double input);
extern double activationSoftplusDeri(void* data, double input);
extern activation activation_make_softplus();

#endif
