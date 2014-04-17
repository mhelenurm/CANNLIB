#include "activation.h"
#include <math.h>
#include <stdlib.h>

double activationEval(activation act, double input)
{
  return act.function(act.data, input);
}

double activationDerEval(activation act, double input)
{
  return act.derivative(act.data, input);
}

void activation_free(activation act)
{
  free(act.data);
}

double activationSigmoidFunc(void* data, double input)
{
  return 1.0/(1.0 + exp(-((double*)data)[0]*input));
}

double activationSigmoidDeri(void* data, double input)
{
  double func = 1.0/(1.0+exp(-((double*)data)[0]*input));
  return ((double*)data)[0] * func * (1.0-func);
}

activation activation_make_sigmoid(double k)
{
  double* stuff = (double*)malloc(sizeof(double));
  stuff[0] = k;
  activation act = (activation){stuff, &activationSigmoidFunc, &activationSigmoidDeri};
  return act;
}

double activationTanhFunc(void* data, double input)
{
  return tanh(input);
}

double activationTanhDeri(void* data, double input)
{
  double tanhr = tanh(input);
  return 1.0-tanhr*tanhr;
}

activation activation_make_tanh()
{
  activation act = (activation){0, &activationTanhFunc, &activationTanhDeri};
  return act;
}

double activationStepFunc(void* data, double input)
{
  return (input<0)?0:1;
}

double activationStepDeri(void* data, double input)
{
  return 0;
}

activation activation_make_step()
{
  return (activation){0, &activationStepFunc, &activationStepDeri};
}

double activationLinearFunc(void* data, double input)
{
  return ((double*)data)[0] * input + ((double*)data)[1];
}

double activationLinearDeri(void* data, double input)
{
  return ((double*)data)[0];
}

activation activation_make_linear(double a, double b)
{
  double* stuff = (double*)malloc(sizeof(double)*2);
  stuff[0] = a;
  stuff[1] = b;
  return (activation){stuff, &activationLinearFunc, &activationLinearDeri};
}

double activationRectifierFunc(void* data, double input)
{
  return (input<0)?0:input;
}

double activationRectifierDeri(void* data, double input)
{
  return (input<0)?0:1;
}

activation activation_make_rectifier()
{
  return (activation){0, &activationRectifierFunc, &activationRectifierDeri};
}

double activationSoftplusFunc(void* data, double input)
{
  return log(1.0 + exp(input));
}

double activationSoftplusDeri(void* data, double input)
{
  return 1.0/(1.0 + exp(-input));
}

activation activation_make_softplus()
{
  return (activation){0, &activationSoftplusFunc, &activationSoftplusDeri};
}
