#include "activation.h"
#include <math.h>
#include <stdlib.h>

decimal activationEval(activation act, decimal input)
{
  return act.function(act.data, input);
}

decimal activationFuncMax(activation act)
{
  if(act.function == &activationSigmoidFunc)
    return 1.0;
  else if(act.function == &activationTanhFunc)
    return 1.0;
  else if(act.function == &activationStepFunc)
    return 1.0;
  else if(act.function == &activationLinearFunc)
    return sqrt(-1);
  else if(act.function == &activationRectifierFunc)
    return sqrt(-1);
  else if(act.function == &activationSoftplusFunc)
    return sqrt(-1);
  else if(act.function == &activationInverseAbsFunc)
    return 1.0;
  else
    return sqrt(-1);
}

decimal activationFuncMin(activation act)
{
  if(act.function == &activationSigmoidFunc)
    return 0.0;
  else if(act.function == &activationTanhFunc)
    return 0.0;
  else if(act.function == &activationStepFunc) 
    return 0.0;
  else if(act.function == &activationLinearFunc)
    return sqrt(-1);
  else if(act.function == &activationRectifierFunc)
    return 0.0;
  else if(act.function == &activationSoftplusFunc)
    return 0.0;
  else if(act.function == &activationInverseAbsFunc)
    return -1.0;
  else
    return sqrt(-1);
} 

decimal activationDerEval(activation act, decimal input)
{
  return act.derivative(act.data, input);
}

void activation_free(activation act)
{
  free(act.data);
}

decimal activationSigmoidFunc(void* data, decimal input)
{
#ifdef USE_DOUBLE
  return 1.0/(1.0 + exp(-((decimal*)data)[0]*input));
#endif
#ifndef USE_DOUBLE
  return 1.0/(1.0 + expf(-((decimal*)data)[0]*input));
#endif
}

decimal activationSigmoidDeri(void* data, decimal input)
{
#ifdef USE_DOUBLE
  decimal func = 1.0/(1.0+exp(-((decimal*)data)[0]*input));
#endif
#ifndef USE_DOUBLE
  decimal func = 1.0/(1.0+exp(-((decimal*)data)[0]*input));
#endif
  return ((decimal*)data)[0] * func * (1.0-func);
}

activation activation_make_sigmoid(decimal k)
{
  decimal* stuff = (decimal*)malloc(sizeof(decimal));
  stuff[0] = k;
  activation act = (activation){stuff, &activationSigmoidFunc, &activationSigmoidDeri};
  return act;
}

decimal activationTanhFunc(void* data, decimal input)
{
#ifdef USE_DOUBLE
  return (tanh(input)+1.0)/2.0;
#endif
#ifndef USE_DOUBLE
  return tanhf(input);
#endif
}

decimal activationTanhDeri(void* data, decimal input)
{
#ifdef USE_DOUBLE
  decimal tanhr = tanh(input);
#endif
#ifndef USE_DOUBLE
  decimal tanhr = tanhf(input);
#endif
  return .5*(1.0-tanhr*tanhr);
}

activation activation_make_tanh()
{
  activation act = (activation){0, &activationTanhFunc, &activationTanhDeri};
  return act;
}

decimal activationStepFunc(void* data, decimal input)
{
  return (input<0)?0:1;
}

decimal activationStepDeri(void* data, decimal input)
{
  return 0;
}

activation activation_make_step()
{
  return (activation){0, &activationStepFunc, &activationStepDeri};
}

decimal activationLinearFunc(void* data, decimal input)
{
  return ((decimal*)data)[0] * input + ((decimal*)data)[1];
}

decimal activationLinearDeri(void* data, decimal input)
{
  return ((decimal*)data)[0];
}

activation activation_make_linear(decimal a, decimal b)
{
  decimal* stuff = (decimal*)malloc(sizeof(decimal)*2);
  stuff[0] = a;
  stuff[1] = b;
  return (activation){stuff, &activationLinearFunc, &activationLinearDeri};
}

decimal activationRectifierFunc(void* data, decimal input)
{
  return (input<0)?0:input;
}

decimal activationRectifierDeri(void* data, decimal input)
{
  return (input<0)?0:1;
}

activation activation_make_rectifier()
{
  return (activation){0, &activationRectifierFunc, &activationRectifierDeri};
}

decimal activationSoftplusFunc(void* data, decimal input)
{
#ifdef USE_DOUBLE
  return log(1.0 + exp(input));
#endif
#ifndef USE_DOUBLE
  return logf(1.0+expf(input));
#endif
}

decimal activationSoftplusDeri(void* data, decimal input)
{
#ifdef USE_DOUBLE
  return 1.0/(1.0 + exp(-input));
#endif
#ifndef USE_DOUBLE
  return 1.0/(1.0+exp(-input));
#endif
}

activation activation_make_softplus()
{
  return (activation){0, &activationSoftplusFunc, &activationSoftplusDeri};
}

decimal activationInverseAbsFunc(void* data, decimal input)
{
#ifdef USE_DOUBLE
  return input/(1+fabs(input));
#endif
#ifndef USE_DOUBLE
  return input/(1+fabsf(input));
#endif
}

decimal activationInverseAbsDeri(void* data, decimal input)
{
  if(input<0)
    input*=-1;
  decimal p1 = input+1;
  return 1.0/p1 - input/(p1*p1);
}

activation activation_make_inverseAbs()
{
  return (activation){0, &activationInverseAbsFunc, &activationInverseAbsDeri};
}
