#include <stdio.h>
#include <stdlib.h>
#include "neural/activation.h"

void printResults(activation act, double x)
{
  printf("For x=%f, F=%f, and F'=%f\n", x, activationEval(act, x), activationDerEval(act, x));
}

int main()
{
  activation rectifier = activation_make_rectifier();
  activation softplus = activation_make_softplus();

  double x = 3.3;
  printf("REC: ");
  printResults(rectifier, x);
  printf("S+: ");
  printResults(softplus, x);  
  x = -2.0;
  printf("REC: ");
  printResults(rectifier, x);
  printf("S+: ");
  printResults(softplus, x);
  return 0;
}
