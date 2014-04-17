#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "neural/neuron.h"

double rand_double()
{
  return 2.0*((double)rand()/(double)RAND_MAX - 0.5) * 100000.0;
}

int main()
{
  //seed the random number
  srand(time(NULL));
  //TESTS!!!
  activation sigmoid = activation_make_sigmoid(.1);
  activation softplus = activation_make_softplus();

  printf("\nTesting input neurons...\n");
  neuron input1 = neuron_make_input(&sigmoid, 0.0, 0);
  for(int i = 0; i < 100; i++)
  {
    double value = rand_double();
    neuron_set_input(&input1, value);

    if(activationEval(sigmoid,value) != neuron_output(&input1))
    {
      printf("Input neurons failed on case: %f. Expected %f, Return %f\n\n", value, activationEval(sigmoid, value), neuron_output(&input1));
      return 1;
    }
  }
  printf("Input neurons good!\n\n");
  printf("Testing simple network...\n");
  //COMPUTE VALUES OF SIGMOID FUNCTION
  neuron input2 = neuron_make_input(&sigmoid, 1.0, 1);
  neuron input3 = neuron_make_input(&sigmoid, 1.0, 1);
  neuron output1 = neuron_make(&sigmoid, 2, 1);
  neuron output2 = neuron_make(&softplus, 1, 0);
  neuron_set_connection(&input2, 0, &output1, 0, 7.0);
  neuron_set_connection(&input3, 0, &output1, 1, 5.0);
  neuron_set_connection(&output1, 0, &output2, 0, 0.39);

  for(int i = 0; i < 100; i++)
  {
    double val1 = rand_double();
    double val2 = rand_double();
    neuron_set_input(&input2, val1);
    neuron_set_input(&input3, val2);

    double ret = neuron_output(&output2);
    double fxn = activationEval(sigmoid,val1)*7.0+activationEval(sigmoid,val2)*5.0;
    fxn = 1.0/(1.0 + exp(-.1*fxn));
    fxn = log(1.0+exp(.39*fxn));    
    if(ret != fxn)
    {
      printf("Simple network failed. Out was: %f, should be: %f, for ins: %f, %f\n\n", ret, fxn, val1, val2);
      return 0;
    }
  }
  printf("Simple network good!\n\n");
  return 0;
}
