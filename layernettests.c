#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "neural/activation.h"
#include "neural/neuron.h"
#include "neural/layered_network.h"
#include "neural/random.h"

int main()
{
  unsigned long random_int = random_make_time();

  unsigned int layers = 3;
  unsigned int* layersizes = (unsigned int*)malloc(sizeof(unsigned int)*layers);
  layersizes[0] = 2;
  layersizes[1] = 4;
  layersizes[2] = 1;
  int sum = 0;
  for(int i = 0; i < layers; i++)
  {
    sum += layersizes[i];
  }
  layered_network_sig nn = layered_network_sig_make(layers, layersizes);

  decimal* inputs = (decimal*)malloc(sizeof(decimal)*layersizes[0]);
  decimal* outputs = (decimal*)malloc(sizeof(decimal)*layersizes[layers-1]);
  decimal* expoutputs = (decimal*)malloc(sizeof(decimal)*layersizes[layers-1]);

  //here's the actual game:
  unsigned int trainings = 1000000000;
  unsigned int per10 = trainings/100;
  decimal learningrate = 0.05;
  for(int i = 1; i <= trainings; i++)
  {
    if(i%per10 == 0)
    {
      printf("%d%c\n", i/per10, '%');
    }
    decimal x = 2.0*(random_next_float(&random_int)-0.5);
    decimal y = 2.0*(random_next_float(&random_int)-0.5);
    inputs[0] = x;
    inputs[1] = y;
    expoutputs[0] = (x*x+y*y<=1.0)?1.0f:0.0f;
    layered_network_sig_train(&nn, inputs, expoutputs, learningrate);
  }

  printf("Final Percentage Test: \n");

  unsigned int totalruns = 100000;
  unsigned int successes = 0;

  for(int i = 0; i <totalruns; i++)
  {
    decimal x = 2.0*(random_next_float(&random_int)-0.5);
    decimal y = 2.0*(random_next_float(&random_int)-0.5);
    inputs[0] = x;
    inputs[1] = y;
    layered_network_sig_set_input(&nn, inputs);
    layered_network_sig_get_output(&nn, outputs);
    if(outputs[0] >= .5)
    {
      if(x * x + y * y <= 1.0)
      {
        successes++;
      }
    } else 
    {
      if(x*x+y*y>1.0)
      {
        successes++;
      }
    }
  }

  printf("Successes: %f\n", ((decimal)successes/(decimal)totalruns));
  //FREE RESOURCES
  layered_network_sig_free(nn);
  free(inputs);
  free(outputs);
  free(layersizes);
  return 0;
}
