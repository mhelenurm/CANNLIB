#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "neural/activation.h"
#include "neural/neuron.h"
#include "neural/layered_network.h"

int main()
{
  unsigned int layers = 4;
  unsigned int* layersizes = (unsigned int*)malloc(sizeof(unsigned int)*layers);
  layersizes[0] = 1;
  layersizes[1] = 2;
  layersizes[2] = 3;
  layersizes[3] = 1;
  int sum = 0;
  for(int i = 0; i < layers; i++)
  {
    sum += layersizes[i];
  }
  layered_network_sig nn = layered_network_sig_make(layers, layersizes);
  free(layersizes); //LAYERSIZES FREED

  double* inputs = (double*)malloc(sizeof(double)*layersizes[0]);
  double* outputs = (double*)malloc(sizeof(double)*layersizes[layers-1]);
  for(int i = 0; i < layersizes[0]; i++)
  {
    inputs[i] = 0;
  }

  layered_network_sig_set_input(&nn, inputs);
  layered_network_sig_get_output(&nn, outputs);

  for(int j = 0; j < sum; j++)
  {
    printf("Node %d: %f\n", j, nn.nodes[j].output_value);
  }

  //PRINT OUTPUTS
  for(int i = 0; i < layersizes[layers-1]; i++)
  {
    printf("%f\n", outputs[i]);
  }

  //FREE RESOURCES
  layered_network_sig_free(nn);
  free(inputs);
  free(outputs);
  return 0;
}
