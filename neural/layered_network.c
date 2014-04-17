#include "layered_network.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

layered_network_sig layered_network_sig_make(unsigned int layers, unsigned int* nodes_per_layer)
{
  activation* act = (activation*)malloc(sizeof(activation));
  act[0] = activation_make_sigmoid(1.0);

  neuron collection = neuron_make(act, nodes_per_layer[layers-1], 0);
  layered_network_sig nn = (layered_network_sig){layers, 0, 0, collection, act};
  unsigned int* npl = (unsigned int*)malloc(sizeof(unsigned int)*layers);
  int sum = 0;
  for(int i = 0; i < layers; i++)
  {
    npl[i] = nodes_per_layer[i];
    sum+=npl[i];
  }
  neuron* nodes = (neuron*)malloc(sizeof(neuron)*sum);

  unsigned int index = 0;
  unsigned int lastlayerstart = 0;
  for(int i = 0; i < layers; i++)
  {
    for(int j = 0; j < npl[i]; j++)
    {
      if(i == 0)
      {
        nodes[index] = neuron_make_input(nn.act, 0.0, (i==layers-1)?0:npl[i+1]);
      } else
      {
        nodes[index] = neuron_make(nn.act, npl[i-1], (i==layers-1)?1:npl[i+1]);
        for(int k = lastlayerstart; k < lastlayerstart+npl[i-1]; k++)
        {
          printf(".\n");
          neuron_set_connection(&(nodes[k]), j, &(nodes[index]), k-lastlayerstart, 1.0);
        }
        if(i==layers-1)
        {
          printf("* %d\n", (sum-1-index));
          neuron_set_connection(&(nodes[index]), 0, &collection, sum-1-index, 1.0);
        }
      }
      index++;
    }
    if(i!=0)
    {
      lastlayerstart += npl[i-1];
    }
  }
  nn.nodes_per_layer = npl;
  nn.nodes = nodes;
  return nn;
}

void layered_network_sig_set_input(layered_network_sig* n, double* inputvals) //of size npl[0]
{
  for(int i = 0; i < n->nodes_per_layer[0]; i++)
  {
    neuron_set_input(&(n->nodes[i]), inputvals[0]);
  }
}

void layered_network_sig_get_output(layered_network_sig* n, double* outputvals) //of size npl[layers-1]
{
  int sum = 0;
  for(int i = 0; i < n->layers-1; i++)
  {
    sum+=n->nodes_per_layer[i];
  }
  neuron_output(&(n->collection));
  for(int i = 0; i < n->nodes_per_layer[n->layers - 1]; i++)
  {
    outputvals[i] = (n->nodes[sum+i]).output_value; //these will be correct 'cause we just calculated them!
  }
}


void layered_network_sig_free(layered_network_sig n)
{
  free(n.nodes);
  free(n.nodes_per_layer);
  n.layers = 0;
  activation_free(*(n.act));
  free(n.act);
}
