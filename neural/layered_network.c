#include "layered_network.h"
#include <stdlib.h>
#include <stdio.h>
#include "random.h"

layered_network layered_network_make(unsigned int layers, unsigned int* nodes_per_layer, activation h)
{
  unsigned long random_int = random_make_time();
  activation* act = (activation*)malloc(sizeof(activation));
  act[0] = h;

  neuron collection = neuron_make(act, nodes_per_layer[layers-1], 0);
  layered_network nn = (layered_network){layers, 0, 0, collection, act};
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
          neuron_set_connection(&(nodes[k]), j, &(nodes[index]), k-lastlayerstart, 2.0*(random_next_float(&random_int)-0.5));
        }
        if(i==layers-1)
        {
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

void layered_network_set_input(layered_network* n, decimal* inputvals) //of size npl[0]
{
  for(int i = 0; i < n->nodes_per_layer[0]; i++)
  {
    neuron_set_input(&(n->nodes[i]), inputvals[i]);
  }
}

void layered_network_get_output(layered_network* n, decimal* outputvals) //of size npl[layers-1]
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

void layered_network_train(layered_network* n, decimal* input, decimal* target, decimal learnrate)
{
  
  for(int i = 0; i < n->nodes_per_layer[0]; i++)
  {
    neuron_set_input(&(n->nodes[i]), input[i]);
  }
  int sum = 0;
  for(int i = 0; i < n->layers-1; i++)
  {
    sum+=n->nodes_per_layer[i];
  }
  neuron_output(&(n->collection));
  unsigned int totalnodesdelta = sum + n->nodes_per_layer[n->layers-1] - n->nodes_per_layer[0];
  decimal littledelta[totalnodesdelta];
  for(int i = 0; i < totalnodesdelta; i++)
  {
    littledelta[i] = 0;
  }
  
  unsigned int curlayerindex = totalnodesdelta;
  for(int i = n->layers-1; i >= 1; i--) //for each layer ascending
  { 
    curlayerindex-=n->nodes_per_layer[i];
    if(i == n->layers-1) //calculate delta for the first layer of nodes
    {
      for(int j = 0; j <n->nodes_per_layer[i]; j++)
      {
        decimal d = (n->nodes[curlayerindex+j+n->nodes_per_layer[0]]).output_value;
        decimal preout = (n->nodes[curlayerindex+j+n->nodes_per_layer[0]]).pre_output;
        littledelta[j+curlayerindex] = activationDerEval(*(n->nodes[curlayerindex+j+n->nodes_per_layer[0]]).act_func, preout) * (target[j] - d);
      }
    }
    for(int j = 0; j < n->nodes_per_layer[i]; j++)
    {
      //we need to: finish calculating this layer; update input weights, update next layer
      if(i != n->layers-1)
      {
         decimal pre_out = (n->nodes[curlayerindex+j+n->nodes_per_layer[0]]).pre_output;
         littledelta[j+curlayerindex] *= activationDerEval(*(n->nodes[curlayerindex+j+n->nodes_per_layer[0]]).act_func, pre_out);
      }
      //update upstream weights
      neuron* c_neu = &(n->nodes[curlayerindex+j+n->nodes_per_layer[0]]);
      for(int k = 0; k < c_neu->inputs_count; k++)
      {
        c_neu->inputweights[k] += (c_neu->inputs[k])->output_value * learnrate * littledelta[j+curlayerindex];
      }
      //update upstream littledeltas
      if(i != 1)
      {
        for(int k = 0; k < n->nodes_per_layer[i-1]; k++)
        {
          littledelta[curlayerindex-n->nodes_per_layer[i-1] + k] += littledelta[curlayerindex+j]*(n->nodes[curlayerindex+j+n->nodes_per_layer[0]]).inputweights[k];
        }
      }
    }
  }
}

void layered_network_free(layered_network n)
{
  free(n.nodes);
  free(n.nodes_per_layer);
  n.layers = 0;
  activation_free(*(n.act));
  free(n.act);
}
