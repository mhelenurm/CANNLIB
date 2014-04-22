#ifndef MHLIB_LAYERED_NETWORK_H
#define MHLIB_LAYERED_NETWORK_H
#include "neuron.h"
#include "activation.h"
#include "neural_types.h"

typedef struct layered_network
{
  unsigned int layers; //number of layers in the network
  unsigned int* nodes_per_layer; //number of nodes per layer
  neuron* nodes; //total neurons; sum of elements of nodes_per_layer
  neuron collection;
  activation* act;
} layered_network;

typedef struct layered_network lann;

extern lann layered_network_make(unsigned int layers, unsigned int* nodes_per_layer, activation h);
extern void layered_network_set_input(lann* n, decimal* inputvals); //len is nodes in layer 0
extern void layered_network_get_output(lann* n, decimal* slot); //len is nodes in last layer

extern void layered_network_train(lann* n, decimal* input, decimal* target, decimal learnrate);

extern void layered_network_free(lann n);
#endif
