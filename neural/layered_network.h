#ifndef MHLIB_LAYERED_NETWORK_H
#define MHLIB_LAYERED_NETWORK_H
#include "neuron.h"
#include "activation.h"
#include "types.h"

typedef struct layered_network_sig
{
  unsigned int layers; //number of layers in the network
  unsigned int* nodes_per_layer; //number of nodes per layer
  neuron* nodes; //total neurons; sum of elements of nodes_per_layer
  neuron collection;
  activation* act;
} layered_network_sig;

typedef struct layered_network_sig lann;

extern lann layered_network_sig_make(unsigned int layers, unsigned int* nodes_per_layer);
extern void layered_network_sig_set_input(lann* n, decimal* inputvals); //len is nodes in layer 0
extern void layered_network_sig_get_output(lann* n, decimal* slot); //len is nodes in last layer

extern void layered_network_sig_train(lann* n, decimal* input, decimal* target, decimal learnrate);

extern void layered_network_sig_free(lann n);
#endif
