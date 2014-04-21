#ifndef MHLIB_NEURON_H
#define MHLIB_NEURON_H
#include <stdbool.h>
#include "activation.h"
#include "types.h"

typedef struct neuron
{
  bool is_input; //whether this node's output is set by hand
  decimal output_value; //if is_input, set this to the input
  activation* act_func; //activation function (pointer so reuse is avail.)

  unsigned int inputs_count; //number of inputs
  struct neuron** inputs; //list of inputs
  decimal* inputweights; //list of input weights

  unsigned int outputs_count; //number of outputs
  struct neuron** outputs; //list of outputs
} neuron;

extern neuron neuron_make_input(activation* act, decimal input, unsigned int outputs);
extern neuron neuron_make(activation* act, unsigned int inputs, unsigned int outputs);

extern void neuron_set_input_neuron(neuron* n, unsigned int index, neuron* add);
extern void neuron_set_input_neuron_weight(neuron* n, unsigned int index, decimal weight);
extern void neuron_set_output_neuron(neuron* n, unsigned int index, neuron* add);
extern void neuron_set_connection(neuron* a, unsigned int inda, neuron* b, unsigned int indb, decimal weight);

extern void neuron_set_input(neuron* n, decimal value); //assuming input neuron
extern decimal neuron_output(neuron* n);
#endif
