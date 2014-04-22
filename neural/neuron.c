#include <stdlib.h>
#include <stdio.h>
#include "neuron.h"

neuron neuron_make_input(activation* act, decimal input, unsigned int outputs)
{
  return (neuron){true, input, activationEval(*act, input), act, 0, 0, 0, outputs, (neuron**)malloc(sizeof(neuron*)*outputs)};
}

neuron neuron_make(activation* act, unsigned int inputs, unsigned int outputs)
{
  return (neuron){false, 0.0, 0.0, act, inputs, (neuron**)malloc(sizeof(neuron*)*inputs),
    (decimal*)malloc(sizeof(decimal)*inputs), outputs, (neuron**)malloc(sizeof(neuron*)*outputs)};
}

void neuron_set_input_neuron(neuron* n, unsigned int index, neuron* add)
{
  if(index<n->inputs_count)
  {
    n->inputs[index] = add;
  }
}

void neuron_set_input_neuron_weight(neuron* n, unsigned int index, decimal weight)
{
  if(index<n->inputs_count)
  {
    n->inputweights[index] = weight;
  }
}

void neuron_set_output_neuron(neuron* n, unsigned int index, neuron* add)
{
  if(index<n->outputs_count)
  {
    n->outputs[index] = add;
  }
}

void neuron_set_connection(neuron* a, unsigned int inda, neuron* b, unsigned int indb, decimal weight)
{
  if(inda < a->outputs_count && indb < b->inputs_count)
  {
    a->outputs[inda] = b;
    b->inputs[indb] = a;
    b->inputweights[indb] = weight;
  }
}
void neuron_set_input(neuron* n, decimal value)
{
  n->pre_output = value;
  n->output_value = activationEval(*(n->act_func), value);
}

decimal neuron_output(neuron* n)
{
  if(n->is_input)
  {
    return n->output_value;
  }
  decimal sum = 0.0;
  for(int i = 0; i < n->inputs_count; i++)
  {
    sum += neuron_output((n->inputs)[i]) * (n->inputweights)[i];
  }
  n->pre_output = sum;
  n->output_value = activationEval(*(n->act_func), sum);
  return n->output_value;
}
