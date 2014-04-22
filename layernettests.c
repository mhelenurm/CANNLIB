#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "neural/activation.h"
#include "neural/neuron.h"
#include "neural/layered_network.h"
#include "neural/random.h"

#include "bitmap/colors.h"
#include "bitmap/bitmap.h"

int main()
{
  activation h = activation_make_tanh();
  decimal max = activationFuncMax(h);
  decimal min = activationFuncMin(h);
  decimal middle = (max+min)/2.0;
  //printf("%f, %f, %f\n", max, min, middle);

  unsigned long random_int = random_make_time();

  //graphing details
  double g_x = -1;
  double g_y = -1;
  unsigned int image_width = 200;
  unsigned int image_height = 200;
  double x_inc = 2.0/(image_width-1);
  double y_inc = 2.0/(image_height-1);
  unsigned int index = 0;
  pixel* data = (pixel*)malloc(sizeof(pixel)*image_width*image_height);

  unsigned int layers = 4;
  unsigned int* layersizes = (unsigned int*)malloc(sizeof(unsigned int)*layers);
  layersizes[0] = 2;
  layersizes[1] = 3;
  layersizes[2] = 3;
  layersizes[3] = 1;
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
  unsigned long trainings = 10000000;
  unsigned long per10 = trainings/100;
  decimal learningrate = 0.01;

  time_t lastpercent = time(NULL);
  unsigned int printed_time = 0;
  unsigned int output_per_percent = 0;

  decimal x, y;

  for(unsigned long i = 1; i <= trainings; i++)
  {
    if(i%per10 == 0)
    {
      time_t next = time(NULL);
      printf("%ld%c\n", i/per10, '%');
      if(next-lastpercent >= 10 && !printed_time)
      {
        printed_time = 1;
        decimal distgone = next-lastpercent;
        distgone/=(decimal)(i/per10);
        distgone*=(decimal)(100-i/per10);
        printf("about %d seconds left.\n", (int)distgone);
      }
      if(output_per_percent)
      {
        g_x = -1;
        g_y = -1;
        index = 0;

        for(int i = 0; i < image_height; i++)
        {
          for(int j = 0; j < image_width; j++)
          {
            //get color now
            inputs[0] = g_x;
            inputs[1] = g_y;
            layered_network_sig_set_input(&nn, inputs);
            layered_network_sig_get_output(&nn, outputs);
            unsigned char color = (unsigned char)(255*(1.0-(outputs[0]-min)/(max-min)));
            data[index] = pixel_make(color, color, color, 255);
            if(fabs(outputs[0]-middle) <= .01)
            {
              data[index] = pixel_make(255, 0, 0, 255);
            }
            g_x+=x_inc;
            index++;
          }
          g_x = -1;
          g_y+=y_inc;
        }
        char s[5+10];
        sprintf(s, "graph%ld.bmp", i/per10);
        
        bitmap_write(image_width, image_height, data, BITMAP_16BPP, s);
      }
    }

    x = 2.0*(random_next_float(&random_int)-0.5);
    y = 2.0*(random_next_float(&random_int)-0.5);
    inputs[0] = x;
    inputs[1] = y;
    expoutputs[0] = (x*x+y*y<=1.0)?max:min;
    layered_network_sig_train(&nn, inputs, expoutputs, learningrate);
  }

  printf("Final Percentage Test: \n");

  unsigned int totalruns = 100000;
  unsigned int successes = 0;

  for(int i = 0; i <totalruns; i++)
  {
    x = 2.0*(random_next_float(&random_int)-0.5);
    y = 2.0*(random_next_float(&random_int)-0.5);
    inputs[0] = x;
    inputs[1] = y;
    layered_network_sig_set_input(&nn, inputs);
    layered_network_sig_get_output(&nn, outputs);
    if(outputs[0] >= middle)
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
  //now we make a graph of the data. Start from lower-left, go row by row
  g_x = -1;
  g_y = -1;
  index = 0;

  for(int i = 0; i < image_height; i++)
  {
    for(int j = 0; j < image_width; j++)
    {
      //get color now
      inputs[0] = g_x;
      inputs[1] = g_y;
      layered_network_sig_set_input(&nn, inputs);
      layered_network_sig_get_output(&nn, outputs);
      unsigned char color = (unsigned char)(255*(1.0-(outputs[0]-min)/(max-min)));
      data[index] = pixel_make(color, color, color, 255);
      if(fabs(outputs[0]-middle) <= .01)
      {
        data[index] = pixel_make(255, 0, 0, 255);
      }
      g_x+=x_inc;
      index++;
    }
    g_x = -1;
    g_y+=y_inc;
  }

  bitmap_write(image_width, image_height, data, BITMAP_16BPP, "graph.bmp");

  printf("Successes: %f\n", ((decimal)successes/(decimal)totalruns));
  //FREE RESOURCES
  free(data);
  layered_network_sig_free(nn);
  free(inputs);
  free(outputs);
  free(layersizes);
  return 0;
}
