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

//PRINTING AND OUTPUT
#define PRINT_SHIT 1
#define IMAGE_PER_PERCENT 0
#define OUTPUT_FINAL_IMAGE 1

//IMAGES 'N SHIT
#define IMAGE_WIDTH 200
#define IMAGE_HEIGHT 200

//NEURAL VARIABLES 'N SHIT
#define TRAINING_ITERATIONS 1000000
#define LEARNING_RATE 0.01

int main()
{
  activation h = activation_make_tanh();
  decimal max = activationFuncMax(h);
  decimal min = activationFuncMin(h);
  decimal middle = (max+min)/2.0;

  unsigned long random_int = random_make_time();

  //graphing details
  double g_x = -1;
  double g_y = -1;
  double x_inc = 2.0/(IMAGE_WIDTH-1);
  double y_inc = 2.0/(IMAGE_HEIGHT-1);
  unsigned int index = 0;
  pixel* data = (pixel*)malloc(sizeof(pixel)*IMAGE_WIDTH*IMAGE_HEIGHT);

  unsigned int layers = 4;
  unsigned int layersizes[] = {2, 3, 3, 1};

  layered_network nn = layered_network_make(layers, layersizes, h);

  decimal* inputs = (decimal*)malloc(sizeof(decimal)*layersizes[0]);
  decimal* outputs = (decimal*)malloc(sizeof(decimal)*layersizes[layers-1]);
  decimal* expoutputs = (decimal*)malloc(sizeof(decimal)*layersizes[layers-1]);

  //here's the actual game:
  unsigned long per_percent = TRAINING_ITERATIONS/100;

  time_t lastpercent = time(NULL);
  unsigned int printed_time = 0;

  decimal x, y;

  for(unsigned long i = 1; i <= TRAINING_ITERATIONS; i++)
  {
    if(i%per_percent == 0)
    {
      time_t next = time(NULL);
      if(PRINT_SHIT)
        printf("%ld%c\n", i/per_percent, '%');
      if(next-lastpercent >= 10 && !printed_time)
      {
        printed_time = 1;
        decimal distgone = next-lastpercent;
        distgone/=(decimal)(i/per_percent);
        distgone*=(decimal)(100-i/per_percent);
        if(PRINT_SHIT)
          printf("about %d seconds left.\n", (int)distgone);
      }
      if(IMAGE_PER_PERCENT)
      {
        g_x = -1;
        g_y = -1;
        index = 0;

        for(int i = 0; i < IMAGE_HEIGHT; i++)
        {
          for(int j = 0; j < IMAGE_WIDTH; j++)
          {
            //get color now
            inputs[0] = g_x;
            inputs[1] = g_y;
            layered_network_set_input(&nn, inputs);
            layered_network_get_output(&nn, outputs);
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
        char s[13];
        sprintf(s, "graph%ld.bmp", i/per_percent);
        
        bitmap_write(IMAGE_WIDTH, IMAGE_HEIGHT, data, BITMAP_16BPP, s);
      }
    }

    x = 2.0*(random_next_float(&random_int)-0.5);
    y = 2.0*(random_next_float(&random_int)-0.5);
    inputs[0] = x;
    inputs[1] = y;
    expoutputs[0] = (x*x+y*y<=1.0)?max:min;
    layered_network_train(&nn, inputs, expoutputs, LEARNING_RATE);
  }
  if(PRINT_SHIT)
    printf("Final Percentage Test: \n");

  unsigned int totalruns = 100000;
  unsigned int successes = 0;

  for(int i = 0; i <totalruns; i++)
  {
    x = 2.0*(random_next_float(&random_int)-0.5);
    y = 2.0*(random_next_float(&random_int)-0.5);
    inputs[0] = x;
    inputs[1] = y;
    layered_network_set_input(&nn, inputs);
    layered_network_get_output(&nn, outputs);
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
  if(OUTPUT_FINAL_IMAGE)
  {
    //now we make a graph of the data. Start from lower-left, go row by row
    g_x = -1;
    g_y = -1;
    index = 0;
    
    for(int i = 0; i < IMAGE_HEIGHT; i++)
    {
      for(int j = 0; j < IMAGE_WIDTH; j++)
      {
        //get color now
        inputs[0] = g_x;
        inputs[1] = g_y;
        layered_network_set_input(&nn, inputs);
        layered_network_get_output(&nn, outputs);
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
    
    bitmap_write(IMAGE_WIDTH, IMAGE_HEIGHT, data, BITMAP_16BPP, "graph.bmp");
  }
  if(PRINT_SHIT)
    printf("Successes: %f\n", ((decimal)successes/(decimal)totalruns));
  
  //FREE RESOURCES
  free(data);
  layered_network_free(nn);
  free(inputs);
  free(outputs);
  return 0;
}
