#include "random.h"
#include <time.h>

unsigned long random_make_time()
{
  return (unsigned long)time(NULL);
}

unsigned long random_make(unsigned long seed)
{
  return (seed^0x5DEECE66DL) & RANDOM_48BIT_MASK;
}

unsigned int random_next_int(unsigned long* seed)
{
  seed[0] = (seed[0] * 0x5DEECE66DL + 0xBL) & RANDOM_48BIT_MASK;
  return (unsigned int)(seed[0] >> 16);
}

decimal random_next_float(unsigned long* seed)
{
  seed[0] = (seed[0] * 0x5DEECE66DL + 0xBL) & RANDOM_48BIT_MASK;
  return ((decimal)(unsigned int)(seed[0] >> 24))/((decimal)RANDOM_24BIT_DIVIDE);
}
