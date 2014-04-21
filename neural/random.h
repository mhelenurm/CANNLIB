#ifndef MHLIB_RANDOM_H
#define MHLIB_RANDOM_H

#include "types.h"

#define RANDOM_48BIT_MASK 0xFFFFFFFFFFFFL
#define RANDOM_24BIT_DIVIDE 0x1000000
extern unsigned long random_make_time();
extern unsigned long random_make(unsigned long seed);

extern unsigned int random_next_int(unsigned long* seed);
extern decimal random_next_float(unsigned long* seed);
#endif
