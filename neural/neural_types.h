#ifndef MHLIB_TYPES_H
#define MHLIB_TYPES_H

#define USE_DOUBLE 1 //comment this out when using floats

#ifdef USE_DOUBLE
typedef double decimal;
#endif

#ifndef USE_DOUBLE
typedef float decimal;
#endif

#endif
