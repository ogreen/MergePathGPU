#include <sys/time.h>
#include <stdio.h>
#include "util.cuh"


struct timeval tv;

double firsttic = 0;
double lasttic = 0;

void tic_reset() {
	gettimeofday(&tv, NULL);
	firsttic = (double)tv.tv_sec + 1.0e-6 * (double)tv.tv_usec;
	lasttic = firsttic;
}
double tic_total() {
	gettimeofday(&tv, NULL);
	lasttic = (double)tv.tv_sec + 1.0e-6 * (double)tv.tv_usec;
	return lasttic - firsttic;
}
double tic_sincelast() {
	gettimeofday(&tv, NULL);
	double rtnval = ((double)tv.tv_sec + 1.0e-6 * (double)tv.tv_usec) - lasttic;
	lasttic = (double)tv.tv_sec + 1.0e-6 * (double)tv.tv_usec;
	return rtnval;
}
