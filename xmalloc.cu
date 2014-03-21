/* -*- mode: C; fill-column: 70; -*- */
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <cuda.h>

#include "xmalloc.cuh"

void *
xmalloc (size_t sz)
{
    void * out;
    out = malloc (sz);
    if (!out) {
	perror ("Failed xmalloc: ");
	abort ();
    }
    return out;
}
void *
xcalloc (size_t nelem, size_t sz)
{
    void * out;
    out = calloc (nelem, sz);
    if (!out) {
	perror ("Failed xcalloc: ");
	abort ();
    }
    return out;
}
void *
xrealloc (void *p, size_t sz)
{
    void * out;
    out = realloc (p, sz);
    if (!out) {
	perror ("Failed xrealloc: ");
	abort ();
    }
    return out;
}

