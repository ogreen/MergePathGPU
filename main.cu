/*
 * Authors: 
 *  Oded Green (ogreen@gatech.edu), Rob McColl (robert.c.mccoll@gmail.com)
 *  High Performance Computing Lab, Georgia Tech
 *
 * Future Publication:
 * GPU MergePath: A GPU Merging Algorithm
 * ACM International Conference on Supercomputing 2012
 * June 25-29 2012, San Servolo, Venice, Italy
 * 
 * Copyright (c) 2012 Georgia Institute of Technology
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 * 
 * - Redistributions of source code must retain the above copyright notice, 
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice, 
 *   this list of conditions and the following disclaimer in the documentation 
 *   and/or other materials provided with the distribution.
 * - Neither the name of the Georgia Institute of Technology nor the names of 
 *   its contributors may be used to endorse or promote products derived from 
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <float.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/generate.h>
#include <thrust/merge.h>

#define CSV 0
#if(CSV)
#define PS(X, S) std::cout << X << ", " << S << ", "; fflush(stdout);
#define PV(X) std::cout << X << ", "; fflush(stdout);
#else
#define PS(X, S) std::cout << X << " " << S <<" :\n"; fflush(stdout);
#define PV(X) std::cout << "\t" << #X << " \t: " << X << "\n"; fflush(stdout);
#endif

/* GLOBAL FUNCTION DECLARATIONS */
template<typename vec_t>
__global__ void cudaWorkloadDiagonals(vec_t * A, uint32_t A_length, vec_t * B, uint32_t B_length, 
					    uint32_t * diagonal_path_intersections);

template<typename vec_t, bool timesections, bool countloops>
__global__ void cudaMergeSinglePath(vec_t * A, uint32_t A_length, vec_t * B, uint32_t B_length, 
				    uint32_t * diagonal_path_intersections, vec_t * C, uint32_t C_length, 
				    float * times, uint32_t * loopCount);

/* POSITIVEINFINITY
 * Returns maximum value of a type
 */
template<typename vec_t>
__host__ __device__ vec_t getPositiveInfinity() {
  vec_t tmp = 0;
  return positiveInfinity(tmp);
}
__host__ __device__ float positiveInfinity(float tmp) {
  return FLT_MAX;
}
__host__ __device__ double positiveInfinity(double tmp) {
  return DBL_MAX;
}
__host__ __device__ uint32_t positiveInfinity(uint32_t tmp) {
  return 0xFFFFFFFFUL;
}
__host__ __device__ uint64_t positiveInfinity(uint64_t tmp) {
  return 0xFFFFFFFFFFFFFFFFUL;
}
/* NEGATIVEINFINITY
 * Returns minimum value of a type
 */
template<typename vec_t>
__host__ __device__ vec_t getNegativeInfinity() {
  vec_t tmp = 0;
  return negativeInfinity(tmp);
}
__host__ __device__ float negativeInfinity(float tmp) {
  return FLT_MIN;
}
__host__ __device__ double negativeInfinity(double tmp) {
  return DBL_MIN;
}
__host__ __device__ uint32_t negativeInfinity(uint32_t tmp) {
  return 0;
}
__host__ __device__ uint64_t negativeInfinity(uint64_t tmp) {
  return 0;
}

/* RAND64
 * Gives up to 64-bits of pseudo-randomness
 * Note: not very "good" or "random" 
 */
template<typename vec_t>
vec_t rand64() {
  vec_t rtn;
  do {
    uint32_t * rtn32 = (uint32_t *)&rtn;
    rtn32[0] = rand();
    if(sizeof(vec_t) > 4) rtn32[1] = rand();
  } while(!(rtn < getPositiveInfinity<vec_t>() &&
	    rtn > getNegativeInfinity<vec_t>()));
  return rtn;
}

/* MERGETYPE
 * Performs <runs> merges of two sorted pseudorandom <vec_t> arrays of length <size> 
 * Times the runs and reports on the average time
 * Checks the output of each merge for correctness
 */
#define PADDING 1024
template<typename vec_t, uint32_t blocks, uint32_t threads, uint32_t runs>
void mergeType(uint64_t size) {
  // Prepare host and device vectors
  thrust::host_vector<vec_t>hostA(size + (PADDING));
  thrust::host_vector<vec_t>hostB(size + (PADDING));
  thrust::host_vector<vec_t>hostC(2*size + (PADDING));

  thrust::device_vector<vec_t>A;
  thrust::device_vector<vec_t>B;
  thrust::device_vector<vec_t>C(2*size + (PADDING));
  thrust::device_vector<uint32_t> diagonal_path_intersections(2 * (blocks + 1));

  float diag = 0;
  float merge = 0;
  uint32_t errors = 0;

  // Fore each run
  for(uint32_t i = 0; i < runs; i++) {
    // Generate two sorted psuedorandom arrays
    thrust::generate(hostA.begin(), hostA.end(), rand64<vec_t>);
    thrust::generate(hostB.begin(), hostB.end(), rand64<vec_t>);

    thrust::fill(hostA.begin() + size, hostA.end(), getPositiveInfinity<vec_t>());
    thrust::fill(hostB.begin() + size, hostB.end(), getPositiveInfinity<vec_t>());
    
    A = hostA;
    B = hostB;

    thrust::sort(A.begin(), A.end());
    thrust::sort(B.begin(), B.end());

    // Perform the global diagonal intersection serach to divide work among SMs
    float temp;
    {
      cudaEvent_t start_event, stop_event;
      cudaEventCreate(&start_event);
      cudaEventCreate(&stop_event);
      cudaEventRecord(start_event, 0);
      cudaWorkloadDiagonals<vec_t><<<blocks, 32>>>
	    (A.data().get(), size, B.data().get(),
	     size, diagonal_path_intersections.data().get());
      cudaEventRecord(stop_event, 0);
      cudaEventSynchronize(stop_event);
      cudaEventElapsedTime(&temp, start_event, stop_event);
      diag += temp;
    }

    // Merge between global diagonals independently on each block
    {
      cudaEvent_t start_event, stop_event;
      cudaEventCreate(&start_event);
      cudaEventCreate(&stop_event);
      cudaEventRecord(start_event, 0);
      cudaMergeSinglePath<vec_t,false,false><<<blocks, threads>>>
	      (A.data().get(), size, B.data().get(), size, diagonal_path_intersections.data().get(),
	       C.data().get(), size * 2, NULL, NULL); 
      cudaEventRecord(stop_event, 0);
      cudaEventSynchronize(stop_event);
      cudaEventElapsedTime(&temp, start_event, stop_event);
      merge += temp;
    }

    // Test for errors
    hostC = C;
    for(uint32_t i = 1; i < size; i++) {
      errors += hostC[i] < hostC[i-1];
    }
  }

  // Print timing results
  diag /= runs;
  merge /= runs;
  float total = diag + merge;
  PV(diag);
  PV(merge);
  PV(total);

  PV(errors);
}

/* MERGEALLTYPES 
 * Performs <runs> merge tests for each type at a given size
 */
template<uint32_t blocks, uint32_t threads, uint32_t runs>
void mergeAllTypes(uint64_t size) {
  PS("uint32_t", size)  mergeType<uint32_t, blocks, threads, runs>(size); printf("\n");
  PS("float", size)	mergeType<float, blocks, threads, runs>(size);    printf("\n");
  PS("uint64_t", size)  mergeType<uint64_t, blocks, threads, runs>(size); printf("\n");
  PS("double", size)    mergeType<double, blocks, threads, runs>(size);   printf("\n");
}

/* MAIN
 * Generates random arrays, merges them.
 */
int main(int argc, char *argv[]) {
  #define blocks  112
  #define threads 128
  #define runs 10
  mergeAllTypes<blocks, threads, runs>(1000000);
  mergeAllTypes<blocks, threads, runs>(10000000);
  mergeAllTypes<blocks, threads, runs>(100000000);
}


/* CUDAWORKLOADDIAGONALS
 * Performs a 32-wide binary search on one glboal diagonal per block to find the intersection with the path.
 * This divides the workload into independent merges for the next step 
 */
#define MAX(X,Y) (((X) > (Y)) ? (X) : (Y))
#define MIN(X,Y) (((X) < (Y)) ? (X) : (Y))
template<typename vec_t>
__global__ void cudaWorkloadDiagonals(vec_t * A, uint32_t A_length, vec_t * B, uint32_t B_length, 
					    uint32_t * diagonal_path_intersections) {

  // Calculate combined index around the MergePath "matrix"
  int32_t combinedIndex = (uint64_t)blockIdx.x * ((uint64_t)A_length + (uint64_t)B_length) / (uint64_t)gridDim.x;
  __shared__ int32_t x_top, y_top, x_bottom, y_bottom,  found;
  __shared__ int32_t oneorzero[32];

  int threadOffset = threadIdx.x - 16;

  // Figure out the coordinates of our diagonal
  x_top = MIN(combinedIndex, A_length);
  y_top = combinedIndex > (A_length) ? combinedIndex - (A_length) : 0;
  x_bottom = y_top;
  y_bottom = x_top;

  found = 0;

  // Search the diagonal
  while(!found) {
    // Update our coordinates within the 32-wide section of the diagonal 
    int32_t current_x = x_top - ((x_top - x_bottom) >> 1) - threadOffset;
    int32_t current_y = y_top + ((y_bottom - y_top) >> 1) + threadOffset;

    // Are we a '1' or '0' with respect to A[x] <= B[x]
    if(current_x >= A_length || current_y < 0) {
      oneorzero[threadIdx.x] = 0;
    } else if(current_y >= B_length || current_x < 1) {
      oneorzero[threadIdx.x] = 1;
    } else {
      oneorzero[threadIdx.x] = (A[current_x-1] <= B[current_y]) ? 1 : 0;
    }

    __syncthreads();

    // If we find the meeting of the '1's and '0's, we found the 
    // intersection of the path and diagonal
    if(threadIdx.x > 0 && (oneorzero[threadIdx.x] != oneorzero[threadIdx.x-1])) {
      found = 1;
      diagonal_path_intersections[blockIdx.x] = current_x;
      diagonal_path_intersections[blockIdx.x + gridDim.x + 1] = current_y;
    }

    __syncthreads();

    // Adjust the search window on the diagonal
    if(threadIdx.x == 16) {
      if(oneorzero[31] != 0) {
	x_bottom = current_x;
	y_bottom = current_y;
      } else {
	x_top = current_x;
	y_top = current_y;
      }
    }
    __syncthreads();
  }

  // Set the boundary diagonals (through 0,0 and A_length,B_length)
  if(threadIdx.x == 0 && blockIdx.x == 0) {
    diagonal_path_intersections[0] = 0;
    diagonal_path_intersections[gridDim.x + 1] = 0;
    diagonal_path_intersections[gridDim.x] = A_length;
    diagonal_path_intersections[gridDim.x + gridDim.x + 1] = B_length;
  }
}

/* CUDAMERGESINGLEPATH
 * Performs merge windows within a thread block from that block's global diagonal 
 * intersection to the next 
 */
#define K 512
template<typename vec_t, bool timesections, bool countloops>
__global__ void cudaMergeSinglePath(vec_t * A, uint32_t A_length, vec_t * B, uint32_t B_length, 
				    uint32_t * diagonal_path_intersections, vec_t * C, uint32_t C_length, 
				    float * times, uint32_t * loopCount) {

  // Setup timers
  clock_t temp, memread = 0, cshared = 0, cglobal = 0;
  __shared__ clock_t search, update;
  search = 0;
  update = 0;
  clock_t init;
  if(timesections) {
    init = clock();
  }

  // Storage space for local merge window
  __shared__ vec_t A_shared[K+2 << 1];
  vec_t* B_shared = A_shared + K+2;

  __shared__ uint32_t x_block_top, y_block_top, x_block_stop, y_block_stop;

  // Pre-calculate reused indices
  uint32_t threadIdX4 = threadIdx.x + threadIdx.x;
  threadIdX4 = threadIdX4 + threadIdX4;
  uint32_t threadIdX4p1 = threadIdX4 + 1;
  uint32_t threadIdX4p2 = threadIdX4p1 + 1;
  uint32_t threadIdX4p3 = threadIdX4p2 + 1;
  uint32_t Ax, Bx;

  // Define global window and create sentinels
  switch(threadIdx.x) {
    case 0:
      x_block_top = diagonal_path_intersections[blockIdx.x];
      A_shared[0] = getNegativeInfinity<vec_t>();
      break;
    case 64:
      y_block_top = diagonal_path_intersections[blockIdx.x + gridDim.x + 1];
      A_shared[K+1] = getPositiveInfinity<vec_t>();
      break;
    case 32:
      x_block_stop = diagonal_path_intersections[blockIdx.x + 1];
      B_shared[0] = getNegativeInfinity<vec_t>();
      break;
    case 96:
      y_block_stop = diagonal_path_intersections[blockIdx.x + gridDim.x + 2];
      B_shared[K+1] = getPositiveInfinity<vec_t>();
      break;
    default:
      break;
  }

  A--;
  B--;

  __syncthreads();

  if(timesections) {
    init = clock() - init;
  }

  if(countloops) {
    if(threadIdx.x == 0) loopCount[blockIdx.x] = 0;
  }

  // Construct and merge windows from diagonal_path_intersections[blockIdx.x] 
  // to diagonal_path_intersections[blockIdx.x+1]
  while(((x_block_top < x_block_stop) || (y_block_top < y_block_stop))) {

    if(countloops) {
      if(threadIdx.x == 0) loopCount[blockIdx.x]++;
    }

    if(timesections) {
      temp = clock();
    }

    // Load current local window
    {
      vec_t * Atemp = A + x_block_top;
      vec_t * Btemp = B + y_block_top;
      uint32_t sharedX = threadIdx.x+1;

      A_shared[sharedX] = Atemp[sharedX];
      B_shared[sharedX] = Btemp[sharedX];
      sharedX += blockDim.x;
      A_shared[sharedX] = Atemp[sharedX];
      B_shared[sharedX] = Btemp[sharedX];
      sharedX += blockDim.x;
      A_shared[sharedX] = Atemp[sharedX];
      B_shared[sharedX] = Btemp[sharedX];
      sharedX += blockDim.x;
      A_shared[sharedX] = Atemp[sharedX];
      B_shared[sharedX] = Btemp[sharedX];
    }

    // Make sure this is before the sync
    vec_t *Ctemp = C + x_block_top + y_block_top;

    __syncthreads();

    if(timesections) {
      memread += clock() - temp;
      temp = clock();
    }

    // Binary search diagonal in the local window for path
    {
      int32_t offset = threadIdX4 >> 1;
      Ax = offset + 1;
      vec_t * BSm1 = B_shared + threadIdX4p2;
      vec_t * BS = BSm1 + 1;
      while(true) {
	offset = ((offset+1) >> 1);
	if(A_shared[Ax] > BSm1[~Ax]) {
	  if(A_shared[Ax-1] <= BS[~Ax]) {
	    //Found it
	    break;
	  }
	  Ax -= offset;
	} else {
	  Ax += offset;
	}
      }
    }

    Bx = threadIdX4p2 - Ax;

    if(timesections) {
      if(threadIdx.x == 127) search += clock() - temp;
      temp = clock();
    }

    // Merge four elements starting at the found path intersection
    vec_t Ai, Bi, Ci;
    Ai = A_shared[Ax];
    Bi = B_shared[Bx];
    if(Ai > Bi) {Ci = Bi; Bx++; Bi = B_shared[Bx];} else {Ci = Ai; Ax++; Ai = A_shared[Ax];}
    Ctemp[threadIdX4] = Ci;
    if(Ai > Bi) {Ci = Bi; Bx++; Bi = B_shared[Bx];} else {Ci = Ai; Ax++; Ai = A_shared[Ax];}
    Ctemp[threadIdX4p1] = Ci;
    if(Ai > Bi) {Ci = Bi; Bx++; Bi = B_shared[Bx];} else {Ci = Ai; Ax++; Ai = A_shared[Ax];}
    Ctemp[threadIdX4p2] = Ci;
    Ctemp[threadIdX4p3] = Ai > Bi ? Bi : Ai;

    if(timesections) {
      if(threadIdx.x == 0) cglobal += clock() - temp;
      temp = clock();
    }

    // Update for next window
    if(threadIdx.x == 127) {
      x_block_top += Ax - 1;
      y_block_top += Bx - 1;
    }

    if(timesections) {
      if(threadIdx.x == 127) update += clock() - temp;
    }
    __syncthreads();
  } // Go to next window

  if(timesections) {
    float total = memread + search + cshared + cglobal + update;
    if(threadIdx.x == 0) {
      times[blockIdx.x] = memread/total;
      times[blockIdx.x + blockDim.x] = search/total;
      times[blockIdx.x + 2*blockDim.x] = cshared/total;
      times[blockIdx.x + 3*blockDim.x] = cglobal/total;
      times[blockIdx.x + 4*blockDim.x] = update/total;
    }
  }
}
