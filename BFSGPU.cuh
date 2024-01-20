#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/unique.h>
#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include <thrust/extrema.h>

using namespace std;

// Deklaracje funkcji kernelowych CUDA
__global__ void BFSPrescan(int iteration, int* queue, int num_vertices, int* adj_list_start, int* adj_list_end, int* adj_list_begin, int* distance, int* newFrontier, bool* d_change, int number_of_edges, int* prev);
__global__ void BFSLayers(int iteration, int* frontier, int num_vertices, int* adj_list_start, int* adj_list_end, int* adj_list_begin, int* distance, int* newFrontier, bool* d_change, int number_of_edges, int* prev);
__global__ void BFSAtomicOpp(int* queue_prev, int num_vertecies, int* adj_list_start, int* adj_list_end, int* adj_list_begin, int* distance, bool* d_change, int number_of_edges, int* d_size_of_new_queue, int* d_size_of_old_queue, int iteration, int* prev, bool* d_possible);
__global__ void BFSAtomicOppGlobalMemory(int* queue_prev, int* new_queue, int* pos, int num_vertecies, int* adj_list_start, int* adj_list_end, int* adj_list_begin, int* distance, bool* d_change, int number_of_edges, int* d_size_of_old_queue, int iteration, int* prev);
__global__ void kernel_cuda_generate_begin_list(int* prefix_sum, int* is_there, int* begin_lists, int num_vertices);
__global__ void kernel_cuda_generate_queue(int* prefix_sum, int* frontier, int* queue, int num_vertices);