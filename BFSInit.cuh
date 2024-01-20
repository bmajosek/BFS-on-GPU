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

using namespace std;

void BFSsequential(const vector<int>& startEdge, const vector<int>& endEdge, int u, int v, ostream& file);
void findPath(int* prev, int u, int v, std::vector<int>& path);
cudaError_t BFSCuda(vector<int>& startEdge, vector<int>& endEdge, int u, int v, ofstream& file);
cudaError_t BFSQueueVersion(thrust::device_vector<int> d_startEdge, thrust::device_vector<int> d_endEdge, thrust::device_vector<int> d_queue, thrust::device_vector<int> d_prefix_sums, thrust::device_vector<int> d_frontier, thrust::device_vector<int> d_newFrontier, bool* d_change, int size, int block_size, int num_blocks, int iteration, thrust::device_vector<int> begin_list, thrust::device_vector<int> distance, int number_of_edges, int v, thrust::device_vector<int> d_prev, ofstream& file, int u);
cudaError_t BFSLayersVersion(thrust::device_vector<int> d_startEdge, thrust::device_vector<int> d_endEdge, thrust::device_vector<int> d_frontier, thrust::device_vector<int> d_newFrontier, bool* d_change, int size, int block_size, int num_blocks, int iteration, thrust::device_vector<int> begin_list, thrust::device_vector<int> distance, int number_of_edges, int v, thrust::device_vector<int> d_prev, ofstream& file, int u);
cudaError_t BFSAtomicOppGlobalVersion(thrust::device_vector<int> d_startEdge, thrust::device_vector<int> d_endEdge, thrust::device_vector<int> d_queue, thrust::device_vector<int> d_size_of_old_queue, bool* d_change, int size, int block_size, int num_blocks, thrust::device_vector<int> begin_list, thrust::device_vector<int> distance, int number_of_edges, int iteration, int v, thrust::device_vector<int> d_prev, ofstream& file, int u);
cudaError_t BFSAtomicOppVersion(thrust::device_vector<int> d_startEdge, thrust::device_vector<int> d_endEdge, thrust::device_vector<int> d_queue, thrust::device_vector<int> d_size_of_new_queue, thrust::device_vector<int> d_size_of_old_queue, bool* d_change, int size, int block_size, int num_blocks, thrust::device_vector<int> begin_list, thrust::device_vector<int> distance, int number_of_edges, int iteration, int v, thrust::device_vector<int> d_prev, ofstream& file, int u, bool* d_possible);