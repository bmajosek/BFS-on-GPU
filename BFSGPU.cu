#include "BFSGPU.cuh"
#include "GlobalVariables.hpp"

__global__ void BFSPrescan(int iteration, int* queue, int num_vertices, int* adj_list_start, int* adj_list_end, int* adj_list_begin, int* distance, int* newFrontier, bool* d_change, int number_of_edges, int* prev)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	// sprawdzam, czy watek miesci sie w kolejce (queue[0] - dlugosc kolejki)
	if (tid < num_vertices && queue[0] > tid)
	{
		// pobieram z kolejki wierzcholek
		int v = queue[tid + 1];
		// przegladam sasiadow
		for (int i = adj_list_begin[v]; i < number_of_edges && adj_list_start[i] == v; ++i)
		{
			int u = adj_list_end[i];
			// jezeli sasiad jest nieodwiedzony zmieniam jego odleglosc i ustawiam poprzednika
			if (distance[u] == MAX_DISTANCE)
			{
				distance[u] = iteration + 1;
				newFrontier[u] = 1;
				prev[u] = v;
			}
		}
	}
}
__global__ void BFSLayers(int iteration, int* frontier, int num_vertices, int* adj_list_start, int* adj_list_end, int* adj_list_begin, int* distance, int* newFrontier, bool* d_change, int number_of_edges, int* prev)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	// sprawdzam, czy wierzcholek zostal odwiedzony w poprzednim kroku
	if (tid < num_vertices && frontier[tid])
	{
		// przegladam sasiadow
		for (int i = adj_list_begin[tid]; i < number_of_edges && adj_list_start[i] == tid; ++i)
		{
			int u = adj_list_end[i];
			// jezeli sasiad jest nieodwiedzony zmieniam jego odleglosc i ustawiam poprzednika
			if (distance[u] == MAX_DISTANCE)
			{
				distance[u] = iteration + 1;
				newFrontier[u] = 1;
				prev[u] = tid;
				*d_change = true;
			}
		}
	}
}
__global__ void BFSAtomicOpp(int* queue_prev, int num_vertecies, int* adj_list_start, int* adj_list_end, int* adj_list_begin, int* distance, bool* d_change, int number_of_edges, int* d_size_of_new_queue, int* d_size_of_old_queue, int iteration, int* prev, bool* d_possible)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	__shared__ int pos;
	__shared__ int new_queue[MAX_BLOCK_QUEUE_SIZE];
	__shared__ int stride;
	if (threadIdx.x == 0)
	{
		pos = 0;
	}
	__syncthreads();
	// sprawdzam, czy watek miesci sie w kolejce
	if (tid < *d_size_of_old_queue)
	{
		// pobieram wierzcholek z kolejki
		int v = queue_prev[tid];
		if (v >= 0)
		{
			// przelgladam sasiadow
			for (int i = adj_list_begin[v]; i < number_of_edges && adj_list_start[i] == v; ++i)
			{
				int u = adj_list_end[i];
				// jezeli sasiad jest nieodwiedzony poprawiam jego odleglosc i poprzednika
				if (distance[u] == MAX_DISTANCE)
				{
					distance[u] = iteration + 1;
					// jezeli pos jest wieksze niz MAX_BLOCK_QUEUE_SIZE oznacza to ze kolejka lokalna nie miesci sie w shared memory, czyli tym sposobem nie da sie zrobic BFS
					if (pos >= MAX_BLOCK_QUEUE_SIZE)
					{
						*d_possible = false;
						return;
					}
					else
					{
						int oldPosition = atomicAdd(&pos, 1);
						new_queue[oldPosition] = u;
						prev[u] = v;
						*d_change = true;
					}
				}
			}
		}
	}
	__syncthreads();
	// obliczam miejsce gdzie zapisac do kolejki globalnej
	if (threadIdx.x == 0)
	{
		stride = atomicAdd(d_size_of_new_queue, pos);
	}
	__syncthreads();
	// watki przepisuja wierzcholki do kolejki globalnej
	for (int i = threadIdx.x; i < pos; i += blockDim.x)
	{
		queue_prev[i + stride] = new_queue[i];
	}
}
__global__ void BFSAtomicOppGlobalMemory(int* queue_prev, int* new_queue, int* pos, int num_vertecies, int* adj_list_start, int* adj_list_end, int* adj_list_begin, int* distance, bool* d_change, int number_of_edges, int* d_size_of_old_queue, int iteration, int* prev)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	// sprawdzam, czy watek miesci sie w kolejce
	if (tid < *d_size_of_old_queue)
	{
		int v = queue_prev[tid];
		if (v >= 0)
		{
			// przelgladam sasiadow
			for (int i = adj_list_begin[v]; i < number_of_edges && adj_list_start[i] == v; ++i)
			{
				int u = adj_list_end[i];
				// jezeli sasiad jest nieodwiedzony poprawiam jego odleglosc i poprzednika i powiekszam dlugosc kolejki
				if (distance[u] == MAX_DISTANCE)
				{
					distance[u] = iteration + 1;
					int oldPosition = atomicAdd(pos, 1);
					new_queue[oldPosition] = u;
					prev[u] = v;
					*d_change = true;
				}
			}
		}
	}
}
__global__ void kernel_cuda_generate_queue(int* prefix_sum, int* frontier, int* queue, int num_vertices) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// jezeli wierzcholek zostal odwiedzony w poprzednim kroku to oznacza ze musi zostac dodany do kolejki, obliczam jego miejsce i dodaje go
	if (tid < num_vertices && frontier[tid] == 1)
	{
		queue[prefix_sum[tid] + 1] = tid;
	}
	// ostatni watek zapisuje dlugosc kolejki
	if (tid == num_vertices - 1)
	{
		queue[0] = prefix_sum[tid] + frontier[tid];
	}
}
__global__ void kernel_cuda_generate_begin_list(int* prefix_sum, int* is_there, int* begin_lists, int num_vertices) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < num_vertices && is_there[tid])
	{
		begin_lists[prefix_sum[tid]] = tid;
	}
}