#include "GlobalVariables.hpp"
#include "BFSInit.cuh"
#include "BFSGPU.cuh"

void BFSsequential(const vector<int>& startEdge, const vector<int>& endEdge, int u, int v, ostream& file) {
	int numVertices = *max_element(endEdge.begin(), endEdge.end()) + 1;
	vector<vector<int>> adjacencyList(numVertices);
	for (size_t i = 0; i < startEdge.size(); ++i) {
		adjacencyList[startEdge[i]].push_back(endEdge[i]);
	}

	vector<bool> visited(numVertices, false);
	queue<int> que;
	que.push(u);
	visited[u] = true;
	int* h_prev = new int[numVertices];
	for (int i = 0; i < numVertices; ++i)
		h_prev[i] = -1;
	while (!que.empty() && !visited[v]) {
		int currentVertex = que.front();
		que.pop();
		// Przeszukiwanie s¹siadów
		for (int neighbor : adjacencyList[currentVertex]) {
			if (!visited[neighbor]) {
				h_prev[neighbor] = currentVertex;
				visited[neighbor] = true;
				que.push(neighbor);
			}
		}
	}
	vector<int> path;
	findPath(h_prev, u, v, path);
	if (!path.empty())
	{
		file << "\n Sequential BFS\n";
		std::copy(path.begin(), path.end(), std::ostream_iterator<int>(file, " "));
		file << "\n";
	}
	else
	{
		file << "Nie znaleziono sciezki. -- Algorytm sekwencyjny\n";
	}
	delete[] h_prev;
}
void findPath(int* prev, int u, int v, std::vector<int>& path) {
	int current = v;
	path.clear();

	while (current != -1 && current != u) {
		path.push_back(current);
		current = prev[current];
	}

	if (current == u) {
		path.push_back(u);
		std::reverse(path.begin(), path.end());
	}
	else {
		path.clear();
	}
}

// Funkcja uruchamiaj¹ca algorytmy BFS na GPU
cudaError_t BFSCuda(vector<int>& startEdge, vector<int>& endEdge, int u, int v, ofstream& file) {
	// inicjalizuje zmienne
	cudaError_t cudaStatus;
	cudaEvent_t startEvent, stopEvent;
	float elapsedMilliseconds;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventRecord(startEvent, 0);
	int size = *thrust::max_element(endEdge.begin(), endEdge.end());
	int iteration = 0;
	int* frontier = new int[size + 1];
	frontier[u] = 1;
	thrust::device_vector<int> d_size_of_new_queue(1);
	thrust::device_vector<int> d_size_of_old_queue(1);
	thrust::device_vector<int> d_startEdge(startEdge.begin(), startEdge.end());
	thrust::device_vector<int> d_endEdge(endEdge.begin(), endEdge.end());
	thrust::device_vector<int> d_queue(size);
	thrust::device_vector<int> d_prefix_sums(size + 1);
	thrust::device_vector<int> d_frontier(size + 1);
	thrust::device_vector<int> output(size);
	thrust::device_vector<int> begin_list(d_startEdge.size());
	thrust::device_vector<int> adj_vector(d_startEdge.size());
	thrust::device_vector<int> distance(size + 1);
	thrust::device_vector<int> d_newFrontier(size + 1);
	thrust::device_vector<int> d_prev(size + 1);
	bool change = 1, * d_change, * d_possible, possible = true;
	int block_size = BLOCK_SIZE;
	int num_blocks = (size + block_size - 1) / block_size;
	int block_size_edges = BLOCK_SIZE;
	int num_blocks_edges = (d_startEdge.size() + block_size_edges - 1) / block_size_edges;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&d_change, sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for d_change: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_possible, sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for d_possible: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaMemcpy(d_change, &change, sizeof(bool), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for d_change!");
		goto Error;
	}

	cudaMemcpy(d_possible, &possible, sizeof(bool), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for d_possible!");
		goto Error;
	}

	// ustawiam poczatkowa dlugosc kolejki
	thrust::fill(d_size_of_old_queue.begin(), d_size_of_old_queue.end(), 1);
	// ustawiam poprzednikow wierzcholkow
	thrust::fill(d_prev.begin(), d_prev.end(), -1);
	// tworze poczatkowa kolejke
	thrust::fill(d_queue.begin(), d_queue.begin() + 1, u);
	thrust::fill(d_queue.begin() + 1, d_queue.end(), -1);
	// ustawiam odleglosci do wierzcholkow
	thrust::fill(distance.begin(), distance.begin() + u, MAX_DISTANCE);
	thrust::fill(distance.begin() + u + 1, distance.end(), MAX_DISTANCE);
	// ustawiam tablcie wierzcholkow ktore maja byc odwiedzone w kolejnym kroku
	thrust::fill(d_frontier.begin(), d_frontier.begin() + u, 0);
	thrust::fill(d_frontier.begin() + u + 1, d_frontier.end(), 0);
	thrust::fill(d_frontier.begin() + u, d_frontier.begin() + u + 1, 1);

	// tworze tablice gdzie mam posortowane krawedzie (poczatki i konce w drugiej tablicy), a nastepnie obliczam gdzie zaczyna sie kazdy wierzcholek
	thrust::sort_by_key(d_startEdge.begin(), d_startEdge.end(), d_endEdge.begin());
	thrust::adjacent_difference(d_startEdge.begin(), d_startEdge.end(), adj_vector.begin());
	kernel_cuda_generate_begin_list << <num_blocks_edges, block_size_edges >> > (thrust::raw_pointer_cast(d_startEdge.data()), thrust::raw_pointer_cast(adj_vector.data()), thrust::raw_pointer_cast(begin_list.data()), d_startEdge.size());
	if (cudaGetLastError() != cudaSuccess)
		goto Error;
	if (cudaDeviceSynchronize() != cudaSuccess)
		goto Error;
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedMilliseconds, startEvent, stopEvent);
	std::cout << "Initialize Time: " << elapsedMilliseconds << " ms\n" << std::endl;

	// Wersja z kolejka bez atomowych operacji
	cudaEventRecord(startEvent, 0);
	cudaStatus = BFSQueueVersion(d_startEdge, d_endEdge, d_queue, d_prefix_sums, d_frontier, d_newFrontier, d_change, size, block_size, num_blocks, iteration, begin_list, distance, d_startEdge.size(), v, d_prev, file, u);
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedMilliseconds, startEvent, stopEvent);
	std::cout << "BFSQueueVersion Time: " << elapsedMilliseconds << " ms\n" << std::endl;

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "BFSQueueVersion failed! Error: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// Wersja bez kolejki, za kazdym razem przegladam nieodwiedzonych sasiadow
	cudaEventRecord(startEvent, 0);
	cudaStatus = BFSLayersVersion(d_startEdge, d_endEdge, d_frontier, d_newFrontier, d_change, size, block_size, num_blocks, iteration, begin_list, distance, d_startEdge.size(), v, d_prev, file, u);
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedMilliseconds, startEvent, stopEvent);
	std::cout << "BFSLayersVersion Time: " << elapsedMilliseconds << " ms\n" << std::endl;

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "BFSLayersVersion failed! Error: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// Wersja z kolejkami lokalnymi z atomowymi operacjami (nie dziala gdy graf ma duzo krawedzi, nie miesci sie w shared memory)
	cudaEventRecord(startEvent, 0);
	cudaStatus = BFSAtomicOppVersion(d_startEdge, d_endEdge, d_queue, d_size_of_new_queue, d_size_of_old_queue, d_change, size, block_size, num_blocks, begin_list, distance, d_startEdge.size(), iteration, v, d_prev, file, u, d_possible);
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedMilliseconds, startEvent, stopEvent);
	std::cout << "BFSAtomicOppVersion Time: " << elapsedMilliseconds << " ms\n" << std::endl;

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "BFSAtomicOppVersion failed! Error: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// Wersja z kolejka globalna z atomowymi operacjami
	cudaEventRecord(startEvent, 0);
	cudaStatus = BFSAtomicOppGlobalVersion(d_startEdge, d_endEdge, d_queue, d_size_of_old_queue, d_change, size, block_size, num_blocks, begin_list, distance, d_startEdge.size(), iteration, v, d_prev, file, u);
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedMilliseconds, startEvent, stopEvent);
	std::cout << "BFSAtomicOppGlobalVersion Time: " << elapsedMilliseconds << " ms\n" << std::endl;

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "BFSAtomicOppGlobalVersion failed! Error: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

Error:
	delete[] frontier;
	cudaFree(d_change);
	cudaFree(d_possible);
	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);

	return cudaStatus;
}
cudaError_t BFSQueueVersion(thrust::device_vector<int> d_startEdge, thrust::device_vector<int> d_endEdge, thrust::device_vector<int> d_queue, thrust::device_vector<int> d_prefix_sums, thrust::device_vector<int> d_frontier, thrust::device_vector<int> d_newFrontier, bool* d_change, int size, int block_size, int num_blocks, int iteration, thrust::device_vector<int> begin_list, thrust::device_vector<int> distance, int number_of_edges, int v, thrust::device_vector<int> d_prev, ofstream& file, int u)
{
	cudaError_t cudaStatus = cudaSuccess;
	bool change = true;
	int distance_v = MAX_DISTANCE;
	cudaEvent_t startQueueEvent, stopQueueEvent, startLoopEvent, stopLoopEvent;
	float elapsedQueueMilliseconds_Sum = 0.0f;
	float elapsedLoopMilliseconds_Sum = 0.0f;
	int* d_queue_act_size;
	int queue_act_size = 1;
	cudaStatus = cudaMalloc((void**)&d_queue_act_size, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for d_change: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	cudaEventCreate(&startQueueEvent);
	cudaEventCreate(&stopQueueEvent);
	cudaEventCreate(&startLoopEvent);
	cudaEventCreate(&stopLoopEvent);
	// petla ktora dziala do momentu az odleglosc do ostatniego wierzcholka sie zmieni
	while (distance_v == MAX_DISTANCE)
	{
		// poczatek mierzenia czasu tworzenia kolejki
		cudaEventRecord(startQueueEvent, 0);
		// tworze tablice sum prefiksowych, aby pozniej wiedziec ile jest elementow w kolejce i latwo obliczyc miejsce w kolejce
		thrust::exclusive_scan(d_frontier.begin(), d_frontier.begin() + size, d_prefix_sums.begin());
		// tworze kolejke
		kernel_cuda_generate_queue << <num_blocks, block_size >> > (thrust::raw_pointer_cast(d_prefix_sums.data()), thrust::raw_pointer_cast(d_frontier.data()), thrust::raw_pointer_cast(d_queue.data()), size);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			return cudaStatus;

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
			return cudaStatus;
		cudaEventRecord(stopQueueEvent, 0);
		cudaEventSynchronize(stopQueueEvent);
		float elapsedQueueMilliseconds;
		cudaEventElapsedTime(&elapsedQueueMilliseconds, startQueueEvent, stopQueueEvent);
		elapsedQueueMilliseconds_Sum += elapsedQueueMilliseconds;
		cudaMemcpy(&queue_act_size, thrust::raw_pointer_cast(&(d_queue[0])), sizeof(int), cudaMemcpyDeviceToHost);
		// jezeli kolejka jest pusta koncze wykonywanie petli
		if (queue_act_size == 0)
		{
			break;
		}
		// poczatek mierzenia czasu przegladania sasiadow
		cudaEventRecord(startLoopEvent, 0);
		int num_b = (queue_act_size + block_size - 1) / block_size;
		// przegladam sasiadow wierzcholkow znajdujacych sie w kolejce
		BFSPrescan << <num_b, block_size >> > (iteration++, thrust::raw_pointer_cast(d_queue.data()), size, thrust::raw_pointer_cast(d_startEdge.data()), thrust::raw_pointer_cast(d_endEdge.data()), thrust::raw_pointer_cast(begin_list.data()), thrust::raw_pointer_cast(distance.data()), thrust::raw_pointer_cast(d_newFrontier.data()), d_change, d_startEdge.size(), thrust::raw_pointer_cast(d_prev.data()));

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			return cudaStatus;

		// przygotowuje tablice Frontier do stworzenia kolejki w kolejnym kroku
		thrust::copy(d_newFrontier.begin(), d_newFrontier.begin() + size, d_frontier.begin());
		thrust::fill(d_newFrontier.begin(), d_newFrontier.begin() + size, 0);
		cudaMemcpy(&change, d_change, sizeof(bool) * 1, cudaMemcpyDeviceToHost);
		cudaEventRecord(stopLoopEvent, 0);
		cudaEventSynchronize(stopLoopEvent);
		float elapsedLoopMilliseconds;
		cudaEventElapsedTime(&elapsedLoopMilliseconds, startLoopEvent, stopLoopEvent);
		elapsedLoopMilliseconds_Sum += elapsedLoopMilliseconds;
		cudaMemcpy(&distance_v, thrust::raw_pointer_cast(&(distance[v])), sizeof(int), cudaMemcpyDeviceToHost);
	}

	std::cout << "Calkowity czas tworzenia kolejki: " << elapsedQueueMilliseconds_Sum << " ms" << std::endl;
	std::cout << "Calkowity czas BFSPrescan'u: " << elapsedLoopMilliseconds_Sum << " ms" << std::endl;

	cudaEventDestroy(startQueueEvent);
	cudaEventDestroy(stopQueueEvent);
	cudaEventDestroy(startLoopEvent);
	cudaEventDestroy(stopLoopEvent);
	vector<int> path;

	int* h_prev = new int[size + 1];
	cudaMemcpy(h_prev, thrust::raw_pointer_cast(d_prev.data()), (size + 1) * sizeof(int), cudaMemcpyDeviceToHost);

	// Znajdz i wypisz sciezke
	findPath(h_prev, u, v, path);

	if (!path.empty())
	{
		file << "\nQueue BFS\n";
		std::copy(path.begin(), path.end(), std::ostream_iterator<int>(file, " "));
		file << "\n";
	}
	else
	{
		file << "Nie znaleziono sciezki. -- Algorytm CUDA BFS z kolejka\n";
	}

	delete[] h_prev;
	return cudaStatus;
}
cudaError_t BFSLayersVersion(thrust::device_vector<int> d_startEdge, thrust::device_vector<int> d_endEdge, thrust::device_vector<int> d_frontier, thrust::device_vector<int> d_newFrontier, bool* d_change, int size, int block_size, int num_blocks, int iteration, thrust::device_vector<int> begin_list, thrust::device_vector<int> distance, int number_of_edges, int v, thrust::device_vector<int> d_prev, ofstream& file, int u)
{
	bool change = true;
	int distance_v = MAX_DISTANCE;
	cudaError_t cudaStatus = cudaSuccess;
	// petla ktora dziala do momentu az ostatni wierzcholek jest nieodwiedzony i jest jeszcze nieodwiedzony sasaid
	while (change && distance_v == MAX_DISTANCE)
	{
		change = 0;
		cudaMemcpy(d_change, &change, sizeof(bool) * 1, cudaMemcpyHostToDevice);
		// przegladam nieodwiedzonych sasiadow
		BFSLayers << <num_blocks, block_size >> > (iteration++, thrust::raw_pointer_cast(d_frontier.data()), size, thrust::raw_pointer_cast(d_startEdge.data()), thrust::raw_pointer_cast(d_endEdge.data()), thrust::raw_pointer_cast(begin_list.data()), thrust::raw_pointer_cast(distance.data()), thrust::raw_pointer_cast(d_newFrontier.data()), d_change, d_startEdge.size(), thrust::raw_pointer_cast(d_prev.data()));

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			return cudaStatus;

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
			return cudaStatus;
		// przygotowuje tablcie Frontier do nastepnego kroku
		thrust::copy(d_newFrontier.begin(), d_newFrontier.end(), d_frontier.begin());
		thrust::fill(d_newFrontier.begin(), d_newFrontier.end(), 0);
		cudaMemcpy(&change, d_change, sizeof(bool) * 1, cudaMemcpyDeviceToHost);
		cudaMemcpy(&distance_v, thrust::raw_pointer_cast(&(distance[v])), sizeof(int), cudaMemcpyDeviceToHost);
	}
	vector<int> path;
	int* h_prev = new int[size + 1];
	cudaMemcpy(h_prev, thrust::raw_pointer_cast(d_prev.data()), (size + 1) * sizeof(int), cudaMemcpyDeviceToHost);

	// Znajdz i wypisz sciezke
	findPath(h_prev, u, v, path);
	if (!path.empty())
	{
		file << "\nFrontiers BFS\n";
		std::copy(path.begin(), path.end(), std::ostream_iterator<int>(file, " "));
		file << "\n";
	}
	else
	{
		file << "Nie znaleziono sciezki. -- Algorytm CUDA BFS Frontiers\n";
	}

	delete[] h_prev;
	return cudaStatus;
}
cudaError_t BFSAtomicOppVersion(thrust::device_vector<int> d_startEdge, thrust::device_vector<int> d_endEdge, thrust::device_vector<int> d_queue, thrust::device_vector<int> d_size_of_new_queue, thrust::device_vector<int> d_size_of_old_queue, bool* d_change, int size, int block_size, int num_blocks, thrust::device_vector<int> begin_list, thrust::device_vector<int> distance, int number_of_edges, int iteration, int v, thrust::device_vector<int> d_prev, ofstream& file, int u, bool* d_possible)
{
	cudaError_t cudaStatus = cudaSuccess;
	bool change = true;
	bool possible = true;
	int zero = 0, distance_v = MAX_DISTANCE;
	// petla ktora dziala do momentu az ostatni wierzcholek jest nieodwiedzony i jest jeszcze nieodwiedzony sasiad
	while (change && possible && distance_v == MAX_DISTANCE) {
		change = false;
		cudaMemcpy(d_change, &change, sizeof(bool), cudaMemcpyHostToDevice);

		// wlaczam BFS z atomowymi operacjami i lokalna kolejka
		BFSAtomicOpp << <num_blocks, block_size >> > (
			thrust::raw_pointer_cast(d_queue.data()),
			size,
			thrust::raw_pointer_cast(d_startEdge.data()),
			thrust::raw_pointer_cast(d_endEdge.data()),
			thrust::raw_pointer_cast(begin_list.data()),
			thrust::raw_pointer_cast(distance.data()),
			d_change,
			d_startEdge.size(),
			thrust::raw_pointer_cast(d_size_of_new_queue.data()),
			thrust::raw_pointer_cast(d_size_of_old_queue.data()),
			iteration++, thrust::raw_pointer_cast(d_prev.data()),
			d_possible
			);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			return cudaStatus;

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
			return cudaStatus;

		cudaMemcpy(&change, d_change, sizeof(bool), cudaMemcpyDeviceToHost);
		cudaMemcpy(d_size_of_old_queue.data().get(), d_size_of_new_queue.data().get(), sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_size_of_new_queue.data().get(), &zero, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(&distance_v, thrust::raw_pointer_cast(&(distance[v])), sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&possible, d_possible, sizeof(bool), cudaMemcpyDeviceToHost);
	}

	if (possible)
	{
		vector<int> path;
		int* h_prev = new int[size + 1];
		cudaMemcpy(h_prev, thrust::raw_pointer_cast(d_prev.data()), (size + 1) * sizeof(int), cudaMemcpyDeviceToHost);

		// Znajdz i wypisz sciezke
		findPath(h_prev, u, v, path);
		if (!path.empty())
		{
			file << "\nAtomic Opperations BFS\n";
			std::copy(path.begin(), path.end(), std::ostream_iterator<int>(file, " "));
			file << "\n";
		}
		else
		{
			file << "Nie znaleziono sciezki. -- Algorytm CUDA BFS Atomowe operacje z shared memory\n";
		}
		delete[] h_prev;
	}
	else
	{
		file << "Nie jest mozliwe znalezienie sciezki tym sposobem. -- Algorytm CUDA BFS Atomowe operacje z shared memory\n";
	}
	return cudaStatus;
}
cudaError_t BFSAtomicOppGlobalVersion(thrust::device_vector<int> d_startEdge, thrust::device_vector<int> d_endEdge, thrust::device_vector<int> d_queue, thrust::device_vector<int> d_size_of_old_queue, bool* d_change, int size, int block_size, int num_blocks, thrust::device_vector<int> begin_list, thrust::device_vector<int> distance, int number_of_edges, int iteration, int v, thrust::device_vector<int> d_prev, ofstream& file, int u)
{
	cudaError_t cudaStatus = cudaSuccess;
	bool change = true;
	int zero = 0, distance_v = MAX_DISTANCE;
	int* pos;
	thrust::device_vector<int> d_new_queue(size);
	thrust::fill(d_new_queue.begin(), d_new_queue.begin() + size, -1);
	cudaStatus = cudaMalloc((void**)&pos, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for pos: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	// petla ktora dziala do momentu az ostatni wierzcholek jest nieodwiedzony i jest jeszcze nieodwiedzony sasiad
	while (change && distance_v == MAX_DISTANCE) {
		change = false;
		cudaMemcpy(d_change, &change, sizeof(bool), cudaMemcpyHostToDevice);
		cudaMemcpy(pos, &zero, sizeof(bool), cudaMemcpyHostToDevice);

		// wlaczam BFS z atomowymi operacjami i globalna kolejka
		BFSAtomicOppGlobalMemory << <num_blocks, block_size >> > (
			thrust::raw_pointer_cast(d_queue.data()),
			thrust::raw_pointer_cast(d_new_queue.data()),
			pos,
			size,
			thrust::raw_pointer_cast(d_startEdge.data()),
			thrust::raw_pointer_cast(d_endEdge.data()),
			thrust::raw_pointer_cast(begin_list.data()),
			thrust::raw_pointer_cast(distance.data()),
			d_change,
			d_startEdge.size(),
			thrust::raw_pointer_cast(d_size_of_old_queue.data()),
			iteration++,
			thrust::raw_pointer_cast(d_prev.data())
			);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			return cudaStatus;

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
			return cudaStatus;

		cudaMemcpy(&change, d_change, sizeof(bool), cudaMemcpyDeviceToHost);
		cudaMemcpy(d_size_of_old_queue.data().get(), pos, sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(&distance_v, thrust::raw_pointer_cast(&(distance[v])), sizeof(int), cudaMemcpyDeviceToHost);
		thrust::copy(d_new_queue.begin(), d_new_queue.end(), d_queue.begin());
		thrust::fill(d_new_queue.begin(), d_new_queue.begin() + size, -1);
	}
	vector<int> path;
	int* h_prev = new int[size + 1];
	cudaMemcpy(h_prev, thrust::raw_pointer_cast(d_prev.data()), (size + 1) * sizeof(int), cudaMemcpyDeviceToHost);

	// Znajdz i wypisz sciezke
	findPath(h_prev, u, v, path);
	if (!path.empty())
	{
		file << "\n Global Atomic Opperations BFS\n";
		std::copy(path.begin(), path.end(), std::ostream_iterator<int>(file, " "));
		file << "\n";
	}
	else
	{
		file << "Nie znaleziono sciezki. -- Algorytm CUDA BFS Atomowe operacje globalna pamiec\n";
	}
	delete[] h_prev;
	return cudaStatus;
}