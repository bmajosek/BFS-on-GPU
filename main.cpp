#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include "cuda_runtime.h"
#include "BFSInit.cuh"

using namespace std;

void showUsage() {
	cout << "Uzycie: [nazwa programu] -f [nazwa pliku grafu] -o [nazwa pliku wyjsciowego] -s [wierzcholek startowy] -e [wierzcholek koncowy]\n";
	cout << "Opcje:\n";
}

int main(int argc, char* argv[]) {
	if (argc != 9) {
		showUsage();
		return 1;
	}

	string inputFile, outputFile;
	int u = -1, v = -1;

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-f") == 0) {
			inputFile = argv[++i];
		}
		else if (strcmp(argv[i], "-o") == 0) {
			outputFile = argv[++i];
		}
		else if (strcmp(argv[i], "-s") == 0) {
			u = atoi(argv[++i]);
		}
		else if (strcmp(argv[i], "-e") == 0) {
			v = atoi(argv[++i]);
		}
		else if (strcmp(argv[i], "-help") == 0) {
			showUsage();
			return 0;
		}
	}

	ifstream file(inputFile);
	if (!file.is_open()) {
		cerr << "Nie mozna otworzyc pliku " << inputFile << "!" << endl;
		return 1;
	}

	ofstream fileOut(outputFile, ios::app);
	vector<int> startEdge, endEdge;

	int start, end;
	while (file >> start >> end) {
		startEdge.push_back(start);
		endEdge.push_back(end);
	}

	file.close();
	cout << "Wczytano dane z pliku " << inputFile << endl;
	cudaError_t cudaStatus = BFSCuda(startEdge, endEdge, u, v, fileOut);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "BFSCuda failed!");
		return 1;
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	auto startTimer = std::chrono::high_resolution_clock::now();
	BFSsequential(startEdge, endEdge, u, v, fileOut);

	auto endTime = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTimer).count();
	cout << "sequential BFS: " << duration / 1000.f << " milliseconds" << endl;

	cout << "\nwynik zostal zapisany w pliku wynik.txt\n";

	fileOut.close();
	return 0;
}