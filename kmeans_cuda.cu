#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <limits>
#include <string>
#include <chrono>
#include <cfloat>

#define THREADS_PER_BLOCK 256

using namespace std;

// Global variables
std::string input_file_name = "input.csv";
std::string output_file_name = "output.csv";

struct Point {
    double x, y;
    int cluster;
	Point(double _x = 0, double _y = 0, int _cluster = -1) 
        : x(_x), y(_y), cluster(_cluster) {}
};

// Reading the input ".csv" file and converting it into vector of points
vector<Point> readcsv(string input_file_name) 
{
    vector<Point> pointsVec;
    string line;
    ifstream file(input_file_name);
    while (getline(file, line)) 
    {
        stringstream lineStream(line);
        string bit;
        double x, y;
        getline(lineStream, bit, ',');
        x = stof(bit);
        getline(lineStream, bit, '\n');
        y = stof(bit);
        pointsVec.push_back(Point(x, y));
    }
    return pointsVec;
}

// Device function to compute Euclidean distance (x2-x1)^2 + (y2-y1)^2
__device__ double euclideanDistance(Point p1, Point p2) {
    return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}

// Kernel to assign points to the nearest centroid
__global__ void assignClusters(Point* points, int numPoints, Point* centroids, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numPoints) {
        double minDist = DBL_MAX;
        int bestCluster = -1;

        for (int j = 0; j < k; j++) {
            double dist = euclideanDistance(points[idx], centroids[j]);
            if (dist < minDist) {
                minDist = dist;
                bestCluster = j;
            }
        }
        points[idx].cluster = bestCluster;
    }
}

// Kernel to recompute centroids
__global__ void recomputeCentroids(Point* points, int numPoints, Point* centroids, int* counts, int k) {
	extern __shared__ double sharedMem[]; 
    double* sumX = sharedMem;
    double* sumY = &sharedMem[k];
    int* clusterCount = (int*)&sharedMem[2 * k];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < k) {
        sumX[idx] = 0.0;
        sumY[idx] = 0.0;
        clusterCount[idx] = 0;

    }
    __syncthreads();
	//Accumulate XY values for each cluster
    if (idx < numPoints) {
        int cluster = points[idx].cluster;
        atomicAdd(&sumX[cluster], points[idx].x);
        atomicAdd(&sumY[cluster], points[idx].y);
        atomicAdd(&clusterCount[cluster], 1);
    }
    __syncthreads();
	// Finalize centroids by averaging accumulated values
    if (idx < k) {
        centroids[idx].x = sumX[idx] / clusterCount[idx];
        centroids[idx].y = sumY[idx] / clusterCount[idx];
    }
}

// Host function to run K-Means clustering
void kMeansClusteringCuda(std::vector<Point>& points, int numIterations, int nClusters, int threadsPerBlockXdim) {
    int numPoints = points.size();

    // Allocate memory on the device
    Point* d_points;
    Point* d_centroids;
    int* d_counts;
    cudaMalloc(&d_points, numPoints * sizeof(Point));
    cudaMalloc(&d_centroids, nClusters * sizeof(Point));
    cudaMalloc(&d_counts, nClusters * sizeof(int));

    // Initialize centroids randomly
    std::vector<Point> centroids(nClusters);
    srand(time(0));
    for (int i = 0; i < nClusters; ++i) {
        centroids[i] = points[rand() % numPoints];
    }

    // Copy data to device
    cudaMemcpy(d_points, points.data(), numPoints * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids.data(), nClusters * sizeof(Point), cudaMemcpyHostToDevice);
		
	dim3 threadsPerBlock(threadsPerBlockXdim, 1, 1);
	dim3 numBlocks((numPoints + threadsPerBlock.x - 1)/threadsPerBlock.x, 1, 1);
	size_t sharedMemSize = (2 * nClusters * sizeof(double)) + (nClusters * sizeof(int));
			
	//do the Kmeans clustering process many times
	for (int i = 0; i < numIterations; ++i) {
		// Launch kernel to assign clusters
		assignClusters<<<numBlocks, threadsPerBlock>>>(d_points, numPoints, d_centroids, nClusters);

		// Reset counts
		cudaMemset(d_counts, 0, nClusters * sizeof(int));

		// Launch kernel to recompute centroids
		recomputeCentroids<<<numBlocks, threadsPerBlock, sharedMemSize>>>(d_points, numPoints, d_centroids, d_counts, nClusters);
	}
	

    // Copy results back to host
    cudaMemcpy(points.data(), d_points, numPoints * sizeof(Point), cudaMemcpyDeviceToHost);
    cudaMemcpy(centroids.data(), d_centroids, nClusters * sizeof(Point), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_counts);

    // Write results to output file
    std::ofstream output_file("output_cuda.csv");
    output_file << "x,y,cluster_id\n";
    for (const auto& point : points) {
        output_file << point.x << "," << point.y << "," << point.cluster << "\n";
    }
    output_file.close();
}

int main(int argc, char *argv[]) {
	int numIterations = 0;
    int nClusters = 0;
	
	if(argc < 4){
        cerr << "Invalid options." << endl << 
        "<program> <Input Data Filename> <Output Filename> <Number of Iterations> <Number of Clusters> [-t <num_threads>]" << endl;
        exit(1);
    }
	
	input_file_name = argv[1];
    output_file_name = argv[2];
    numIterations = atoi(argv[3]);
    nClusters = atoi(argv[4]);
	
	 // Reading data file and populating points in a vector
    vector<Point> points = readcsv(input_file_name);
	int numPoints = points.size();
	
	//-------run kMeansClustering mutliple times with different block dimensions-----
	int loopIterator = 0;
	int threadsPerBlockXdim = 8;
	while(loopIterator < 5){
		auto t1 = std::chrono::high_resolution_clock::now();
		
		dim3 threadsPerBlock(threadsPerBlockXdim, 1, 1);
		dim3 numBlocks((numPoints + threadsPerBlock.x - 1)/threadsPerBlock.x, 1, 1);
		//size_t sharedMemSize = (2 * nClusters * sizeof(double)) + (nClusters * sizeof(int));
				
		kMeansClusteringCuda(points, numIterations, nClusters, threadsPerBlockXdim);
		
		auto t2 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> runTime = t2 - t1;
		printf("Application: csv file. Data file: %s, Time: %.3f, using blockSize (%d, %d, %d) \n", 
				input_file_name.c_str(), runTime, threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);

		
		threadsPerBlockXdim = threadsPerBlockXdim + 8;
		loopIterator++;
	}

    return 0;
}
