#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>
#include <omp.h>
#include <cfloat>
#include <cstdlib> 

using namespace std;

/// Global variable
int nthreads = 0 ;
string output_file_name = "output.csv";

/// Structure to store features of a point
struct Point {
    double x, y;     /// coordinates
    int cluster;     /// cluster ID
    double minDist;  /// Minimun distance to the nearest cluster

    /// Setting the default values
    Point() : x(0.0), y(0.0), cluster(-1), minDist(__DBL_MAX__) {}
    Point(double x, double y) : x(x), y(y), cluster(-1), minDist(__DBL_MAX__) {}

    
    /// Compute the Distance using Euclidean Distance formula between 2 points
    double distance(Point p) {
        return ((p.x - x) * (p.x - x) + (p.y - y) * (p.y - y));
    }
};

/// Reading the input ".csv" file and converting it into vector of points
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

void plotResults(string outputFileName) {
    string plotCommand =
        "gnuplot -e \"" 
        "set terminal png size 800,600; "      // Output to PNG
        "set output 'kmeans_plot.png'; "       // Name of output file
        "set title 'K-Means Clustering'; "     // Title of plot
        "set xlabel 'X'; set ylabel 'Y'; "    // Axis labels
        "set grid; "                           // Add grid lines
        "set datafile separator ','; "         // Specify space as separator
		"plot '" + outputFileName + "_serial.csv' every ::1 using 1:2:3 with points pt 7 palette title 'Clusters';\"";
		
		
	//  "plot '" + outputFileName + "_serial.csv' every ::1 using 1:2 with points pt 7 palette title 'Clusters';\""; // Plot the CSV file

    system(plotCommand.c_str());
}


/// Perform k-means clustering serially
/// @param pointsVec - pointer to vector of points
/// @param iterations - number of k means iterations
/// @param ncluster - the number of initial centroids (# of clusters we want)

void kMeansClusteringSerial(vector<Point>* pointsVec, int numIterations, int nClusters) {
    int n = (*pointsVec).size();

    // Randomly initialise ncluster many centroids (randomly select ncluster many initial points)
		// The index of the centroid represents the cluster ID
    vector<Point> centroids;
    srand(time(0));
    for (int i = 0; i < nClusters; ++i) {
        centroids.push_back((*pointsVec).at(rand() % n));
    }
	
	//do the Kmeans clustering process multiple times
    for (int i = 0; i < numIterations; ++i) {
		
        //---------assigning each point to the nearest centroid-------------
		
		//For each centroid, compute the distance between each point and the centroid
        for (vector<Point>::iterator centrIterator = begin(centroids); centrIterator != end(centroids); ++centrIterator) {
            int clusterId = centrIterator - begin(centroids);
			
			//for each point: if it's closest to this centroid then assign it to this centroid
            for (vector<Point>::iterator vecIterator = (*pointsVec).begin(); vecIterator != (*pointsVec).end(); vecIterator++) {
				//vecIterator contains this current point
                double dist = (*centrIterator).distance(*vecIterator);
				// updating this point's values to see if it should get assigned to this centroid
				if (dist < (*vecIterator).minDist) {
					(*vecIterator).minDist = dist;
					(*vecIterator).cluster = clusterId; //assigning this point to this centroid
				}
            }
        }
		
		//-----------recomputing centroids----------------
			//(recalc the position of each centroid to be the mean of all its points)
		
        // Vectors to keep track of the data needed to compute new centroids
        vector<int> nPointsInCluster; 
        vector<double> sumX, sumY; 
        for (int j = 0; j < nClusters; ++j) {
            nPointsInCluster.push_back(0);
            sumX.push_back(0.0);
            sumY.push_back(0.0);
        }
		//accumulate stats for each cluster
        for (vector<Point>::iterator vecIterator = (*pointsVec).begin(); vecIterator != (*pointsVec).end(); vecIterator++) {
			//each point, in the universe, already belongs to a cluster (using clusterID)
            int clusterId = (*vecIterator).cluster; 
            nPointsInCluster[clusterId] += 1;
			
			//add this point's x and y values to this point's sumX and sumY
            sumX[clusterId] += (*vecIterator).x;
            sumY[clusterId] += (*vecIterator).y;

            (*vecIterator).minDist = __DBL_MAX__;  // reset min distance, to get ready for next iteration
        }

		//for each centroid, compute its new centroid using the mean of all its points
        for (vector<Point>::iterator centrIterator = begin(centroids); centrIterator != end(centroids); ++centrIterator) {
            int clusterId = centrIterator - begin(centroids);
            (*centrIterator).x = sumX[clusterId] / nPointsInCluster[clusterId];
            (*centrIterator).y = sumY[clusterId] / nPointsInCluster[clusterId];
        }
    }

    // Writing output to csv
    ofstream output_file;
    output_file.open(output_file_name + "_serial.csv");
    output_file << "x,y,cluster_id" << endl;

    for (vector<Point>::iterator vecIterator = (*pointsVec).begin(); vecIterator != (*pointsVec).end(); vecIterator++) {
        output_file << (*vecIterator).x << "," << (*vecIterator).y << "," << (*vecIterator).cluster << endl;
    }
    output_file.close();
}

int main(int argc, char *argv[]) 
{
    int no_iterations = 0;
    int no_clusters = 0;
    string input_file_name = "input.csv";

    /// Setting command line arguments
    if(argc < 4)
    {
        cerr << "Invalid options." << endl << 
        "<program> <Input Data Filename> <Output Filename> <Number of Iterations> <Number of Clusters> [-t <num_threads>]" << endl;
        exit(1);
    }

    input_file_name = argv[1];
    output_file_name = argv[2];
    no_iterations = atoi(argv[3]);
    no_clusters = atoi(argv[4]);

    if(argc == 7 && strcasecmp(argv[5], "-t") == 0){
        nthreads = atoi(argv[6]);
        omp_set_num_threads(nthreads);
    }

    /// Reading data file and populating points in a vector
    vector<Point> points = readcsv(input_file_name);
    
    /// Running K-Means algorithm for Customer Segmentation
    /// Serial Code
    double t1,t2;
    t1 = omp_get_wtime();
    kMeansClusteringSerial(&points, no_iterations, no_clusters);
    t2 = omp_get_wtime();
	printf("serial algorithm. Data file: %s, Time: %.3f\n", input_file_name.c_str(), t2-t1);
	plotResults(output_file_name);

}
