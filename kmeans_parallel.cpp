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
        "set terminal png size 800,600; "              // Output to PNG
        "set output 'kmeans_plot.png'; "               // Name of output file
        "set title 'K-Means Clustering'; "             // Title of plot
        "set xlabel 'X'; set ylabel 'Y'; "             // Axis labels
        "set grid; "                                   // Add grid lines
        "plot '" + outputFileName + "_serial.csv' "    // Plot the CSV file
        "using 1:2:3 with points pt 7 palette title 'Clusters';\"";

    system(plotCommand.c_str());
}

/// Perform k-means clustering Parallely
/// @param pointsVec - pointer to vector of points
/// @param iterations - number of k means iterations
/// @param ncluster - the number of initial centroids

void kMeansClusteringParallel (vector<Point>* pointsVec, int iterations, int ncluster) 
{
    int n = (*pointsVec).size(); /// Size of the data

    /// Vectors to keep track of data needed to compute new centroids
    vector<int> nPoints;
    vector<double> sumX, sumY;

    /// Randomly initialise centroids
    /// The index of the centroid represents the cluster ID
    vector<Point> centroids;
    srand(time(0));
    for (int i = 0; i < ncluster; ++i) 
    {
        centroids.push_back((*pointsVec).at(rand() % n));
    }

    for (int i = 0; i < iterations; ++i) 
    {
        /// For each centroid, compute distance between each point and centroid
        /// and update min Distance and Cluster ID, if changed
        for (unsigned int j=0; j < centroids.size(); j++) 
        {
            int clusterId = j;
                #pragma omp parallel for
                for(unsigned int i = 0; i < (*pointsVec).size(); i++)
                {
                    double dist = centroids[j].distance((*pointsVec)[i]);
                    if (dist < (*pointsVec)[i].minDist) 
                    {
                        (*pointsVec)[i].minDist = dist;
                        (*pointsVec)[i].cluster = clusterId;
                    }
                }
        }
    
        #pragma omp parallel
        {
            /// cout << "\n Thread Count : " << omp_get_num_threads();
            #pragma omp single 
            {
                for (int j = 0; j < ncluster; ++j) 
                {
                    nPoints.push_back(0);
                    sumX.push_back(0.0);
                    sumY.push_back(0.0);
                }
            }
            #pragma omp barrier

            #pragma omp for
            for (unsigned int j=0; j < (*pointsVec).size(); j++) 
            {
                int clusterId = (*pointsVec)[j].cluster;
                nPoints[clusterId] += 1;
                sumX[clusterId] += (*pointsVec)[j].x;
                sumY[clusterId] += (*pointsVec)[j].y;

                (*pointsVec)[j].minDist = __DBL_MAX__;  /// reset min distance
            }
            #pragma omp barrier

            /// Compute the new centroids
            #pragma omp for
            for (unsigned int j=0; j < centroids.size(); j++) 
            {
                int clusterId =j;
                centroids[j].x = sumX[clusterId] / nPoints[clusterId];
                centroids[j].y = sumY[clusterId] / nPoints[clusterId];
            }
        }
    }

    /// Writing output to csv
    ofstream output_file;        
    output_file.open(output_file_name + "_parallel.csv");
    output_file << "x,y,cluster_id" << endl;

    for (vector<Point>::iterator it = (*pointsVec).begin(); it != (*pointsVec).end(); ++it) 
    {
        output_file << it->x << "," << it->y << "," << it->cluster << endl;
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

    //cout << "algo_type,threads,in_data_file,time\n" << "serial," << nthreads << ",\"" << input_file_name << "\"," << (t2-t1) << "\n" ;
    /// Parallel Code
     t1 = omp_get_wtime();
     kMeansClusteringParallel(&points, no_iterations, no_clusters);
     t2 = omp_get_wtime();
	printf("openMP algorithm. Data file: %s, Time: %.3f, number threads: %d\n", input_file_name.c_str(), t2-t1, nthreads);

}