#include <cstdlib>
#include <iostream>
#include <fstream>
using namespace std;

int main(int argc, char *argv[]) 
{

    /// Setting command line arguments
    if(argc < 7)
    {
        cerr << "Invalid options." << endl << 
        "<program> <output_file_name> <rec_count> <low rang1> <high range1> <low rang2> <high range2>]" << endl;
        exit(1);
    }

    string output_file_name = argv[1];
    int rec_count=atoi(argv[2]);
    int f1 = atoi(argv[3]);
    int f2 = atoi(argv[4]);
    int s1 = atoi(argv[5]);
    int s2 = atoi(argv[6]);

     /// Writing output to csv
    ofstream output_file;        
    output_file.open(output_file_name);
    // output_file << "x,y" << endl;

    for (int i=0; i < rec_count; i++) 
    {
        output_file << (int)((rand()%(f2-f1))+f1) << "," << (int)((rand()%(s2-s1))+s1) << endl;
    }
    output_file.close();
}
