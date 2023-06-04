#include <iostream>
#include <omp.h>
#include "compressor.hpp"
#include <string>

int main(int argc,char *argv[]) {

    if (argc < 3) {
        printf("incorrect arguments.\n");
        printf("<input_path> <output_path> [K] [H]\n");
        abort();
    }
    std::string input_path(argv[1]);
    std::string output_path(argv[2]);

    int _K = -1, _H = -1;

    if (argc > 3) {
        _K = std::atoi(argv[3]);
        _H = std::atoi(argv[4]);
    }
    
    auto compressor = Compressor(_K, _H);
    compressor.load_graph(input_path);

    printf("%s graph loaded.\n", input_path.c_str());
    compressor.compress();
    printf("Compression completed.\n");
    compressor.write_cgr(output_path);
    printf("Generation completed.\n");

    return 0;
}
