#include <fstream>
#include <iostream>
#include <string>

#include "runner.hpp"

// Read config file
// Read input data
// Run benchmark
// Write output data
// Write benchmark results

int main(int argc, char* argv[])
{
    // Check if the correct number of arguments is given
    if (argc != 3)
    {
        std::cout << "Wrong number of arguments. Expected 2, got " << argc - 1
                  << std::endl;
        std::cout << "Usage: " << argv[0] << " <config_file_path>"
                  << "<out_path>" << std::endl;
        return 1;
    }
    std::string config_file_path = argv[1];
    std::string out_path = argv[2];
    // Create a runner
    runner<float> r(config_file_path, out_path);
    // Run the benchmark
    r.run();
    return 0;
}