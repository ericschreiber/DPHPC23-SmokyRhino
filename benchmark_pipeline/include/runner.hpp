#ifndef RUNNER_HPP
#define RUNNER_HPP

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

template <typename T>
class runner
{
    public:
        runner(std::string config_file_path, std::string out_path);
        ~runner();
        void run();

    private:
        void write_result();

        std::string _out_path;
        std::string _results_file_path;
        // A list of tuples to be run. Each tuple contains:
        // - The name of the function class to be run
        // - The name of the dataset to be used
        std::vector<std::tuple<std::string, std::string>> _functions_to_run;
        // A list of tuples containing the results of the runs. Each tuple contains:
        // - The name of the function class that was run
        // - The name of the dataset that was used
        // - Where the results are stored
        std::vector<std::tuple<std::string, std::string, std::string>> _results;
};

#endif  // RUNNER_HPP
