#include "config_helper.hpp"

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

// *************************************************************************************************
//
// The format of the config file is as follows:
// - Each line contains a function class and a dataset separated by a comma
// - The dataset is comprised of 3 paths separated by a comma
//
// *************************************************************************************************

void read_config_file(std::string config_file_path, std::vector<std::tuple<std::string, std::string, dataset_paths>>& functions_to_run)
{
    assert(check_config_file(config_file_path) && "Config file has problems");
    // Open the file for reading
    std::ifstream config_file(config_file_path);
    // Read the file line by line
    std::string line;
    while (std::getline(config_file, line))
    {
        // Check if the line is empty
        if (line.empty())
        {
            continue;
        }
        // Check if the line is a comment
        if (line[0] == '#')
        {
            continue;
        }
        // Split the line
        std::vector<std::string> line_split;
        std::stringstream line_stream(line);
        std::string cell;
        while (std::getline(line_stream, cell, ','))
        {
            // Remove all whitespace from the cell
            cell.erase(std::remove_if(cell.begin(), cell.end(), ::isspace), cell.end());
            line_split.push_back(cell);
        }
        // Add the line to the list of functions to run
        dataset_paths dataset = dataset_paths(line_split[2], line_split[3], line_split[4]);
        functions_to_run.push_back(std::make_tuple(line_split[0], line_split[1], dataset));
    }
    // Close the file
    config_file.close();
}

// *************************************************************************************************
//
// The format of the config file is as follows:
// - Each line contains a function class and a dataset separated by a comma
// - The dataset is comprised of 3 paths separated by a comma
//
// We check the following:
// - The file can be opened
// - The file is not empty
// - Each line has four values separated by a comma
//
// *************************************************************************************************
bool check_config_file(std::string config_file_path)
{
    // Open the file for reading
    std::ifstream config_file(config_file_path);
    // Check if the file is open
    if (!config_file.is_open())
    {
        std::cerr << "Could not open config file: " << config_file_path << std::endl;
        return false;
    }
    // Check if the file is empty
    if (config_file.peek() == std::ifstream::traits_type::eof())
    {
        std::cerr << "Config file is empty: " << config_file_path << std::endl;
        return false;
    }
    // Read the file line by line
    std::string line;
    while (std::getline(config_file, line))
    {
        // Check if the line is empty
        if (line.empty())
        {
            continue;
        }
        // Check if the line is a comment
        if (line[0] == '#')
        {
            continue;
        }
        // Check if the line is valid
        std::vector<std::string> line_split;
        std::stringstream line_stream(line);
        std::string cell;
        while (std::getline(line_stream, cell, ','))
        {
            line_split.push_back(cell);
        }
        if (line_split.size() != 5)
        {
            std::cerr << "Invalid line in config file: " << line << std::endl;
            return false;
        }
        // check that all the paths exist (cell 2, 3, 4)
        for (int i = 2; i < 5; i++)
        {
            // Remove all whitespace from the cell
            line_split[i].erase(std::remove_if(line_split[i].begin(), line_split[i].end(), ::isspace), line_split[i].end());
            std::ifstream file(line_split[i]);
            if (!file.good())
            {
                std::cerr << "File does not exist: " << line_split[i] << std::endl;
                return false;
            }
        }
    }
    // Close the file
    config_file.close();
    // Return true if all checks passed
    return true;
}
