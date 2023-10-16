#ifndef CONFIG_HELPER_HPP
#define CONFIG_HELPER_HPP

#include <string>
#include <tuple>
#include <vector>

void read_config_file(std::string config_file_path, std::vector<std::tuple<std::string, std::string>>& functions_to_run);

bool check_config_file(std::string config_file_path);

#endif  // CONFIG_HELPER_HPP