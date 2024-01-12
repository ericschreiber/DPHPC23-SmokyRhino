#!/bin/bash

# Creates the config files for a run

# Set what to run

# Will generate the config files to set directory, and create run_on_cluster_test.sh file
generate_configs=true
# If ture it will delete all the old results in the results folder
delete_old_runs=false


#Paths to declare
downloaded_matrix_path="/scratch/eschreib/matrix_market_matrices/"
generated_matrix_path="/scratch/eschreib/matrices/Dataset_generated_matrices/"
config_dir="/scratch/eschreib/matrices/matrix_market_config_files/"
results_dir="/scratch/eschreib/correct_SDDMM_results/results_matrixmarket_different_k_100k/"
run_script_path="/users/eschreib/marcus/DPHPC23-SmokyRhino/benchmark_pipeline/Pipeline/run_on_cluster_test.sh"

# Set matrix type on same index for the corresponding function
functions=("better_naive_CSR_SDDMM_GPU" "coo_opt_vectorization_SDDMM_GPU")
matrixType=("CSRMatrix" "COOMatrix")

# ~10k x 10k matrices
#matrices=( "nemeth13/nemeth13_fixed_0005" "c-41/c-41_0001_fixed_0001"  "bloweybq/bloweybq_00005_fixed_00005" "bcsstm38/bcsstm38_000016_fixed_000016")
#matrix_shapes=("9506x9506" "9769x9769" "10001x10001" "8032x8032")
#sparsities=("0.005" "0.001" "0.0005" "0.00016")

# ~24k x 24k matrices
#matrices=("TSOPF_RS_b2052_c1/TSOPF_RS_b2052_c1_001" "smt/smt_fixed_00056" "crystm03/crystm03_000096_fixed_000096" "g7jac080/g7jac080_000046_fixed_000046" "qpband/qpband_00001_fixed_00001")


#matrices=("nemeth20/nemeth20_001_fixed_001" "nemeth13/nemeth13_fixed_0005" "c-41/c-41_0001_fixed_0001" "bloweybq/bloweybq_00005_fixed_00005"  "bcsstm38/bcsstm38_000016_fixed_000016")
#matrix_shapes=( "25626x25626" "25710x25710" "24696x24696" "23670x23670" "20000x20000")
#sparsities=( "0.01" "0.0056" "0.00096" "0.00046" "0.0001")

# 100kx 100k matrices
matrices=("m_t1/m_t1_fixed_0001" "Si34H36/Si34H36_fixed_00005" "ASIC_100k/ASIC_100k_fixed_0000095") 
matrix_shapes=("97578x97578" "97569x97569" "99340x99340")
sparsities=("0.001" "0.0005" "0.000095")


ks=(50 100 500 1000)


#Generate config files and load them in a common run script

if "$generate_configs"
then
    #Run on cluster path delete old version
     > "$run_script_path"
    for k in "${ks[@]}"
    do
        for func_i in "${!functions[@]}"
        do
        for matrix_i in "${!matrices[@]}"
        do  

            #Get matrix dimension
            nm_dim=(${matrix_shapes[$matrix_i]//"x"/ })
            
            #Define config, result path
            config_path=$config_dir"config_"${functions[$func_i]}"_"${matrix_shapes[$matrix_i]}"_k"$k".txt"
            result_path=$results_dir"${functions[$func_i]}"/"${matrix_shapes[$matrix_i]}"

            # Clear existing config file if it exists
            > "$config_path"

            if [ ! -d "$result_path" ]
            then 
            mkdir -p "$result_path"
            fi

            if "$delete_old_runs"
            then
            rm $result_path"/"*
            fi 

            #Get sparsity
            sparse_nbr="${sparsities[$matrix_i]//.}"

            #Get downloaded matrix
            sparse_matrix_path=$downloaded_matrix_path${matrices[$matrix_i]}".mtx"

            #Get path to matrix A and B (square matrices)
            #A_matrix_path=$generated_matrix_path${matrix_shapes[$matrix_i]}"/n_"${nm_dim[0]}"_m_"${nm_dim[1]}"_sparsity_1"
            #B_matrix_path=$generated_matrix_path${matrix_shapes[$matrix_i]}"/n_"${nm_dim[0]}"_m_"${nm_dim[1]}"_sparsity_1"


            #With different k
            A_matrix_path=$generated_matrix_path${matrix_shapes[$matrix_i]}"/n_"${nm_dim[0]}"_m_"${k}"_sparsity_1"
            B_matrix_path=$generated_matrix_path${matrix_shapes[$matrix_i]}"/n_"${k}"_m_"${nm_dim[1]}"_sparsity_1"

            echo "${functions[$func_i]}", "${matrixType[$func_i]}", "$A_matrix_path", "$B_matrix_path", "$sparse_matrix_path" >> "$config_path"
            #Write to the run on cluster filem
            echo "/users/eschreib/marcus/DPHPC23-SmokyRhino/benchmark_pipeline/Pipeline/benchmark_run.sh "$config_path" "$result_path >> "$run_script_path"
        done

        
        

        done

    done
fi