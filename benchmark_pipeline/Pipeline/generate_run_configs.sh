#!/bin/bash

# Creates the config files for a run

# Set what to run
# Will generate matrices according to the specifications and folder structure
generate_matricies=false
# Will generate teh config files to set directory, and create run_on_cluster_test.sh file
generate_configs=true
# If ture it will delete all the old results in the results folder
delete_old_runs=false
# Running the matrices of the downloaded format (This reqires a bit different running scheme)
generate_downloaded_sizes_configs=false


#Paths to declare
generated_matrix_path="/scratch/eschreib/matrices/Dataset_generated_matrices/"
config_dir="/scratch/eschreib/matrices/Dataset_generated_matrices/config_files/"
results_dir="/scratch/eschreib/correct_SDDMM_results/results_different_k_1k_10k_20k/"
run_script_path="/users/eschreib/marcus/DPHPC23-SmokyRhino/benchmark_pipeline/Pipeline/run_on_cluster_test.sh"

# Set matrix type on same index for the corresponding function
functions=("better_naive_CSR_SDDMM_GPU" "coo_opt_vectorization_SDDMM_GPU")
matrixType=("CSRMatrix" "COOMatrix")

# For each function it will loop over the different sparsities and matrix sizes
sparsities=("0.01" "0.005" "0.001" "0.0005" "0.0001" "0.00001")
matrix_shapes=("1000x1000" "10000x10000" "20000x20000")

# The k denotes the inner dimension of A and B matrix, which the script loops over (for running square change lower in this script)
ks=(50 100 500 1000)


if "$generate_matricies"
then 
    #Block to generate all the matrices
    for matrix_shape in "${matrix_shapes[@]}"
    do 
        #Matrix dimensions
        nm_dim=(${matrix_shape//"x"/ })

        # Create the folder if it doesnt exist
        if [ ! -d "$generated_matrix_path$matrix_shape" ]
        then 
        mkdir "$generated_matrix_path$matrix_shape"
        fi

        # Create the matrix
        for sparsity in "${sparsities[@]}"
        do
            #Generate sparse matrix
            #bash /users/eschreib/marcus/DPHPC23-SmokyRhino/benchmark_pipeline/Pipeline/generateMatrix.sh "$generated_matrix_path$matrix_shape" "$matrix_shape" "$sparsity"
            echo "fill loop"
        done

        #Generate dense matrix as A
        bash /users/eschreib/marcus/DPHPC23-SmokyRhino/benchmark_pipeline/Pipeline/generateMatrix.sh "$generated_matrix_path$matrix_shape" "${nm_dim[1]}x${k}" "1"
        # Generate a square matrix
        #bash /users/eschreib/marcus/DPHPC23-SmokyRhino/benchmark_pipeline/Pipeline/generateMatrix.sh "$generated_matrix_path$matrix_shape" "${nm_dim[0]}x${nm_dim[1]}" "1"
        #Generate dense matrix as B
        bash /users/eschreib/marcus/DPHPC23-SmokyRhino/benchmark_pipeline/Pipeline/generateMatrix.sh "$generated_matrix_path$matrix_shape" "${k}x${nm_dim[1]}" "1"
            


    done
fi 


#Generate config files and load them in a common run script
if "$generate_configs"
then
    #Define run on cluster path, and delete old version
    > "$run_script_path"
    for k in "${ks[@]}"
    do

        for i in "${!functions[@]}"
        do
        #Block to go through the matrices
        for matrix_shape in "${matrix_shapes[@]}"
        do 
            nm_dim=(${matrix_shape//"x"/ })

            #Define config, result path
            config_path=$config_dir"config_"${functions[$i]}"_"$matrix_shape"_k"$k".txt"
            result_path=$results_dir"${functions[$i]}"/"$matrix_shape"

            if [ ! -d "$result_path" ]
            then 
            mkdir -p "$result_path"
            fi

            if "$delete_old_runs"
            then
            rm $result_path"/"*
            fi 

            
            # Clear existing config file if it exists
            > "$config_path"

            # Go through the sparcities
            for sparsity in "${sparsities[@]}"
            do
                # Define the path to the sparse matrix
                sparse_nbr="${sparsity//.}"
    
                sparse_matrix_path="$generated_matrix_path$matrix_shape"/n_"${nm_dim[0]}"_m_"${nm_dim[1]}"_sparsity_$sparse_nbr".mtx"

                # For a given k
                A_matrix_path=$generated_matrix_path$matrix_shape"/n_"${nm_dim[0]}"_m_"$k"_sparsity_1"
                B_matrix_path=$generated_matrix_path$matrix_shape"/n_"$k"_m_"${nm_dim[1]}"_sparsity_1"
                
                # For square matrices
                #A_matrix_path=$generated_matrix_path$matrix_shape"/n_"${nm_dim[0]}"_m_"${nm_dim[1]}"_sparsity_1"
                #B_matrix_path=$generated_matrix_path$matrix_shape"/n_"${nm_dim[0]}"_m_"${nm_dim[1]}"_sparsity_1"

                # Create config file for each matrix
                echo "${functions[$i]}", "${matrixType[$i]}", "$A_matrix_path", "$B_matrix_path", "$sparse_matrix_path" >> "$config_path"
                
            done
            #Write to the run on cluster filem
            echo "/users/eschreib/marcus/DPHPC23-SmokyRhino/benchmark_pipeline/Pipeline/benchmark_run.sh "$config_path" "$result_path >> "$run_script_path"
        done
        done
    done
fi

#Generate config files and load them in a common run script
if "$generate_downloaded_sizes_configs"
then

    #Define run on cluster path, and delete old version
    > "$run_script_path"

    for i in "${!functions[@]}"
    do
    #Block to go through the matrices
        for matrix_i in "${!matrix_shapes[@]}"
        do 
            matrix_shape="${matrix_shapes[$matrix_i]}"
            sparsity="${sparsities[$matrix_i]}"

            nm_dim=(${matrix_shape//"x"/ })

            #Define config, result path
            config_path=$config_dir"config_"${functions[$i]}"_"$matrix_shape"_k"$k".txt"
            result_path=$results_dir"${functions[$i]}"/"$matrix_shape"

            if [ ! -d "$result_path" ]
            then 
            mkdir -p "$result_path"
            fi

            if "$delete_old_runs"
            then
            rm $result_path"/"*
            fi 

            
            # Clear existing config file if it exists
            > "$config_path"

            
            # Define the path to the sparse matrix
            sparse_nbr="${sparsity//.}"
    
            sparse_matrix_path="$generated_matrix_path$matrix_shape"/n_"${nm_dim[0]}"_m_"${nm_dim[1]}"_sparsity_$sparse_nbr".mtx"

            A_matrix_path=$generated_matrix_path$matrix_shape"/n_"${nm_dim[0]}"_m_"${nm_dim[1]}"_sparsity_1"
            B_matrix_path=$generated_matrix_path$matrix_shape"/n_"${nm_dim[0]}"_m_"${nm_dim[1]}"_sparsity_1"

                # Create config file for each matrix
            echo "${functions[$i]}", "${matrixType[$i]}", "$A_matrix_path", "$B_matrix_path", "$sparse_matrix_path" >> "$config_path"
                
            
            #Write to the run on cluster filem
            echo "/users/eschreib/marcus/DPHPC23-SmokyRhino/benchmark_pipeline/Pipeline/benchmark_run.sh "$config_path" "$result_path >> "$run_script_path"
        done
    done
fi