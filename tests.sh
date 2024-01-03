#!/bin/bash


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

TESTFILES="
"
GPU_TESTFILES="
${SCRIPT_DIR}/build/SDDMMlib/tests/SDDMM/test_tiled_tiles
${SCRIPT_DIR}/build/SDDMMlib/tests/SDDMM/test_naive_CPU_SDDMM 
${SCRIPT_DIR}/build/SDDMMlib/tests/SDDMM/test_naive_coo_SDDMM_GPU 
${SCRIPT_DIR}/build/SDDMMlib/tests/SDDMM/test_cache_coo_SDDMM_GPU
${SCRIPT_DIR}/build/SDDMMlib/tests/SDDMM/test_coo_opt_vectorization_SDDMM_GPU
${SCRIPT_DIR}/build/SDDMMlib/tests/SDDMM/test_coo_opt_loop_unrolled_SDDMM_GPU
${SCRIPT_DIR}/build/SDDMMlib/tests/SDDMM/test_coo_tiling_naive_gpu
${SCRIPT_DIR}/build/SDDMMlib/tests/SDDMM/semi_naive_CSR_GPU/test_semi_naive_CSR_SDDMM_GPU
${SCRIPT_DIR}/build/SDDMMlib/tests/SDDMM/naive_dense_dense_gpu/test_compile
${SCRIPT_DIR}/build/SDDMMlib/tests/SDDMM/test_cusparse_baseline
"
# ${SCRIPT_DIR}/build/SDDMMlib/tests/SDDMM/test_naive_csr_via_coo_SDDMM_GPU // is not working

# Detect if GPU is available
if command -v nvidia-smi &> /dev/null
then
    echo "GPU detected"
    TESTFILES="${TESTFILES} ${GPU_TESTFILES}"
else
    echo "GPU not detected"
fi

NUM_FAILS=0

for testfile in ${TESTFILES}
do
    echo "Running ${testfile}"
    ${testfile}
    if [ $? -ne 0 ]; then
        NUM_FAILS=$((NUM_FAILS+1))
        echo "Test ${testfile} failed"
        exit 1
    fi
done

 echo "************************ RESULT ************************"
if [ ${NUM_FAILS} -eq 0 ]; then
    echo "All tests passed"
else
    echo "${NUM_FAILS} tests failed"
fi
 echo "************************ RESULT ************************"