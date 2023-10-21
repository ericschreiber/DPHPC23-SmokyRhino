Execute generateMatrices.sh to generate test matrices. Might need to execute "chmod +x generateMatrices.sh" first. 

The parameters that will be iterated over when generating the different matrices can be modified in generateMatrices.sh.

Each matrix is being saved to it's own file since I think we will probably generate very large matrices and then I think we do not want to have multiple of those in the same file (but maybe that is not a good idea and needs to be changed). 

Resulting matrices will be saved intp generated_matrices. If this folder already contains files these files will not be deleted.