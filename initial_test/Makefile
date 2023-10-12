CUFLAGS = -O3 -std=c++11 --compiler-options "-Wall -Wextra" -I../../common/inc -I../include \
					-use_fast_math -lcublas -gencode=arch=compute_86,code=sm_86 \
					-lineinfo
.PHONY: all clean

all: main.exe
	@true

ptx: my_dgemm.ptx sampling.ptx
	@true

clean:
	rm -f *.o *.ptx *.exe

%.o: %.cu
	nvcc $(CUFLAGS) $< -c -o $@

%.exe: %.o main.o
	nvcc $(CUFLAGS) $^ -o $@

%.ptx: %.cu
	nvcc $(CUFLAGS) -ptx $^ -o $@
