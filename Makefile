CC=nvcc

all: main.o poseidon.o
	nvcc $^ -o main

main.o: main.cu field.cuh
	nvcc -dc main.cu

poseidon.o: poseidon.cu
	nvcc -dc $^

clean:
	rm -f *.o main
