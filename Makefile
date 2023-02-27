CC=nvcc

all: main.o poseidon.o
	nvcc $^ -o main

main.o: main.cu
	nvcc -dc $^

poseidon.o: poseidon.cu
	nvcc -dc $^
