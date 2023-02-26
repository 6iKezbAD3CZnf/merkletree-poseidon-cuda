all: main.cu test.cu poseidon.cu
	nvcc $^ -o main
