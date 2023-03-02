CC=nvcc

all: uint128_t.o field.o poseidon.o merkle_tree.o main.o
	${CC} $^ -o main

%.o: %.cu
	${CC} -dc $^

clean:
	rm -f *.o main
