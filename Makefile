CC=nvcc

all: uint128_t.o field.o poseidon.o main.o
	${CC} $^ -o main

%.o: %.cu
	${CC} -dc $^

clean:
	rm -f *.o main
