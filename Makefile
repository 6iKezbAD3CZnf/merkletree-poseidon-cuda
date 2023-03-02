BIN = main
OBJ = field.o poseidon.o merkle_tree.o main.o
CC = nvcc

${BIN}: ${OBJ}
	${CC} $^ -o ${BIN}

main.o: main.cu
	${CC} -dc $^

%.o: %.cu %.cuh uint128_t.cuh
	${CC} -dc $<

clean:
	rm -f *.o ${BIN}
