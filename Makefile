BIN = main
OBJ = uint128_t.o field.o poseidon.o merkle_tree.o main.o
CC = nvcc

${BIN}: ${OBJ}
	${CC} $^ -o ${BIN}

main.o: main.cu
	${CC} -dc $^

%.o: %.cu %.cuh
	${CC} -dc $<

clean:
	rm -f *.o ${BIN}
