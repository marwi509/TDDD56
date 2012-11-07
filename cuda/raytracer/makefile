all:
	nvcc -o main main.cu -lcudart -lglut -lGL
	nvcc -o cpumain cpumain.cu -lcudart -lglut -lGL
cpu:
	nvcc -o cpumain cpumain.cu -lcudart -lglut -lGL
gpu:
	nvcc -o main main.cu -lcudart -lglut -lGL
