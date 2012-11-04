#include "mandelbrot.h"

struct mandelbrot_param_offset* popStack();

void pushStack(struct mandelbrot_param* parametersIn, int heightOffset);

struct mandelbrot_param_offset
{
	struct mandelbrot_param theParam;
	int heightOffset;
};

void initStack();
void clearStack();
