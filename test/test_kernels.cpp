#include <kernels.hpp>
#include "assert.h"
#include <iostream>
int main()
{
    std::cout << "Test output" << std::endl;
    assert(simpleKernel(5) == 6);
}

