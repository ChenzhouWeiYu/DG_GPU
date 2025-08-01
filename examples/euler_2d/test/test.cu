#include <stdio.h>

template<int Order>
__global__ void test_kernel() {
    printf("Running kernel of order %d\n", Order);
}

template<int Order>
void launch() {
    test_kernel<Order><<<1, 1>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

template void launch<2>();


int main(){
    launch<1>();
    return 0;
}
