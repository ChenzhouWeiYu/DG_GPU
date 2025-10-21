#include "runner/run_compressible_euler/run_compressible_euler_impl.cuh"

#define Explicit_For_Flux(NAME,Order) \
template void RunCompressibleEuler<Order,NAME##53C,true>(uInt N, FilesystemManager& fsm, LoggerSystem& logger, uInt);\
template void RunCompressibleEuler<Order,NAME##75C,true>(uInt N, FilesystemManager& fsm, LoggerSystem& logger, uInt);\
template void RunCompressibleEuler<Order,NAME##53C,false>(uInt N, FilesystemManager& fsm, LoggerSystem& logger, uInt);\
template void RunCompressibleEuler<Order,NAME##75C,false>(uInt N, FilesystemManager& fsm, LoggerSystem& logger, uInt);


Explicit_For_Flux(HLLEM,1)
Explicit_For_Flux(HLLEM,2)
Explicit_For_Flux(HLLEM,3)
// Explicit_For_Flux(HLLEM,4)
// Explicit_For_Flux(HLLEM,5)

#undef Explicit_For_Flux