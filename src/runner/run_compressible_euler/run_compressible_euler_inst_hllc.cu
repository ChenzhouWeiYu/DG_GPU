#include "runner/run_compressible_euler/run_compressible_euler_impl.h"

#define Explicit_For_Flux(NAME,Order) \
template void RunCompressibleEuler<Order,NAME##53C,true>(uInt N, FilesystemManager& fsm, LoggerSystem& logger, uInt);\
template void RunCompressibleEuler<Order,NAME##75C,true>(uInt N, FilesystemManager& fsm, LoggerSystem& logger, uInt);\
template void RunCompressibleEuler<Order,NAME##53C,false>(uInt N, FilesystemManager& fsm, LoggerSystem& logger, uInt);\
template void RunCompressibleEuler<Order,NAME##75C,false>(uInt N, FilesystemManager& fsm, LoggerSystem& logger, uInt);


Explicit_For_Flux(HLLC,1)
Explicit_For_Flux(HLLC,2)
Explicit_For_Flux(HLLC,3)
// Explicit_For_Flux(HLLC,4)
// Explicit_For_Flux(HLLC,5)

#undef Explicit_For_Flux