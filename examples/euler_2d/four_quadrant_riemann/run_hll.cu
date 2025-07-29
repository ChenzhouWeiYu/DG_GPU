#include "run_impl.h"

#define Explicit_For_Flux(NAME,Order) \
template void Run<Order,NAME##53C>(uInt N, FilesystemManager& fsm, LoggerSystem& logger);\
template void Run<Order,NAME##75C>(uInt N, FilesystemManager& fsm, LoggerSystem& logger);

Explicit_For_Flux(HLL,1)
Explicit_For_Flux(HLL,2)
Explicit_For_Flux(HLL,3)
// Explicit_For_Flux(HLL,4)
// Explicit_For_Flux(HLL,5)