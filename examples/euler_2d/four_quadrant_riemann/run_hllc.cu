#include "run_impl.h"

#define Explicit_For_Flux(NAME,Order) \
template void Run<Order,NAME##53C>(uInt N, FilesystemManager& fsm, LoggerSystem& logger);\
template void Run<Order,NAME##75C>(uInt N, FilesystemManager& fsm, LoggerSystem& logger);

Explicit_For_Flux(HLLC,1)
Explicit_For_Flux(HLLC,2)
Explicit_For_Flux(HLLC,3)
// Explicit_For_Flux(HLLC,4)
// Explicit_For_Flux(HLLC,5)