#include "run_impl.h"

#define Explicit_For_Flux(NAME,Order) \
template void Run<Order,NAME##53C>(uInt N, FilesystemManager& fsm, LoggerSystem& logger);\
template void Run<Order,NAME##75C>(uInt N, FilesystemManager& fsm, LoggerSystem& logger);

Explicit_For_Flux(HLLEM,1)
Explicit_For_Flux(HLLEM,2)
Explicit_For_Flux(HLLEM,3)
// Explicit_For_Flux(HLLEM,4)
// Explicit_For_Flux(HLLEM,5)