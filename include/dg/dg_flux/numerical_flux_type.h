// include/DG/Flux/NumericalFluxType.h
#pragma once
#include "base/type.h"
#include "matrix/matrix.h"

enum class NumericalFluxType : uint8_t {LF,LaxFriedrichs,Roe,HLL,HLLC,RHLLC,HLLEM};
#define FOREACH_FLUX_TYPE(F,SecondParams) \
    F(LF,SecondParams) \
    F(LaxFriedrichs,SecondParams) \
    F(Roe,SecondParams) \
    F(HLL,SecondParams) \
    F(HLLC,SecondParams) \
    F(RHLLC,SecondParams) \
    F(HLLEM,SecondParams)