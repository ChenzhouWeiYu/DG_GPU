// include/DG/DG_Basis/DG_Basis_Func_2D.h
#pragma once
#include "base/type.h"
#include "dg/dg_basis/dg_basis.h"


// Basis 0
template<>
struct DGBasis2D<0> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            0,
            0
        };
    }
    static constexpr uInt Order = 0;
};

// Basis 1
template<>
struct DGBasis2D<1> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return 3*x - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            3,
            0
        };
    }
    static constexpr uInt Order = 1;
};

// Basis 2
template<>
struct DGBasis2D<2> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return x + 2*y - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            1,
            2
        };
    }
    static constexpr uInt Order = 1;
};

// Basis 3
template<>
struct DGBasis2D<3> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return 10*ipow<2>(x) - 8*x + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            20*x - 8,
            0
        };
    }
    static constexpr uInt Order = 1;
};

// Basis 4
template<>
struct DGBasis2D<4> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (5*x - 1)*(x + 2*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            10*x + 10*y - 6,
            10*x - 2
        };
    }
    static constexpr uInt Order = 2;
};

// Basis 5
template<>
struct DGBasis2D<5> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return ipow<2>(x) + x*(6*y - 2) + 6*ipow<2>(y) - 6*y + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            2*x + 6*y - 2,
            6*x + 12*y - 6
        };
    }
    static constexpr uInt Order = 2;
};

// Basis 6
template<>
struct DGBasis2D<6> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return 35*ipow<3>(x) - 45*ipow<2>(x) + 15*x - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            105*ipow<2>(x) - 90*x + 15,
            0
        };
    }
    static constexpr uInt Order = 2;
};

// Basis 7
template<>
struct DGBasis2D<7> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (x + 2*y - 1)*(21*ipow<2>(x) - 12*x + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            21*ipow<2>(x) - 12*x + (42*x - 12)*(x + 2*y - 1) + 1,
            42*ipow<2>(x) - 24*x + 2
        };
    }
    static constexpr uInt Order = 2;
};

// Basis 8
template<>
struct DGBasis2D<8> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (7*x - 1)*(ipow<2>(x) + x*(6*y - 2) + 6*ipow<2>(y) - 6*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            7*ipow<2>(x) + 7*x*(6*y - 2) + 42*ipow<2>(y) - 42*y + (7*x - 1)*(2*x + 6*y - 2) + 7,
            (7*x - 1)*(6*x + 12*y - 6)
        };
    }
    static constexpr uInt Order = 2;
};

// Basis 9
template<>
struct DGBasis2D<9> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return ipow<3>(x) + 3*ipow<2>(x)*(4*y - 1) + 3*x*(10*ipow<2>(y) - 8*y + 1) + 20*ipow<3>(y) - 30*ipow<2>(y) + 12*y - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            3*ipow<2>(x) + 6*x*(4*y - 1) + 30*ipow<2>(y) - 24*y + 3,
            12*ipow<2>(x) + 3*x*(20*y - 8) + 60*ipow<2>(y) - 60*y + 12
        };
    }
    static constexpr uInt Order = 2;
};

// Basis 10
template<>
struct DGBasis2D<10> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return 126*ipow<4>(x) - 224*ipow<3>(x) + 126*ipow<2>(x) - 24*x + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            504*ipow<3>(x) - 672*ipow<2>(x) + 252*x - 24,
            0
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 11
template<>
struct DGBasis2D<11> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (x + 2*y - 1)*(84*ipow<3>(x) - 84*ipow<2>(x) + 21*x - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            84*ipow<3>(x) - 84*ipow<2>(x) + 21*x + (x + 2*y - 1)*(252*ipow<2>(x) - 168*x + 21) - 1,
            168*ipow<3>(x) - 168*ipow<2>(x) + 42*x - 2
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 12
template<>
struct DGBasis2D<12> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (36*ipow<2>(x) - 16*x + 1)*(ipow<2>(x) + x*(6*y - 2) + 6*ipow<2>(y) - 6*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (72*x - 16)*(ipow<2>(x) + x*(6*y - 2) + 6*ipow<2>(y) - 6*y + 1) + (2*x + 6*y - 2)*(36*ipow<2>(x) - 16*x + 1),
            (6*x + 12*y - 6)*(36*ipow<2>(x) - 16*x + 1)
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 13
template<>
struct DGBasis2D<13> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (9*x - 1)*(ipow<3>(x) + 3*ipow<2>(x)*(4*y - 1) + 3*x*(10*ipow<2>(y) - 8*y + 1) + 20*ipow<3>(y) - 30*ipow<2>(y) + 12*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            9*ipow<3>(x) + 27*ipow<2>(x)*(4*y - 1) + 27*x*(10*ipow<2>(y) - 8*y + 1) + 180*ipow<3>(y) - 270*ipow<2>(y) + 108*y + (9*x - 1)*(3*ipow<2>(x) + 6*x*(4*y - 1) + 30*ipow<2>(y) - 24*y + 3) - 9,
            (9*x - 1)*(12*ipow<2>(x) + 3*x*(20*y - 8) + 60*ipow<2>(y) - 60*y + 12)
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 14
template<>
struct DGBasis2D<14> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return ipow<4>(x) + 4*ipow<3>(x)*(5*y - 1) + ipow<2>(x)*(90*ipow<2>(y) - 60*y + 6) + 4*x*(35*ipow<3>(y) - 45*ipow<2>(y) + 15*y - 1) + 70*ipow<4>(y) - 140*ipow<3>(y) + 90*ipow<2>(y) - 20*y + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            4*ipow<3>(x) + 12*ipow<2>(x)*(5*y - 1) + 2*x*(90*ipow<2>(y) - 60*y + 6) + 140*ipow<3>(y) - 180*ipow<2>(y) + 60*y - 4,
            20*ipow<3>(x) + ipow<2>(x)*(180*y - 60) + 4*x*(105*ipow<2>(y) - 90*y + 15) + 280*ipow<3>(y) - 420*ipow<2>(y) + 180*y - 20
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 15
template<>
struct DGBasis2D<15> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return 462*ipow<5>(x) - 1050*ipow<4>(x) + 840*ipow<3>(x) - 280*ipow<2>(x) + 35*x - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            2310*ipow<4>(x) - 4200*ipow<3>(x) + 2520*ipow<2>(x) - 560*x + 35,
            0
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 16
template<>
struct DGBasis2D<16> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (x + 2*y - 1)*(330*ipow<4>(x) - 480*ipow<3>(x) + 216*ipow<2>(x) - 32*x + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            330*ipow<4>(x) - 480*ipow<3>(x) + 216*ipow<2>(x) - 32*x + (x + 2*y - 1)*(1320*ipow<3>(x) - 1440*ipow<2>(x) + 432*x - 32) + 1,
            660*ipow<4>(x) - 960*ipow<3>(x) + 432*ipow<2>(x) - 64*x + 2
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 17
template<>
struct DGBasis2D<17> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (165*ipow<3>(x) - 135*ipow<2>(x) + 27*x - 1)*(ipow<2>(x) + x*(6*y - 2) + 6*ipow<2>(y) - 6*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (2*x + 6*y - 2)*(165*ipow<3>(x) - 135*ipow<2>(x) + 27*x - 1) + (495*ipow<2>(x) - 270*x + 27)*(ipow<2>(x) + x*(6*y - 2) + 6*ipow<2>(y) - 6*y + 1),
            (6*x + 12*y - 6)*(165*ipow<3>(x) - 135*ipow<2>(x) + 27*x - 1)
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 18
template<>
struct DGBasis2D<18> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (55*ipow<2>(x) - 20*x + 1)*(ipow<3>(x) + 3*ipow<2>(x)*(4*y - 1) + 3*x*(10*ipow<2>(y) - 8*y + 1) + 20*ipow<3>(y) - 30*ipow<2>(y) + 12*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (110*x - 20)*(ipow<3>(x) + 3*ipow<2>(x)*(4*y - 1) + 3*x*(10*ipow<2>(y) - 8*y + 1) + 20*ipow<3>(y) - 30*ipow<2>(y) + 12*y - 1) + (55*ipow<2>(x) - 20*x + 1)*(3*ipow<2>(x) + 6*x*(4*y - 1) + 30*ipow<2>(y) - 24*y + 3),
            (55*ipow<2>(x) - 20*x + 1)*(12*ipow<2>(x) + 3*x*(20*y - 8) + 60*ipow<2>(y) - 60*y + 12)
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 19
template<>
struct DGBasis2D<19> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (11*x - 1)*(ipow<4>(x) + 4*ipow<3>(x)*(5*y - 1) + ipow<2>(x)*(90*ipow<2>(y) - 60*y + 6) + 4*x*(35*ipow<3>(y) - 45*ipow<2>(y) + 15*y - 1) + 70*ipow<4>(y) - 140*ipow<3>(y) + 90*ipow<2>(y) - 20*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            11*ipow<4>(x) + 44*ipow<3>(x)*(5*y - 1) + 11*ipow<2>(x)*(90*ipow<2>(y) - 60*y + 6) + 44*x*(35*ipow<3>(y) - 45*ipow<2>(y) + 15*y - 1) + 770*ipow<4>(y) - 1540*ipow<3>(y) + 990*ipow<2>(y) - 220*y + (11*x - 1)*(4*ipow<3>(x) + 12*ipow<2>(x)*(5*y - 1) + 2*x*(90*ipow<2>(y) - 60*y + 6) + 140*ipow<3>(y) - 180*ipow<2>(y) + 60*y - 4) + 11,
            (11*x - 1)*(20*ipow<3>(x) + ipow<2>(x)*(180*y - 60) + 4*x*(105*ipow<2>(y) - 90*y + 15) + 280*ipow<3>(y) - 420*ipow<2>(y) + 180*y - 20)
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 20
template<>
struct DGBasis2D<20> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return ipow<5>(x) + 5*ipow<4>(x)*(6*y - 1) + 10*ipow<3>(x)*(21*ipow<2>(y) - 12*y + 1) + 10*ipow<2>(x)*(56*ipow<3>(y) - 63*ipow<2>(y) + 18*y - 1) + 5*x*(126*ipow<4>(y) - 224*ipow<3>(y) + 126*ipow<2>(y) - 24*y + 1) + 252*ipow<5>(y) - 630*ipow<4>(y) + 560*ipow<3>(y) - 210*ipow<2>(y) + 30*y - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            5*ipow<4>(x) + 20*ipow<3>(x)*(6*y - 1) + 30*ipow<2>(x)*(21*ipow<2>(y) - 12*y + 1) + 20*x*(56*ipow<3>(y) - 63*ipow<2>(y) + 18*y - 1) + 630*ipow<4>(y) - 1120*ipow<3>(y) + 630*ipow<2>(y) - 120*y + 5,
            30*ipow<4>(x) + 10*ipow<3>(x)*(42*y - 12) + 10*ipow<2>(x)*(168*ipow<2>(y) - 126*y + 18) + 5*x*(504*ipow<3>(y) - 672*ipow<2>(y) + 252*y - 24) + 1260*ipow<4>(y) - 2520*ipow<3>(y) + 1680*ipow<2>(y) - 420*y + 30
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 21
template<>
struct DGBasis2D<21> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return 1716*ipow<6>(x) - 4752*ipow<5>(x) + 4950*ipow<4>(x) - 2400*ipow<3>(x) + 540*ipow<2>(x) - 48*x + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            10296*ipow<5>(x) - 23760*ipow<4>(x) + 19800*ipow<3>(x) - 7200*ipow<2>(x) + 1080*x - 48,
            0
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 22
template<>
struct DGBasis2D<22> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (x + 2*y - 1)*(1287*ipow<5>(x) - 2475*ipow<4>(x) + 1650*ipow<3>(x) - 450*ipow<2>(x) + 45*x - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            1287*ipow<5>(x) - 2475*ipow<4>(x) + 1650*ipow<3>(x) - 450*ipow<2>(x) + 45*x + (x + 2*y - 1)*(6435*ipow<4>(x) - 9900*ipow<3>(x) + 4950*ipow<2>(x) - 900*x + 45) - 1,
            2574*ipow<5>(x) - 4950*ipow<4>(x) + 3300*ipow<3>(x) - 900*ipow<2>(x) + 90*x - 2
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 23
template<>
struct DGBasis2D<23> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (ipow<2>(x) + x*(6*y - 2) + 6*ipow<2>(y) - 6*y + 1)*(715*ipow<4>(x) - 880*ipow<3>(x) + 330*ipow<2>(x) - 40*x + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (2*x + 6*y - 2)*(715*ipow<4>(x) - 880*ipow<3>(x) + 330*ipow<2>(x) - 40*x + 1) + (2860*ipow<3>(x) - 2640*ipow<2>(x) + 660*x - 40)*(ipow<2>(x) + x*(6*y - 2) + 6*ipow<2>(y) - 6*y + 1),
            (6*x + 12*y - 6)*(715*ipow<4>(x) - 880*ipow<3>(x) + 330*ipow<2>(x) - 40*x + 1)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 24
template<>
struct DGBasis2D<24> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (286*ipow<3>(x) - 198*ipow<2>(x) + 33*x - 1)*(ipow<3>(x) + 3*ipow<2>(x)*(4*y - 1) + 3*x*(10*ipow<2>(y) - 8*y + 1) + 20*ipow<3>(y) - 30*ipow<2>(y) + 12*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (858*ipow<2>(x) - 396*x + 33)*(ipow<3>(x) + 3*ipow<2>(x)*(4*y - 1) + 3*x*(10*ipow<2>(y) - 8*y + 1) + 20*ipow<3>(y) - 30*ipow<2>(y) + 12*y - 1) + (286*ipow<3>(x) - 198*ipow<2>(x) + 33*x - 1)*(3*ipow<2>(x) + 6*x*(4*y - 1) + 30*ipow<2>(y) - 24*y + 3),
            (286*ipow<3>(x) - 198*ipow<2>(x) + 33*x - 1)*(12*ipow<2>(x) + 3*x*(20*y - 8) + 60*ipow<2>(y) - 60*y + 12)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 25
template<>
struct DGBasis2D<25> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (78*ipow<2>(x) - 24*x + 1)*(ipow<4>(x) + 4*ipow<3>(x)*(5*y - 1) + ipow<2>(x)*(90*ipow<2>(y) - 60*y + 6) + 4*x*(35*ipow<3>(y) - 45*ipow<2>(y) + 15*y - 1) + 70*ipow<4>(y) - 140*ipow<3>(y) + 90*ipow<2>(y) - 20*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (156*x - 24)*(ipow<4>(x) + 4*ipow<3>(x)*(5*y - 1) + ipow<2>(x)*(90*ipow<2>(y) - 60*y + 6) + 4*x*(35*ipow<3>(y) - 45*ipow<2>(y) + 15*y - 1) + 70*ipow<4>(y) - 140*ipow<3>(y) + 90*ipow<2>(y) - 20*y + 1) + (78*ipow<2>(x) - 24*x + 1)*(4*ipow<3>(x) + 12*ipow<2>(x)*(5*y - 1) + 2*x*(90*ipow<2>(y) - 60*y + 6) + 140*ipow<3>(y) - 180*ipow<2>(y) + 60*y - 4),
            (78*ipow<2>(x) - 24*x + 1)*(20*ipow<3>(x) + ipow<2>(x)*(180*y - 60) + 4*x*(105*ipow<2>(y) - 90*y + 15) + 280*ipow<3>(y) - 420*ipow<2>(y) + 180*y - 20)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 26
template<>
struct DGBasis2D<26> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (13*x - 1)*(ipow<5>(x) + 5*ipow<4>(x)*(6*y - 1) + 10*ipow<3>(x)*(21*ipow<2>(y) - 12*y + 1) + 10*ipow<2>(x)*(56*ipow<3>(y) - 63*ipow<2>(y) + 18*y - 1) + 5*x*(126*ipow<4>(y) - 224*ipow<3>(y) + 126*ipow<2>(y) - 24*y + 1) + 252*ipow<5>(y) - 630*ipow<4>(y) + 560*ipow<3>(y) - 210*ipow<2>(y) + 30*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            13*ipow<5>(x) + 65*ipow<4>(x)*(6*y - 1) + 130*ipow<3>(x)*(21*ipow<2>(y) - 12*y + 1) + 130*ipow<2>(x)*(56*ipow<3>(y) - 63*ipow<2>(y) + 18*y - 1) + 65*x*(126*ipow<4>(y) - 224*ipow<3>(y) + 126*ipow<2>(y) - 24*y + 1) + 3276*ipow<5>(y) - 8190*ipow<4>(y) + 7280*ipow<3>(y) - 2730*ipow<2>(y) + 390*y + (13*x - 1)*(5*ipow<4>(x) + 20*ipow<3>(x)*(6*y - 1) + 30*ipow<2>(x)*(21*ipow<2>(y) - 12*y + 1) + 20*x*(56*ipow<3>(y) - 63*ipow<2>(y) + 18*y - 1) + 630*ipow<4>(y) - 1120*ipow<3>(y) + 630*ipow<2>(y) - 120*y + 5) - 13,
            (13*x - 1)*(30*ipow<4>(x) + 10*ipow<3>(x)*(42*y - 12) + 10*ipow<2>(x)*(168*ipow<2>(y) - 126*y + 18) + 5*x*(504*ipow<3>(y) - 672*ipow<2>(y) + 252*y - 24) + 1260*ipow<4>(y) - 2520*ipow<3>(y) + 1680*ipow<2>(y) - 420*y + 30)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 27
template<>
struct DGBasis2D<27> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return ipow<6>(x) + 6*ipow<5>(x)*(7*y - 1) + 15*ipow<4>(x)*(28*ipow<2>(y) - 14*y + 1) + 20*ipow<3>(x)*(84*ipow<3>(y) - 84*ipow<2>(y) + 21*y - 1) + 15*ipow<2>(x)*(210*ipow<4>(y) - 336*ipow<3>(y) + 168*ipow<2>(y) - 28*y + 1) + 6*x*(462*ipow<5>(y) - 1050*ipow<4>(y) + 840*ipow<3>(y) - 280*ipow<2>(y) + 35*y - 1) + 924*ipow<6>(y) - 2772*ipow<5>(y) + 3150*ipow<4>(y) - 1680*ipow<3>(y) + 420*ipow<2>(y) - 42*y + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            6*ipow<5>(x) + 30*ipow<4>(x)*(7*y - 1) + 60*ipow<3>(x)*(28*ipow<2>(y) - 14*y + 1) + 60*ipow<2>(x)*(84*ipow<3>(y) - 84*ipow<2>(y) + 21*y - 1) + 30*x*(210*ipow<4>(y) - 336*ipow<3>(y) + 168*ipow<2>(y) - 28*y + 1) + 2772*ipow<5>(y) - 6300*ipow<4>(y) + 5040*ipow<3>(y) - 1680*ipow<2>(y) + 210*y - 6,
            42*ipow<5>(x) + 15*ipow<4>(x)*(56*y - 14) + 20*ipow<3>(x)*(252*ipow<2>(y) - 168*y + 21) + 15*ipow<2>(x)*(840*ipow<3>(y) - 1008*ipow<2>(y) + 336*y - 28) + 6*x*(2310*ipow<4>(y) - 4200*ipow<3>(y) + 2520*ipow<2>(y) - 560*y + 35) + 5544*ipow<5>(y) - 13860*ipow<4>(y) + 12600*ipow<3>(y) - 5040*ipow<2>(y) + 840*y - 42
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 28
template<>
struct DGBasis2D<28> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return 6435*ipow<7>(x) - 21021*ipow<6>(x) + 27027*ipow<5>(x) - 17325*ipow<4>(x) + 5775*ipow<3>(x) - 945*ipow<2>(x) + 63*x - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            45045*ipow<6>(x) - 126126*ipow<5>(x) + 135135*ipow<4>(x) - 69300*ipow<3>(x) + 17325*ipow<2>(x) - 1890*x + 63,
            0
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 29
template<>
struct DGBasis2D<29> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (x + 2*y - 1)*(5005*ipow<6>(x) - 12012*ipow<5>(x) + 10725*ipow<4>(x) - 4400*ipow<3>(x) + 825*ipow<2>(x) - 60*x + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            5005*ipow<6>(x) - 12012*ipow<5>(x) + 10725*ipow<4>(x) - 4400*ipow<3>(x) + 825*ipow<2>(x) - 60*x + (x + 2*y - 1)*(30030*ipow<5>(x) - 60060*ipow<4>(x) + 42900*ipow<3>(x) - 13200*ipow<2>(x) + 1650*x - 60) + 1,
            10010*ipow<6>(x) - 24024*ipow<5>(x) + 21450*ipow<4>(x) - 8800*ipow<3>(x) + 1650*ipow<2>(x) - 120*x + 2
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 30
template<>
struct DGBasis2D<30> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (ipow<2>(x) + x*(6*y - 2) + 6*ipow<2>(y) - 6*y + 1)*(3003*ipow<5>(x) - 5005*ipow<4>(x) + 2860*ipow<3>(x) - 660*ipow<2>(x) + 55*x - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (2*x + 6*y - 2)*(3003*ipow<5>(x) - 5005*ipow<4>(x) + 2860*ipow<3>(x) - 660*ipow<2>(x) + 55*x - 1) + (ipow<2>(x) + x*(6*y - 2) + 6*ipow<2>(y) - 6*y + 1)*(15015*ipow<4>(x) - 20020*ipow<3>(x) + 8580*ipow<2>(x) - 1320*x + 55),
            (6*x + 12*y - 6)*(3003*ipow<5>(x) - 5005*ipow<4>(x) + 2860*ipow<3>(x) - 660*ipow<2>(x) + 55*x - 1)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 31
template<>
struct DGBasis2D<31> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (1365*ipow<4>(x) - 1456*ipow<3>(x) + 468*ipow<2>(x) - 48*x + 1)*(ipow<3>(x) + 3*ipow<2>(x)*(4*y - 1) + 3*x*(10*ipow<2>(y) - 8*y + 1) + 20*ipow<3>(y) - 30*ipow<2>(y) + 12*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (5460*ipow<3>(x) - 4368*ipow<2>(x) + 936*x - 48)*(ipow<3>(x) + 3*ipow<2>(x)*(4*y - 1) + 3*x*(10*ipow<2>(y) - 8*y + 1) + 20*ipow<3>(y) - 30*ipow<2>(y) + 12*y - 1) + (3*ipow<2>(x) + 6*x*(4*y - 1) + 30*ipow<2>(y) - 24*y + 3)*(1365*ipow<4>(x) - 1456*ipow<3>(x) + 468*ipow<2>(x) - 48*x + 1),
            (12*ipow<2>(x) + 3*x*(20*y - 8) + 60*ipow<2>(y) - 60*y + 12)*(1365*ipow<4>(x) - 1456*ipow<3>(x) + 468*ipow<2>(x) - 48*x + 1)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 32
template<>
struct DGBasis2D<32> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (455*ipow<3>(x) - 273*ipow<2>(x) + 39*x - 1)*(ipow<4>(x) + 4*ipow<3>(x)*(5*y - 1) + ipow<2>(x)*(90*ipow<2>(y) - 60*y + 6) + 4*x*(35*ipow<3>(y) - 45*ipow<2>(y) + 15*y - 1) + 70*ipow<4>(y) - 140*ipow<3>(y) + 90*ipow<2>(y) - 20*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (1365*ipow<2>(x) - 546*x + 39)*(ipow<4>(x) + 4*ipow<3>(x)*(5*y - 1) + ipow<2>(x)*(90*ipow<2>(y) - 60*y + 6) + 4*x*(35*ipow<3>(y) - 45*ipow<2>(y) + 15*y - 1) + 70*ipow<4>(y) - 140*ipow<3>(y) + 90*ipow<2>(y) - 20*y + 1) + (455*ipow<3>(x) - 273*ipow<2>(x) + 39*x - 1)*(4*ipow<3>(x) + 12*ipow<2>(x)*(5*y - 1) + 2*x*(90*ipow<2>(y) - 60*y + 6) + 140*ipow<3>(y) - 180*ipow<2>(y) + 60*y - 4),
            (455*ipow<3>(x) - 273*ipow<2>(x) + 39*x - 1)*(20*ipow<3>(x) + ipow<2>(x)*(180*y - 60) + 4*x*(105*ipow<2>(y) - 90*y + 15) + 280*ipow<3>(y) - 420*ipow<2>(y) + 180*y - 20)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 33
template<>
struct DGBasis2D<33> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (105*ipow<2>(x) - 28*x + 1)*(ipow<5>(x) + 5*ipow<4>(x)*(6*y - 1) + 10*ipow<3>(x)*(21*ipow<2>(y) - 12*y + 1) + 10*ipow<2>(x)*(56*ipow<3>(y) - 63*ipow<2>(y) + 18*y - 1) + 5*x*(126*ipow<4>(y) - 224*ipow<3>(y) + 126*ipow<2>(y) - 24*y + 1) + 252*ipow<5>(y) - 630*ipow<4>(y) + 560*ipow<3>(y) - 210*ipow<2>(y) + 30*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (210*x - 28)*(ipow<5>(x) + 5*ipow<4>(x)*(6*y - 1) + 10*ipow<3>(x)*(21*ipow<2>(y) - 12*y + 1) + 10*ipow<2>(x)*(56*ipow<3>(y) - 63*ipow<2>(y) + 18*y - 1) + 5*x*(126*ipow<4>(y) - 224*ipow<3>(y) + 126*ipow<2>(y) - 24*y + 1) + 252*ipow<5>(y) - 630*ipow<4>(y) + 560*ipow<3>(y) - 210*ipow<2>(y) + 30*y - 1) + (105*ipow<2>(x) - 28*x + 1)*(5*ipow<4>(x) + 20*ipow<3>(x)*(6*y - 1) + 30*ipow<2>(x)*(21*ipow<2>(y) - 12*y + 1) + 20*x*(56*ipow<3>(y) - 63*ipow<2>(y) + 18*y - 1) + 630*ipow<4>(y) - 1120*ipow<3>(y) + 630*ipow<2>(y) - 120*y + 5),
            (105*ipow<2>(x) - 28*x + 1)*(30*ipow<4>(x) + 10*ipow<3>(x)*(42*y - 12) + 10*ipow<2>(x)*(168*ipow<2>(y) - 126*y + 18) + 5*x*(504*ipow<3>(y) - 672*ipow<2>(y) + 252*y - 24) + 1260*ipow<4>(y) - 2520*ipow<3>(y) + 1680*ipow<2>(y) - 420*y + 30)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 34
template<>
struct DGBasis2D<34> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (15*x - 1)*(ipow<6>(x) + 6*ipow<5>(x)*(7*y - 1) + 15*ipow<4>(x)*(28*ipow<2>(y) - 14*y + 1) + 20*ipow<3>(x)*(84*ipow<3>(y) - 84*ipow<2>(y) + 21*y - 1) + 15*ipow<2>(x)*(210*ipow<4>(y) - 336*ipow<3>(y) + 168*ipow<2>(y) - 28*y + 1) + 6*x*(462*ipow<5>(y) - 1050*ipow<4>(y) + 840*ipow<3>(y) - 280*ipow<2>(y) + 35*y - 1) + 924*ipow<6>(y) - 2772*ipow<5>(y) + 3150*ipow<4>(y) - 1680*ipow<3>(y) + 420*ipow<2>(y) - 42*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            15*ipow<6>(x) + 90*ipow<5>(x)*(7*y - 1) + 225*ipow<4>(x)*(28*ipow<2>(y) - 14*y + 1) + 300*ipow<3>(x)*(84*ipow<3>(y) - 84*ipow<2>(y) + 21*y - 1) + 225*ipow<2>(x)*(210*ipow<4>(y) - 336*ipow<3>(y) + 168*ipow<2>(y) - 28*y + 1) + 90*x*(462*ipow<5>(y) - 1050*ipow<4>(y) + 840*ipow<3>(y) - 280*ipow<2>(y) + 35*y - 1) + 13860*ipow<6>(y) - 41580*ipow<5>(y) + 47250*ipow<4>(y) - 25200*ipow<3>(y) + 6300*ipow<2>(y) - 630*y + (15*x - 1)*(6*ipow<5>(x) + 30*ipow<4>(x)*(7*y - 1) + 60*ipow<3>(x)*(28*ipow<2>(y) - 14*y + 1) + 60*ipow<2>(x)*(84*ipow<3>(y) - 84*ipow<2>(y) + 21*y - 1) + 30*x*(210*ipow<4>(y) - 336*ipow<3>(y) + 168*ipow<2>(y) - 28*y + 1) + 2772*ipow<5>(y) - 6300*ipow<4>(y) + 5040*ipow<3>(y) - 1680*ipow<2>(y) + 210*y - 6) + 15,
            (15*x - 1)*(42*ipow<5>(x) + 15*ipow<4>(x)*(56*y - 14) + 20*ipow<3>(x)*(252*ipow<2>(y) - 168*y + 21) + 15*ipow<2>(x)*(840*ipow<3>(y) - 1008*ipow<2>(y) + 336*y - 28) + 6*x*(2310*ipow<4>(y) - 4200*ipow<3>(y) + 2520*ipow<2>(y) - 560*y + 35) + 5544*ipow<5>(y) - 13860*ipow<4>(y) + 12600*ipow<3>(y) - 5040*ipow<2>(y) + 840*y - 42)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 35
template<>
struct DGBasis2D<35> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return ipow<7>(x) + 7*ipow<6>(x)*(8*y - 1) + 21*ipow<5>(x)*(36*ipow<2>(y) - 16*y + 1) + 35*ipow<4>(x)*(120*ipow<3>(y) - 108*ipow<2>(y) + 24*y - 1) + 35*ipow<3>(x)*(330*ipow<4>(y) - 480*ipow<3>(y) + 216*ipow<2>(y) - 32*y + 1) + 21*ipow<2>(x)*(792*ipow<5>(y) - 1650*ipow<4>(y) + 1200*ipow<3>(y) - 360*ipow<2>(y) + 40*y - 1) + 7*x*(1716*ipow<6>(y) - 4752*ipow<5>(y) + 4950*ipow<4>(y) - 2400*ipow<3>(y) + 540*ipow<2>(y) - 48*y + 1) + 3432*ipow<7>(y) - 12012*ipow<6>(y) + 16632*ipow<5>(y) - 11550*ipow<4>(y) + 4200*ipow<3>(y) - 756*ipow<2>(y) + 56*y - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            7*ipow<6>(x) + 42*ipow<5>(x)*(8*y - 1) + 105*ipow<4>(x)*(36*ipow<2>(y) - 16*y + 1) + 140*ipow<3>(x)*(120*ipow<3>(y) - 108*ipow<2>(y) + 24*y - 1) + 105*ipow<2>(x)*(330*ipow<4>(y) - 480*ipow<3>(y) + 216*ipow<2>(y) - 32*y + 1) + 42*x*(792*ipow<5>(y) - 1650*ipow<4>(y) + 1200*ipow<3>(y) - 360*ipow<2>(y) + 40*y - 1) + 12012*ipow<6>(y) - 33264*ipow<5>(y) + 34650*ipow<4>(y) - 16800*ipow<3>(y) + 3780*ipow<2>(y) - 336*y + 7,
            56*ipow<6>(x) + 21*ipow<5>(x)*(72*y - 16) + 35*ipow<4>(x)*(360*ipow<2>(y) - 216*y + 24) + 35*ipow<3>(x)*(1320*ipow<3>(y) - 1440*ipow<2>(y) + 432*y - 32) + 21*ipow<2>(x)*(3960*ipow<4>(y) - 6600*ipow<3>(y) + 3600*ipow<2>(y) - 720*y + 40) + 7*x*(10296*ipow<5>(y) - 23760*ipow<4>(y) + 19800*ipow<3>(y) - 7200*ipow<2>(y) + 1080*y - 48) + 24024*ipow<6>(y) - 72072*ipow<5>(y) + 83160*ipow<4>(y) - 46200*ipow<3>(y) + 12600*ipow<2>(y) - 1512*y + 56
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 36
template<>
struct DGBasis2D<36> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return 24310*ipow<8>(x) - 91520*ipow<7>(x) + 140140*ipow<6>(x) - 112112*ipow<5>(x) + 50050*ipow<4>(x) - 12320*ipow<3>(x) + 1540*ipow<2>(x) - 80*x + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            194480*ipow<7>(x) - 640640*ipow<6>(x) + 840840*ipow<5>(x) - 560560*ipow<4>(x) + 200200*ipow<3>(x) - 36960*ipow<2>(x) + 3080*x - 80,
            0
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 37
template<>
struct DGBasis2D<37> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (x + 2*y - 1)*(19448*ipow<7>(x) - 56056*ipow<6>(x) + 63063*ipow<5>(x) - 35035*ipow<4>(x) + 10010*ipow<3>(x) - 1386*ipow<2>(x) + 77*x - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            19448*ipow<7>(x) - 56056*ipow<6>(x) + 63063*ipow<5>(x) - 35035*ipow<4>(x) + 10010*ipow<3>(x) - 1386*ipow<2>(x) + 77*x + (x + 2*y - 1)*(136136*ipow<6>(x) - 336336*ipow<5>(x) + 315315*ipow<4>(x) - 140140*ipow<3>(x) + 30030*ipow<2>(x) - 2772*x + 77) - 1,
            38896*ipow<7>(x) - 112112*ipow<6>(x) + 126126*ipow<5>(x) - 70070*ipow<4>(x) + 20020*ipow<3>(x) - 2772*ipow<2>(x) + 154*x - 2
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 38
template<>
struct DGBasis2D<38> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (ipow<2>(x) + x*(6*y - 2) + 6*ipow<2>(y) - 6*y + 1)*(12376*ipow<6>(x) - 26208*ipow<5>(x) + 20475*ipow<4>(x) - 7280*ipow<3>(x) + 1170*ipow<2>(x) - 72*x + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (2*x + 6*y - 2)*(12376*ipow<6>(x) - 26208*ipow<5>(x) + 20475*ipow<4>(x) - 7280*ipow<3>(x) + 1170*ipow<2>(x) - 72*x + 1) + (ipow<2>(x) + x*(6*y - 2) + 6*ipow<2>(y) - 6*y + 1)*(74256*ipow<5>(x) - 131040*ipow<4>(x) + 81900*ipow<3>(x) - 21840*ipow<2>(x) + 2340*x - 72),
            (6*x + 12*y - 6)*(12376*ipow<6>(x) - 26208*ipow<5>(x) + 20475*ipow<4>(x) - 7280*ipow<3>(x) + 1170*ipow<2>(x) - 72*x + 1)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 39
template<>
struct DGBasis2D<39> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (6188*ipow<5>(x) - 9100*ipow<4>(x) + 4550*ipow<3>(x) - 910*ipow<2>(x) + 65*x - 1)*(ipow<3>(x) + 3*ipow<2>(x)*(4*y - 1) + 3*x*(10*ipow<2>(y) - 8*y + 1) + 20*ipow<3>(y) - 30*ipow<2>(y) + 12*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (3*ipow<2>(x) + 6*x*(4*y - 1) + 30*ipow<2>(y) - 24*y + 3)*(6188*ipow<5>(x) - 9100*ipow<4>(x) + 4550*ipow<3>(x) - 910*ipow<2>(x) + 65*x - 1) + (30940*ipow<4>(x) - 36400*ipow<3>(x) + 13650*ipow<2>(x) - 1820*x + 65)*(ipow<3>(x) + 3*ipow<2>(x)*(4*y - 1) + 3*x*(10*ipow<2>(y) - 8*y + 1) + 20*ipow<3>(y) - 30*ipow<2>(y) + 12*y - 1),
            (12*ipow<2>(x) + 3*x*(20*y - 8) + 60*ipow<2>(y) - 60*y + 12)*(6188*ipow<5>(x) - 9100*ipow<4>(x) + 4550*ipow<3>(x) - 910*ipow<2>(x) + 65*x - 1)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 40
template<>
struct DGBasis2D<40> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (2380*ipow<4>(x) - 2240*ipow<3>(x) + 630*ipow<2>(x) - 56*x + 1)*(ipow<4>(x) + 4*ipow<3>(x)*(5*y - 1) + ipow<2>(x)*(90*ipow<2>(y) - 60*y + 6) + 4*x*(35*ipow<3>(y) - 45*ipow<2>(y) + 15*y - 1) + 70*ipow<4>(y) - 140*ipow<3>(y) + 90*ipow<2>(y) - 20*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (9520*ipow<3>(x) - 6720*ipow<2>(x) + 1260*x - 56)*(ipow<4>(x) + 4*ipow<3>(x)*(5*y - 1) + ipow<2>(x)*(90*ipow<2>(y) - 60*y + 6) + 4*x*(35*ipow<3>(y) - 45*ipow<2>(y) + 15*y - 1) + 70*ipow<4>(y) - 140*ipow<3>(y) + 90*ipow<2>(y) - 20*y + 1) + (2380*ipow<4>(x) - 2240*ipow<3>(x) + 630*ipow<2>(x) - 56*x + 1)*(4*ipow<3>(x) + 12*ipow<2>(x)*(5*y - 1) + 2*x*(90*ipow<2>(y) - 60*y + 6) + 140*ipow<3>(y) - 180*ipow<2>(y) + 60*y - 4),
            (2380*ipow<4>(x) - 2240*ipow<3>(x) + 630*ipow<2>(x) - 56*x + 1)*(20*ipow<3>(x) + ipow<2>(x)*(180*y - 60) + 4*x*(105*ipow<2>(y) - 90*y + 15) + 280*ipow<3>(y) - 420*ipow<2>(y) + 180*y - 20)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 41
template<>
struct DGBasis2D<41> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (680*ipow<3>(x) - 360*ipow<2>(x) + 45*x - 1)*(ipow<5>(x) + 5*ipow<4>(x)*(6*y - 1) + 10*ipow<3>(x)*(21*ipow<2>(y) - 12*y + 1) + 10*ipow<2>(x)*(56*ipow<3>(y) - 63*ipow<2>(y) + 18*y - 1) + 5*x*(126*ipow<4>(y) - 224*ipow<3>(y) + 126*ipow<2>(y) - 24*y + 1) + 252*ipow<5>(y) - 630*ipow<4>(y) + 560*ipow<3>(y) - 210*ipow<2>(y) + 30*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (2040*ipow<2>(x) - 720*x + 45)*(ipow<5>(x) + 5*ipow<4>(x)*(6*y - 1) + 10*ipow<3>(x)*(21*ipow<2>(y) - 12*y + 1) + 10*ipow<2>(x)*(56*ipow<3>(y) - 63*ipow<2>(y) + 18*y - 1) + 5*x*(126*ipow<4>(y) - 224*ipow<3>(y) + 126*ipow<2>(y) - 24*y + 1) + 252*ipow<5>(y) - 630*ipow<4>(y) + 560*ipow<3>(y) - 210*ipow<2>(y) + 30*y - 1) + (680*ipow<3>(x) - 360*ipow<2>(x) + 45*x - 1)*(5*ipow<4>(x) + 20*ipow<3>(x)*(6*y - 1) + 30*ipow<2>(x)*(21*ipow<2>(y) - 12*y + 1) + 20*x*(56*ipow<3>(y) - 63*ipow<2>(y) + 18*y - 1) + 630*ipow<4>(y) - 1120*ipow<3>(y) + 630*ipow<2>(y) - 120*y + 5),
            (680*ipow<3>(x) - 360*ipow<2>(x) + 45*x - 1)*(30*ipow<4>(x) + 10*ipow<3>(x)*(42*y - 12) + 10*ipow<2>(x)*(168*ipow<2>(y) - 126*y + 18) + 5*x*(504*ipow<3>(y) - 672*ipow<2>(y) + 252*y - 24) + 1260*ipow<4>(y) - 2520*ipow<3>(y) + 1680*ipow<2>(y) - 420*y + 30)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 42
template<>
struct DGBasis2D<42> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (136*ipow<2>(x) - 32*x + 1)*(ipow<6>(x) + 6*ipow<5>(x)*(7*y - 1) + 15*ipow<4>(x)*(28*ipow<2>(y) - 14*y + 1) + 20*ipow<3>(x)*(84*ipow<3>(y) - 84*ipow<2>(y) + 21*y - 1) + 15*ipow<2>(x)*(210*ipow<4>(y) - 336*ipow<3>(y) + 168*ipow<2>(y) - 28*y + 1) + 6*x*(462*ipow<5>(y) - 1050*ipow<4>(y) + 840*ipow<3>(y) - 280*ipow<2>(y) + 35*y - 1) + 924*ipow<6>(y) - 2772*ipow<5>(y) + 3150*ipow<4>(y) - 1680*ipow<3>(y) + 420*ipow<2>(y) - 42*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (272*x - 32)*(ipow<6>(x) + 6*ipow<5>(x)*(7*y - 1) + 15*ipow<4>(x)*(28*ipow<2>(y) - 14*y + 1) + 20*ipow<3>(x)*(84*ipow<3>(y) - 84*ipow<2>(y) + 21*y - 1) + 15*ipow<2>(x)*(210*ipow<4>(y) - 336*ipow<3>(y) + 168*ipow<2>(y) - 28*y + 1) + 6*x*(462*ipow<5>(y) - 1050*ipow<4>(y) + 840*ipow<3>(y) - 280*ipow<2>(y) + 35*y - 1) + 924*ipow<6>(y) - 2772*ipow<5>(y) + 3150*ipow<4>(y) - 1680*ipow<3>(y) + 420*ipow<2>(y) - 42*y + 1) + (136*ipow<2>(x) - 32*x + 1)*(6*ipow<5>(x) + 30*ipow<4>(x)*(7*y - 1) + 60*ipow<3>(x)*(28*ipow<2>(y) - 14*y + 1) + 60*ipow<2>(x)*(84*ipow<3>(y) - 84*ipow<2>(y) + 21*y - 1) + 30*x*(210*ipow<4>(y) - 336*ipow<3>(y) + 168*ipow<2>(y) - 28*y + 1) + 2772*ipow<5>(y) - 6300*ipow<4>(y) + 5040*ipow<3>(y) - 1680*ipow<2>(y) + 210*y - 6),
            (136*ipow<2>(x) - 32*x + 1)*(42*ipow<5>(x) + 15*ipow<4>(x)*(56*y - 14) + 20*ipow<3>(x)*(252*ipow<2>(y) - 168*y + 21) + 15*ipow<2>(x)*(840*ipow<3>(y) - 1008*ipow<2>(y) + 336*y - 28) + 6*x*(2310*ipow<4>(y) - 4200*ipow<3>(y) + 2520*ipow<2>(y) - 560*y + 35) + 5544*ipow<5>(y) - 13860*ipow<4>(y) + 12600*ipow<3>(y) - 5040*ipow<2>(y) + 840*y - 42)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 43
template<>
struct DGBasis2D<43> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (17*x - 1)*(ipow<7>(x) + 7*ipow<6>(x)*(8*y - 1) + 21*ipow<5>(x)*(36*ipow<2>(y) - 16*y + 1) + 35*ipow<4>(x)*(120*ipow<3>(y) - 108*ipow<2>(y) + 24*y - 1) + 35*ipow<3>(x)*(330*ipow<4>(y) - 480*ipow<3>(y) + 216*ipow<2>(y) - 32*y + 1) + 21*ipow<2>(x)*(792*ipow<5>(y) - 1650*ipow<4>(y) + 1200*ipow<3>(y) - 360*ipow<2>(y) + 40*y - 1) + 7*x*(1716*ipow<6>(y) - 4752*ipow<5>(y) + 4950*ipow<4>(y) - 2400*ipow<3>(y) + 540*ipow<2>(y) - 48*y + 1) + 3432*ipow<7>(y) - 12012*ipow<6>(y) + 16632*ipow<5>(y) - 11550*ipow<4>(y) + 4200*ipow<3>(y) - 756*ipow<2>(y) + 56*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            17*ipow<7>(x) + 119*ipow<6>(x)*(8*y - 1) + 357*ipow<5>(x)*(36*ipow<2>(y) - 16*y + 1) + 595*ipow<4>(x)*(120*ipow<3>(y) - 108*ipow<2>(y) + 24*y - 1) + 595*ipow<3>(x)*(330*ipow<4>(y) - 480*ipow<3>(y) + 216*ipow<2>(y) - 32*y + 1) + 357*ipow<2>(x)*(792*ipow<5>(y) - 1650*ipow<4>(y) + 1200*ipow<3>(y) - 360*ipow<2>(y) + 40*y - 1) + 119*x*(1716*ipow<6>(y) - 4752*ipow<5>(y) + 4950*ipow<4>(y) - 2400*ipow<3>(y) + 540*ipow<2>(y) - 48*y + 1) + 58344*ipow<7>(y) - 204204*ipow<6>(y) + 282744*ipow<5>(y) - 196350*ipow<4>(y) + 71400*ipow<3>(y) - 12852*ipow<2>(y) + 952*y + (17*x - 1)*(7*ipow<6>(x) + 42*ipow<5>(x)*(8*y - 1) + 105*ipow<4>(x)*(36*ipow<2>(y) - 16*y + 1) + 140*ipow<3>(x)*(120*ipow<3>(y) - 108*ipow<2>(y) + 24*y - 1) + 105*ipow<2>(x)*(330*ipow<4>(y) - 480*ipow<3>(y) + 216*ipow<2>(y) - 32*y + 1) + 42*x*(792*ipow<5>(y) - 1650*ipow<4>(y) + 1200*ipow<3>(y) - 360*ipow<2>(y) + 40*y - 1) + 12012*ipow<6>(y) - 33264*ipow<5>(y) + 34650*ipow<4>(y) - 16800*ipow<3>(y) + 3780*ipow<2>(y) - 336*y + 7) - 17,
            (17*x - 1)*(56*ipow<6>(x) + 21*ipow<5>(x)*(72*y - 16) + 35*ipow<4>(x)*(360*ipow<2>(y) - 216*y + 24) + 35*ipow<3>(x)*(1320*ipow<3>(y) - 1440*ipow<2>(y) + 432*y - 32) + 21*ipow<2>(x)*(3960*ipow<4>(y) - 6600*ipow<3>(y) + 3600*ipow<2>(y) - 720*y + 40) + 7*x*(10296*ipow<5>(y) - 23760*ipow<4>(y) + 19800*ipow<3>(y) - 7200*ipow<2>(y) + 1080*y - 48) + 24024*ipow<6>(y) - 72072*ipow<5>(y) + 83160*ipow<4>(y) - 46200*ipow<3>(y) + 12600*ipow<2>(y) - 1512*y + 56)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 44
template<>
struct DGBasis2D<44> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return ipow<8>(x) + 8*ipow<7>(x)*(9*y - 1) + 28*ipow<6>(x)*(45*ipow<2>(y) - 18*y + 1) + 56*ipow<5>(x)*(165*ipow<3>(y) - 135*ipow<2>(y) + 27*y - 1) + 70*ipow<4>(x)*(495*ipow<4>(y) - 660*ipow<3>(y) + 270*ipow<2>(y) - 36*y + 1) + 56*ipow<3>(x)*(1287*ipow<5>(y) - 2475*ipow<4>(y) + 1650*ipow<3>(y) - 450*ipow<2>(y) + 45*y - 1) + 28*ipow<2>(x)*(3003*ipow<6>(y) - 7722*ipow<5>(y) + 7425*ipow<4>(y) - 3300*ipow<3>(y) + 675*ipow<2>(y) - 54*y + 1) + 8*x*(6435*ipow<7>(y) - 21021*ipow<6>(y) + 27027*ipow<5>(y) - 17325*ipow<4>(y) + 5775*ipow<3>(y) - 945*ipow<2>(y) + 63*y - 1) + 12870*ipow<8>(y) - 51480*ipow<7>(y) + 84084*ipow<6>(y) - 72072*ipow<5>(y) + 34650*ipow<4>(y) - 9240*ipow<3>(y) + 1260*ipow<2>(y) - 72*y + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            8*ipow<7>(x) + 56*ipow<6>(x)*(9*y - 1) + 168*ipow<5>(x)*(45*ipow<2>(y) - 18*y + 1) + 280*ipow<4>(x)*(165*ipow<3>(y) - 135*ipow<2>(y) + 27*y - 1) + 280*ipow<3>(x)*(495*ipow<4>(y) - 660*ipow<3>(y) + 270*ipow<2>(y) - 36*y + 1) + 168*ipow<2>(x)*(1287*ipow<5>(y) - 2475*ipow<4>(y) + 1650*ipow<3>(y) - 450*ipow<2>(y) + 45*y - 1) + 56*x*(3003*ipow<6>(y) - 7722*ipow<5>(y) + 7425*ipow<4>(y) - 3300*ipow<3>(y) + 675*ipow<2>(y) - 54*y + 1) + 51480*ipow<7>(y) - 168168*ipow<6>(y) + 216216*ipow<5>(y) - 138600*ipow<4>(y) + 46200*ipow<3>(y) - 7560*ipow<2>(y) + 504*y - 8,
            72*ipow<7>(x) + 28*ipow<6>(x)*(90*y - 18) + 56*ipow<5>(x)*(495*ipow<2>(y) - 270*y + 27) + 70*ipow<4>(x)*(1980*ipow<3>(y) - 1980*ipow<2>(y) + 540*y - 36) + 56*ipow<3>(x)*(6435*ipow<4>(y) - 9900*ipow<3>(y) + 4950*ipow<2>(y) - 900*y + 45) + 28*ipow<2>(x)*(18018*ipow<5>(y) - 38610*ipow<4>(y) + 29700*ipow<3>(y) - 9900*ipow<2>(y) + 1350*y - 54) + 8*x*(45045*ipow<6>(y) - 126126*ipow<5>(y) + 135135*ipow<4>(y) - 69300*ipow<3>(y) + 17325*ipow<2>(y) - 1890*y + 63) + 102960*ipow<7>(y) - 360360*ipow<6>(y) + 504504*ipow<5>(y) - 360360*ipow<4>(y) + 138600*ipow<3>(y) - 27720*ipow<2>(y) + 2520*y - 72
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 45
template<>
struct DGBasis2D<45> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return 92378*ipow<9>(x) - 393822*ipow<8>(x) + 700128*ipow<7>(x) - 672672*ipow<6>(x) + 378378*ipow<5>(x) - 126126*ipow<4>(x) + 24024*ipow<3>(x) - 2376*ipow<2>(x) + 99*x - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            831402*ipow<8>(x) - 3150576*ipow<7>(x) + 4900896*ipow<6>(x) - 4036032*ipow<5>(x) + 1891890*ipow<4>(x) - 504504*ipow<3>(x) + 72072*ipow<2>(x) - 4752*x + 99,
            0
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 46
template<>
struct DGBasis2D<46> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (x + 2*y - 1)*(75582*ipow<8>(x) - 254592*ipow<7>(x) + 346528*ipow<6>(x) - 244608*ipow<5>(x) + 95550*ipow<4>(x) - 20384*ipow<3>(x) + 2184*ipow<2>(x) - 96*x + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            75582*ipow<8>(x) - 254592*ipow<7>(x) + 346528*ipow<6>(x) - 244608*ipow<5>(x) + 95550*ipow<4>(x) - 20384*ipow<3>(x) + 2184*ipow<2>(x) - 96*x + (x + 2*y - 1)*(604656*ipow<7>(x) - 1782144*ipow<6>(x) + 2079168*ipow<5>(x) - 1223040*ipow<4>(x) + 382200*ipow<3>(x) - 61152*ipow<2>(x) + 4368*x - 96) + 1,
            151164*ipow<8>(x) - 509184*ipow<7>(x) + 693056*ipow<6>(x) - 489216*ipow<5>(x) + 191100*ipow<4>(x) - 40768*ipow<3>(x) + 4368*ipow<2>(x) - 192*x + 2
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 47
template<>
struct DGBasis2D<47> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (ipow<2>(x) + x*(6*y - 2) + 6*ipow<2>(y) - 6*y + 1)*(50388*ipow<7>(x) - 129948*ipow<6>(x) + 129948*ipow<5>(x) - 63700*ipow<4>(x) + 15925*ipow<3>(x) - 1911*ipow<2>(x) + 91*x - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (2*x + 6*y - 2)*(50388*ipow<7>(x) - 129948*ipow<6>(x) + 129948*ipow<5>(x) - 63700*ipow<4>(x) + 15925*ipow<3>(x) - 1911*ipow<2>(x) + 91*x - 1) + (ipow<2>(x) + x*(6*y - 2) + 6*ipow<2>(y) - 6*y + 1)*(352716*ipow<6>(x) - 779688*ipow<5>(x) + 649740*ipow<4>(x) - 254800*ipow<3>(x) + 47775*ipow<2>(x) - 3822*x + 91),
            (6*x + 12*y - 6)*(50388*ipow<7>(x) - 129948*ipow<6>(x) + 129948*ipow<5>(x) - 63700*ipow<4>(x) + 15925*ipow<3>(x) - 1911*ipow<2>(x) + 91*x - 1)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 48
template<>
struct DGBasis2D<48> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (ipow<3>(x) + 3*ipow<2>(x)*(4*y - 1) + 3*x*(10*ipow<2>(y) - 8*y + 1) + 20*ipow<3>(y) - 30*ipow<2>(y) + 12*y - 1)*(27132*ipow<6>(x) - 51408*ipow<5>(x) + 35700*ipow<4>(x) - 11200*ipow<3>(x) + 1575*ipow<2>(x) - 84*x + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (3*ipow<2>(x) + 6*x*(4*y - 1) + 30*ipow<2>(y) - 24*y + 3)*(27132*ipow<6>(x) - 51408*ipow<5>(x) + 35700*ipow<4>(x) - 11200*ipow<3>(x) + 1575*ipow<2>(x) - 84*x + 1) + (162792*ipow<5>(x) - 257040*ipow<4>(x) + 142800*ipow<3>(x) - 33600*ipow<2>(x) + 3150*x - 84)*(ipow<3>(x) + 3*ipow<2>(x)*(4*y - 1) + 3*x*(10*ipow<2>(y) - 8*y + 1) + 20*ipow<3>(y) - 30*ipow<2>(y) + 12*y - 1),
            (12*ipow<2>(x) + 3*x*(20*y - 8) + 60*ipow<2>(y) - 60*y + 12)*(27132*ipow<6>(x) - 51408*ipow<5>(x) + 35700*ipow<4>(x) - 11200*ipow<3>(x) + 1575*ipow<2>(x) - 84*x + 1)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 49
template<>
struct DGBasis2D<49> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (11628*ipow<5>(x) - 15300*ipow<4>(x) + 6800*ipow<3>(x) - 1200*ipow<2>(x) + 75*x - 1)*(ipow<4>(x) + 4*ipow<3>(x)*(5*y - 1) + ipow<2>(x)*(90*ipow<2>(y) - 60*y + 6) + 4*x*(35*ipow<3>(y) - 45*ipow<2>(y) + 15*y - 1) + 70*ipow<4>(y) - 140*ipow<3>(y) + 90*ipow<2>(y) - 20*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (58140*ipow<4>(x) - 61200*ipow<3>(x) + 20400*ipow<2>(x) - 2400*x + 75)*(ipow<4>(x) + 4*ipow<3>(x)*(5*y - 1) + ipow<2>(x)*(90*ipow<2>(y) - 60*y + 6) + 4*x*(35*ipow<3>(y) - 45*ipow<2>(y) + 15*y - 1) + 70*ipow<4>(y) - 140*ipow<3>(y) + 90*ipow<2>(y) - 20*y + 1) + (11628*ipow<5>(x) - 15300*ipow<4>(x) + 6800*ipow<3>(x) - 1200*ipow<2>(x) + 75*x - 1)*(4*ipow<3>(x) + 12*ipow<2>(x)*(5*y - 1) + 2*x*(90*ipow<2>(y) - 60*y + 6) + 140*ipow<3>(y) - 180*ipow<2>(y) + 60*y - 4),
            (11628*ipow<5>(x) - 15300*ipow<4>(x) + 6800*ipow<3>(x) - 1200*ipow<2>(x) + 75*x - 1)*(20*ipow<3>(x) + ipow<2>(x)*(180*y - 60) + 4*x*(105*ipow<2>(y) - 90*y + 15) + 280*ipow<3>(y) - 420*ipow<2>(y) + 180*y - 20)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 50
template<>
struct DGBasis2D<50> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (3876*ipow<4>(x) - 3264*ipow<3>(x) + 816*ipow<2>(x) - 64*x + 1)*(ipow<5>(x) + 5*ipow<4>(x)*(6*y - 1) + 10*ipow<3>(x)*(21*ipow<2>(y) - 12*y + 1) + 10*ipow<2>(x)*(56*ipow<3>(y) - 63*ipow<2>(y) + 18*y - 1) + 5*x*(126*ipow<4>(y) - 224*ipow<3>(y) + 126*ipow<2>(y) - 24*y + 1) + 252*ipow<5>(y) - 630*ipow<4>(y) + 560*ipow<3>(y) - 210*ipow<2>(y) + 30*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (15504*ipow<3>(x) - 9792*ipow<2>(x) + 1632*x - 64)*(ipow<5>(x) + 5*ipow<4>(x)*(6*y - 1) + 10*ipow<3>(x)*(21*ipow<2>(y) - 12*y + 1) + 10*ipow<2>(x)*(56*ipow<3>(y) - 63*ipow<2>(y) + 18*y - 1) + 5*x*(126*ipow<4>(y) - 224*ipow<3>(y) + 126*ipow<2>(y) - 24*y + 1) + 252*ipow<5>(y) - 630*ipow<4>(y) + 560*ipow<3>(y) - 210*ipow<2>(y) + 30*y - 1) + (3876*ipow<4>(x) - 3264*ipow<3>(x) + 816*ipow<2>(x) - 64*x + 1)*(5*ipow<4>(x) + 20*ipow<3>(x)*(6*y - 1) + 30*ipow<2>(x)*(21*ipow<2>(y) - 12*y + 1) + 20*x*(56*ipow<3>(y) - 63*ipow<2>(y) + 18*y - 1) + 630*ipow<4>(y) - 1120*ipow<3>(y) + 630*ipow<2>(y) - 120*y + 5),
            (3876*ipow<4>(x) - 3264*ipow<3>(x) + 816*ipow<2>(x) - 64*x + 1)*(30*ipow<4>(x) + 10*ipow<3>(x)*(42*y - 12) + 10*ipow<2>(x)*(168*ipow<2>(y) - 126*y + 18) + 5*x*(504*ipow<3>(y) - 672*ipow<2>(y) + 252*y - 24) + 1260*ipow<4>(y) - 2520*ipow<3>(y) + 1680*ipow<2>(y) - 420*y + 30)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 51
template<>
struct DGBasis2D<51> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (969*ipow<3>(x) - 459*ipow<2>(x) + 51*x - 1)*(ipow<6>(x) + 6*ipow<5>(x)*(7*y - 1) + 15*ipow<4>(x)*(28*ipow<2>(y) - 14*y + 1) + 20*ipow<3>(x)*(84*ipow<3>(y) - 84*ipow<2>(y) + 21*y - 1) + 15*ipow<2>(x)*(210*ipow<4>(y) - 336*ipow<3>(y) + 168*ipow<2>(y) - 28*y + 1) + 6*x*(462*ipow<5>(y) - 1050*ipow<4>(y) + 840*ipow<3>(y) - 280*ipow<2>(y) + 35*y - 1) + 924*ipow<6>(y) - 2772*ipow<5>(y) + 3150*ipow<4>(y) - 1680*ipow<3>(y) + 420*ipow<2>(y) - 42*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (2907*ipow<2>(x) - 918*x + 51)*(ipow<6>(x) + 6*ipow<5>(x)*(7*y - 1) + 15*ipow<4>(x)*(28*ipow<2>(y) - 14*y + 1) + 20*ipow<3>(x)*(84*ipow<3>(y) - 84*ipow<2>(y) + 21*y - 1) + 15*ipow<2>(x)*(210*ipow<4>(y) - 336*ipow<3>(y) + 168*ipow<2>(y) - 28*y + 1) + 6*x*(462*ipow<5>(y) - 1050*ipow<4>(y) + 840*ipow<3>(y) - 280*ipow<2>(y) + 35*y - 1) + 924*ipow<6>(y) - 2772*ipow<5>(y) + 3150*ipow<4>(y) - 1680*ipow<3>(y) + 420*ipow<2>(y) - 42*y + 1) + (969*ipow<3>(x) - 459*ipow<2>(x) + 51*x - 1)*(6*ipow<5>(x) + 30*ipow<4>(x)*(7*y - 1) + 60*ipow<3>(x)*(28*ipow<2>(y) - 14*y + 1) + 60*ipow<2>(x)*(84*ipow<3>(y) - 84*ipow<2>(y) + 21*y - 1) + 30*x*(210*ipow<4>(y) - 336*ipow<3>(y) + 168*ipow<2>(y) - 28*y + 1) + 2772*ipow<5>(y) - 6300*ipow<4>(y) + 5040*ipow<3>(y) - 1680*ipow<2>(y) + 210*y - 6),
            (969*ipow<3>(x) - 459*ipow<2>(x) + 51*x - 1)*(42*ipow<5>(x) + 15*ipow<4>(x)*(56*y - 14) + 20*ipow<3>(x)*(252*ipow<2>(y) - 168*y + 21) + 15*ipow<2>(x)*(840*ipow<3>(y) - 1008*ipow<2>(y) + 336*y - 28) + 6*x*(2310*ipow<4>(y) - 4200*ipow<3>(y) + 2520*ipow<2>(y) - 560*y + 35) + 5544*ipow<5>(y) - 13860*ipow<4>(y) + 12600*ipow<3>(y) - 5040*ipow<2>(y) + 840*y - 42)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 52
template<>
struct DGBasis2D<52> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (171*ipow<2>(x) - 36*x + 1)*(ipow<7>(x) + 7*ipow<6>(x)*(8*y - 1) + 21*ipow<5>(x)*(36*ipow<2>(y) - 16*y + 1) + 35*ipow<4>(x)*(120*ipow<3>(y) - 108*ipow<2>(y) + 24*y - 1) + 35*ipow<3>(x)*(330*ipow<4>(y) - 480*ipow<3>(y) + 216*ipow<2>(y) - 32*y + 1) + 21*ipow<2>(x)*(792*ipow<5>(y) - 1650*ipow<4>(y) + 1200*ipow<3>(y) - 360*ipow<2>(y) + 40*y - 1) + 7*x*(1716*ipow<6>(y) - 4752*ipow<5>(y) + 4950*ipow<4>(y) - 2400*ipow<3>(y) + 540*ipow<2>(y) - 48*y + 1) + 3432*ipow<7>(y) - 12012*ipow<6>(y) + 16632*ipow<5>(y) - 11550*ipow<4>(y) + 4200*ipow<3>(y) - 756*ipow<2>(y) + 56*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (342*x - 36)*(ipow<7>(x) + 7*ipow<6>(x)*(8*y - 1) + 21*ipow<5>(x)*(36*ipow<2>(y) - 16*y + 1) + 35*ipow<4>(x)*(120*ipow<3>(y) - 108*ipow<2>(y) + 24*y - 1) + 35*ipow<3>(x)*(330*ipow<4>(y) - 480*ipow<3>(y) + 216*ipow<2>(y) - 32*y + 1) + 21*ipow<2>(x)*(792*ipow<5>(y) - 1650*ipow<4>(y) + 1200*ipow<3>(y) - 360*ipow<2>(y) + 40*y - 1) + 7*x*(1716*ipow<6>(y) - 4752*ipow<5>(y) + 4950*ipow<4>(y) - 2400*ipow<3>(y) + 540*ipow<2>(y) - 48*y + 1) + 3432*ipow<7>(y) - 12012*ipow<6>(y) + 16632*ipow<5>(y) - 11550*ipow<4>(y) + 4200*ipow<3>(y) - 756*ipow<2>(y) + 56*y - 1) + (171*ipow<2>(x) - 36*x + 1)*(7*ipow<6>(x) + 42*ipow<5>(x)*(8*y - 1) + 105*ipow<4>(x)*(36*ipow<2>(y) - 16*y + 1) + 140*ipow<3>(x)*(120*ipow<3>(y) - 108*ipow<2>(y) + 24*y - 1) + 105*ipow<2>(x)*(330*ipow<4>(y) - 480*ipow<3>(y) + 216*ipow<2>(y) - 32*y + 1) + 42*x*(792*ipow<5>(y) - 1650*ipow<4>(y) + 1200*ipow<3>(y) - 360*ipow<2>(y) + 40*y - 1) + 12012*ipow<6>(y) - 33264*ipow<5>(y) + 34650*ipow<4>(y) - 16800*ipow<3>(y) + 3780*ipow<2>(y) - 336*y + 7),
            (171*ipow<2>(x) - 36*x + 1)*(56*ipow<6>(x) + 21*ipow<5>(x)*(72*y - 16) + 35*ipow<4>(x)*(360*ipow<2>(y) - 216*y + 24) + 35*ipow<3>(x)*(1320*ipow<3>(y) - 1440*ipow<2>(y) + 432*y - 32) + 21*ipow<2>(x)*(3960*ipow<4>(y) - 6600*ipow<3>(y) + 3600*ipow<2>(y) - 720*y + 40) + 7*x*(10296*ipow<5>(y) - 23760*ipow<4>(y) + 19800*ipow<3>(y) - 7200*ipow<2>(y) + 1080*y - 48) + 24024*ipow<6>(y) - 72072*ipow<5>(y) + 83160*ipow<4>(y) - 46200*ipow<3>(y) + 12600*ipow<2>(y) - 1512*y + 56)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 53
template<>
struct DGBasis2D<53> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (19*x - 1)*(ipow<8>(x) + 8*ipow<7>(x)*(9*y - 1) + 28*ipow<6>(x)*(45*ipow<2>(y) - 18*y + 1) + 56*ipow<5>(x)*(165*ipow<3>(y) - 135*ipow<2>(y) + 27*y - 1) + 70*ipow<4>(x)*(495*ipow<4>(y) - 660*ipow<3>(y) + 270*ipow<2>(y) - 36*y + 1) + 56*ipow<3>(x)*(1287*ipow<5>(y) - 2475*ipow<4>(y) + 1650*ipow<3>(y) - 450*ipow<2>(y) + 45*y - 1) + 28*ipow<2>(x)*(3003*ipow<6>(y) - 7722*ipow<5>(y) + 7425*ipow<4>(y) - 3300*ipow<3>(y) + 675*ipow<2>(y) - 54*y + 1) + 8*x*(6435*ipow<7>(y) - 21021*ipow<6>(y) + 27027*ipow<5>(y) - 17325*ipow<4>(y) + 5775*ipow<3>(y) - 945*ipow<2>(y) + 63*y - 1) + 12870*ipow<8>(y) - 51480*ipow<7>(y) + 84084*ipow<6>(y) - 72072*ipow<5>(y) + 34650*ipow<4>(y) - 9240*ipow<3>(y) + 1260*ipow<2>(y) - 72*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            19*ipow<8>(x) + 152*ipow<7>(x)*(9*y - 1) + 532*ipow<6>(x)*(45*ipow<2>(y) - 18*y + 1) + 1064*ipow<5>(x)*(165*ipow<3>(y) - 135*ipow<2>(y) + 27*y - 1) + 1330*ipow<4>(x)*(495*ipow<4>(y) - 660*ipow<3>(y) + 270*ipow<2>(y) - 36*y + 1) + 1064*ipow<3>(x)*(1287*ipow<5>(y) - 2475*ipow<4>(y) + 1650*ipow<3>(y) - 450*ipow<2>(y) + 45*y - 1) + 532*ipow<2>(x)*(3003*ipow<6>(y) - 7722*ipow<5>(y) + 7425*ipow<4>(y) - 3300*ipow<3>(y) + 675*ipow<2>(y) - 54*y + 1) + 152*x*(6435*ipow<7>(y) - 21021*ipow<6>(y) + 27027*ipow<5>(y) - 17325*ipow<4>(y) + 5775*ipow<3>(y) - 945*ipow<2>(y) + 63*y - 1) + 244530*ipow<8>(y) - 978120*ipow<7>(y) + 1597596*ipow<6>(y) - 1369368*ipow<5>(y) + 658350*ipow<4>(y) - 175560*ipow<3>(y) + 23940*ipow<2>(y) - 1368*y + (19*x - 1)*(8*ipow<7>(x) + 56*ipow<6>(x)*(9*y - 1) + 168*ipow<5>(x)*(45*ipow<2>(y) - 18*y + 1) + 280*ipow<4>(x)*(165*ipow<3>(y) - 135*ipow<2>(y) + 27*y - 1) + 280*ipow<3>(x)*(495*ipow<4>(y) - 660*ipow<3>(y) + 270*ipow<2>(y) - 36*y + 1) + 168*ipow<2>(x)*(1287*ipow<5>(y) - 2475*ipow<4>(y) + 1650*ipow<3>(y) - 450*ipow<2>(y) + 45*y - 1) + 56*x*(3003*ipow<6>(y) - 7722*ipow<5>(y) + 7425*ipow<4>(y) - 3300*ipow<3>(y) + 675*ipow<2>(y) - 54*y + 1) + 51480*ipow<7>(y) - 168168*ipow<6>(y) + 216216*ipow<5>(y) - 138600*ipow<4>(y) + 46200*ipow<3>(y) - 7560*ipow<2>(y) + 504*y - 8) + 19,
            (19*x - 1)*(72*ipow<7>(x) + 28*ipow<6>(x)*(90*y - 18) + 56*ipow<5>(x)*(495*ipow<2>(y) - 270*y + 27) + 70*ipow<4>(x)*(1980*ipow<3>(y) - 1980*ipow<2>(y) + 540*y - 36) + 56*ipow<3>(x)*(6435*ipow<4>(y) - 9900*ipow<3>(y) + 4950*ipow<2>(y) - 900*y + 45) + 28*ipow<2>(x)*(18018*ipow<5>(y) - 38610*ipow<4>(y) + 29700*ipow<3>(y) - 9900*ipow<2>(y) + 1350*y - 54) + 8*x*(45045*ipow<6>(y) - 126126*ipow<5>(y) + 135135*ipow<4>(y) - 69300*ipow<3>(y) + 17325*ipow<2>(y) - 1890*y + 63) + 102960*ipow<7>(y) - 360360*ipow<6>(y) + 504504*ipow<5>(y) - 360360*ipow<4>(y) + 138600*ipow<3>(y) - 27720*ipow<2>(y) + 2520*y - 72)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 54
template<>
struct DGBasis2D<54> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return ipow<9>(x) + ipow<8>(x)*(90*y - 9) + 36*ipow<7>(x)*(55*ipow<2>(y) - 20*y + 1) + 84*ipow<6>(x)*(220*ipow<3>(y) - 165*ipow<2>(y) + 30*y - 1) + 126*ipow<5>(x)*(715*ipow<4>(y) - 880*ipow<3>(y) + 330*ipow<2>(y) - 40*y + 1) + 126*ipow<4>(x)*(2002*ipow<5>(y) - 3575*ipow<4>(y) + 2200*ipow<3>(y) - 550*ipow<2>(y) + 50*y - 1) + 84*ipow<3>(x)*(5005*ipow<6>(y) - 12012*ipow<5>(y) + 10725*ipow<4>(y) - 4400*ipow<3>(y) + 825*ipow<2>(y) - 60*y + 1) + 36*ipow<2>(x)*(11440*ipow<7>(y) - 35035*ipow<6>(y) + 42042*ipow<5>(y) - 25025*ipow<4>(y) + 7700*ipow<3>(y) - 1155*ipow<2>(y) + 70*y - 1) + 9*x*(24310*ipow<8>(y) - 91520*ipow<7>(y) + 140140*ipow<6>(y) - 112112*ipow<5>(y) + 50050*ipow<4>(y) - 12320*ipow<3>(y) + 1540*ipow<2>(y) - 80*y + 1) + 48620*ipow<9>(y) - 218790*ipow<8>(y) + 411840*ipow<7>(y) - 420420*ipow<6>(y) + 252252*ipow<5>(y) - 90090*ipow<4>(y) + 18480*ipow<3>(y) - 1980*ipow<2>(y) + 90*y - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            9*ipow<8>(x) + 8*ipow<7>(x)*(90*y - 9) + 252*ipow<6>(x)*(55*ipow<2>(y) - 20*y + 1) + 504*ipow<5>(x)*(220*ipow<3>(y) - 165*ipow<2>(y) + 30*y - 1) + 630*ipow<4>(x)*(715*ipow<4>(y) - 880*ipow<3>(y) + 330*ipow<2>(y) - 40*y + 1) + 504*ipow<3>(x)*(2002*ipow<5>(y) - 3575*ipow<4>(y) + 2200*ipow<3>(y) - 550*ipow<2>(y) + 50*y - 1) + 252*ipow<2>(x)*(5005*ipow<6>(y) - 12012*ipow<5>(y) + 10725*ipow<4>(y) - 4400*ipow<3>(y) + 825*ipow<2>(y) - 60*y + 1) + 72*x*(11440*ipow<7>(y) - 35035*ipow<6>(y) + 42042*ipow<5>(y) - 25025*ipow<4>(y) + 7700*ipow<3>(y) - 1155*ipow<2>(y) + 70*y - 1) + 218790*ipow<8>(y) - 823680*ipow<7>(y) + 1261260*ipow<6>(y) - 1009008*ipow<5>(y) + 450450*ipow<4>(y) - 110880*ipow<3>(y) + 13860*ipow<2>(y) - 720*y + 9,
            90*ipow<8>(x) + 36*ipow<7>(x)*(110*y - 20) + 84*ipow<6>(x)*(660*ipow<2>(y) - 330*y + 30) + 126*ipow<5>(x)*(2860*ipow<3>(y) - 2640*ipow<2>(y) + 660*y - 40) + 126*ipow<4>(x)*(10010*ipow<4>(y) - 14300*ipow<3>(y) + 6600*ipow<2>(y) - 1100*y + 50) + 84*ipow<3>(x)*(30030*ipow<5>(y) - 60060*ipow<4>(y) + 42900*ipow<3>(y) - 13200*ipow<2>(y) + 1650*y - 60) + 36*ipow<2>(x)*(80080*ipow<6>(y) - 210210*ipow<5>(y) + 210210*ipow<4>(y) - 100100*ipow<3>(y) + 23100*ipow<2>(y) - 2310*y + 70) + 9*x*(194480*ipow<7>(y) - 640640*ipow<6>(y) + 840840*ipow<5>(y) - 560560*ipow<4>(y) + 200200*ipow<3>(y) - 36960*ipow<2>(y) + 3080*y - 80) + 437580*ipow<8>(y) - 1750320*ipow<7>(y) + 2882880*ipow<6>(y) - 2522520*ipow<5>(y) + 1261260*ipow<4>(y) - 360360*ipow<3>(y) + 55440*ipow<2>(y) - 3960*y + 90
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 55
template<>
struct DGBasis2D<55> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return 352716*ipow<10>(x) - 1679600*ipow<9>(x) + 3401190*ipow<8>(x) - 3818880*ipow<7>(x) + 2598960*ipow<6>(x) - 1100736*ipow<5>(x) + 286650*ipow<4>(x) - 43680*ipow<3>(x) + 3510*ipow<2>(x) - 120*x + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            3527160*ipow<9>(x) - 15116400*ipow<8>(x) + 27209520*ipow<7>(x) - 26732160*ipow<6>(x) + 15593760*ipow<5>(x) - 5503680*ipow<4>(x) + 1146600*ipow<3>(x) - 131040*ipow<2>(x) + 7020*x - 120,
            0
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 56
template<>
struct DGBasis2D<56> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (x + 2*y - 1)*(293930*ipow<9>(x) - 1133730*ipow<8>(x) + 1813968*ipow<7>(x) - 1559376*ipow<6>(x) + 779688*ipow<5>(x) - 229320*ipow<4>(x) + 38220*ipow<3>(x) - 3276*ipow<2>(x) + 117*x - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            293930*ipow<9>(x) - 1133730*ipow<8>(x) + 1813968*ipow<7>(x) - 1559376*ipow<6>(x) + 779688*ipow<5>(x) - 229320*ipow<4>(x) + 38220*ipow<3>(x) - 3276*ipow<2>(x) + 117*x + (x + 2*y - 1)*(2645370*ipow<8>(x) - 9069840*ipow<7>(x) + 12697776*ipow<6>(x) - 9356256*ipow<5>(x) + 3898440*ipow<4>(x) - 917280*ipow<3>(x) + 114660*ipow<2>(x) - 6552*x + 117) - 1,
            587860*ipow<9>(x) - 2267460*ipow<8>(x) + 3627936*ipow<7>(x) - 3118752*ipow<6>(x) + 1559376*ipow<5>(x) - 458640*ipow<4>(x) + 76440*ipow<3>(x) - 6552*ipow<2>(x) + 234*x - 2
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 57
template<>
struct DGBasis2D<57> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (ipow<2>(x) + x*(6*y - 2) + 6*ipow<2>(y) - 6*y + 1)*(203490*ipow<8>(x) - 620160*ipow<7>(x) + 759696*ipow<6>(x) - 479808*ipow<5>(x) + 166600*ipow<4>(x) - 31360*ipow<3>(x) + 2940*ipow<2>(x) - 112*x + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (2*x + 6*y - 2)*(203490*ipow<8>(x) - 620160*ipow<7>(x) + 759696*ipow<6>(x) - 479808*ipow<5>(x) + 166600*ipow<4>(x) - 31360*ipow<3>(x) + 2940*ipow<2>(x) - 112*x + 1) + (ipow<2>(x) + x*(6*y - 2) + 6*ipow<2>(y) - 6*y + 1)*(1627920*ipow<7>(x) - 4341120*ipow<6>(x) + 4558176*ipow<5>(x) - 2399040*ipow<4>(x) + 666400*ipow<3>(x) - 94080*ipow<2>(x) + 5880*x - 112),
            (6*x + 12*y - 6)*(203490*ipow<8>(x) - 620160*ipow<7>(x) + 759696*ipow<6>(x) - 479808*ipow<5>(x) + 166600*ipow<4>(x) - 31360*ipow<3>(x) + 2940*ipow<2>(x) - 112*x + 1)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 58
template<>
struct DGBasis2D<58> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (ipow<3>(x) + 3*ipow<2>(x)*(4*y - 1) + 3*x*(10*ipow<2>(y) - 8*y + 1) + 20*ipow<3>(y) - 30*ipow<2>(y) + 12*y - 1)*(116280*ipow<7>(x) - 271320*ipow<6>(x) + 244188*ipow<5>(x) - 107100*ipow<4>(x) + 23800*ipow<3>(x) - 2520*ipow<2>(x) + 105*x - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (3*ipow<2>(x) + 6*x*(4*y - 1) + 30*ipow<2>(y) - 24*y + 3)*(116280*ipow<7>(x) - 271320*ipow<6>(x) + 244188*ipow<5>(x) - 107100*ipow<4>(x) + 23800*ipow<3>(x) - 2520*ipow<2>(x) + 105*x - 1) + (ipow<3>(x) + 3*ipow<2>(x)*(4*y - 1) + 3*x*(10*ipow<2>(y) - 8*y + 1) + 20*ipow<3>(y) - 30*ipow<2>(y) + 12*y - 1)*(813960*ipow<6>(x) - 1627920*ipow<5>(x) + 1220940*ipow<4>(x) - 428400*ipow<3>(x) + 71400*ipow<2>(x) - 5040*x + 105),
            (12*ipow<2>(x) + 3*x*(20*y - 8) + 60*ipow<2>(y) - 60*y + 12)*(116280*ipow<7>(x) - 271320*ipow<6>(x) + 244188*ipow<5>(x) - 107100*ipow<4>(x) + 23800*ipow<3>(x) - 2520*ipow<2>(x) + 105*x - 1)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 59
template<>
struct DGBasis2D<59> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (54264*ipow<6>(x) - 93024*ipow<5>(x) + 58140*ipow<4>(x) - 16320*ipow<3>(x) + 2040*ipow<2>(x) - 96*x + 1)*(ipow<4>(x) + 4*ipow<3>(x)*(5*y - 1) + ipow<2>(x)*(90*ipow<2>(y) - 60*y + 6) + 4*x*(35*ipow<3>(y) - 45*ipow<2>(y) + 15*y - 1) + 70*ipow<4>(y) - 140*ipow<3>(y) + 90*ipow<2>(y) - 20*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (325584*ipow<5>(x) - 465120*ipow<4>(x) + 232560*ipow<3>(x) - 48960*ipow<2>(x) + 4080*x - 96)*(ipow<4>(x) + 4*ipow<3>(x)*(5*y - 1) + ipow<2>(x)*(90*ipow<2>(y) - 60*y + 6) + 4*x*(35*ipow<3>(y) - 45*ipow<2>(y) + 15*y - 1) + 70*ipow<4>(y) - 140*ipow<3>(y) + 90*ipow<2>(y) - 20*y + 1) + (4*ipow<3>(x) + 12*ipow<2>(x)*(5*y - 1) + 2*x*(90*ipow<2>(y) - 60*y + 6) + 140*ipow<3>(y) - 180*ipow<2>(y) + 60*y - 4)*(54264*ipow<6>(x) - 93024*ipow<5>(x) + 58140*ipow<4>(x) - 16320*ipow<3>(x) + 2040*ipow<2>(x) - 96*x + 1),
            (20*ipow<3>(x) + ipow<2>(x)*(180*y - 60) + 4*x*(105*ipow<2>(y) - 90*y + 15) + 280*ipow<3>(y) - 420*ipow<2>(y) + 180*y - 20)*(54264*ipow<6>(x) - 93024*ipow<5>(x) + 58140*ipow<4>(x) - 16320*ipow<3>(x) + 2040*ipow<2>(x) - 96*x + 1)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 60
template<>
struct DGBasis2D<60> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (20349*ipow<5>(x) - 24225*ipow<4>(x) + 9690*ipow<3>(x) - 1530*ipow<2>(x) + 85*x - 1)*(ipow<5>(x) + 5*ipow<4>(x)*(6*y - 1) + 10*ipow<3>(x)*(21*ipow<2>(y) - 12*y + 1) + 10*ipow<2>(x)*(56*ipow<3>(y) - 63*ipow<2>(y) + 18*y - 1) + 5*x*(126*ipow<4>(y) - 224*ipow<3>(y) + 126*ipow<2>(y) - 24*y + 1) + 252*ipow<5>(y) - 630*ipow<4>(y) + 560*ipow<3>(y) - 210*ipow<2>(y) + 30*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (101745*ipow<4>(x) - 96900*ipow<3>(x) + 29070*ipow<2>(x) - 3060*x + 85)*(ipow<5>(x) + 5*ipow<4>(x)*(6*y - 1) + 10*ipow<3>(x)*(21*ipow<2>(y) - 12*y + 1) + 10*ipow<2>(x)*(56*ipow<3>(y) - 63*ipow<2>(y) + 18*y - 1) + 5*x*(126*ipow<4>(y) - 224*ipow<3>(y) + 126*ipow<2>(y) - 24*y + 1) + 252*ipow<5>(y) - 630*ipow<4>(y) + 560*ipow<3>(y) - 210*ipow<2>(y) + 30*y - 1) + (20349*ipow<5>(x) - 24225*ipow<4>(x) + 9690*ipow<3>(x) - 1530*ipow<2>(x) + 85*x - 1)*(5*ipow<4>(x) + 20*ipow<3>(x)*(6*y - 1) + 30*ipow<2>(x)*(21*ipow<2>(y) - 12*y + 1) + 20*x*(56*ipow<3>(y) - 63*ipow<2>(y) + 18*y - 1) + 630*ipow<4>(y) - 1120*ipow<3>(y) + 630*ipow<2>(y) - 120*y + 5),
            (20349*ipow<5>(x) - 24225*ipow<4>(x) + 9690*ipow<3>(x) - 1530*ipow<2>(x) + 85*x - 1)*(30*ipow<4>(x) + 10*ipow<3>(x)*(42*y - 12) + 10*ipow<2>(x)*(168*ipow<2>(y) - 126*y + 18) + 5*x*(504*ipow<3>(y) - 672*ipow<2>(y) + 252*y - 24) + 1260*ipow<4>(y) - 2520*ipow<3>(y) + 1680*ipow<2>(y) - 420*y + 30)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 61
template<>
struct DGBasis2D<61> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (5985*ipow<4>(x) - 4560*ipow<3>(x) + 1026*ipow<2>(x) - 72*x + 1)*(ipow<6>(x) + 6*ipow<5>(x)*(7*y - 1) + 15*ipow<4>(x)*(28*ipow<2>(y) - 14*y + 1) + 20*ipow<3>(x)*(84*ipow<3>(y) - 84*ipow<2>(y) + 21*y - 1) + 15*ipow<2>(x)*(210*ipow<4>(y) - 336*ipow<3>(y) + 168*ipow<2>(y) - 28*y + 1) + 6*x*(462*ipow<5>(y) - 1050*ipow<4>(y) + 840*ipow<3>(y) - 280*ipow<2>(y) + 35*y - 1) + 924*ipow<6>(y) - 2772*ipow<5>(y) + 3150*ipow<4>(y) - 1680*ipow<3>(y) + 420*ipow<2>(y) - 42*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (23940*ipow<3>(x) - 13680*ipow<2>(x) + 2052*x - 72)*(ipow<6>(x) + 6*ipow<5>(x)*(7*y - 1) + 15*ipow<4>(x)*(28*ipow<2>(y) - 14*y + 1) + 20*ipow<3>(x)*(84*ipow<3>(y) - 84*ipow<2>(y) + 21*y - 1) + 15*ipow<2>(x)*(210*ipow<4>(y) - 336*ipow<3>(y) + 168*ipow<2>(y) - 28*y + 1) + 6*x*(462*ipow<5>(y) - 1050*ipow<4>(y) + 840*ipow<3>(y) - 280*ipow<2>(y) + 35*y - 1) + 924*ipow<6>(y) - 2772*ipow<5>(y) + 3150*ipow<4>(y) - 1680*ipow<3>(y) + 420*ipow<2>(y) - 42*y + 1) + (5985*ipow<4>(x) - 4560*ipow<3>(x) + 1026*ipow<2>(x) - 72*x + 1)*(6*ipow<5>(x) + 30*ipow<4>(x)*(7*y - 1) + 60*ipow<3>(x)*(28*ipow<2>(y) - 14*y + 1) + 60*ipow<2>(x)*(84*ipow<3>(y) - 84*ipow<2>(y) + 21*y - 1) + 30*x*(210*ipow<4>(y) - 336*ipow<3>(y) + 168*ipow<2>(y) - 28*y + 1) + 2772*ipow<5>(y) - 6300*ipow<4>(y) + 5040*ipow<3>(y) - 1680*ipow<2>(y) + 210*y - 6),
            (5985*ipow<4>(x) - 4560*ipow<3>(x) + 1026*ipow<2>(x) - 72*x + 1)*(42*ipow<5>(x) + 15*ipow<4>(x)*(56*y - 14) + 20*ipow<3>(x)*(252*ipow<2>(y) - 168*y + 21) + 15*ipow<2>(x)*(840*ipow<3>(y) - 1008*ipow<2>(y) + 336*y - 28) + 6*x*(2310*ipow<4>(y) - 4200*ipow<3>(y) + 2520*ipow<2>(y) - 560*y + 35) + 5544*ipow<5>(y) - 13860*ipow<4>(y) + 12600*ipow<3>(y) - 5040*ipow<2>(y) + 840*y - 42)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 62
template<>
struct DGBasis2D<62> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (1330*ipow<3>(x) - 570*ipow<2>(x) + 57*x - 1)*(ipow<7>(x) + 7*ipow<6>(x)*(8*y - 1) + 21*ipow<5>(x)*(36*ipow<2>(y) - 16*y + 1) + 35*ipow<4>(x)*(120*ipow<3>(y) - 108*ipow<2>(y) + 24*y - 1) + 35*ipow<3>(x)*(330*ipow<4>(y) - 480*ipow<3>(y) + 216*ipow<2>(y) - 32*y + 1) + 21*ipow<2>(x)*(792*ipow<5>(y) - 1650*ipow<4>(y) + 1200*ipow<3>(y) - 360*ipow<2>(y) + 40*y - 1) + 7*x*(1716*ipow<6>(y) - 4752*ipow<5>(y) + 4950*ipow<4>(y) - 2400*ipow<3>(y) + 540*ipow<2>(y) - 48*y + 1) + 3432*ipow<7>(y) - 12012*ipow<6>(y) + 16632*ipow<5>(y) - 11550*ipow<4>(y) + 4200*ipow<3>(y) - 756*ipow<2>(y) + 56*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (3990*ipow<2>(x) - 1140*x + 57)*(ipow<7>(x) + 7*ipow<6>(x)*(8*y - 1) + 21*ipow<5>(x)*(36*ipow<2>(y) - 16*y + 1) + 35*ipow<4>(x)*(120*ipow<3>(y) - 108*ipow<2>(y) + 24*y - 1) + 35*ipow<3>(x)*(330*ipow<4>(y) - 480*ipow<3>(y) + 216*ipow<2>(y) - 32*y + 1) + 21*ipow<2>(x)*(792*ipow<5>(y) - 1650*ipow<4>(y) + 1200*ipow<3>(y) - 360*ipow<2>(y) + 40*y - 1) + 7*x*(1716*ipow<6>(y) - 4752*ipow<5>(y) + 4950*ipow<4>(y) - 2400*ipow<3>(y) + 540*ipow<2>(y) - 48*y + 1) + 3432*ipow<7>(y) - 12012*ipow<6>(y) + 16632*ipow<5>(y) - 11550*ipow<4>(y) + 4200*ipow<3>(y) - 756*ipow<2>(y) + 56*y - 1) + (1330*ipow<3>(x) - 570*ipow<2>(x) + 57*x - 1)*(7*ipow<6>(x) + 42*ipow<5>(x)*(8*y - 1) + 105*ipow<4>(x)*(36*ipow<2>(y) - 16*y + 1) + 140*ipow<3>(x)*(120*ipow<3>(y) - 108*ipow<2>(y) + 24*y - 1) + 105*ipow<2>(x)*(330*ipow<4>(y) - 480*ipow<3>(y) + 216*ipow<2>(y) - 32*y + 1) + 42*x*(792*ipow<5>(y) - 1650*ipow<4>(y) + 1200*ipow<3>(y) - 360*ipow<2>(y) + 40*y - 1) + 12012*ipow<6>(y) - 33264*ipow<5>(y) + 34650*ipow<4>(y) - 16800*ipow<3>(y) + 3780*ipow<2>(y) - 336*y + 7),
            (1330*ipow<3>(x) - 570*ipow<2>(x) + 57*x - 1)*(56*ipow<6>(x) + 21*ipow<5>(x)*(72*y - 16) + 35*ipow<4>(x)*(360*ipow<2>(y) - 216*y + 24) + 35*ipow<3>(x)*(1320*ipow<3>(y) - 1440*ipow<2>(y) + 432*y - 32) + 21*ipow<2>(x)*(3960*ipow<4>(y) - 6600*ipow<3>(y) + 3600*ipow<2>(y) - 720*y + 40) + 7*x*(10296*ipow<5>(y) - 23760*ipow<4>(y) + 19800*ipow<3>(y) - 7200*ipow<2>(y) + 1080*y - 48) + 24024*ipow<6>(y) - 72072*ipow<5>(y) + 83160*ipow<4>(y) - 46200*ipow<3>(y) + 12600*ipow<2>(y) - 1512*y + 56)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 63
template<>
struct DGBasis2D<63> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (210*ipow<2>(x) - 40*x + 1)*(ipow<8>(x) + 8*ipow<7>(x)*(9*y - 1) + 28*ipow<6>(x)*(45*ipow<2>(y) - 18*y + 1) + 56*ipow<5>(x)*(165*ipow<3>(y) - 135*ipow<2>(y) + 27*y - 1) + 70*ipow<4>(x)*(495*ipow<4>(y) - 660*ipow<3>(y) + 270*ipow<2>(y) - 36*y + 1) + 56*ipow<3>(x)*(1287*ipow<5>(y) - 2475*ipow<4>(y) + 1650*ipow<3>(y) - 450*ipow<2>(y) + 45*y - 1) + 28*ipow<2>(x)*(3003*ipow<6>(y) - 7722*ipow<5>(y) + 7425*ipow<4>(y) - 3300*ipow<3>(y) + 675*ipow<2>(y) - 54*y + 1) + 8*x*(6435*ipow<7>(y) - 21021*ipow<6>(y) + 27027*ipow<5>(y) - 17325*ipow<4>(y) + 5775*ipow<3>(y) - 945*ipow<2>(y) + 63*y - 1) + 12870*ipow<8>(y) - 51480*ipow<7>(y) + 84084*ipow<6>(y) - 72072*ipow<5>(y) + 34650*ipow<4>(y) - 9240*ipow<3>(y) + 1260*ipow<2>(y) - 72*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (420*x - 40)*(ipow<8>(x) + 8*ipow<7>(x)*(9*y - 1) + 28*ipow<6>(x)*(45*ipow<2>(y) - 18*y + 1) + 56*ipow<5>(x)*(165*ipow<3>(y) - 135*ipow<2>(y) + 27*y - 1) + 70*ipow<4>(x)*(495*ipow<4>(y) - 660*ipow<3>(y) + 270*ipow<2>(y) - 36*y + 1) + 56*ipow<3>(x)*(1287*ipow<5>(y) - 2475*ipow<4>(y) + 1650*ipow<3>(y) - 450*ipow<2>(y) + 45*y - 1) + 28*ipow<2>(x)*(3003*ipow<6>(y) - 7722*ipow<5>(y) + 7425*ipow<4>(y) - 3300*ipow<3>(y) + 675*ipow<2>(y) - 54*y + 1) + 8*x*(6435*ipow<7>(y) - 21021*ipow<6>(y) + 27027*ipow<5>(y) - 17325*ipow<4>(y) + 5775*ipow<3>(y) - 945*ipow<2>(y) + 63*y - 1) + 12870*ipow<8>(y) - 51480*ipow<7>(y) + 84084*ipow<6>(y) - 72072*ipow<5>(y) + 34650*ipow<4>(y) - 9240*ipow<3>(y) + 1260*ipow<2>(y) - 72*y + 1) + (210*ipow<2>(x) - 40*x + 1)*(8*ipow<7>(x) + 56*ipow<6>(x)*(9*y - 1) + 168*ipow<5>(x)*(45*ipow<2>(y) - 18*y + 1) + 280*ipow<4>(x)*(165*ipow<3>(y) - 135*ipow<2>(y) + 27*y - 1) + 280*ipow<3>(x)*(495*ipow<4>(y) - 660*ipow<3>(y) + 270*ipow<2>(y) - 36*y + 1) + 168*ipow<2>(x)*(1287*ipow<5>(y) - 2475*ipow<4>(y) + 1650*ipow<3>(y) - 450*ipow<2>(y) + 45*y - 1) + 56*x*(3003*ipow<6>(y) - 7722*ipow<5>(y) + 7425*ipow<4>(y) - 3300*ipow<3>(y) + 675*ipow<2>(y) - 54*y + 1) + 51480*ipow<7>(y) - 168168*ipow<6>(y) + 216216*ipow<5>(y) - 138600*ipow<4>(y) + 46200*ipow<3>(y) - 7560*ipow<2>(y) + 504*y - 8),
            (210*ipow<2>(x) - 40*x + 1)*(72*ipow<7>(x) + 28*ipow<6>(x)*(90*y - 18) + 56*ipow<5>(x)*(495*ipow<2>(y) - 270*y + 27) + 70*ipow<4>(x)*(1980*ipow<3>(y) - 1980*ipow<2>(y) + 540*y - 36) + 56*ipow<3>(x)*(6435*ipow<4>(y) - 9900*ipow<3>(y) + 4950*ipow<2>(y) - 900*y + 45) + 28*ipow<2>(x)*(18018*ipow<5>(y) - 38610*ipow<4>(y) + 29700*ipow<3>(y) - 9900*ipow<2>(y) + 1350*y - 54) + 8*x*(45045*ipow<6>(y) - 126126*ipow<5>(y) + 135135*ipow<4>(y) - 69300*ipow<3>(y) + 17325*ipow<2>(y) - 1890*y + 63) + 102960*ipow<7>(y) - 360360*ipow<6>(y) + 504504*ipow<5>(y) - 360360*ipow<4>(y) + 138600*ipow<3>(y) - 27720*ipow<2>(y) + 2520*y - 72)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 64
template<>
struct DGBasis2D<64> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (21*x - 1)*(ipow<9>(x) + ipow<8>(x)*(90*y - 9) + 36*ipow<7>(x)*(55*ipow<2>(y) - 20*y + 1) + 84*ipow<6>(x)*(220*ipow<3>(y) - 165*ipow<2>(y) + 30*y - 1) + 126*ipow<5>(x)*(715*ipow<4>(y) - 880*ipow<3>(y) + 330*ipow<2>(y) - 40*y + 1) + 126*ipow<4>(x)*(2002*ipow<5>(y) - 3575*ipow<4>(y) + 2200*ipow<3>(y) - 550*ipow<2>(y) + 50*y - 1) + 84*ipow<3>(x)*(5005*ipow<6>(y) - 12012*ipow<5>(y) + 10725*ipow<4>(y) - 4400*ipow<3>(y) + 825*ipow<2>(y) - 60*y + 1) + 36*ipow<2>(x)*(11440*ipow<7>(y) - 35035*ipow<6>(y) + 42042*ipow<5>(y) - 25025*ipow<4>(y) + 7700*ipow<3>(y) - 1155*ipow<2>(y) + 70*y - 1) + 9*x*(24310*ipow<8>(y) - 91520*ipow<7>(y) + 140140*ipow<6>(y) - 112112*ipow<5>(y) + 50050*ipow<4>(y) - 12320*ipow<3>(y) + 1540*ipow<2>(y) - 80*y + 1) + 48620*ipow<9>(y) - 218790*ipow<8>(y) + 411840*ipow<7>(y) - 420420*ipow<6>(y) + 252252*ipow<5>(y) - 90090*ipow<4>(y) + 18480*ipow<3>(y) - 1980*ipow<2>(y) + 90*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            21*ipow<9>(x) + 21*ipow<8>(x)*(90*y - 9) + 756*ipow<7>(x)*(55*ipow<2>(y) - 20*y + 1) + 1764*ipow<6>(x)*(220*ipow<3>(y) - 165*ipow<2>(y) + 30*y - 1) + 2646*ipow<5>(x)*(715*ipow<4>(y) - 880*ipow<3>(y) + 330*ipow<2>(y) - 40*y + 1) + 2646*ipow<4>(x)*(2002*ipow<5>(y) - 3575*ipow<4>(y) + 2200*ipow<3>(y) - 550*ipow<2>(y) + 50*y - 1) + 1764*ipow<3>(x)*(5005*ipow<6>(y) - 12012*ipow<5>(y) + 10725*ipow<4>(y) - 4400*ipow<3>(y) + 825*ipow<2>(y) - 60*y + 1) + 756*ipow<2>(x)*(11440*ipow<7>(y) - 35035*ipow<6>(y) + 42042*ipow<5>(y) - 25025*ipow<4>(y) + 7700*ipow<3>(y) - 1155*ipow<2>(y) + 70*y - 1) + 189*x*(24310*ipow<8>(y) - 91520*ipow<7>(y) + 140140*ipow<6>(y) - 112112*ipow<5>(y) + 50050*ipow<4>(y) - 12320*ipow<3>(y) + 1540*ipow<2>(y) - 80*y + 1) + 1021020*ipow<9>(y) - 4594590*ipow<8>(y) + 8648640*ipow<7>(y) - 8828820*ipow<6>(y) + 5297292*ipow<5>(y) - 1891890*ipow<4>(y) + 388080*ipow<3>(y) - 41580*ipow<2>(y) + 1890*y + (21*x - 1)*(9*ipow<8>(x) + 8*ipow<7>(x)*(90*y - 9) + 252*ipow<6>(x)*(55*ipow<2>(y) - 20*y + 1) + 504*ipow<5>(x)*(220*ipow<3>(y) - 165*ipow<2>(y) + 30*y - 1) + 630*ipow<4>(x)*(715*ipow<4>(y) - 880*ipow<3>(y) + 330*ipow<2>(y) - 40*y + 1) + 504*ipow<3>(x)*(2002*ipow<5>(y) - 3575*ipow<4>(y) + 2200*ipow<3>(y) - 550*ipow<2>(y) + 50*y - 1) + 252*ipow<2>(x)*(5005*ipow<6>(y) - 12012*ipow<5>(y) + 10725*ipow<4>(y) - 4400*ipow<3>(y) + 825*ipow<2>(y) - 60*y + 1) + 72*x*(11440*ipow<7>(y) - 35035*ipow<6>(y) + 42042*ipow<5>(y) - 25025*ipow<4>(y) + 7700*ipow<3>(y) - 1155*ipow<2>(y) + 70*y - 1) + 218790*ipow<8>(y) - 823680*ipow<7>(y) + 1261260*ipow<6>(y) - 1009008*ipow<5>(y) + 450450*ipow<4>(y) - 110880*ipow<3>(y) + 13860*ipow<2>(y) - 720*y + 9) - 21,
            (21*x - 1)*(90*ipow<8>(x) + 36*ipow<7>(x)*(110*y - 20) + 84*ipow<6>(x)*(660*ipow<2>(y) - 330*y + 30) + 126*ipow<5>(x)*(2860*ipow<3>(y) - 2640*ipow<2>(y) + 660*y - 40) + 126*ipow<4>(x)*(10010*ipow<4>(y) - 14300*ipow<3>(y) + 6600*ipow<2>(y) - 1100*y + 50) + 84*ipow<3>(x)*(30030*ipow<5>(y) - 60060*ipow<4>(y) + 42900*ipow<3>(y) - 13200*ipow<2>(y) + 1650*y - 60) + 36*ipow<2>(x)*(80080*ipow<6>(y) - 210210*ipow<5>(y) + 210210*ipow<4>(y) - 100100*ipow<3>(y) + 23100*ipow<2>(y) - 2310*y + 70) + 9*x*(194480*ipow<7>(y) - 640640*ipow<6>(y) + 840840*ipow<5>(y) - 560560*ipow<4>(y) + 200200*ipow<3>(y) - 36960*ipow<2>(y) + 3080*y - 80) + 437580*ipow<8>(y) - 1750320*ipow<7>(y) + 2882880*ipow<6>(y) - 2522520*ipow<5>(y) + 1261260*ipow<4>(y) - 360360*ipow<3>(y) + 55440*ipow<2>(y) - 3960*y + 90)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 65
template<>
struct DGBasis2D<65> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return ipow<10>(x) + 10*ipow<9>(x)*(11*y - 1) + 45*ipow<8>(x)*(66*ipow<2>(y) - 22*y + 1) + 120*ipow<7>(x)*(286*ipow<3>(y) - 198*ipow<2>(y) + 33*y - 1) + 210*ipow<6>(x)*(1001*ipow<4>(y) - 1144*ipow<3>(y) + 396*ipow<2>(y) - 44*y + 1) + 252*ipow<5>(x)*(3003*ipow<5>(y) - 5005*ipow<4>(y) + 2860*ipow<3>(y) - 660*ipow<2>(y) + 55*y - 1) + 210*ipow<4>(x)*(8008*ipow<6>(y) - 18018*ipow<5>(y) + 15015*ipow<4>(y) - 5720*ipow<3>(y) + 990*ipow<2>(y) - 66*y + 1) + 120*ipow<3>(x)*(19448*ipow<7>(y) - 56056*ipow<6>(y) + 63063*ipow<5>(y) - 35035*ipow<4>(y) + 10010*ipow<3>(y) - 1386*ipow<2>(y) + 77*y - 1) + 45*ipow<2>(x)*(43758*ipow<8>(y) - 155584*ipow<7>(y) + 224224*ipow<6>(y) - 168168*ipow<5>(y) + 70070*ipow<4>(y) - 16016*ipow<3>(y) + 1848*ipow<2>(y) - 88*y + 1) + 10*x*(92378*ipow<9>(y) - 393822*ipow<8>(y) + 700128*ipow<7>(y) - 672672*ipow<6>(y) + 378378*ipow<5>(y) - 126126*ipow<4>(y) + 24024*ipow<3>(y) - 2376*ipow<2>(y) + 99*y - 1) + 184756*ipow<10>(y) - 923780*ipow<9>(y) + 1969110*ipow<8>(y) - 2333760*ipow<7>(y) + 1681680*ipow<6>(y) - 756756*ipow<5>(y) + 210210*ipow<4>(y) - 34320*ipow<3>(y) + 2970*ipow<2>(y) - 110*y + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            10*ipow<9>(x) + 90*ipow<8>(x)*(11*y - 1) + 360*ipow<7>(x)*(66*ipow<2>(y) - 22*y + 1) + 840*ipow<6>(x)*(286*ipow<3>(y) - 198*ipow<2>(y) + 33*y - 1) + 1260*ipow<5>(x)*(1001*ipow<4>(y) - 1144*ipow<3>(y) + 396*ipow<2>(y) - 44*y + 1) + 1260*ipow<4>(x)*(3003*ipow<5>(y) - 5005*ipow<4>(y) + 2860*ipow<3>(y) - 660*ipow<2>(y) + 55*y - 1) + 840*ipow<3>(x)*(8008*ipow<6>(y) - 18018*ipow<5>(y) + 15015*ipow<4>(y) - 5720*ipow<3>(y) + 990*ipow<2>(y) - 66*y + 1) + 360*ipow<2>(x)*(19448*ipow<7>(y) - 56056*ipow<6>(y) + 63063*ipow<5>(y) - 35035*ipow<4>(y) + 10010*ipow<3>(y) - 1386*ipow<2>(y) + 77*y - 1) + 90*x*(43758*ipow<8>(y) - 155584*ipow<7>(y) + 224224*ipow<6>(y) - 168168*ipow<5>(y) + 70070*ipow<4>(y) - 16016*ipow<3>(y) + 1848*ipow<2>(y) - 88*y + 1) + 923780*ipow<9>(y) - 3938220*ipow<8>(y) + 7001280*ipow<7>(y) - 6726720*ipow<6>(y) + 3783780*ipow<5>(y) - 1261260*ipow<4>(y) + 240240*ipow<3>(y) - 23760*ipow<2>(y) + 990*y - 10,
            110*ipow<9>(x) + 45*ipow<8>(x)*(132*y - 22) + 120*ipow<7>(x)*(858*ipow<2>(y) - 396*y + 33) + 210*ipow<6>(x)*(4004*ipow<3>(y) - 3432*ipow<2>(y) + 792*y - 44) + 252*ipow<5>(x)*(15015*ipow<4>(y) - 20020*ipow<3>(y) + 8580*ipow<2>(y) - 1320*y + 55) + 210*ipow<4>(x)*(48048*ipow<5>(y) - 90090*ipow<4>(y) + 60060*ipow<3>(y) - 17160*ipow<2>(y) + 1980*y - 66) + 120*ipow<3>(x)*(136136*ipow<6>(y) - 336336*ipow<5>(y) + 315315*ipow<4>(y) - 140140*ipow<3>(y) + 30030*ipow<2>(y) - 2772*y + 77) + 45*ipow<2>(x)*(350064*ipow<7>(y) - 1089088*ipow<6>(y) + 1345344*ipow<5>(y) - 840840*ipow<4>(y) + 280280*ipow<3>(y) - 48048*ipow<2>(y) + 3696*y - 88) + 10*x*(831402*ipow<8>(y) - 3150576*ipow<7>(y) + 4900896*ipow<6>(y) - 4036032*ipow<5>(y) + 1891890*ipow<4>(y) - 504504*ipow<3>(y) + 72072*ipow<2>(y) - 4752*y + 99) + 1847560*ipow<9>(y) - 8314020*ipow<8>(y) + 15752880*ipow<7>(y) - 16336320*ipow<6>(y) + 10090080*ipow<5>(y) - 3783780*ipow<4>(y) + 840840*ipow<3>(y) - 102960*ipow<2>(y) + 5940*y - 110
        };
    }
    static constexpr uInt Order = 6;
};
