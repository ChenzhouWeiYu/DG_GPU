#pragma once
#include "base/type.h"
#include "dg/dg_basis/dg_basis.h"

/*
    生成列表 Mathematica
Table[JacobiP[ii, 2*jj + 2*kk + 2, 0, 2 x - 1]*(1 - x)^(jj)*
    JacobiP[jj, 2*kk + 1, 0, 2 y/(1 - x) - 1]*(1 - x - y)^(kk)*
    JacobiP[kk, 0, 0, 2 z/(1 - x - y) - 1], {n, 0, 3}, {kk, 0, 
    n}, {jj, 0, n - kk}, {ii, n - kk - jj, n - kk - jj}] // 
Simplify // Flatten
*/

template<uInt N>
HostDevice constexpr Scalar ipow(Scalar x) {
    static_assert(N >= 0, "N must be non-negative");
    if constexpr (N == 0) return 1.0;
    else if constexpr (N == 1) return x;
    else if constexpr (N % 2 == 0) {
        auto half = ipow<N/2>(x);
        return half * half;
    } else {
        return x * ipow<N-1>(x);
    }
    return 1.0;
}

// Basis 0
template<>
struct DGBasis<0> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return 1;
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            0,
            0,
            0
        };
    }
    static constexpr uInt Order = 0;
};

// Basis 1
template<>
struct DGBasis<1> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return 4*x - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            4,
            0,
            0
        };
    }
    static constexpr uInt Order = 1;
};

// Basis 2
template<>
struct DGBasis<2> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return x + 3*y - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            1,
            3,
            0
        };
    }
    static constexpr uInt Order = 1;
};

// Basis 3
template<>
struct DGBasis<3> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return x + y + 2*z - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            1,
            1,
            2
        };
    }
    static constexpr uInt Order = 1;
};

// Basis 4
template<>
struct DGBasis<4> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return 5*x*(3*x - 2) + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            30*x - 10,
            0,
            0
        };
    }
    static constexpr uInt Order = 2;
};

// Basis 5
template<>
struct DGBasis<5> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (6*x - 1)*(x + 3*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            12*x + 18*y - 7,
            18*x - 3,
            0
        };
    }
    static constexpr uInt Order = 2;
};

// Basis 6
template<>
struct DGBasis<6> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return 10*ipow<2>(y) + y*(8*x - 8) + ipow<2>(x - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            2*x + 8*y - 2,
            8*x + 20*y - 8,
            0
        };
    }
    static constexpr uInt Order = 2;
};

// Basis 7
template<>
struct DGBasis<7> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (6*x - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            12*x + 6*y + 12*z - 7,
            6*x - 1,
            12*x - 2
        };
    }
    static constexpr uInt Order = 2;
};

// Basis 8
template<>
struct DGBasis<8> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x + 5*y - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            2*x + 6*y + 2*z - 2,
            6*x + 10*y + 10*z - 6,
            2*x + 10*y - 2
        };
    }
    static constexpr uInt Order = 2;
};

// Basis 9
template<>
struct DGBasis<9> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return 6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            2*x + 2*y + 6*z - 2,
            2*x + 2*y + 6*z - 2,
            6*x + 6*y + 12*z - 6
        };
    }
    static constexpr uInt Order = 2;
};

// Basis 10
template<>
struct DGBasis<10> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return x*(7*x*(8*x - 9) + 18) - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            7*x*(8*x - 9) + x*(112*x - 63) + 18,
            0,
            0
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 11
template<>
struct DGBasis<11> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (14*x*(2*x - 1) + 1)*(x + 3*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            14*x*(2*x - 1) + (56*x - 14)*(x + 3*y - 1) + 1,
            42*x*(2*x - 1) + 3,
            0
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 12
template<>
struct DGBasis<12> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (8*x - 1)*(10*ipow<2>(y) + y*(8*x - 8) + ipow<2>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            80*ipow<2>(y) + 8*y*(8*x - 8) + 8*ipow<2>(x - 1) + (8*x - 1)*(2*x + 8*y - 2),
            (8*x - 1)*(8*x + 20*y - 8),
            0
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 13
template<>
struct DGBasis<13> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return 35*ipow<3>(y) + ipow<2>(y)*(45*x - 45) + 15*y*ipow<2>(x - 1) + ipow<3>(x - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            45*ipow<2>(y) + 15*y*(2*x - 2) + 3*ipow<2>(x - 1),
            105*ipow<2>(y) + 2*y*(45*x - 45) + 15*ipow<2>(x - 1),
            0
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 14
template<>
struct DGBasis<14> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (14*x*(2*x - 1) + 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            14*x*(2*x - 1) + (56*x - 14)*(x + y + 2*z - 1) + 1,
            14*x*(2*x - 1) + 1,
            28*x*(2*x - 1) + 2
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 15
template<>
struct DGBasis<15> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (8*x - 1)*(x + 5*y - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (8*x - 1)*(x + 5*y - 1) + (8*x - 1)*(x + y + 2*z - 1) + 8*(x + 5*y - 1)*(x + y + 2*z - 1),
            (8*x - 1)*(x + 5*y - 1) + 5*(8*x - 1)*(x + y + 2*z - 1),
            2*(8*x - 1)*(x + 5*y - 1)
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 16
template<>
struct DGBasis<16> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1))*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1) + (2*x + 12*y - 2)*(x + y + 2*z - 1),
            21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1) + (12*x + 42*y - 12)*(x + y + 2*z - 1),
            42*ipow<2>(y) + 2*y*(12*x - 12) + 2*ipow<2>(x - 1)
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 17
template<>
struct DGBasis<17> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (8*x - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            48*ipow<2>(z) + 8*z*(6*x + 6*y - 6) + (8*x - 1)*(2*x + 2*y + 6*z - 2) + 8*ipow<2>(x + y - 1),
            (8*x - 1)*(2*x + 2*y + 6*z - 2),
            (8*x - 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 18
template<>
struct DGBasis<18> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x + 7*y - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1) + (x + 7*y - 1)*(2*x + 2*y + 6*z - 2),
            42*ipow<2>(z) + 7*z*(6*x + 6*y - 6) + 7*ipow<2>(x + y - 1) + (x + 7*y - 1)*(2*x + 2*y + 6*z - 2),
            (x + 7*y - 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 19
template<>
struct DGBasis<19> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return 20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1),
            30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1),
            60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1)
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 20
template<>
struct DGBasis<20> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return 14*x*(3*x*(x*(5*x - 8) + 4) - 2) + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            42*x*(x*(5*x - 8) + 4) + 14*x*(3*x*(5*x - 8) + 3*x*(10*x - 8) + 12) - 28,
            0,
            0
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 21
template<>
struct DGBasis<21> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (12*x*(x*(10*x - 9) + 2) - 1)*(x + 3*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            12*x*(x*(10*x - 9) + 2) + (x + 3*y - 1)*(12*x*(10*x - 9) + 12*x*(20*x - 9) + 24) - 1,
            36*x*(x*(10*x - 9) + 2) - 3,
            0
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 22
template<>
struct DGBasis<22> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (9*x*(5*x - 2) + 1)*(10*ipow<2>(y) + y*(8*x - 8) + ipow<2>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (90*x - 18)*(10*ipow<2>(y) + y*(8*x - 8) + ipow<2>(x - 1)) + (9*x*(5*x - 2) + 1)*(2*x + 8*y - 2),
            (9*x*(5*x - 2) + 1)*(8*x + 20*y - 8),
            0
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 23
template<>
struct DGBasis<23> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (10*x - 1)*(35*ipow<3>(y) + ipow<2>(y)*(45*x - 45) + 15*y*ipow<2>(x - 1) + ipow<3>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            350*ipow<3>(y) + 10*ipow<2>(y)*(45*x - 45) + 150*y*ipow<2>(x - 1) + 10*ipow<3>(x - 1) + (10*x - 1)*(45*ipow<2>(y) + 15*y*(2*x - 2) + 3*ipow<2>(x - 1)),
            (10*x - 1)*(105*ipow<2>(y) + 2*y*(45*x - 45) + 15*ipow<2>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 24
template<>
struct DGBasis<24> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return 126*ipow<4>(y) + ipow<3>(y)*(224*x - 224) + 126*ipow<2>(y)*ipow<2>(x - 1) + 24*y*ipow<3>(x - 1) + ipow<4>(x - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            224*ipow<3>(y) + 126*ipow<2>(y)*(2*x - 2) + 72*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1),
            504*ipow<3>(y) + 3*ipow<2>(y)*(224*x - 224) + 252*y*ipow<2>(x - 1) + 24*ipow<3>(x - 1),
            0
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 25
template<>
struct DGBasis<25> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (12*x*(x*(10*x - 9) + 2) - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            12*x*(x*(10*x - 9) + 2) + (12*x*(10*x - 9) + 12*x*(20*x - 9) + 24)*(x + y + 2*z - 1) - 1,
            12*x*(x*(10*x - 9) + 2) - 1,
            24*x*(x*(10*x - 9) + 2) - 2
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 26
template<>
struct DGBasis<26> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (9*x*(5*x - 2) + 1)*(x + 5*y - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (90*x - 18)*(x + 5*y - 1)*(x + y + 2*z - 1) + (9*x*(5*x - 2) + 1)*(x + 5*y - 1) + (9*x*(5*x - 2) + 1)*(x + y + 2*z - 1),
            (9*x*(5*x - 2) + 1)*(x + 5*y - 1) + 5*(9*x*(5*x - 2) + 1)*(x + y + 2*z - 1),
            2*(9*x*(5*x - 2) + 1)*(x + 5*y - 1)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 27
template<>
struct DGBasis<27> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (10*x - 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1))*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (10*x - 1)*(2*x + 12*y - 2)*(x + y + 2*z - 1) + (10*x - 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1)) + 10*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1))*(x + y + 2*z - 1),
            (10*x - 1)*(12*x + 42*y - 12)*(x + y + 2*z - 1) + (10*x - 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1)),
            2*(10*x - 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1))
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 28
template<>
struct DGBasis<28> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x + y + 2*z - 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1) + (84*ipow<2>(y) + 21*y*(2*x - 2) + 3*ipow<2>(x - 1))*(x + y + 2*z - 1),
            84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1) + (252*ipow<2>(y) + 2*y*(84*x - 84) + 21*ipow<2>(x - 1))*(x + y + 2*z - 1),
            168*ipow<3>(y) + 2*ipow<2>(y)*(84*x - 84) + 42*y*ipow<2>(x - 1) + 2*ipow<3>(x - 1)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 29
template<>
struct DGBasis<29> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (9*x*(5*x - 2) + 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (90*x - 18)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (9*x*(5*x - 2) + 1)*(2*x + 2*y + 6*z - 2),
            (9*x*(5*x - 2) + 1)*(2*x + 2*y + 6*z - 2),
            (9*x*(5*x - 2) + 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 30
template<>
struct DGBasis<30> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (10*x - 1)*(x + 7*y - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (10*x - 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + (10*x - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + 10*(x + 7*y - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)),
            (10*x - 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + 7*(10*x - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)),
            (10*x - 1)*(x + 7*y - 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 31
template<>
struct DGBasis<31> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x + 16*y - 2)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(2*x + 2*y + 6*z - 2),
            (16*x + 72*y - 16)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(2*x + 2*y + 6*z - 2),
            (36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 32
template<>
struct DGBasis<32> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (10*x - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            200*ipow<3>(z) + 10*ipow<2>(z)*(30*x + 30*y - 30) + 120*z*ipow<2>(x + y - 1) + (10*x - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)) + 10*ipow<3>(x + y - 1),
            (10*x - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)),
            (10*x - 1)*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 33
template<>
struct DGBasis<33> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x + 9*y - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1) + (x + 9*y - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)),
            180*ipow<3>(z) + 9*ipow<2>(z)*(30*x + 30*y - 30) + 108*z*ipow<2>(x + y - 1) + 9*ipow<3>(x + y - 1) + (x + 9*y - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)),
            (x + 9*y - 1)*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 34
template<>
struct DGBasis<34> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return 70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1),
            140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1),
            280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 35
template<>
struct DGBasis<35> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return 2*x*(3*x*(x*(11*x*(12*x - 25) + 200) - 60) + 20) - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            6*x*(x*(11*x*(12*x - 25) + 200) - 60) + 2*x*(3*x*(11*x*(12*x - 25) + 200) + 3*x*(11*x*(12*x - 25) + x*(264*x - 275) + 200) - 180) + 40,
            0,
            0
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 36
template<>
struct DGBasis<36> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (3*x*(5*x*(11*x*(3*x - 4) + 18) - 12) + 1)*(x + 3*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            3*x*(5*x*(11*x*(3*x - 4) + 18) - 12) + (x + 3*y - 1)*(15*x*(11*x*(3*x - 4) + 18) + 3*x*(55*x*(3*x - 4) + 5*x*(66*x - 44) + 90) - 36) + 1,
            9*x*(5*x*(11*x*(3*x - 4) + 18) - 12) + 3,
            0
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 37
template<>
struct DGBasis<37> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (5*x*(11*x*(4*x - 3) + 6) - 1)*(10*ipow<2>(y) + y*(8*x - 8) + ipow<2>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (5*x*(11*x*(4*x - 3) + 6) - 1)*(2*x + 8*y - 2) + (10*ipow<2>(y) + y*(8*x - 8) + ipow<2>(x - 1))*(55*x*(4*x - 3) + 5*x*(88*x - 33) + 30),
            (5*x*(11*x*(4*x - 3) + 6) - 1)*(8*x + 20*y - 8),
            0
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 38
template<>
struct DGBasis<38> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (22*x*(3*x - 1) + 1)*(35*ipow<3>(y) + ipow<2>(y)*(45*x - 45) + 15*y*ipow<2>(x - 1) + ipow<3>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (132*x - 22)*(35*ipow<3>(y) + ipow<2>(y)*(45*x - 45) + 15*y*ipow<2>(x - 1) + ipow<3>(x - 1)) + (22*x*(3*x - 1) + 1)*(45*ipow<2>(y) + 15*y*(2*x - 2) + 3*ipow<2>(x - 1)),
            (22*x*(3*x - 1) + 1)*(105*ipow<2>(y) + 2*y*(45*x - 45) + 15*ipow<2>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 39
template<>
struct DGBasis<39> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (12*x - 1)*(126*ipow<4>(y) + ipow<3>(y)*(224*x - 224) + 126*ipow<2>(y)*ipow<2>(x - 1) + 24*y*ipow<3>(x - 1) + ipow<4>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            1512*ipow<4>(y) + 12*ipow<3>(y)*(224*x - 224) + 1512*ipow<2>(y)*ipow<2>(x - 1) + 288*y*ipow<3>(x - 1) + 12*ipow<4>(x - 1) + (12*x - 1)*(224*ipow<3>(y) + 126*ipow<2>(y)*(2*x - 2) + 72*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1)),
            (12*x - 1)*(504*ipow<3>(y) + 3*ipow<2>(y)*(224*x - 224) + 252*y*ipow<2>(x - 1) + 24*ipow<3>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 40
template<>
struct DGBasis<40> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return 462*ipow<5>(y) + ipow<4>(y)*(1050*x - 1050) + 840*ipow<3>(y)*ipow<2>(x - 1) + 280*ipow<2>(y)*ipow<3>(x - 1) + 35*y*ipow<4>(x - 1) + ipow<5>(x - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            1050*ipow<4>(y) + 840*ipow<3>(y)*(2*x - 2) + 840*ipow<2>(y)*ipow<2>(x - 1) + 140*y*ipow<3>(x - 1) + 5*ipow<4>(x - 1),
            2310*ipow<4>(y) + 4*ipow<3>(y)*(1050*x - 1050) + 2520*ipow<2>(y)*ipow<2>(x - 1) + 560*y*ipow<3>(x - 1) + 35*ipow<4>(x - 1),
            0
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 41
template<>
struct DGBasis<41> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (3*x*(5*x*(11*x*(3*x - 4) + 18) - 12) + 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            3*x*(5*x*(11*x*(3*x - 4) + 18) - 12) + (15*x*(11*x*(3*x - 4) + 18) + 3*x*(55*x*(3*x - 4) + 5*x*(66*x - 44) + 90) - 36)*(x + y + 2*z - 1) + 1,
            3*x*(5*x*(11*x*(3*x - 4) + 18) - 12) + 1,
            6*x*(5*x*(11*x*(3*x - 4) + 18) - 12) + 2
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 42
template<>
struct DGBasis<42> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (5*x*(11*x*(4*x - 3) + 6) - 1)*(x + 5*y - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (5*x*(11*x*(4*x - 3) + 6) - 1)*(x + 5*y - 1) + (5*x*(11*x*(4*x - 3) + 6) - 1)*(x + y + 2*z - 1) + (x + 5*y - 1)*(55*x*(4*x - 3) + 5*x*(88*x - 33) + 30)*(x + y + 2*z - 1),
            (5*x*(11*x*(4*x - 3) + 6) - 1)*(x + 5*y - 1) + 5*(5*x*(11*x*(4*x - 3) + 6) - 1)*(x + y + 2*z - 1),
            2*(5*x*(11*x*(4*x - 3) + 6) - 1)*(x + 5*y - 1)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 43
template<>
struct DGBasis<43> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (22*x*(3*x - 1) + 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1))*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (132*x - 22)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1))*(x + y + 2*z - 1) + (22*x*(3*x - 1) + 1)*(2*x + 12*y - 2)*(x + y + 2*z - 1) + (22*x*(3*x - 1) + 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1)),
            (22*x*(3*x - 1) + 1)*(12*x + 42*y - 12)*(x + y + 2*z - 1) + (22*x*(3*x - 1) + 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1)),
            2*(22*x*(3*x - 1) + 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1))
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 44
template<>
struct DGBasis<44> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (12*x - 1)*(x + y + 2*z - 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (12*x - 1)*(84*ipow<2>(y) + 21*y*(2*x - 2) + 3*ipow<2>(x - 1))*(x + y + 2*z - 1) + (12*x - 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1)) + 12*(x + y + 2*z - 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (12*x - 1)*(252*ipow<2>(y) + 2*y*(84*x - 84) + 21*ipow<2>(x - 1))*(x + y + 2*z - 1) + (12*x - 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            2*(12*x - 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1))
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 45
template<>
struct DGBasis<45> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x + y + 2*z - 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1) + (x + y + 2*z - 1)*(480*ipow<3>(y) + 216*ipow<2>(y)*(2*x - 2) + 96*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1)),
            330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1) + (x + y + 2*z - 1)*(1320*ipow<3>(y) + 3*ipow<2>(y)*(480*x - 480) + 432*y*ipow<2>(x - 1) + 32*ipow<3>(x - 1)),
            660*ipow<4>(y) + 2*ipow<3>(y)*(480*x - 480) + 432*ipow<2>(y)*ipow<2>(x - 1) + 64*y*ipow<3>(x - 1) + 2*ipow<4>(x - 1)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 46
template<>
struct DGBasis<46> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (5*x*(11*x*(4*x - 3) + 6) - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (5*x*(11*x*(4*x - 3) + 6) - 1)*(2*x + 2*y + 6*z - 2) + (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(55*x*(4*x - 3) + 5*x*(88*x - 33) + 30),
            (5*x*(11*x*(4*x - 3) + 6) - 1)*(2*x + 2*y + 6*z - 2),
            (5*x*(11*x*(4*x - 3) + 6) - 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 47
template<>
struct DGBasis<47> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (22*x*(3*x - 1) + 1)*(x + 7*y - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (132*x - 22)*(x + 7*y - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (22*x*(3*x - 1) + 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + (22*x*(3*x - 1) + 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)),
            (22*x*(3*x - 1) + 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + 7*(22*x*(3*x - 1) + 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)),
            (22*x*(3*x - 1) + 1)*(x + 7*y - 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 48
template<>
struct DGBasis<48> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (12*x - 1)*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (12*x - 1)*(2*x + 16*y - 2)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (12*x - 1)*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(2*x + 2*y + 6*z - 2) + 12*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)),
            (12*x - 1)*(16*x + 72*y - 16)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (12*x - 1)*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(2*x + 2*y + 6*z - 2),
            (12*x - 1)*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 49
template<>
struct DGBasis<49> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (135*ipow<2>(y) + 27*y*(2*x - 2) + 3*ipow<2>(x - 1))*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (2*x + 2*y + 6*z - 2)*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (495*ipow<2>(y) + 2*y*(135*x - 135) + 27*ipow<2>(x - 1))*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (2*x + 2*y + 6*z - 2)*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (6*x + 6*y + 12*z - 6)*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1))
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 50
template<>
struct DGBasis<50> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (22*x*(3*x - 1) + 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (132*x - 22)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (22*x*(3*x - 1) + 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)),
            (22*x*(3*x - 1) + 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)),
            (22*x*(3*x - 1) + 1)*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 51
template<>
struct DGBasis<51> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (12*x - 1)*(x + 9*y - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (12*x - 1)*(x + 9*y - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)) + (12*x - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + 12*(x + 9*y - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (12*x - 1)*(x + 9*y - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)) + 9*(12*x - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (12*x - 1)*(x + 9*y - 1)*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 52
template<>
struct DGBasis<52> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x + 20*y - 2)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)),
            (20*x + 110*y - 20)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)),
            (55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 53
template<>
struct DGBasis<53> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (12*x - 1)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            840*ipow<4>(z) + 12*ipow<3>(z)*(140*x + 140*y - 140) + 1080*ipow<2>(z)*ipow<2>(x + y - 1) + 240*z*ipow<3>(x + y - 1) + (12*x - 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)) + 12*ipow<4>(x + y - 1),
            (12*x - 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)),
            (12*x - 1)*(280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 54
template<>
struct DGBasis<54> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x + 11*y - 1)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1) + (x + 11*y - 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)),
            770*ipow<4>(z) + 11*ipow<3>(z)*(140*x + 140*y - 140) + 990*ipow<2>(z)*ipow<2>(x + y - 1) + 220*z*ipow<3>(x + y - 1) + 11*ipow<4>(x + y - 1) + (x + 11*y - 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)),
            (x + 11*y - 1)*(280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 55
template<>
struct DGBasis<55> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return 252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1),
            630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1),
            1260*ipow<4>(z) + 4*ipow<3>(z)*(630*x + 630*y - 630) + 1680*ipow<2>(z)*ipow<2>(x + y - 1) + 420*z*ipow<3>(x + y - 1) + 30*ipow<4>(x + y - 1)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 56
template<>
struct DGBasis<56> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return 3*x*(x*(11*x*(x*(13*x*(7*x - 18) + 225) - 100) + 225) - 18) + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            3*x*(11*x*(x*(13*x*(7*x - 18) + 225) - 100) + 225) + 3*x*(11*x*(x*(13*x*(7*x - 18) + 225) - 100) + x*(11*x*(13*x*(7*x - 18) + 225) + 11*x*(13*x*(7*x - 18) + x*(182*x - 234) + 225) - 1100) + 225) - 54,
            0,
            0
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 57
template<>
struct DGBasis<57> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x*(11*x*(x*(13*x*(14*x - 25) + 200) - 50) + 50) - 1)*(x + 3*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            x*(11*x*(x*(13*x*(14*x - 25) + 200) - 50) + 50) + (x + 3*y - 1)*(11*x*(x*(13*x*(14*x - 25) + 200) - 50) + x*(11*x*(13*x*(14*x - 25) + 200) + 11*x*(13*x*(14*x - 25) + x*(364*x - 325) + 200) - 550) + 50) - 1,
            3*x*(11*x*(x*(13*x*(14*x - 25) + 200) - 50) + 50) - 3,
            0
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 58
template<>
struct DGBasis<58> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (11*x*(x*(13*x*(7*x - 8) + 36) - 4) + 1)*(10*ipow<2>(y) + y*(8*x - 8) + ipow<2>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (11*x*(x*(13*x*(7*x - 8) + 36) - 4) + 1)*(2*x + 8*y - 2) + (10*ipow<2>(y) + y*(8*x - 8) + ipow<2>(x - 1))*(11*x*(13*x*(7*x - 8) + 36) + 11*x*(13*x*(7*x - 8) + x*(182*x - 104) + 36) - 44),
            (11*x*(x*(13*x*(7*x - 8) + 36) - 4) + 1)*(8*x + 20*y - 8),
            0
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 59
template<>
struct DGBasis<59> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (2*x*(13*x*(14*x - 9) + 18) - 1)*(35*ipow<3>(y) + ipow<2>(y)*(45*x - 45) + 15*y*ipow<2>(x - 1) + ipow<3>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x*(13*x*(14*x - 9) + 18) - 1)*(45*ipow<2>(y) + 15*y*(2*x - 2) + 3*ipow<2>(x - 1)) + (26*x*(14*x - 9) + 2*x*(364*x - 117) + 36)*(35*ipow<3>(y) + ipow<2>(y)*(45*x - 45) + 15*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (2*x*(13*x*(14*x - 9) + 18) - 1)*(105*ipow<2>(y) + 2*y*(45*x - 45) + 15*ipow<2>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 60
template<>
struct DGBasis<60> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (13*x*(7*x - 2) + 1)*(126*ipow<4>(y) + ipow<3>(y)*(224*x - 224) + 126*ipow<2>(y)*ipow<2>(x - 1) + 24*y*ipow<3>(x - 1) + ipow<4>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (182*x - 26)*(126*ipow<4>(y) + ipow<3>(y)*(224*x - 224) + 126*ipow<2>(y)*ipow<2>(x - 1) + 24*y*ipow<3>(x - 1) + ipow<4>(x - 1)) + (13*x*(7*x - 2) + 1)*(224*ipow<3>(y) + 126*ipow<2>(y)*(2*x - 2) + 72*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1)),
            (13*x*(7*x - 2) + 1)*(504*ipow<3>(y) + 3*ipow<2>(y)*(224*x - 224) + 252*y*ipow<2>(x - 1) + 24*ipow<3>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 61
template<>
struct DGBasis<61> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (14*x - 1)*(462*ipow<5>(y) + ipow<4>(y)*(1050*x - 1050) + 840*ipow<3>(y)*ipow<2>(x - 1) + 280*ipow<2>(y)*ipow<3>(x - 1) + 35*y*ipow<4>(x - 1) + ipow<5>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            6468*ipow<5>(y) + 14*ipow<4>(y)*(1050*x - 1050) + 11760*ipow<3>(y)*ipow<2>(x - 1) + 3920*ipow<2>(y)*ipow<3>(x - 1) + 490*y*ipow<4>(x - 1) + 14*ipow<5>(x - 1) + (14*x - 1)*(1050*ipow<4>(y) + 840*ipow<3>(y)*(2*x - 2) + 840*ipow<2>(y)*ipow<2>(x - 1) + 140*y*ipow<3>(x - 1) + 5*ipow<4>(x - 1)),
            (14*x - 1)*(2310*ipow<4>(y) + 4*ipow<3>(y)*(1050*x - 1050) + 2520*ipow<2>(y)*ipow<2>(x - 1) + 560*y*ipow<3>(x - 1) + 35*ipow<4>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 62
template<>
struct DGBasis<62> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return 1716*ipow<6>(y) + ipow<5>(y)*(4752*x - 4752) + 4950*ipow<4>(y)*ipow<2>(x - 1) + 2400*ipow<3>(y)*ipow<3>(x - 1) + 540*ipow<2>(y)*ipow<4>(x - 1) + 48*y*ipow<5>(x - 1) + ipow<6>(x - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            4752*ipow<5>(y) + 4950*ipow<4>(y)*(2*x - 2) + 7200*ipow<3>(y)*ipow<2>(x - 1) + 2160*ipow<2>(y)*ipow<3>(x - 1) + 240*y*ipow<4>(x - 1) + 6*ipow<5>(x - 1),
            10296*ipow<5>(y) + 5*ipow<4>(y)*(4752*x - 4752) + 19800*ipow<3>(y)*ipow<2>(x - 1) + 7200*ipow<2>(y)*ipow<3>(x - 1) + 1080*y*ipow<4>(x - 1) + 48*ipow<5>(x - 1),
            0
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 63
template<>
struct DGBasis<63> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x*(11*x*(x*(13*x*(14*x - 25) + 200) - 50) + 50) - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            x*(11*x*(x*(13*x*(14*x - 25) + 200) - 50) + 50) + (11*x*(x*(13*x*(14*x - 25) + 200) - 50) + x*(11*x*(13*x*(14*x - 25) + 200) + 11*x*(13*x*(14*x - 25) + x*(364*x - 325) + 200) - 550) + 50)*(x + y + 2*z - 1) - 1,
            x*(11*x*(x*(13*x*(14*x - 25) + 200) - 50) + 50) - 1,
            2*x*(11*x*(x*(13*x*(14*x - 25) + 200) - 50) + 50) - 2
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 64
template<>
struct DGBasis<64> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (11*x*(x*(13*x*(7*x - 8) + 36) - 4) + 1)*(x + 5*y - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (11*x*(x*(13*x*(7*x - 8) + 36) - 4) + 1)*(x + 5*y - 1) + (11*x*(x*(13*x*(7*x - 8) + 36) - 4) + 1)*(x + y + 2*z - 1) + (x + 5*y - 1)*(11*x*(13*x*(7*x - 8) + 36) + 11*x*(13*x*(7*x - 8) + x*(182*x - 104) + 36) - 44)*(x + y + 2*z - 1),
            (11*x*(x*(13*x*(7*x - 8) + 36) - 4) + 1)*(x + 5*y - 1) + 5*(11*x*(x*(13*x*(7*x - 8) + 36) - 4) + 1)*(x + y + 2*z - 1),
            2*(11*x*(x*(13*x*(7*x - 8) + 36) - 4) + 1)*(x + 5*y - 1)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 65
template<>
struct DGBasis<65> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (2*x*(13*x*(14*x - 9) + 18) - 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1))*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x*(13*x*(14*x - 9) + 18) - 1)*(2*x + 12*y - 2)*(x + y + 2*z - 1) + (2*x*(13*x*(14*x - 9) + 18) - 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1)) + (21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1))*(26*x*(14*x - 9) + 2*x*(364*x - 117) + 36)*(x + y + 2*z - 1),
            (2*x*(13*x*(14*x - 9) + 18) - 1)*(12*x + 42*y - 12)*(x + y + 2*z - 1) + (2*x*(13*x*(14*x - 9) + 18) - 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1)),
            2*(2*x*(13*x*(14*x - 9) + 18) - 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 66
template<>
struct DGBasis<66> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (13*x*(7*x - 2) + 1)*(x + y + 2*z - 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (182*x - 26)*(x + y + 2*z - 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1)) + (13*x*(7*x - 2) + 1)*(84*ipow<2>(y) + 21*y*(2*x - 2) + 3*ipow<2>(x - 1))*(x + y + 2*z - 1) + (13*x*(7*x - 2) + 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (13*x*(7*x - 2) + 1)*(252*ipow<2>(y) + 2*y*(84*x - 84) + 21*ipow<2>(x - 1))*(x + y + 2*z - 1) + (13*x*(7*x - 2) + 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            2*(13*x*(7*x - 2) + 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 67
template<>
struct DGBasis<67> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (14*x - 1)*(x + y + 2*z - 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (14*x - 1)*(x + y + 2*z - 1)*(480*ipow<3>(y) + 216*ipow<2>(y)*(2*x - 2) + 96*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1)) + (14*x - 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1)) + 14*(x + y + 2*z - 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (14*x - 1)*(x + y + 2*z - 1)*(1320*ipow<3>(y) + 3*ipow<2>(y)*(480*x - 480) + 432*y*ipow<2>(x - 1) + 32*ipow<3>(x - 1)) + (14*x - 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            2*(14*x - 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 68
template<>
struct DGBasis<68> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x + y + 2*z - 1)*(1287*ipow<5>(y) + ipow<4>(y)*(2475*x - 2475) + 1650*ipow<3>(y)*ipow<2>(x - 1) + 450*ipow<2>(y)*ipow<3>(x - 1) + 45*y*ipow<4>(x - 1) + ipow<5>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            1287*ipow<5>(y) + ipow<4>(y)*(2475*x - 2475) + 1650*ipow<3>(y)*ipow<2>(x - 1) + 450*ipow<2>(y)*ipow<3>(x - 1) + 45*y*ipow<4>(x - 1) + ipow<5>(x - 1) + (x + y + 2*z - 1)*(2475*ipow<4>(y) + 1650*ipow<3>(y)*(2*x - 2) + 1350*ipow<2>(y)*ipow<2>(x - 1) + 180*y*ipow<3>(x - 1) + 5*ipow<4>(x - 1)),
            1287*ipow<5>(y) + ipow<4>(y)*(2475*x - 2475) + 1650*ipow<3>(y)*ipow<2>(x - 1) + 450*ipow<2>(y)*ipow<3>(x - 1) + 45*y*ipow<4>(x - 1) + ipow<5>(x - 1) + (x + y + 2*z - 1)*(6435*ipow<4>(y) + 4*ipow<3>(y)*(2475*x - 2475) + 4950*ipow<2>(y)*ipow<2>(x - 1) + 900*y*ipow<3>(x - 1) + 45*ipow<4>(x - 1)),
            2574*ipow<5>(y) + 2*ipow<4>(y)*(2475*x - 2475) + 3300*ipow<3>(y)*ipow<2>(x - 1) + 900*ipow<2>(y)*ipow<3>(x - 1) + 90*y*ipow<4>(x - 1) + 2*ipow<5>(x - 1)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 69
template<>
struct DGBasis<69> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (11*x*(x*(13*x*(7*x - 8) + 36) - 4) + 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (11*x*(x*(13*x*(7*x - 8) + 36) - 4) + 1)*(2*x + 2*y + 6*z - 2) + (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(11*x*(13*x*(7*x - 8) + 36) + 11*x*(13*x*(7*x - 8) + x*(182*x - 104) + 36) - 44),
            (11*x*(x*(13*x*(7*x - 8) + 36) - 4) + 1)*(2*x + 2*y + 6*z - 2),
            (11*x*(x*(13*x*(7*x - 8) + 36) - 4) + 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 70
template<>
struct DGBasis<70> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (2*x*(13*x*(14*x - 9) + 18) - 1)*(x + 7*y - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x*(13*x*(14*x - 9) + 18) - 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + (2*x*(13*x*(14*x - 9) + 18) - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (x + 7*y - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(26*x*(14*x - 9) + 2*x*(364*x - 117) + 36),
            (2*x*(13*x*(14*x - 9) + 18) - 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + 7*(2*x*(13*x*(14*x - 9) + 18) - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)),
            (2*x*(13*x*(14*x - 9) + 18) - 1)*(x + 7*y - 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 71
template<>
struct DGBasis<71> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (13*x*(7*x - 2) + 1)*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (182*x - 26)*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (13*x*(7*x - 2) + 1)*(2*x + 16*y - 2)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (13*x*(7*x - 2) + 1)*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(2*x + 2*y + 6*z - 2),
            (13*x*(7*x - 2) + 1)*(16*x + 72*y - 16)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (13*x*(7*x - 2) + 1)*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(2*x + 2*y + 6*z - 2),
            (13*x*(7*x - 2) + 1)*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 72
template<>
struct DGBasis<72> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (14*x - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (14*x - 1)*(135*ipow<2>(y) + 27*y*(2*x - 2) + 3*ipow<2>(x - 1))*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (14*x - 1)*(2*x + 2*y + 6*z - 2)*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1)) + 14*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (14*x - 1)*(495*ipow<2>(y) + 2*y*(135*x - 135) + 27*ipow<2>(x - 1))*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (14*x - 1)*(2*x + 2*y + 6*z - 2)*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (14*x - 1)*(6*x + 6*y + 12*z - 6)*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 73
template<>
struct DGBasis<73> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(715*ipow<4>(y) + ipow<3>(y)*(880*x - 880) + 330*ipow<2>(y)*ipow<2>(x - 1) + 40*y*ipow<3>(x - 1) + ipow<4>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(880*ipow<3>(y) + 330*ipow<2>(y)*(2*x - 2) + 120*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1)) + (2*x + 2*y + 6*z - 2)*(715*ipow<4>(y) + ipow<3>(y)*(880*x - 880) + 330*ipow<2>(y)*ipow<2>(x - 1) + 40*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(2860*ipow<3>(y) + 3*ipow<2>(y)*(880*x - 880) + 660*y*ipow<2>(x - 1) + 40*ipow<3>(x - 1)) + (2*x + 2*y + 6*z - 2)*(715*ipow<4>(y) + ipow<3>(y)*(880*x - 880) + 330*ipow<2>(y)*ipow<2>(x - 1) + 40*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (6*x + 6*y + 12*z - 6)*(715*ipow<4>(y) + ipow<3>(y)*(880*x - 880) + 330*ipow<2>(y)*ipow<2>(x - 1) + 40*y*ipow<3>(x - 1) + ipow<4>(x - 1))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 74
template<>
struct DGBasis<74> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (2*x*(13*x*(14*x - 9) + 18) - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x*(13*x*(14*x - 9) + 18) - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)) + (26*x*(14*x - 9) + 2*x*(364*x - 117) + 36)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (2*x*(13*x*(14*x - 9) + 18) - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)),
            (2*x*(13*x*(14*x - 9) + 18) - 1)*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 75
template<>
struct DGBasis<75> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (13*x*(7*x - 2) + 1)*(x + 9*y - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (182*x - 26)*(x + 9*y - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (13*x*(7*x - 2) + 1)*(x + 9*y - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)) + (13*x*(7*x - 2) + 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (13*x*(7*x - 2) + 1)*(x + 9*y - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)) + 9*(13*x*(7*x - 2) + 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (13*x*(7*x - 2) + 1)*(x + 9*y - 1)*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 76
template<>
struct DGBasis<76> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (14*x - 1)*(55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (14*x - 1)*(2*x + 20*y - 2)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (14*x - 1)*(55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)) + 14*(55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (14*x - 1)*(20*x + 110*y - 20)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (14*x - 1)*(55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)),
            (14*x - 1)*(55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 77
template<>
struct DGBasis<77> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (286*ipow<3>(y) + ipow<2>(y)*(198*x - 198) + 33*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (198*ipow<2>(y) + 33*y*(2*x - 2) + 3*ipow<2>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(286*ipow<3>(y) + ipow<2>(y)*(198*x - 198) + 33*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (858*ipow<2>(y) + 2*y*(198*x - 198) + 33*ipow<2>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(286*ipow<3>(y) + ipow<2>(y)*(198*x - 198) + 33*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))*(286*ipow<3>(y) + ipow<2>(y)*(198*x - 198) + 33*y*ipow<2>(x - 1) + ipow<3>(x - 1))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 78
template<>
struct DGBasis<78> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (13*x*(7*x - 2) + 1)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (182*x - 26)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (13*x*(7*x - 2) + 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)),
            (13*x*(7*x - 2) + 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)),
            (13*x*(7*x - 2) + 1)*(280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 79
template<>
struct DGBasis<79> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (14*x - 1)*(x + 11*y - 1)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (14*x - 1)*(x + 11*y - 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)) + (14*x - 1)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + 14*(x + 11*y - 1)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)),
            (14*x - 1)*(x + 11*y - 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)) + 11*(14*x - 1)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)),
            (14*x - 1)*(x + 11*y - 1)*(280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 80
template<>
struct DGBasis<80> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (78*ipow<2>(y) + y*(24*x - 24) + ipow<2>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x + 24*y - 2)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (78*ipow<2>(y) + y*(24*x - 24) + ipow<2>(x - 1))*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)),
            (24*x + 156*y - 24)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (78*ipow<2>(y) + y*(24*x - 24) + ipow<2>(x - 1))*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)),
            (78*ipow<2>(y) + y*(24*x - 24) + ipow<2>(x - 1))*(280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 81
template<>
struct DGBasis<81> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (14*x - 1)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            3528*ipow<5>(z) + 14*ipow<4>(z)*(630*x + 630*y - 630) + 7840*ipow<3>(z)*ipow<2>(x + y - 1) + 2940*ipow<2>(z)*ipow<3>(x + y - 1) + 420*z*ipow<4>(x + y - 1) + (14*x - 1)*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)) + 14*ipow<5>(x + y - 1),
            (14*x - 1)*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)),
            (14*x - 1)*(1260*ipow<4>(z) + 4*ipow<3>(z)*(630*x + 630*y - 630) + 1680*ipow<2>(z)*ipow<2>(x + y - 1) + 420*z*ipow<3>(x + y - 1) + 30*ipow<4>(x + y - 1))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 82
template<>
struct DGBasis<82> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x + 13*y - 1)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1) + (x + 13*y - 1)*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)),
            3276*ipow<5>(z) + 13*ipow<4>(z)*(630*x + 630*y - 630) + 7280*ipow<3>(z)*ipow<2>(x + y - 1) + 2730*ipow<2>(z)*ipow<3>(x + y - 1) + 390*z*ipow<4>(x + y - 1) + 13*ipow<5>(x + y - 1) + (x + 13*y - 1)*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)),
            (x + 13*y - 1)*(1260*ipow<4>(z) + 4*ipow<3>(z)*(630*x + 630*y - 630) + 1680*ipow<2>(z)*ipow<2>(x + y - 1) + 420*z*ipow<3>(x + y - 1) + 30*ipow<4>(x + y - 1))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 83
template<>
struct DGBasis<83> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return 924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1),
            2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1),
            5544*ipow<5>(z) + 5*ipow<4>(z)*(2772*x + 2772*y - 2772) + 12600*ipow<3>(z)*ipow<2>(x + y - 1) + 5040*ipow<2>(z)*ipow<3>(x + y - 1) + 840*z*ipow<4>(x + y - 1) + 42*ipow<5>(x + y - 1)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 84
template<>
struct DGBasis<84> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return x*(11*x*(x*(13*x*(x*(5*x*(16*x - 49) + 294) - 175) + 700) - 105) + 70) - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            11*x*(x*(13*x*(x*(5*x*(16*x - 49) + 294) - 175) + 700) - 105) + x*(11*x*(13*x*(x*(5*x*(16*x - 49) + 294) - 175) + 700) + 11*x*(13*x*(x*(5*x*(16*x - 49) + 294) - 175) + x*(13*x*(5*x*(16*x - 49) + 294) + 13*x*(5*x*(16*x - 49) + x*(160*x - 245) + 294) - 2275) + 700) - 1155) + 70,
            0,
            0
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 85
template<>
struct DGBasis<85> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (11*x*(2*x - 1)*(13*x*(7*x*(x*(4*x - 7) + 4) - 6) + 6) + 1)*(x + 3*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            11*x*(2*x - 1)*(13*x*(7*x*(x*(4*x - 7) + 4) - 6) + 6) + (x + 3*y - 1)*(11*x*(2*x - 1)*(91*x*(x*(4*x - 7) + 4) + 13*x*(7*x*(4*x - 7) + 7*x*(8*x - 7) + 28) - 78) + 22*x*(13*x*(7*x*(x*(4*x - 7) + 4) - 6) + 6) + 11*(2*x - 1)*(13*x*(7*x*(x*(4*x - 7) + 4) - 6) + 6)) + 1,
            33*x*(2*x - 1)*(13*x*(7*x*(x*(4*x - 7) + 4) - 6) + 6) + 3,
            0
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 86
template<>
struct DGBasis<86> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x*(13*x*(7*x*(48*ipow<2>(x) - 75*x + 40) - 60) + 60) - 1)*(10*ipow<2>(y) + y*(8*x - 8) + ipow<2>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (x*(13*x*(7*x*(48*ipow<2>(x) - 75*x + 40) - 60) + 60) - 1)*(2*x + 8*y - 2) + (10*ipow<2>(y) + y*(8*x - 8) + ipow<2>(x - 1))*(13*x*(7*x*(48*ipow<2>(x) - 75*x + 40) - 60) + x*(91*x*(48*ipow<2>(x) - 75*x + 40) + 13*x*(336*ipow<2>(x) + 7*x*(96*x - 75) - 525*x + 280) - 780) + 60),
            (x*(13*x*(7*x*(48*ipow<2>(x) - 75*x + 40) - 60) + 60) - 1)*(8*x + 20*y - 8),
            0
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 87
template<>
struct DGBasis<87> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(35*ipow<3>(y) + ipow<2>(y)*(45*x - 45) + 15*y*ipow<2>(x - 1) + ipow<3>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(45*ipow<2>(y) + 15*y*(2*x - 2) + 3*ipow<2>(x - 1)) + (182*x*(x*(10*x - 10) + 3) + 26*x*(7*x*(10*x - 10) + 7*x*(20*x - 10) + 21) - 52)*(35*ipow<3>(y) + ipow<2>(y)*(45*x - 45) + 15*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(105*ipow<2>(y) + 2*y*(45*x - 45) + 15*ipow<2>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 88
template<>
struct DGBasis<88> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (7*x*(5*x*(16*x - 9) + 6) - 1)*(126*ipow<4>(y) + ipow<3>(y)*(224*x - 224) + 126*ipow<2>(y)*ipow<2>(x - 1) + 24*y*ipow<3>(x - 1) + ipow<4>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(224*ipow<3>(y) + 126*ipow<2>(y)*(2*x - 2) + 72*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1)) + (35*x*(16*x - 9) + 7*x*(160*x - 45) + 42)*(126*ipow<4>(y) + ipow<3>(y)*(224*x - 224) + 126*ipow<2>(y)*ipow<2>(x - 1) + 24*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(504*ipow<3>(y) + 3*ipow<2>(y)*(224*x - 224) + 252*y*ipow<2>(x - 1) + 24*ipow<3>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 89
template<>
struct DGBasis<89> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (30*x*(4*x - 1) + 1)*(462*ipow<5>(y) + ipow<4>(y)*(1050*x - 1050) + 840*ipow<3>(y)*ipow<2>(x - 1) + 280*ipow<2>(y)*ipow<3>(x - 1) + 35*y*ipow<4>(x - 1) + ipow<5>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (240*x - 30)*(462*ipow<5>(y) + ipow<4>(y)*(1050*x - 1050) + 840*ipow<3>(y)*ipow<2>(x - 1) + 280*ipow<2>(y)*ipow<3>(x - 1) + 35*y*ipow<4>(x - 1) + ipow<5>(x - 1)) + (30*x*(4*x - 1) + 1)*(1050*ipow<4>(y) + 840*ipow<3>(y)*(2*x - 2) + 840*ipow<2>(y)*ipow<2>(x - 1) + 140*y*ipow<3>(x - 1) + 5*ipow<4>(x - 1)),
            (30*x*(4*x - 1) + 1)*(2310*ipow<4>(y) + 4*ipow<3>(y)*(1050*x - 1050) + 2520*ipow<2>(y)*ipow<2>(x - 1) + 560*y*ipow<3>(x - 1) + 35*ipow<4>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 90
template<>
struct DGBasis<90> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (16*x - 1)*(1716*ipow<6>(y) + ipow<5>(y)*(4752*x - 4752) + 4950*ipow<4>(y)*ipow<2>(x - 1) + 2400*ipow<3>(y)*ipow<3>(x - 1) + 540*ipow<2>(y)*ipow<4>(x - 1) + 48*y*ipow<5>(x - 1) + ipow<6>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            27456*ipow<6>(y) + 16*ipow<5>(y)*(4752*x - 4752) + 79200*ipow<4>(y)*ipow<2>(x - 1) + 38400*ipow<3>(y)*ipow<3>(x - 1) + 8640*ipow<2>(y)*ipow<4>(x - 1) + 768*y*ipow<5>(x - 1) + 16*ipow<6>(x - 1) + (16*x - 1)*(4752*ipow<5>(y) + 4950*ipow<4>(y)*(2*x - 2) + 7200*ipow<3>(y)*ipow<2>(x - 1) + 2160*ipow<2>(y)*ipow<3>(x - 1) + 240*y*ipow<4>(x - 1) + 6*ipow<5>(x - 1)),
            (16*x - 1)*(10296*ipow<5>(y) + 5*ipow<4>(y)*(4752*x - 4752) + 19800*ipow<3>(y)*ipow<2>(x - 1) + 7200*ipow<2>(y)*ipow<3>(x - 1) + 1080*y*ipow<4>(x - 1) + 48*ipow<5>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 91
template<>
struct DGBasis<91> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return 6435*ipow<7>(y) + ipow<6>(y)*(21021*x - 21021) + 27027*ipow<5>(y)*ipow<2>(x - 1) + 17325*ipow<4>(y)*ipow<3>(x - 1) + 5775*ipow<3>(y)*ipow<4>(x - 1) + 945*ipow<2>(y)*ipow<5>(x - 1) + 63*y*ipow<6>(x - 1) + ipow<7>(x - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            21021*ipow<6>(y) + 27027*ipow<5>(y)*(2*x - 2) + 51975*ipow<4>(y)*ipow<2>(x - 1) + 23100*ipow<3>(y)*ipow<3>(x - 1) + 4725*ipow<2>(y)*ipow<4>(x - 1) + 378*y*ipow<5>(x - 1) + 7*ipow<6>(x - 1),
            45045*ipow<6>(y) + 6*ipow<5>(y)*(21021*x - 21021) + 135135*ipow<4>(y)*ipow<2>(x - 1) + 69300*ipow<3>(y)*ipow<3>(x - 1) + 17325*ipow<2>(y)*ipow<4>(x - 1) + 1890*y*ipow<5>(x - 1) + 63*ipow<6>(x - 1),
            0
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 92
template<>
struct DGBasis<92> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (11*x*(2*x - 1)*(13*x*(7*x*(x*(4*x - 7) + 4) - 6) + 6) + 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            11*x*(2*x - 1)*(13*x*(7*x*(x*(4*x - 7) + 4) - 6) + 6) + (11*x*(2*x - 1)*(91*x*(x*(4*x - 7) + 4) + 13*x*(7*x*(4*x - 7) + 7*x*(8*x - 7) + 28) - 78) + 22*x*(13*x*(7*x*(x*(4*x - 7) + 4) - 6) + 6) + 11*(2*x - 1)*(13*x*(7*x*(x*(4*x - 7) + 4) - 6) + 6))*(x + y + 2*z - 1) + 1,
            11*x*(2*x - 1)*(13*x*(7*x*(x*(4*x - 7) + 4) - 6) + 6) + 1,
            22*x*(2*x - 1)*(13*x*(7*x*(x*(4*x - 7) + 4) - 6) + 6) + 2
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 93
template<>
struct DGBasis<93> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x*(13*x*(7*x*(48*ipow<2>(x) - 75*x + 40) - 60) + 60) - 1)*(x + 5*y - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (x*(13*x*(7*x*(48*ipow<2>(x) - 75*x + 40) - 60) + 60) - 1)*(x + 5*y - 1) + (x*(13*x*(7*x*(48*ipow<2>(x) - 75*x + 40) - 60) + 60) - 1)*(x + y + 2*z - 1) + (x + 5*y - 1)*(13*x*(7*x*(48*ipow<2>(x) - 75*x + 40) - 60) + x*(91*x*(48*ipow<2>(x) - 75*x + 40) + 13*x*(336*ipow<2>(x) + 7*x*(96*x - 75) - 525*x + 280) - 780) + 60)*(x + y + 2*z - 1),
            (x*(13*x*(7*x*(48*ipow<2>(x) - 75*x + 40) - 60) + 60) - 1)*(x + 5*y - 1) + 5*(x*(13*x*(7*x*(48*ipow<2>(x) - 75*x + 40) - 60) + 60) - 1)*(x + y + 2*z - 1),
            2*(x*(13*x*(7*x*(48*ipow<2>(x) - 75*x + 40) - 60) + 60) - 1)*(x + 5*y - 1)
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 94
template<>
struct DGBasis<94> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1))*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(2*x + 12*y - 2)*(x + y + 2*z - 1) + (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1)) + (21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1))*(182*x*(x*(10*x - 10) + 3) + 26*x*(7*x*(10*x - 10) + 7*x*(20*x - 10) + 21) - 52)*(x + y + 2*z - 1),
            (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(12*x + 42*y - 12)*(x + y + 2*z - 1) + (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1)),
            2*(26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 95
template<>
struct DGBasis<95> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (7*x*(5*x*(16*x - 9) + 6) - 1)*(x + y + 2*z - 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(84*ipow<2>(y) + 21*y*(2*x - 2) + 3*ipow<2>(x - 1))*(x + y + 2*z - 1) + (7*x*(5*x*(16*x - 9) + 6) - 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1)) + (35*x*(16*x - 9) + 7*x*(160*x - 45) + 42)*(x + y + 2*z - 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(252*ipow<2>(y) + 2*y*(84*x - 84) + 21*ipow<2>(x - 1))*(x + y + 2*z - 1) + (7*x*(5*x*(16*x - 9) + 6) - 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            2*(7*x*(5*x*(16*x - 9) + 6) - 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 96
template<>
struct DGBasis<96> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (30*x*(4*x - 1) + 1)*(x + y + 2*z - 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (240*x - 30)*(x + y + 2*z - 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1)) + (30*x*(4*x - 1) + 1)*(x + y + 2*z - 1)*(480*ipow<3>(y) + 216*ipow<2>(y)*(2*x - 2) + 96*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1)) + (30*x*(4*x - 1) + 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (30*x*(4*x - 1) + 1)*(x + y + 2*z - 1)*(1320*ipow<3>(y) + 3*ipow<2>(y)*(480*x - 480) + 432*y*ipow<2>(x - 1) + 32*ipow<3>(x - 1)) + (30*x*(4*x - 1) + 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            2*(30*x*(4*x - 1) + 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 97
template<>
struct DGBasis<97> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (16*x - 1)*(x + y + 2*z - 1)*(1287*ipow<5>(y) + ipow<4>(y)*(2475*x - 2475) + 1650*ipow<3>(y)*ipow<2>(x - 1) + 450*ipow<2>(y)*ipow<3>(x - 1) + 45*y*ipow<4>(x - 1) + ipow<5>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (16*x - 1)*(x + y + 2*z - 1)*(2475*ipow<4>(y) + 1650*ipow<3>(y)*(2*x - 2) + 1350*ipow<2>(y)*ipow<2>(x - 1) + 180*y*ipow<3>(x - 1) + 5*ipow<4>(x - 1)) + (16*x - 1)*(1287*ipow<5>(y) + ipow<4>(y)*(2475*x - 2475) + 1650*ipow<3>(y)*ipow<2>(x - 1) + 450*ipow<2>(y)*ipow<3>(x - 1) + 45*y*ipow<4>(x - 1) + ipow<5>(x - 1)) + 16*(x + y + 2*z - 1)*(1287*ipow<5>(y) + ipow<4>(y)*(2475*x - 2475) + 1650*ipow<3>(y)*ipow<2>(x - 1) + 450*ipow<2>(y)*ipow<3>(x - 1) + 45*y*ipow<4>(x - 1) + ipow<5>(x - 1)),
            (16*x - 1)*(x + y + 2*z - 1)*(6435*ipow<4>(y) + 4*ipow<3>(y)*(2475*x - 2475) + 4950*ipow<2>(y)*ipow<2>(x - 1) + 900*y*ipow<3>(x - 1) + 45*ipow<4>(x - 1)) + (16*x - 1)*(1287*ipow<5>(y) + ipow<4>(y)*(2475*x - 2475) + 1650*ipow<3>(y)*ipow<2>(x - 1) + 450*ipow<2>(y)*ipow<3>(x - 1) + 45*y*ipow<4>(x - 1) + ipow<5>(x - 1)),
            2*(16*x - 1)*(1287*ipow<5>(y) + ipow<4>(y)*(2475*x - 2475) + 1650*ipow<3>(y)*ipow<2>(x - 1) + 450*ipow<2>(y)*ipow<3>(x - 1) + 45*y*ipow<4>(x - 1) + ipow<5>(x - 1))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 98
template<>
struct DGBasis<98> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x + y + 2*z - 1)*(5005*ipow<6>(y) + ipow<5>(y)*(12012*x - 12012) + 10725*ipow<4>(y)*ipow<2>(x - 1) + 4400*ipow<3>(y)*ipow<3>(x - 1) + 825*ipow<2>(y)*ipow<4>(x - 1) + 60*y*ipow<5>(x - 1) + ipow<6>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            5005*ipow<6>(y) + ipow<5>(y)*(12012*x - 12012) + 10725*ipow<4>(y)*ipow<2>(x - 1) + 4400*ipow<3>(y)*ipow<3>(x - 1) + 825*ipow<2>(y)*ipow<4>(x - 1) + 60*y*ipow<5>(x - 1) + ipow<6>(x - 1) + (x + y + 2*z - 1)*(12012*ipow<5>(y) + 10725*ipow<4>(y)*(2*x - 2) + 13200*ipow<3>(y)*ipow<2>(x - 1) + 3300*ipow<2>(y)*ipow<3>(x - 1) + 300*y*ipow<4>(x - 1) + 6*ipow<5>(x - 1)),
            5005*ipow<6>(y) + ipow<5>(y)*(12012*x - 12012) + 10725*ipow<4>(y)*ipow<2>(x - 1) + 4400*ipow<3>(y)*ipow<3>(x - 1) + 825*ipow<2>(y)*ipow<4>(x - 1) + 60*y*ipow<5>(x - 1) + ipow<6>(x - 1) + (x + y + 2*z - 1)*(30030*ipow<5>(y) + 5*ipow<4>(y)*(12012*x - 12012) + 42900*ipow<3>(y)*ipow<2>(x - 1) + 13200*ipow<2>(y)*ipow<3>(x - 1) + 1650*y*ipow<4>(x - 1) + 60*ipow<5>(x - 1)),
            10010*ipow<6>(y) + 2*ipow<5>(y)*(12012*x - 12012) + 21450*ipow<4>(y)*ipow<2>(x - 1) + 8800*ipow<3>(y)*ipow<3>(x - 1) + 1650*ipow<2>(y)*ipow<4>(x - 1) + 120*y*ipow<5>(x - 1) + 2*ipow<6>(x - 1)
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 99
template<>
struct DGBasis<99> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x*(13*x*(7*x*(48*ipow<2>(x) - 75*x + 40) - 60) + 60) - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (x*(13*x*(7*x*(48*ipow<2>(x) - 75*x + 40) - 60) + 60) - 1)*(2*x + 2*y + 6*z - 2) + (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(13*x*(7*x*(48*ipow<2>(x) - 75*x + 40) - 60) + x*(91*x*(48*ipow<2>(x) - 75*x + 40) + 13*x*(336*ipow<2>(x) + 7*x*(96*x - 75) - 525*x + 280) - 780) + 60),
            (x*(13*x*(7*x*(48*ipow<2>(x) - 75*x + 40) - 60) + 60) - 1)*(2*x + 2*y + 6*z - 2),
            (x*(13*x*(7*x*(48*ipow<2>(x) - 75*x + 40) - 60) + 60) - 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 100
template<>
struct DGBasis<100> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(x + 7*y - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (x + 7*y - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(182*x*(x*(10*x - 10) + 3) + 26*x*(7*x*(10*x - 10) + 7*x*(20*x - 10) + 21) - 52),
            (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + 7*(26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)),
            (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(x + 7*y - 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 101
template<>
struct DGBasis<101> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (7*x*(5*x*(16*x - 9) + 6) - 1)*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(2*x + 16*y - 2)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (7*x*(5*x*(16*x - 9) + 6) - 1)*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(2*x + 2*y + 6*z - 2) + (36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(35*x*(16*x - 9) + 7*x*(160*x - 45) + 42),
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(16*x + 72*y - 16)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (7*x*(5*x*(16*x - 9) + 6) - 1)*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(2*x + 2*y + 6*z - 2),
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 102
template<>
struct DGBasis<102> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (30*x*(4*x - 1) + 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (240*x - 30)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1)) + (30*x*(4*x - 1) + 1)*(135*ipow<2>(y) + 27*y*(2*x - 2) + 3*ipow<2>(x - 1))*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (30*x*(4*x - 1) + 1)*(2*x + 2*y + 6*z - 2)*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (30*x*(4*x - 1) + 1)*(495*ipow<2>(y) + 2*y*(135*x - 135) + 27*ipow<2>(x - 1))*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (30*x*(4*x - 1) + 1)*(2*x + 2*y + 6*z - 2)*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (30*x*(4*x - 1) + 1)*(6*x + 6*y + 12*z - 6)*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 103
template<>
struct DGBasis<103> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (16*x - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(715*ipow<4>(y) + ipow<3>(y)*(880*x - 880) + 330*ipow<2>(y)*ipow<2>(x - 1) + 40*y*ipow<3>(x - 1) + ipow<4>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (16*x - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(880*ipow<3>(y) + 330*ipow<2>(y)*(2*x - 2) + 120*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1)) + (16*x - 1)*(2*x + 2*y + 6*z - 2)*(715*ipow<4>(y) + ipow<3>(y)*(880*x - 880) + 330*ipow<2>(y)*ipow<2>(x - 1) + 40*y*ipow<3>(x - 1) + ipow<4>(x - 1)) + 16*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(715*ipow<4>(y) + ipow<3>(y)*(880*x - 880) + 330*ipow<2>(y)*ipow<2>(x - 1) + 40*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (16*x - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(2860*ipow<3>(y) + 3*ipow<2>(y)*(880*x - 880) + 660*y*ipow<2>(x - 1) + 40*ipow<3>(x - 1)) + (16*x - 1)*(2*x + 2*y + 6*z - 2)*(715*ipow<4>(y) + ipow<3>(y)*(880*x - 880) + 330*ipow<2>(y)*ipow<2>(x - 1) + 40*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (16*x - 1)*(6*x + 6*y + 12*z - 6)*(715*ipow<4>(y) + ipow<3>(y)*(880*x - 880) + 330*ipow<2>(y)*ipow<2>(x - 1) + 40*y*ipow<3>(x - 1) + ipow<4>(x - 1))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 104
template<>
struct DGBasis<104> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(3003*ipow<5>(y) + ipow<4>(y)*(5005*x - 5005) + 2860*ipow<3>(y)*ipow<2>(x - 1) + 660*ipow<2>(y)*ipow<3>(x - 1) + 55*y*ipow<4>(x - 1) + ipow<5>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(5005*ipow<4>(y) + 2860*ipow<3>(y)*(2*x - 2) + 1980*ipow<2>(y)*ipow<2>(x - 1) + 220*y*ipow<3>(x - 1) + 5*ipow<4>(x - 1)) + (2*x + 2*y + 6*z - 2)*(3003*ipow<5>(y) + ipow<4>(y)*(5005*x - 5005) + 2860*ipow<3>(y)*ipow<2>(x - 1) + 660*ipow<2>(y)*ipow<3>(x - 1) + 55*y*ipow<4>(x - 1) + ipow<5>(x - 1)),
            (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(15015*ipow<4>(y) + 4*ipow<3>(y)*(5005*x - 5005) + 8580*ipow<2>(y)*ipow<2>(x - 1) + 1320*y*ipow<3>(x - 1) + 55*ipow<4>(x - 1)) + (2*x + 2*y + 6*z - 2)*(3003*ipow<5>(y) + ipow<4>(y)*(5005*x - 5005) + 2860*ipow<3>(y)*ipow<2>(x - 1) + 660*ipow<2>(y)*ipow<3>(x - 1) + 55*y*ipow<4>(x - 1) + ipow<5>(x - 1)),
            (6*x + 6*y + 12*z - 6)*(3003*ipow<5>(y) + ipow<4>(y)*(5005*x - 5005) + 2860*ipow<3>(y)*ipow<2>(x - 1) + 660*ipow<2>(y)*ipow<3>(x - 1) + 55*y*ipow<4>(x - 1) + ipow<5>(x - 1))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 105
template<>
struct DGBasis<105> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)) + (182*x*(x*(10*x - 10) + 3) + 26*x*(7*x*(10*x - 10) + 7*x*(20*x - 10) + 21) - 52)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)),
            (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 106
template<>
struct DGBasis<106> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (7*x*(5*x*(16*x - 9) + 6) - 1)*(x + 9*y - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(x + 9*y - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)) + (7*x*(5*x*(16*x - 9) + 6) - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (x + 9*y - 1)*(35*x*(16*x - 9) + 7*x*(160*x - 45) + 42)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(x + 9*y - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)) + 9*(7*x*(5*x*(16*x - 9) + 6) - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(x + 9*y - 1)*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 107
template<>
struct DGBasis<107> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (30*x*(4*x - 1) + 1)*(55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (240*x - 30)*(55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (30*x*(4*x - 1) + 1)*(2*x + 20*y - 2)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (30*x*(4*x - 1) + 1)*(55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)),
            (30*x*(4*x - 1) + 1)*(20*x + 110*y - 20)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (30*x*(4*x - 1) + 1)*(55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)),
            (30*x*(4*x - 1) + 1)*(55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 108
template<>
struct DGBasis<108> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (16*x - 1)*(286*ipow<3>(y) + ipow<2>(y)*(198*x - 198) + 33*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (16*x - 1)*(198*ipow<2>(y) + 33*y*(2*x - 2) + 3*ipow<2>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (16*x - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(286*ipow<3>(y) + ipow<2>(y)*(198*x - 198) + 33*y*ipow<2>(x - 1) + ipow<3>(x - 1)) + 16*(286*ipow<3>(y) + ipow<2>(y)*(198*x - 198) + 33*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (16*x - 1)*(858*ipow<2>(y) + 2*y*(198*x - 198) + 33*ipow<2>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (16*x - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(286*ipow<3>(y) + ipow<2>(y)*(198*x - 198) + 33*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (16*x - 1)*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))*(286*ipow<3>(y) + ipow<2>(y)*(198*x - 198) + 33*y*ipow<2>(x - 1) + ipow<3>(x - 1))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 109
template<>
struct DGBasis<109> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(1365*ipow<4>(y) + ipow<3>(y)*(1456*x - 1456) + 468*ipow<2>(y)*ipow<2>(x - 1) + 48*y*ipow<3>(x - 1) + ipow<4>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(1365*ipow<4>(y) + ipow<3>(y)*(1456*x - 1456) + 468*ipow<2>(y)*ipow<2>(x - 1) + 48*y*ipow<3>(x - 1) + ipow<4>(x - 1)) + (1456*ipow<3>(y) + 468*ipow<2>(y)*(2*x - 2) + 144*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(1365*ipow<4>(y) + ipow<3>(y)*(1456*x - 1456) + 468*ipow<2>(y)*ipow<2>(x - 1) + 48*y*ipow<3>(x - 1) + ipow<4>(x - 1)) + (5460*ipow<3>(y) + 3*ipow<2>(y)*(1456*x - 1456) + 936*y*ipow<2>(x - 1) + 48*ipow<3>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))*(1365*ipow<4>(y) + ipow<3>(y)*(1456*x - 1456) + 468*ipow<2>(y)*ipow<2>(x - 1) + 48*y*ipow<3>(x - 1) + ipow<4>(x - 1))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 110
template<>
struct DGBasis<110> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (7*x*(5*x*(16*x - 9) + 6) - 1)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)) + (35*x*(16*x - 9) + 7*x*(160*x - 45) + 42)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)),
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)),
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 111
template<>
struct DGBasis<111> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (30*x*(4*x - 1) + 1)*(x + 11*y - 1)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (240*x - 30)*(x + 11*y - 1)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (30*x*(4*x - 1) + 1)*(x + 11*y - 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)) + (30*x*(4*x - 1) + 1)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)),
            (30*x*(4*x - 1) + 1)*(x + 11*y - 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)) + 11*(30*x*(4*x - 1) + 1)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)),
            (30*x*(4*x - 1) + 1)*(x + 11*y - 1)*(280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 112
template<>
struct DGBasis<112> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (16*x - 1)*(78*ipow<2>(y) + y*(24*x - 24) + ipow<2>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (16*x - 1)*(2*x + 24*y - 2)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (16*x - 1)*(78*ipow<2>(y) + y*(24*x - 24) + ipow<2>(x - 1))*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)) + 16*(78*ipow<2>(y) + y*(24*x - 24) + ipow<2>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)),
            (16*x - 1)*(24*x + 156*y - 24)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (16*x - 1)*(78*ipow<2>(y) + y*(24*x - 24) + ipow<2>(x - 1))*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)),
            (16*x - 1)*(78*ipow<2>(y) + y*(24*x - 24) + ipow<2>(x - 1))*(280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 113
template<>
struct DGBasis<113> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (455*ipow<3>(y) + ipow<2>(y)*(273*x - 273) + 39*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (273*ipow<2>(y) + 39*y*(2*x - 2) + 3*ipow<2>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (455*ipow<3>(y) + ipow<2>(y)*(273*x - 273) + 39*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)),
            (1365*ipow<2>(y) + 2*y*(273*x - 273) + 39*ipow<2>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (455*ipow<3>(y) + ipow<2>(y)*(273*x - 273) + 39*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)),
            (455*ipow<3>(y) + ipow<2>(y)*(273*x - 273) + 39*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 114
template<>
struct DGBasis<114> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (30*x*(4*x - 1) + 1)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (240*x - 30)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (30*x*(4*x - 1) + 1)*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)),
            (30*x*(4*x - 1) + 1)*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)),
            (30*x*(4*x - 1) + 1)*(1260*ipow<4>(z) + 4*ipow<3>(z)*(630*x + 630*y - 630) + 1680*ipow<2>(z)*ipow<2>(x + y - 1) + 420*z*ipow<3>(x + y - 1) + 30*ipow<4>(x + y - 1))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 115
template<>
struct DGBasis<115> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (16*x - 1)*(x + 13*y - 1)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (16*x - 1)*(x + 13*y - 1)*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)) + (16*x - 1)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + 16*(x + 13*y - 1)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)),
            (16*x - 1)*(x + 13*y - 1)*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)) + 13*(16*x - 1)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)),
            (16*x - 1)*(x + 13*y - 1)*(1260*ipow<4>(z) + 4*ipow<3>(z)*(630*x + 630*y - 630) + 1680*ipow<2>(z)*ipow<2>(x + y - 1) + 420*z*ipow<3>(x + y - 1) + 30*ipow<4>(x + y - 1))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 116
template<>
struct DGBasis<116> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (105*ipow<2>(y) + y*(28*x - 28) + ipow<2>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x + 28*y - 2)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (105*ipow<2>(y) + y*(28*x - 28) + ipow<2>(x - 1))*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)),
            (28*x + 210*y - 28)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (105*ipow<2>(y) + y*(28*x - 28) + ipow<2>(x - 1))*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)),
            (105*ipow<2>(y) + y*(28*x - 28) + ipow<2>(x - 1))*(1260*ipow<4>(z) + 4*ipow<3>(z)*(630*x + 630*y - 630) + 1680*ipow<2>(z)*ipow<2>(x + y - 1) + 420*z*ipow<3>(x + y - 1) + 30*ipow<4>(x + y - 1))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 117
template<>
struct DGBasis<117> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (16*x - 1)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            14784*ipow<6>(z) + 16*ipow<5>(z)*(2772*x + 2772*y - 2772) + 50400*ipow<4>(z)*ipow<2>(x + y - 1) + 26880*ipow<3>(z)*ipow<3>(x + y - 1) + 6720*ipow<2>(z)*ipow<4>(x + y - 1) + 672*z*ipow<5>(x + y - 1) + (16*x - 1)*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)) + 16*ipow<6>(x + y - 1),
            (16*x - 1)*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)),
            (16*x - 1)*(5544*ipow<5>(z) + 5*ipow<4>(z)*(2772*x + 2772*y - 2772) + 12600*ipow<3>(z)*ipow<2>(x + y - 1) + 5040*ipow<2>(z)*ipow<3>(x + y - 1) + 840*z*ipow<4>(x + y - 1) + 42*ipow<5>(x + y - 1))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 118
template<>
struct DGBasis<118> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x + 15*y - 1)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1) + (x + 15*y - 1)*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)),
            13860*ipow<6>(z) + 15*ipow<5>(z)*(2772*x + 2772*y - 2772) + 47250*ipow<4>(z)*ipow<2>(x + y - 1) + 25200*ipow<3>(z)*ipow<3>(x + y - 1) + 6300*ipow<2>(z)*ipow<4>(x + y - 1) + 630*z*ipow<5>(x + y - 1) + 15*ipow<6>(x + y - 1) + (x + 15*y - 1)*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)),
            (x + 15*y - 1)*(5544*ipow<5>(z) + 5*ipow<4>(z)*(2772*x + 2772*y - 2772) + 12600*ipow<3>(z)*ipow<2>(x + y - 1) + 5040*ipow<2>(z)*ipow<3>(x + y - 1) + 840*z*ipow<4>(x + y - 1) + 42*ipow<5>(x + y - 1))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 119
template<>
struct DGBasis<119> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            7*ipow<6>(x) + 42*ipow<5>(x)*y - 42*ipow<5>(x) + 105*ipow<4>(x)*ipow<2>(y) - 210*ipow<4>(x)*y + 105*ipow<4>(x) + 140*ipow<3>(x)*ipow<3>(y) - 420*ipow<3>(x)*ipow<2>(y) + 420*ipow<3>(x)*y - 140*ipow<3>(x) + 105*ipow<2>(x)*ipow<4>(y) - 420*ipow<2>(x)*ipow<3>(y) + 630*ipow<2>(x)*ipow<2>(y) - 420*ipow<2>(x)*y + 105*ipow<2>(x) + 42*x*ipow<5>(y) - 210*x*ipow<4>(y) + 420*x*ipow<3>(y) - 420*x*ipow<2>(y) + 210*x*y - 42*x + 7*ipow<6>(y) - 42*ipow<5>(y) + 105*ipow<4>(y) - 140*ipow<3>(y) + 105*ipow<2>(y) - 42*y + 12012*ipow<6>(z) + 16632*ipow<5>(z)*(2*x + 2*y - 2) + 34650*ipow<4>(z)*ipow<2>(x + y - 1) + 16800*ipow<3>(z)*ipow<3>(x + y - 1) + 3780*ipow<2>(z)*ipow<4>(x + y - 1) + 336*z*ipow<5>(x + y - 1) + 7,
            7*ipow<6>(x) + 42*ipow<5>(x)*y - 42*ipow<5>(x) + 105*ipow<4>(x)*ipow<2>(y) - 210*ipow<4>(x)*y + 105*ipow<4>(x) + 140*ipow<3>(x)*ipow<3>(y) - 420*ipow<3>(x)*ipow<2>(y) + 420*ipow<3>(x)*y - 140*ipow<3>(x) + 105*ipow<2>(x)*ipow<4>(y) - 420*ipow<2>(x)*ipow<3>(y) + 630*ipow<2>(x)*ipow<2>(y) - 420*ipow<2>(x)*y + 105*ipow<2>(x) + 42*x*ipow<5>(y) - 210*x*ipow<4>(y) + 420*x*ipow<3>(y) - 420*x*ipow<2>(y) + 210*x*y - 42*x + 7*ipow<6>(y) - 42*ipow<5>(y) + 105*ipow<4>(y) - 140*ipow<3>(y) + 105*ipow<2>(y) - 42*y + 12012*ipow<6>(z) + 16632*ipow<5>(z)*(2*x + 2*y - 2) + 34650*ipow<4>(z)*ipow<2>(x + y - 1) + 16800*ipow<3>(z)*ipow<3>(x + y - 1) + 3780*ipow<2>(z)*ipow<4>(x + y - 1) + 336*z*ipow<5>(x + y - 1) + 7,
            24024*ipow<6>(z) + 6*ipow<5>(z)*(12012*x + 12012*y - 12012) + 83160*ipow<4>(z)*ipow<2>(x + y - 1) + 46200*ipow<3>(z)*ipow<3>(x + y - 1) + 12600*ipow<2>(z)*ipow<4>(x + y - 1) + 1512*z*ipow<5>(x + y - 1) + 56*ipow<6>(x + y - 1)
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 120
template<>
struct DGBasis<120> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return 22*x*(x*(13*x*(x*(x*(x*(17*x*(9*x - 32) + 784) - 588) + 245) - 56) + 84) - 4) + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            22*x*(13*x*(x*(x*(x*(17*x*(9*x - 32) + 784) - 588) + 245) - 56) + 84) + 22*x*(13*x*(x*(x*(x*(17*x*(9*x - 32) + 784) - 588) + 245) - 56) + x*(13*x*(x*(x*(17*x*(9*x - 32) + 784) - 588) + 245) + 13*x*(x*(x*(17*x*(9*x - 32) + 784) - 588) + x*(x*(17*x*(9*x - 32) + 784) + x*(17*x*(9*x - 32) + x*(306*x - 544) + 784) - 588) + 245) - 728) + 84) - 88,
            0,
            0
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 121
template<>
struct DGBasis<121> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x*(13*x*(x*(x*(8*x*(17*x*(18*x - 49) + 882) - 3675) + 980) - 126) + 84) - 1)*(x + 3*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            x*(13*x*(x*(x*(8*x*(17*x*(18*x - 49) + 882) - 3675) + 980) - 126) + 84) + (x + 3*y - 1)*(13*x*(x*(x*(8*x*(17*x*(18*x - 49) + 882) - 3675) + 980) - 126) + x*(13*x*(x*(8*x*(17*x*(18*x - 49) + 882) - 3675) + 980) + 13*x*(x*(8*x*(17*x*(18*x - 49) + 882) - 3675) + x*(8*x*(17*x*(18*x - 49) + 882) + x*(136*x*(18*x - 49) + 8*x*(612*x - 833) + 7056) - 3675) + 980) - 1638) + 84) - 1,
            3*x*(13*x*(x*(x*(8*x*(17*x*(18*x - 49) + 882) - 3675) + 980) - 126) + 84) - 3,
            0
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 122
template<>
struct DGBasis<122> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (13*x*(7*x*(4*x*(3*x*(x*(17*x - 34) + 25) - 25) + 15) - 6) + 1)*(10*ipow<2>(y) + y*(8*x - 8) + ipow<2>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (13*x*(7*x*(4*x*(3*x*(x*(17*x - 34) + 25) - 25) + 15) - 6) + 1)*(2*x + 8*y - 2) + (10*ipow<2>(y) + y*(8*x - 8) + ipow<2>(x - 1))*(91*x*(4*x*(3*x*(x*(17*x - 34) + 25) - 25) + 15) + 13*x*(28*x*(3*x*(x*(17*x - 34) + 25) - 25) + 7*x*(12*x*(x*(17*x - 34) + 25) + 4*x*(3*x*(17*x - 34) + 3*x*(34*x - 34) + 75) - 100) + 105) - 78),
            (13*x*(7*x*(4*x*(3*x*(x*(17*x - 34) + 25) - 25) + 15) - 6) + 1)*(8*x + 20*y - 8),
            0
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 123
template<>
struct DGBasis<123> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(35*ipow<3>(y) + ipow<2>(y)*(45*x - 45) + 15*y*ipow<2>(x - 1) + ipow<3>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(45*ipow<2>(y) + 15*y*(2*x - 2) + 3*ipow<2>(x - 1)) + (14*x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 14*x*(2*x*(17*x*(18*x - 25) + 200) + x*(34*x*(18*x - 25) + 2*x*(612*x - 425) + 400) - 75) + 70)*(35*ipow<3>(y) + ipow<2>(y)*(45*x - 45) + 15*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(105*ipow<2>(y) + 2*y*(45*x - 45) + 15*ipow<2>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 124
template<>
struct DGBasis<124> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(126*ipow<4>(y) + ipow<3>(y)*(224*x - 224) + 126*ipow<2>(y)*ipow<2>(x - 1) + 24*y*ipow<3>(x - 1) + ipow<4>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(224*ipow<3>(y) + 126*ipow<2>(y)*(2*x - 2) + 72*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1)) + (20*x*(17*x*(9*x - 8) + 36) + 20*x*(17*x*(9*x - 8) + x*(306*x - 136) + 36) - 60)*(126*ipow<4>(y) + ipow<3>(y)*(224*x - 224) + 126*ipow<2>(y)*ipow<2>(x - 1) + 24*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(504*ipow<3>(y) + 3*ipow<2>(y)*(224*x - 224) + 252*y*ipow<2>(x - 1) + 24*ipow<3>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 125
template<>
struct DGBasis<125> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (24*x*(17*x*(2*x - 1) + 2) - 1)*(462*ipow<5>(y) + ipow<4>(y)*(1050*x - 1050) + 840*ipow<3>(y)*ipow<2>(x - 1) + 280*ipow<2>(y)*ipow<3>(x - 1) + 35*y*ipow<4>(x - 1) + ipow<5>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(1050*ipow<4>(y) + 840*ipow<3>(y)*(2*x - 2) + 840*ipow<2>(y)*ipow<2>(x - 1) + 140*y*ipow<3>(x - 1) + 5*ipow<4>(x - 1)) + (408*x*(2*x - 1) + 24*x*(68*x - 17) + 48)*(462*ipow<5>(y) + ipow<4>(y)*(1050*x - 1050) + 840*ipow<3>(y)*ipow<2>(x - 1) + 280*ipow<2>(y)*ipow<3>(x - 1) + 35*y*ipow<4>(x - 1) + ipow<5>(x - 1)),
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(2310*ipow<4>(y) + 4*ipow<3>(y)*(1050*x - 1050) + 2520*ipow<2>(y)*ipow<2>(x - 1) + 560*y*ipow<3>(x - 1) + 35*ipow<4>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 126
template<>
struct DGBasis<126> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (17*x*(9*x - 2) + 1)*(1716*ipow<6>(y) + ipow<5>(y)*(4752*x - 4752) + 4950*ipow<4>(y)*ipow<2>(x - 1) + 2400*ipow<3>(y)*ipow<3>(x - 1) + 540*ipow<2>(y)*ipow<4>(x - 1) + 48*y*ipow<5>(x - 1) + ipow<6>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (306*x - 34)*(1716*ipow<6>(y) + ipow<5>(y)*(4752*x - 4752) + 4950*ipow<4>(y)*ipow<2>(x - 1) + 2400*ipow<3>(y)*ipow<3>(x - 1) + 540*ipow<2>(y)*ipow<4>(x - 1) + 48*y*ipow<5>(x - 1) + ipow<6>(x - 1)) + (17*x*(9*x - 2) + 1)*(4752*ipow<5>(y) + 4950*ipow<4>(y)*(2*x - 2) + 7200*ipow<3>(y)*ipow<2>(x - 1) + 2160*ipow<2>(y)*ipow<3>(x - 1) + 240*y*ipow<4>(x - 1) + 6*ipow<5>(x - 1)),
            (17*x*(9*x - 2) + 1)*(10296*ipow<5>(y) + 5*ipow<4>(y)*(4752*x - 4752) + 19800*ipow<3>(y)*ipow<2>(x - 1) + 7200*ipow<2>(y)*ipow<3>(x - 1) + 1080*y*ipow<4>(x - 1) + 48*ipow<5>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 127
template<>
struct DGBasis<127> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (18*x - 1)*(6435*ipow<7>(y) + ipow<6>(y)*(21021*x - 21021) + 27027*ipow<5>(y)*ipow<2>(x - 1) + 17325*ipow<4>(y)*ipow<3>(x - 1) + 5775*ipow<3>(y)*ipow<4>(x - 1) + 945*ipow<2>(y)*ipow<5>(x - 1) + 63*y*ipow<6>(x - 1) + ipow<7>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            115830*ipow<7>(y) + 18*ipow<6>(y)*(21021*x - 21021) + 486486*ipow<5>(y)*ipow<2>(x - 1) + 311850*ipow<4>(y)*ipow<3>(x - 1) + 103950*ipow<3>(y)*ipow<4>(x - 1) + 17010*ipow<2>(y)*ipow<5>(x - 1) + 1134*y*ipow<6>(x - 1) + 18*ipow<7>(x - 1) + (18*x - 1)*(21021*ipow<6>(y) + 27027*ipow<5>(y)*(2*x - 2) + 51975*ipow<4>(y)*ipow<2>(x - 1) + 23100*ipow<3>(y)*ipow<3>(x - 1) + 4725*ipow<2>(y)*ipow<4>(x - 1) + 378*y*ipow<5>(x - 1) + 7*ipow<6>(x - 1)),
            (18*x - 1)*(45045*ipow<6>(y) + 6*ipow<5>(y)*(21021*x - 21021) + 135135*ipow<4>(y)*ipow<2>(x - 1) + 69300*ipow<3>(y)*ipow<3>(x - 1) + 17325*ipow<2>(y)*ipow<4>(x - 1) + 1890*y*ipow<5>(x - 1) + 63*ipow<6>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 128
template<>
struct DGBasis<128> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return 24310*ipow<8>(y) + ipow<7>(y)*(91520*x - 91520) + 140140*ipow<6>(y)*ipow<2>(x - 1) + 112112*ipow<5>(y)*ipow<3>(x - 1) + 50050*ipow<4>(y)*ipow<4>(x - 1) + 12320*ipow<3>(y)*ipow<5>(x - 1) + 1540*ipow<2>(y)*ipow<6>(x - 1) + 80*y*ipow<7>(x - 1) + ipow<8>(x - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            91520*ipow<7>(y) + 140140*ipow<6>(y)*(2*x - 2) + 336336*ipow<5>(y)*ipow<2>(x - 1) + 200200*ipow<4>(y)*ipow<3>(x - 1) + 61600*ipow<3>(y)*ipow<4>(x - 1) + 9240*ipow<2>(y)*ipow<5>(x - 1) + 560*y*ipow<6>(x - 1) + 8*ipow<7>(x - 1),
            194480*ipow<7>(y) + 7*ipow<6>(y)*(91520*x - 91520) + 840840*ipow<5>(y)*ipow<2>(x - 1) + 560560*ipow<4>(y)*ipow<3>(x - 1) + 200200*ipow<3>(y)*ipow<4>(x - 1) + 36960*ipow<2>(y)*ipow<5>(x - 1) + 3080*y*ipow<6>(x - 1) + 80*ipow<7>(x - 1),
            0
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 129
template<>
struct DGBasis<129> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x*(13*x*(x*(x*(8*x*(17*x*(18*x - 49) + 882) - 3675) + 980) - 126) + 84) - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            x*(13*x*(x*(x*(8*x*(17*x*(18*x - 49) + 882) - 3675) + 980) - 126) + 84) + (13*x*(x*(x*(8*x*(17*x*(18*x - 49) + 882) - 3675) + 980) - 126) + x*(13*x*(x*(8*x*(17*x*(18*x - 49) + 882) - 3675) + 980) + 13*x*(x*(8*x*(17*x*(18*x - 49) + 882) - 3675) + x*(8*x*(17*x*(18*x - 49) + 882) + x*(136*x*(18*x - 49) + 8*x*(612*x - 833) + 7056) - 3675) + 980) - 1638) + 84)*(x + y + 2*z - 1) - 1,
            x*(13*x*(x*(x*(8*x*(17*x*(18*x - 49) + 882) - 3675) + 980) - 126) + 84) - 1,
            2*x*(13*x*(x*(x*(8*x*(17*x*(18*x - 49) + 882) - 3675) + 980) - 126) + 84) - 2
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 130
template<>
struct DGBasis<130> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (13*x*(7*x*(4*x*(3*x*(x*(17*x - 34) + 25) - 25) + 15) - 6) + 1)*(x + 5*y - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (13*x*(7*x*(4*x*(3*x*(x*(17*x - 34) + 25) - 25) + 15) - 6) + 1)*(x + 5*y - 1) + (13*x*(7*x*(4*x*(3*x*(x*(17*x - 34) + 25) - 25) + 15) - 6) + 1)*(x + y + 2*z - 1) + (x + 5*y - 1)*(91*x*(4*x*(3*x*(x*(17*x - 34) + 25) - 25) + 15) + 13*x*(28*x*(3*x*(x*(17*x - 34) + 25) - 25) + 7*x*(12*x*(x*(17*x - 34) + 25) + 4*x*(3*x*(17*x - 34) + 3*x*(34*x - 34) + 75) - 100) + 105) - 78)*(x + y + 2*z - 1),
            (13*x*(7*x*(4*x*(3*x*(x*(17*x - 34) + 25) - 25) + 15) - 6) + 1)*(x + 5*y - 1) + 5*(13*x*(7*x*(4*x*(3*x*(x*(17*x - 34) + 25) - 25) + 15) - 6) + 1)*(x + y + 2*z - 1),
            2*(13*x*(7*x*(4*x*(3*x*(x*(17*x - 34) + 25) - 25) + 15) - 6) + 1)*(x + 5*y - 1)
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 131
template<>
struct DGBasis<131> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1))*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(2*x + 12*y - 2)*(x + y + 2*z - 1) + (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1)) + (21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1))*(14*x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 14*x*(2*x*(17*x*(18*x - 25) + 200) + x*(34*x*(18*x - 25) + 2*x*(612*x - 425) + 400) - 75) + 70)*(x + y + 2*z - 1),
            (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(12*x + 42*y - 12)*(x + y + 2*z - 1) + (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1)),
            2*(14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 132
template<>
struct DGBasis<132> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(x + y + 2*z - 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(84*ipow<2>(y) + 21*y*(2*x - 2) + 3*ipow<2>(x - 1))*(x + y + 2*z - 1) + (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1)) + (20*x*(17*x*(9*x - 8) + 36) + 20*x*(17*x*(9*x - 8) + x*(306*x - 136) + 36) - 60)*(x + y + 2*z - 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(252*ipow<2>(y) + 2*y*(84*x - 84) + 21*ipow<2>(x - 1))*(x + y + 2*z - 1) + (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            2*(20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 133
template<>
struct DGBasis<133> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (24*x*(17*x*(2*x - 1) + 2) - 1)*(x + y + 2*z - 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(x + y + 2*z - 1)*(480*ipow<3>(y) + 216*ipow<2>(y)*(2*x - 2) + 96*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1)) + (24*x*(17*x*(2*x - 1) + 2) - 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1)) + (408*x*(2*x - 1) + 24*x*(68*x - 17) + 48)*(x + y + 2*z - 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(x + y + 2*z - 1)*(1320*ipow<3>(y) + 3*ipow<2>(y)*(480*x - 480) + 432*y*ipow<2>(x - 1) + 32*ipow<3>(x - 1)) + (24*x*(17*x*(2*x - 1) + 2) - 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            2*(24*x*(17*x*(2*x - 1) + 2) - 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 134
template<>
struct DGBasis<134> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (17*x*(9*x - 2) + 1)*(x + y + 2*z - 1)*(1287*ipow<5>(y) + ipow<4>(y)*(2475*x - 2475) + 1650*ipow<3>(y)*ipow<2>(x - 1) + 450*ipow<2>(y)*ipow<3>(x - 1) + 45*y*ipow<4>(x - 1) + ipow<5>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (306*x - 34)*(x + y + 2*z - 1)*(1287*ipow<5>(y) + ipow<4>(y)*(2475*x - 2475) + 1650*ipow<3>(y)*ipow<2>(x - 1) + 450*ipow<2>(y)*ipow<3>(x - 1) + 45*y*ipow<4>(x - 1) + ipow<5>(x - 1)) + (17*x*(9*x - 2) + 1)*(x + y + 2*z - 1)*(2475*ipow<4>(y) + 1650*ipow<3>(y)*(2*x - 2) + 1350*ipow<2>(y)*ipow<2>(x - 1) + 180*y*ipow<3>(x - 1) + 5*ipow<4>(x - 1)) + (17*x*(9*x - 2) + 1)*(1287*ipow<5>(y) + ipow<4>(y)*(2475*x - 2475) + 1650*ipow<3>(y)*ipow<2>(x - 1) + 450*ipow<2>(y)*ipow<3>(x - 1) + 45*y*ipow<4>(x - 1) + ipow<5>(x - 1)),
            (17*x*(9*x - 2) + 1)*(x + y + 2*z - 1)*(6435*ipow<4>(y) + 4*ipow<3>(y)*(2475*x - 2475) + 4950*ipow<2>(y)*ipow<2>(x - 1) + 900*y*ipow<3>(x - 1) + 45*ipow<4>(x - 1)) + (17*x*(9*x - 2) + 1)*(1287*ipow<5>(y) + ipow<4>(y)*(2475*x - 2475) + 1650*ipow<3>(y)*ipow<2>(x - 1) + 450*ipow<2>(y)*ipow<3>(x - 1) + 45*y*ipow<4>(x - 1) + ipow<5>(x - 1)),
            2*(17*x*(9*x - 2) + 1)*(1287*ipow<5>(y) + ipow<4>(y)*(2475*x - 2475) + 1650*ipow<3>(y)*ipow<2>(x - 1) + 450*ipow<2>(y)*ipow<3>(x - 1) + 45*y*ipow<4>(x - 1) + ipow<5>(x - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 135
template<>
struct DGBasis<135> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (18*x - 1)*(x + y + 2*z - 1)*(5005*ipow<6>(y) + ipow<5>(y)*(12012*x - 12012) + 10725*ipow<4>(y)*ipow<2>(x - 1) + 4400*ipow<3>(y)*ipow<3>(x - 1) + 825*ipow<2>(y)*ipow<4>(x - 1) + 60*y*ipow<5>(x - 1) + ipow<6>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (18*x - 1)*(x + y + 2*z - 1)*(12012*ipow<5>(y) + 10725*ipow<4>(y)*(2*x - 2) + 13200*ipow<3>(y)*ipow<2>(x - 1) + 3300*ipow<2>(y)*ipow<3>(x - 1) + 300*y*ipow<4>(x - 1) + 6*ipow<5>(x - 1)) + (18*x - 1)*(5005*ipow<6>(y) + ipow<5>(y)*(12012*x - 12012) + 10725*ipow<4>(y)*ipow<2>(x - 1) + 4400*ipow<3>(y)*ipow<3>(x - 1) + 825*ipow<2>(y)*ipow<4>(x - 1) + 60*y*ipow<5>(x - 1) + ipow<6>(x - 1)) + 18*(x + y + 2*z - 1)*(5005*ipow<6>(y) + ipow<5>(y)*(12012*x - 12012) + 10725*ipow<4>(y)*ipow<2>(x - 1) + 4400*ipow<3>(y)*ipow<3>(x - 1) + 825*ipow<2>(y)*ipow<4>(x - 1) + 60*y*ipow<5>(x - 1) + ipow<6>(x - 1)),
            (18*x - 1)*(x + y + 2*z - 1)*(30030*ipow<5>(y) + 5*ipow<4>(y)*(12012*x - 12012) + 42900*ipow<3>(y)*ipow<2>(x - 1) + 13200*ipow<2>(y)*ipow<3>(x - 1) + 1650*y*ipow<4>(x - 1) + 60*ipow<5>(x - 1)) + (18*x - 1)*(5005*ipow<6>(y) + ipow<5>(y)*(12012*x - 12012) + 10725*ipow<4>(y)*ipow<2>(x - 1) + 4400*ipow<3>(y)*ipow<3>(x - 1) + 825*ipow<2>(y)*ipow<4>(x - 1) + 60*y*ipow<5>(x - 1) + ipow<6>(x - 1)),
            2*(18*x - 1)*(5005*ipow<6>(y) + ipow<5>(y)*(12012*x - 12012) + 10725*ipow<4>(y)*ipow<2>(x - 1) + 4400*ipow<3>(y)*ipow<3>(x - 1) + 825*ipow<2>(y)*ipow<4>(x - 1) + 60*y*ipow<5>(x - 1) + ipow<6>(x - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 136
template<>
struct DGBasis<136> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x + y + 2*z - 1)*(19448*ipow<7>(y) + ipow<6>(y)*(56056*x - 56056) + 63063*ipow<5>(y)*ipow<2>(x - 1) + 35035*ipow<4>(y)*ipow<3>(x - 1) + 10010*ipow<3>(y)*ipow<4>(x - 1) + 1386*ipow<2>(y)*ipow<5>(x - 1) + 77*y*ipow<6>(x - 1) + ipow<7>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            19448*ipow<7>(y) + ipow<6>(y)*(56056*x - 56056) + 63063*ipow<5>(y)*ipow<2>(x - 1) + 35035*ipow<4>(y)*ipow<3>(x - 1) + 10010*ipow<3>(y)*ipow<4>(x - 1) + 1386*ipow<2>(y)*ipow<5>(x - 1) + 77*y*ipow<6>(x - 1) + ipow<7>(x - 1) + (x + y + 2*z - 1)*(56056*ipow<6>(y) + 63063*ipow<5>(y)*(2*x - 2) + 105105*ipow<4>(y)*ipow<2>(x - 1) + 40040*ipow<3>(y)*ipow<3>(x - 1) + 6930*ipow<2>(y)*ipow<4>(x - 1) + 462*y*ipow<5>(x - 1) + 7*ipow<6>(x - 1)),
            19448*ipow<7>(y) + ipow<6>(y)*(56056*x - 56056) + 63063*ipow<5>(y)*ipow<2>(x - 1) + 35035*ipow<4>(y)*ipow<3>(x - 1) + 10010*ipow<3>(y)*ipow<4>(x - 1) + 1386*ipow<2>(y)*ipow<5>(x - 1) + 77*y*ipow<6>(x - 1) + ipow<7>(x - 1) + (x + y + 2*z - 1)*(136136*ipow<6>(y) + 6*ipow<5>(y)*(56056*x - 56056) + 315315*ipow<4>(y)*ipow<2>(x - 1) + 140140*ipow<3>(y)*ipow<3>(x - 1) + 30030*ipow<2>(y)*ipow<4>(x - 1) + 2772*y*ipow<5>(x - 1) + 77*ipow<6>(x - 1)),
            38896*ipow<7>(y) + 2*ipow<6>(y)*(56056*x - 56056) + 126126*ipow<5>(y)*ipow<2>(x - 1) + 70070*ipow<4>(y)*ipow<3>(x - 1) + 20020*ipow<3>(y)*ipow<4>(x - 1) + 2772*ipow<2>(y)*ipow<5>(x - 1) + 154*y*ipow<6>(x - 1) + 2*ipow<7>(x - 1)
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 137
template<>
struct DGBasis<137> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (13*x*(7*x*(4*x*(3*x*(x*(17*x - 34) + 25) - 25) + 15) - 6) + 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (13*x*(7*x*(4*x*(3*x*(x*(17*x - 34) + 25) - 25) + 15) - 6) + 1)*(2*x + 2*y + 6*z - 2) + (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(91*x*(4*x*(3*x*(x*(17*x - 34) + 25) - 25) + 15) + 13*x*(28*x*(3*x*(x*(17*x - 34) + 25) - 25) + 7*x*(12*x*(x*(17*x - 34) + 25) + 4*x*(3*x*(17*x - 34) + 3*x*(34*x - 34) + 75) - 100) + 105) - 78),
            (13*x*(7*x*(4*x*(3*x*(x*(17*x - 34) + 25) - 25) + 15) - 6) + 1)*(2*x + 2*y + 6*z - 2),
            (13*x*(7*x*(4*x*(3*x*(x*(17*x - 34) + 25) - 25) + 15) - 6) + 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 138
template<>
struct DGBasis<138> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(x + 7*y - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (x + 7*y - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(14*x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 14*x*(2*x*(17*x*(18*x - 25) + 200) + x*(34*x*(18*x - 25) + 2*x*(612*x - 425) + 400) - 75) + 70),
            (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + 7*(14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)),
            (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(x + 7*y - 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 139
template<>
struct DGBasis<139> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(2*x + 16*y - 2)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(2*x + 2*y + 6*z - 2) + (36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(20*x*(17*x*(9*x - 8) + 36) + 20*x*(17*x*(9*x - 8) + x*(306*x - 136) + 36) - 60),
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(16*x + 72*y - 16)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(2*x + 2*y + 6*z - 2),
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 140
template<>
struct DGBasis<140> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (24*x*(17*x*(2*x - 1) + 2) - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(135*ipow<2>(y) + 27*y*(2*x - 2) + 3*ipow<2>(x - 1))*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (24*x*(17*x*(2*x - 1) + 2) - 1)*(2*x + 2*y + 6*z - 2)*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1)) + (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(408*x*(2*x - 1) + 24*x*(68*x - 17) + 48)*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(495*ipow<2>(y) + 2*y*(135*x - 135) + 27*ipow<2>(x - 1))*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (24*x*(17*x*(2*x - 1) + 2) - 1)*(2*x + 2*y + 6*z - 2)*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(6*x + 6*y + 12*z - 6)*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 141
template<>
struct DGBasis<141> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (17*x*(9*x - 2) + 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(715*ipow<4>(y) + ipow<3>(y)*(880*x - 880) + 330*ipow<2>(y)*ipow<2>(x - 1) + 40*y*ipow<3>(x - 1) + ipow<4>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (306*x - 34)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(715*ipow<4>(y) + ipow<3>(y)*(880*x - 880) + 330*ipow<2>(y)*ipow<2>(x - 1) + 40*y*ipow<3>(x - 1) + ipow<4>(x - 1)) + (17*x*(9*x - 2) + 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(880*ipow<3>(y) + 330*ipow<2>(y)*(2*x - 2) + 120*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1)) + (17*x*(9*x - 2) + 1)*(2*x + 2*y + 6*z - 2)*(715*ipow<4>(y) + ipow<3>(y)*(880*x - 880) + 330*ipow<2>(y)*ipow<2>(x - 1) + 40*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (17*x*(9*x - 2) + 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(2860*ipow<3>(y) + 3*ipow<2>(y)*(880*x - 880) + 660*y*ipow<2>(x - 1) + 40*ipow<3>(x - 1)) + (17*x*(9*x - 2) + 1)*(2*x + 2*y + 6*z - 2)*(715*ipow<4>(y) + ipow<3>(y)*(880*x - 880) + 330*ipow<2>(y)*ipow<2>(x - 1) + 40*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (17*x*(9*x - 2) + 1)*(6*x + 6*y + 12*z - 6)*(715*ipow<4>(y) + ipow<3>(y)*(880*x - 880) + 330*ipow<2>(y)*ipow<2>(x - 1) + 40*y*ipow<3>(x - 1) + ipow<4>(x - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 142
template<>
struct DGBasis<142> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (18*x - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(3003*ipow<5>(y) + ipow<4>(y)*(5005*x - 5005) + 2860*ipow<3>(y)*ipow<2>(x - 1) + 660*ipow<2>(y)*ipow<3>(x - 1) + 55*y*ipow<4>(x - 1) + ipow<5>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (18*x - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(5005*ipow<4>(y) + 2860*ipow<3>(y)*(2*x - 2) + 1980*ipow<2>(y)*ipow<2>(x - 1) + 220*y*ipow<3>(x - 1) + 5*ipow<4>(x - 1)) + (18*x - 1)*(2*x + 2*y + 6*z - 2)*(3003*ipow<5>(y) + ipow<4>(y)*(5005*x - 5005) + 2860*ipow<3>(y)*ipow<2>(x - 1) + 660*ipow<2>(y)*ipow<3>(x - 1) + 55*y*ipow<4>(x - 1) + ipow<5>(x - 1)) + 18*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(3003*ipow<5>(y) + ipow<4>(y)*(5005*x - 5005) + 2860*ipow<3>(y)*ipow<2>(x - 1) + 660*ipow<2>(y)*ipow<3>(x - 1) + 55*y*ipow<4>(x - 1) + ipow<5>(x - 1)),
            (18*x - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(15015*ipow<4>(y) + 4*ipow<3>(y)*(5005*x - 5005) + 8580*ipow<2>(y)*ipow<2>(x - 1) + 1320*y*ipow<3>(x - 1) + 55*ipow<4>(x - 1)) + (18*x - 1)*(2*x + 2*y + 6*z - 2)*(3003*ipow<5>(y) + ipow<4>(y)*(5005*x - 5005) + 2860*ipow<3>(y)*ipow<2>(x - 1) + 660*ipow<2>(y)*ipow<3>(x - 1) + 55*y*ipow<4>(x - 1) + ipow<5>(x - 1)),
            (18*x - 1)*(6*x + 6*y + 12*z - 6)*(3003*ipow<5>(y) + ipow<4>(y)*(5005*x - 5005) + 2860*ipow<3>(y)*ipow<2>(x - 1) + 660*ipow<2>(y)*ipow<3>(x - 1) + 55*y*ipow<4>(x - 1) + ipow<5>(x - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 143
template<>
struct DGBasis<143> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(12376*ipow<6>(y) + ipow<5>(y)*(26208*x - 26208) + 20475*ipow<4>(y)*ipow<2>(x - 1) + 7280*ipow<3>(y)*ipow<3>(x - 1) + 1170*ipow<2>(y)*ipow<4>(x - 1) + 72*y*ipow<5>(x - 1) + ipow<6>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(26208*ipow<5>(y) + 20475*ipow<4>(y)*(2*x - 2) + 21840*ipow<3>(y)*ipow<2>(x - 1) + 4680*ipow<2>(y)*ipow<3>(x - 1) + 360*y*ipow<4>(x - 1) + 6*ipow<5>(x - 1)) + (2*x + 2*y + 6*z - 2)*(12376*ipow<6>(y) + ipow<5>(y)*(26208*x - 26208) + 20475*ipow<4>(y)*ipow<2>(x - 1) + 7280*ipow<3>(y)*ipow<3>(x - 1) + 1170*ipow<2>(y)*ipow<4>(x - 1) + 72*y*ipow<5>(x - 1) + ipow<6>(x - 1)),
            (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(74256*ipow<5>(y) + 5*ipow<4>(y)*(26208*x - 26208) + 81900*ipow<3>(y)*ipow<2>(x - 1) + 21840*ipow<2>(y)*ipow<3>(x - 1) + 2340*y*ipow<4>(x - 1) + 72*ipow<5>(x - 1)) + (2*x + 2*y + 6*z - 2)*(12376*ipow<6>(y) + ipow<5>(y)*(26208*x - 26208) + 20475*ipow<4>(y)*ipow<2>(x - 1) + 7280*ipow<3>(y)*ipow<3>(x - 1) + 1170*ipow<2>(y)*ipow<4>(x - 1) + 72*y*ipow<5>(x - 1) + ipow<6>(x - 1)),
            (6*x + 6*y + 12*z - 6)*(12376*ipow<6>(y) + ipow<5>(y)*(26208*x - 26208) + 20475*ipow<4>(y)*ipow<2>(x - 1) + 7280*ipow<3>(y)*ipow<3>(x - 1) + 1170*ipow<2>(y)*ipow<4>(x - 1) + 72*y*ipow<5>(x - 1) + ipow<6>(x - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 144
template<>
struct DGBasis<144> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)) + (14*x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 14*x*(2*x*(17*x*(18*x - 25) + 200) + x*(34*x*(18*x - 25) + 2*x*(612*x - 425) + 400) - 75) + 70)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)),
            (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 145
template<>
struct DGBasis<145> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(x + 9*y - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(x + 9*y - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)) + (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (x + 9*y - 1)*(20*x*(17*x*(9*x - 8) + 36) + 20*x*(17*x*(9*x - 8) + x*(306*x - 136) + 36) - 60)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(x + 9*y - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)) + 9*(20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(x + 9*y - 1)*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 146
template<>
struct DGBasis<146> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (24*x*(17*x*(2*x - 1) + 2) - 1)*(55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(2*x + 20*y - 2)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (24*x*(17*x*(2*x - 1) + 2) - 1)*(55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)) + (55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(408*x*(2*x - 1) + 24*x*(68*x - 17) + 48)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(20*x + 110*y - 20)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (24*x*(17*x*(2*x - 1) + 2) - 1)*(55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)),
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 147
template<>
struct DGBasis<147> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (17*x*(9*x - 2) + 1)*(286*ipow<3>(y) + ipow<2>(y)*(198*x - 198) + 33*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (306*x - 34)*(286*ipow<3>(y) + ipow<2>(y)*(198*x - 198) + 33*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (17*x*(9*x - 2) + 1)*(198*ipow<2>(y) + 33*y*(2*x - 2) + 3*ipow<2>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (17*x*(9*x - 2) + 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(286*ipow<3>(y) + ipow<2>(y)*(198*x - 198) + 33*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (17*x*(9*x - 2) + 1)*(858*ipow<2>(y) + 2*y*(198*x - 198) + 33*ipow<2>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (17*x*(9*x - 2) + 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(286*ipow<3>(y) + ipow<2>(y)*(198*x - 198) + 33*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (17*x*(9*x - 2) + 1)*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))*(286*ipow<3>(y) + ipow<2>(y)*(198*x - 198) + 33*y*ipow<2>(x - 1) + ipow<3>(x - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 148
template<>
struct DGBasis<148> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (18*x - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(1365*ipow<4>(y) + ipow<3>(y)*(1456*x - 1456) + 468*ipow<2>(y)*ipow<2>(x - 1) + 48*y*ipow<3>(x - 1) + ipow<4>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (18*x - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(1365*ipow<4>(y) + ipow<3>(y)*(1456*x - 1456) + 468*ipow<2>(y)*ipow<2>(x - 1) + 48*y*ipow<3>(x - 1) + ipow<4>(x - 1)) + (18*x - 1)*(1456*ipow<3>(y) + 468*ipow<2>(y)*(2*x - 2) + 144*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + 18*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(1365*ipow<4>(y) + ipow<3>(y)*(1456*x - 1456) + 468*ipow<2>(y)*ipow<2>(x - 1) + 48*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (18*x - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(1365*ipow<4>(y) + ipow<3>(y)*(1456*x - 1456) + 468*ipow<2>(y)*ipow<2>(x - 1) + 48*y*ipow<3>(x - 1) + ipow<4>(x - 1)) + (18*x - 1)*(5460*ipow<3>(y) + 3*ipow<2>(y)*(1456*x - 1456) + 936*y*ipow<2>(x - 1) + 48*ipow<3>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (18*x - 1)*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))*(1365*ipow<4>(y) + ipow<3>(y)*(1456*x - 1456) + 468*ipow<2>(y)*ipow<2>(x - 1) + 48*y*ipow<3>(x - 1) + ipow<4>(x - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 149
template<>
struct DGBasis<149> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(6188*ipow<5>(y) + ipow<4>(y)*(9100*x - 9100) + 4550*ipow<3>(y)*ipow<2>(x - 1) + 910*ipow<2>(y)*ipow<3>(x - 1) + 65*y*ipow<4>(x - 1) + ipow<5>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(6188*ipow<5>(y) + ipow<4>(y)*(9100*x - 9100) + 4550*ipow<3>(y)*ipow<2>(x - 1) + 910*ipow<2>(y)*ipow<3>(x - 1) + 65*y*ipow<4>(x - 1) + ipow<5>(x - 1)) + (20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(9100*ipow<4>(y) + 4550*ipow<3>(y)*(2*x - 2) + 2730*ipow<2>(y)*ipow<2>(x - 1) + 260*y*ipow<3>(x - 1) + 5*ipow<4>(x - 1)),
            (30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(6188*ipow<5>(y) + ipow<4>(y)*(9100*x - 9100) + 4550*ipow<3>(y)*ipow<2>(x - 1) + 910*ipow<2>(y)*ipow<3>(x - 1) + 65*y*ipow<4>(x - 1) + ipow<5>(x - 1)) + (20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(30940*ipow<4>(y) + 4*ipow<3>(y)*(9100*x - 9100) + 13650*ipow<2>(y)*ipow<2>(x - 1) + 1820*y*ipow<3>(x - 1) + 65*ipow<4>(x - 1)),
            (60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))*(6188*ipow<5>(y) + ipow<4>(y)*(9100*x - 9100) + 4550*ipow<3>(y)*ipow<2>(x - 1) + 910*ipow<2>(y)*ipow<3>(x - 1) + 65*y*ipow<4>(x - 1) + ipow<5>(x - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 150
template<>
struct DGBasis<150> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)) + (20*x*(17*x*(9*x - 8) + 36) + 20*x*(17*x*(9*x - 8) + x*(306*x - 136) + 36) - 60)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)),
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)),
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 151
template<>
struct DGBasis<151> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (24*x*(17*x*(2*x - 1) + 2) - 1)*(x + 11*y - 1)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(x + 11*y - 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)) + (24*x*(17*x*(2*x - 1) + 2) - 1)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (x + 11*y - 1)*(408*x*(2*x - 1) + 24*x*(68*x - 17) + 48)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)),
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(x + 11*y - 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)) + 11*(24*x*(17*x*(2*x - 1) + 2) - 1)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)),
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(x + 11*y - 1)*(280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 152
template<>
struct DGBasis<152> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (17*x*(9*x - 2) + 1)*(78*ipow<2>(y) + y*(24*x - 24) + ipow<2>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (306*x - 34)*(78*ipow<2>(y) + y*(24*x - 24) + ipow<2>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (17*x*(9*x - 2) + 1)*(2*x + 24*y - 2)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (17*x*(9*x - 2) + 1)*(78*ipow<2>(y) + y*(24*x - 24) + ipow<2>(x - 1))*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)),
            (17*x*(9*x - 2) + 1)*(24*x + 156*y - 24)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (17*x*(9*x - 2) + 1)*(78*ipow<2>(y) + y*(24*x - 24) + ipow<2>(x - 1))*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)),
            (17*x*(9*x - 2) + 1)*(78*ipow<2>(y) + y*(24*x - 24) + ipow<2>(x - 1))*(280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 153
template<>
struct DGBasis<153> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (18*x - 1)*(455*ipow<3>(y) + ipow<2>(y)*(273*x - 273) + 39*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (18*x - 1)*(273*ipow<2>(y) + 39*y*(2*x - 2) + 3*ipow<2>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (18*x - 1)*(455*ipow<3>(y) + ipow<2>(y)*(273*x - 273) + 39*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)) + 18*(455*ipow<3>(y) + ipow<2>(y)*(273*x - 273) + 39*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)),
            (18*x - 1)*(1365*ipow<2>(y) + 2*y*(273*x - 273) + 39*ipow<2>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (18*x - 1)*(455*ipow<3>(y) + ipow<2>(y)*(273*x - 273) + 39*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)),
            (18*x - 1)*(455*ipow<3>(y) + ipow<2>(y)*(273*x - 273) + 39*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 154
template<>
struct DGBasis<154> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (2380*ipow<4>(y) + ipow<3>(y)*(2240*x - 2240) + 630*ipow<2>(y)*ipow<2>(x - 1) + 56*y*ipow<3>(x - 1) + ipow<4>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2240*ipow<3>(y) + 630*ipow<2>(y)*(2*x - 2) + 168*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1))*(2380*ipow<4>(y) + ipow<3>(y)*(2240*x - 2240) + 630*ipow<2>(y)*ipow<2>(x - 1) + 56*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (9520*ipow<3>(y) + 3*ipow<2>(y)*(2240*x - 2240) + 1260*y*ipow<2>(x - 1) + 56*ipow<3>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1))*(2380*ipow<4>(y) + ipow<3>(y)*(2240*x - 2240) + 630*ipow<2>(y)*ipow<2>(x - 1) + 56*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))*(2380*ipow<4>(y) + ipow<3>(y)*(2240*x - 2240) + 630*ipow<2>(y)*ipow<2>(x - 1) + 56*y*ipow<3>(x - 1) + ipow<4>(x - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 155
template<>
struct DGBasis<155> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (24*x*(17*x*(2*x - 1) + 2) - 1)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)) + (408*x*(2*x - 1) + 24*x*(68*x - 17) + 48)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)),
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)),
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(1260*ipow<4>(z) + 4*ipow<3>(z)*(630*x + 630*y - 630) + 1680*ipow<2>(z)*ipow<2>(x + y - 1) + 420*z*ipow<3>(x + y - 1) + 30*ipow<4>(x + y - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 156
template<>
struct DGBasis<156> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (17*x*(9*x - 2) + 1)*(x + 13*y - 1)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (306*x - 34)*(x + 13*y - 1)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (17*x*(9*x - 2) + 1)*(x + 13*y - 1)*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)) + (17*x*(9*x - 2) + 1)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)),
            (17*x*(9*x - 2) + 1)*(x + 13*y - 1)*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)) + 13*(17*x*(9*x - 2) + 1)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)),
            (17*x*(9*x - 2) + 1)*(x + 13*y - 1)*(1260*ipow<4>(z) + 4*ipow<3>(z)*(630*x + 630*y - 630) + 1680*ipow<2>(z)*ipow<2>(x + y - 1) + 420*z*ipow<3>(x + y - 1) + 30*ipow<4>(x + y - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 157
template<>
struct DGBasis<157> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (18*x - 1)*(105*ipow<2>(y) + y*(28*x - 28) + ipow<2>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (18*x - 1)*(2*x + 28*y - 2)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (18*x - 1)*(105*ipow<2>(y) + y*(28*x - 28) + ipow<2>(x - 1))*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)) + 18*(105*ipow<2>(y) + y*(28*x - 28) + ipow<2>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)),
            (18*x - 1)*(28*x + 210*y - 28)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (18*x - 1)*(105*ipow<2>(y) + y*(28*x - 28) + ipow<2>(x - 1))*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)),
            (18*x - 1)*(105*ipow<2>(y) + y*(28*x - 28) + ipow<2>(x - 1))*(1260*ipow<4>(z) + 4*ipow<3>(z)*(630*x + 630*y - 630) + 1680*ipow<2>(z)*ipow<2>(x + y - 1) + 420*z*ipow<3>(x + y - 1) + 30*ipow<4>(x + y - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 158
template<>
struct DGBasis<158> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (680*ipow<3>(y) + ipow<2>(y)*(360*x - 360) + 45*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (360*ipow<2>(y) + 45*y*(2*x - 2) + 3*ipow<2>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (680*ipow<3>(y) + ipow<2>(y)*(360*x - 360) + 45*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)),
            (2040*ipow<2>(y) + 2*y*(360*x - 360) + 45*ipow<2>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (680*ipow<3>(y) + ipow<2>(y)*(360*x - 360) + 45*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)),
            (680*ipow<3>(y) + ipow<2>(y)*(360*x - 360) + 45*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(1260*ipow<4>(z) + 4*ipow<3>(z)*(630*x + 630*y - 630) + 1680*ipow<2>(z)*ipow<2>(x + y - 1) + 420*z*ipow<3>(x + y - 1) + 30*ipow<4>(x + y - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 159
template<>
struct DGBasis<159> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (17*x*(9*x - 2) + 1)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (306*x - 34)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)) + (17*x*(9*x - 2) + 1)*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)),
            (17*x*(9*x - 2) + 1)*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)),
            (17*x*(9*x - 2) + 1)*(5544*ipow<5>(z) + 5*ipow<4>(z)*(2772*x + 2772*y - 2772) + 12600*ipow<3>(z)*ipow<2>(x + y - 1) + 5040*ipow<2>(z)*ipow<3>(x + y - 1) + 840*z*ipow<4>(x + y - 1) + 42*ipow<5>(x + y - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 160
template<>
struct DGBasis<160> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (18*x - 1)*(x + 15*y - 1)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (18*x - 1)*(x + 15*y - 1)*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)) + (18*x - 1)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)) + 18*(x + 15*y - 1)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)),
            (18*x - 1)*(x + 15*y - 1)*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)) + 15*(18*x - 1)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)),
            (18*x - 1)*(x + 15*y - 1)*(5544*ipow<5>(z) + 5*ipow<4>(z)*(2772*x + 2772*y - 2772) + 12600*ipow<3>(z)*ipow<2>(x + y - 1) + 5040*ipow<2>(z)*ipow<3>(x + y - 1) + 840*z*ipow<4>(x + y - 1) + 42*ipow<5>(x + y - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 161
template<>
struct DGBasis<161> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (136*ipow<2>(y) + y*(32*x - 32) + ipow<2>(x - 1))*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x + 32*y - 2)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)) + (136*ipow<2>(y) + y*(32*x - 32) + ipow<2>(x - 1))*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)),
            (32*x + 272*y - 32)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)) + (136*ipow<2>(y) + y*(32*x - 32) + ipow<2>(x - 1))*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)),
            (136*ipow<2>(y) + y*(32*x - 32) + ipow<2>(x - 1))*(5544*ipow<5>(z) + 5*ipow<4>(z)*(2772*x + 2772*y - 2772) + 12600*ipow<3>(z)*ipow<2>(x + y - 1) + 5040*ipow<2>(z)*ipow<3>(x + y - 1) + 840*z*ipow<4>(x + y - 1) + 42*ipow<5>(x + y - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 162
template<>
struct DGBasis<162> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (18*x - 1)*(ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            18*ipow<7>(x) + 126*ipow<6>(x)*y - 126*ipow<6>(x) + 378*ipow<5>(x)*ipow<2>(y) - 756*ipow<5>(x)*y + 378*ipow<5>(x) + 630*ipow<4>(x)*ipow<3>(y) - 1890*ipow<4>(x)*ipow<2>(y) + 1890*ipow<4>(x)*y - 630*ipow<4>(x) + 630*ipow<3>(x)*ipow<4>(y) - 2520*ipow<3>(x)*ipow<3>(y) + 3780*ipow<3>(x)*ipow<2>(y) - 2520*ipow<3>(x)*y + 630*ipow<3>(x) + 378*ipow<2>(x)*ipow<5>(y) - 1890*ipow<2>(x)*ipow<4>(y) + 3780*ipow<2>(x)*ipow<3>(y) - 3780*ipow<2>(x)*ipow<2>(y) + 1890*ipow<2>(x)*y - 378*ipow<2>(x) + 126*x*ipow<6>(y) - 756*x*ipow<5>(y) + 1890*x*ipow<4>(y) - 2520*x*ipow<3>(y) + 1890*x*ipow<2>(y) - 756*x*y + 126*x + 18*ipow<7>(y) - 126*ipow<6>(y) + 378*ipow<5>(y) - 630*ipow<4>(y) + 630*ipow<3>(y) - 378*ipow<2>(y) + 126*y + 61776*ipow<7>(z) + 18*ipow<6>(z)*(12012*x + 12012*y - 12012) + 299376*ipow<5>(z)*ipow<2>(x + y - 1) + 207900*ipow<4>(z)*ipow<3>(x + y - 1) + 75600*ipow<3>(z)*ipow<4>(x + y - 1) + 13608*ipow<2>(z)*ipow<5>(x + y - 1) + 1008*z*ipow<6>(x + y - 1) + (18*x - 1)*(7*ipow<6>(x) + 42*ipow<5>(x)*y - 42*ipow<5>(x) + 105*ipow<4>(x)*ipow<2>(y) - 210*ipow<4>(x)*y + 105*ipow<4>(x) + 140*ipow<3>(x)*ipow<3>(y) - 420*ipow<3>(x)*ipow<2>(y) + 420*ipow<3>(x)*y - 140*ipow<3>(x) + 105*ipow<2>(x)*ipow<4>(y) - 420*ipow<2>(x)*ipow<3>(y) + 630*ipow<2>(x)*ipow<2>(y) - 420*ipow<2>(x)*y + 105*ipow<2>(x) + 42*x*ipow<5>(y) - 210*x*ipow<4>(y) + 420*x*ipow<3>(y) - 420*x*ipow<2>(y) + 210*x*y - 42*x + 7*ipow<6>(y) - 42*ipow<5>(y) + 105*ipow<4>(y) - 140*ipow<3>(y) + 105*ipow<2>(y) - 42*y + 12012*ipow<6>(z) + 16632*ipow<5>(z)*(2*x + 2*y - 2) + 34650*ipow<4>(z)*ipow<2>(x + y - 1) + 16800*ipow<3>(z)*ipow<3>(x + y - 1) + 3780*ipow<2>(z)*ipow<4>(x + y - 1) + 336*z*ipow<5>(x + y - 1) + 7) - 18,
            (18*x - 1)*(7*ipow<6>(x) + 42*ipow<5>(x)*y - 42*ipow<5>(x) + 105*ipow<4>(x)*ipow<2>(y) - 210*ipow<4>(x)*y + 105*ipow<4>(x) + 140*ipow<3>(x)*ipow<3>(y) - 420*ipow<3>(x)*ipow<2>(y) + 420*ipow<3>(x)*y - 140*ipow<3>(x) + 105*ipow<2>(x)*ipow<4>(y) - 420*ipow<2>(x)*ipow<3>(y) + 630*ipow<2>(x)*ipow<2>(y) - 420*ipow<2>(x)*y + 105*ipow<2>(x) + 42*x*ipow<5>(y) - 210*x*ipow<4>(y) + 420*x*ipow<3>(y) - 420*x*ipow<2>(y) + 210*x*y - 42*x + 7*ipow<6>(y) - 42*ipow<5>(y) + 105*ipow<4>(y) - 140*ipow<3>(y) + 105*ipow<2>(y) - 42*y + 12012*ipow<6>(z) + 16632*ipow<5>(z)*(2*x + 2*y - 2) + 34650*ipow<4>(z)*ipow<2>(x + y - 1) + 16800*ipow<3>(z)*ipow<3>(x + y - 1) + 3780*ipow<2>(z)*ipow<4>(x + y - 1) + 336*z*ipow<5>(x + y - 1) + 7),
            (18*x - 1)*(24024*ipow<6>(z) + 6*ipow<5>(z)*(12012*x + 12012*y - 12012) + 83160*ipow<4>(z)*ipow<2>(x + y - 1) + 46200*ipow<3>(z)*ipow<3>(x + y - 1) + 12600*ipow<2>(z)*ipow<4>(x + y - 1) + 1512*z*ipow<5>(x + y - 1) + 56*ipow<6>(x + y - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 163
template<>
struct DGBasis<163> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x + 17*y - 1)*(ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) + (x + 17*y - 1)*(7*ipow<6>(x) + 42*ipow<5>(x)*y - 42*ipow<5>(x) + 105*ipow<4>(x)*ipow<2>(y) - 210*ipow<4>(x)*y + 105*ipow<4>(x) + 140*ipow<3>(x)*ipow<3>(y) - 420*ipow<3>(x)*ipow<2>(y) + 420*ipow<3>(x)*y - 140*ipow<3>(x) + 105*ipow<2>(x)*ipow<4>(y) - 420*ipow<2>(x)*ipow<3>(y) + 630*ipow<2>(x)*ipow<2>(y) - 420*ipow<2>(x)*y + 105*ipow<2>(x) + 42*x*ipow<5>(y) - 210*x*ipow<4>(y) + 420*x*ipow<3>(y) - 420*x*ipow<2>(y) + 210*x*y - 42*x + 7*ipow<6>(y) - 42*ipow<5>(y) + 105*ipow<4>(y) - 140*ipow<3>(y) + 105*ipow<2>(y) - 42*y + 12012*ipow<6>(z) + 16632*ipow<5>(z)*(2*x + 2*y - 2) + 34650*ipow<4>(z)*ipow<2>(x + y - 1) + 16800*ipow<3>(z)*ipow<3>(x + y - 1) + 3780*ipow<2>(z)*ipow<4>(x + y - 1) + 336*z*ipow<5>(x + y - 1) + 7) - 1,
            17*ipow<7>(x) + 119*ipow<6>(x)*y - 119*ipow<6>(x) + 357*ipow<5>(x)*ipow<2>(y) - 714*ipow<5>(x)*y + 357*ipow<5>(x) + 595*ipow<4>(x)*ipow<3>(y) - 1785*ipow<4>(x)*ipow<2>(y) + 1785*ipow<4>(x)*y - 595*ipow<4>(x) + 595*ipow<3>(x)*ipow<4>(y) - 2380*ipow<3>(x)*ipow<3>(y) + 3570*ipow<3>(x)*ipow<2>(y) - 2380*ipow<3>(x)*y + 595*ipow<3>(x) + 357*ipow<2>(x)*ipow<5>(y) - 1785*ipow<2>(x)*ipow<4>(y) + 3570*ipow<2>(x)*ipow<3>(y) - 3570*ipow<2>(x)*ipow<2>(y) + 1785*ipow<2>(x)*y - 357*ipow<2>(x) + 119*x*ipow<6>(y) - 714*x*ipow<5>(y) + 1785*x*ipow<4>(y) - 2380*x*ipow<3>(y) + 1785*x*ipow<2>(y) - 714*x*y + 119*x + 17*ipow<7>(y) - 119*ipow<6>(y) + 357*ipow<5>(y) - 595*ipow<4>(y) + 595*ipow<3>(y) - 357*ipow<2>(y) + 119*y + 58344*ipow<7>(z) + 17*ipow<6>(z)*(12012*x + 12012*y - 12012) + 282744*ipow<5>(z)*ipow<2>(x + y - 1) + 196350*ipow<4>(z)*ipow<3>(x + y - 1) + 71400*ipow<3>(z)*ipow<4>(x + y - 1) + 12852*ipow<2>(z)*ipow<5>(x + y - 1) + 952*z*ipow<6>(x + y - 1) + (x + 17*y - 1)*(7*ipow<6>(x) + 42*ipow<5>(x)*y - 42*ipow<5>(x) + 105*ipow<4>(x)*ipow<2>(y) - 210*ipow<4>(x)*y + 105*ipow<4>(x) + 140*ipow<3>(x)*ipow<3>(y) - 420*ipow<3>(x)*ipow<2>(y) + 420*ipow<3>(x)*y - 140*ipow<3>(x) + 105*ipow<2>(x)*ipow<4>(y) - 420*ipow<2>(x)*ipow<3>(y) + 630*ipow<2>(x)*ipow<2>(y) - 420*ipow<2>(x)*y + 105*ipow<2>(x) + 42*x*ipow<5>(y) - 210*x*ipow<4>(y) + 420*x*ipow<3>(y) - 420*x*ipow<2>(y) + 210*x*y - 42*x + 7*ipow<6>(y) - 42*ipow<5>(y) + 105*ipow<4>(y) - 140*ipow<3>(y) + 105*ipow<2>(y) - 42*y + 12012*ipow<6>(z) + 16632*ipow<5>(z)*(2*x + 2*y - 2) + 34650*ipow<4>(z)*ipow<2>(x + y - 1) + 16800*ipow<3>(z)*ipow<3>(x + y - 1) + 3780*ipow<2>(z)*ipow<4>(x + y - 1) + 336*z*ipow<5>(x + y - 1) + 7) - 17,
            (x + 17*y - 1)*(24024*ipow<6>(z) + 6*ipow<5>(z)*(12012*x + 12012*y - 12012) + 83160*ipow<4>(z)*ipow<2>(x + y - 1) + 46200*ipow<3>(z)*ipow<3>(x + y - 1) + 12600*ipow<2>(z)*ipow<4>(x + y - 1) + 1512*z*ipow<5>(x + y - 1) + 56*ipow<6>(x + y - 1))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 164
template<>
struct DGBasis<164> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return ipow<8>(x) + 8*ipow<7>(x)*y - 8*ipow<7>(x) + 28*ipow<6>(x)*ipow<2>(y) - 56*ipow<6>(x)*y + 28*ipow<6>(x) + 56*ipow<5>(x)*ipow<3>(y) - 168*ipow<5>(x)*ipow<2>(y) + 168*ipow<5>(x)*y - 56*ipow<5>(x) + 70*ipow<4>(x)*ipow<4>(y) - 280*ipow<4>(x)*ipow<3>(y) + 420*ipow<4>(x)*ipow<2>(y) - 280*ipow<4>(x)*y + 70*ipow<4>(x) + 56*ipow<3>(x)*ipow<5>(y) - 280*ipow<3>(x)*ipow<4>(y) + 560*ipow<3>(x)*ipow<3>(y) - 560*ipow<3>(x)*ipow<2>(y) + 280*ipow<3>(x)*y - 56*ipow<3>(x) + 28*ipow<2>(x)*ipow<6>(y) - 168*ipow<2>(x)*ipow<5>(y) + 420*ipow<2>(x)*ipow<4>(y) - 560*ipow<2>(x)*ipow<3>(y) + 420*ipow<2>(x)*ipow<2>(y) - 168*ipow<2>(x)*y + 28*ipow<2>(x) + 8*x*ipow<7>(y) - 56*x*ipow<6>(y) + 168*x*ipow<5>(y) - 280*x*ipow<4>(y) + 280*x*ipow<3>(y) - 168*x*ipow<2>(y) + 56*x*y - 8*x + ipow<8>(y) - 8*ipow<7>(y) + 28*ipow<6>(y) - 56*ipow<5>(y) + 70*ipow<4>(y) - 56*ipow<3>(y) + 28*ipow<2>(y) - 8*y + 12870*ipow<8>(z) + ipow<7>(z)*(51480*x + 51480*y - 51480) + 84084*ipow<6>(z)*ipow<2>(x + y - 1) + 72072*ipow<5>(z)*ipow<3>(x + y - 1) + 34650*ipow<4>(z)*ipow<4>(x + y - 1) + 9240*ipow<3>(z)*ipow<5>(x + y - 1) + 1260*ipow<2>(z)*ipow<6>(x + y - 1) + 72*z*ipow<7>(x + y - 1) + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            8*ipow<7>(x) + 56*ipow<6>(x)*y - 56*ipow<6>(x) + 168*ipow<5>(x)*ipow<2>(y) - 336*ipow<5>(x)*y + 168*ipow<5>(x) + 280*ipow<4>(x)*ipow<3>(y) - 840*ipow<4>(x)*ipow<2>(y) + 840*ipow<4>(x)*y - 280*ipow<4>(x) + 280*ipow<3>(x)*ipow<4>(y) - 1120*ipow<3>(x)*ipow<3>(y) + 1680*ipow<3>(x)*ipow<2>(y) - 1120*ipow<3>(x)*y + 280*ipow<3>(x) + 168*ipow<2>(x)*ipow<5>(y) - 840*ipow<2>(x)*ipow<4>(y) + 1680*ipow<2>(x)*ipow<3>(y) - 1680*ipow<2>(x)*ipow<2>(y) + 840*ipow<2>(x)*y - 168*ipow<2>(x) + 56*x*ipow<6>(y) - 336*x*ipow<5>(y) + 840*x*ipow<4>(y) - 1120*x*ipow<3>(y) + 840*x*ipow<2>(y) - 336*x*y + 56*x + 8*ipow<7>(y) - 56*ipow<6>(y) + 168*ipow<5>(y) - 280*ipow<4>(y) + 280*ipow<3>(y) - 168*ipow<2>(y) + 56*y + 51480*ipow<7>(z) + 84084*ipow<6>(z)*(2*x + 2*y - 2) + 216216*ipow<5>(z)*ipow<2>(x + y - 1) + 138600*ipow<4>(z)*ipow<3>(x + y - 1) + 46200*ipow<3>(z)*ipow<4>(x + y - 1) + 7560*ipow<2>(z)*ipow<5>(x + y - 1) + 504*z*ipow<6>(x + y - 1) - 8,
            8*ipow<7>(x) + 56*ipow<6>(x)*y - 56*ipow<6>(x) + 168*ipow<5>(x)*ipow<2>(y) - 336*ipow<5>(x)*y + 168*ipow<5>(x) + 280*ipow<4>(x)*ipow<3>(y) - 840*ipow<4>(x)*ipow<2>(y) + 840*ipow<4>(x)*y - 280*ipow<4>(x) + 280*ipow<3>(x)*ipow<4>(y) - 1120*ipow<3>(x)*ipow<3>(y) + 1680*ipow<3>(x)*ipow<2>(y) - 1120*ipow<3>(x)*y + 280*ipow<3>(x) + 168*ipow<2>(x)*ipow<5>(y) - 840*ipow<2>(x)*ipow<4>(y) + 1680*ipow<2>(x)*ipow<3>(y) - 1680*ipow<2>(x)*ipow<2>(y) + 840*ipow<2>(x)*y - 168*ipow<2>(x) + 56*x*ipow<6>(y) - 336*x*ipow<5>(y) + 840*x*ipow<4>(y) - 1120*x*ipow<3>(y) + 840*x*ipow<2>(y) - 336*x*y + 56*x + 8*ipow<7>(y) - 56*ipow<6>(y) + 168*ipow<5>(y) - 280*ipow<4>(y) + 280*ipow<3>(y) - 168*ipow<2>(y) + 56*y + 51480*ipow<7>(z) + 84084*ipow<6>(z)*(2*x + 2*y - 2) + 216216*ipow<5>(z)*ipow<2>(x + y - 1) + 138600*ipow<4>(z)*ipow<3>(x + y - 1) + 46200*ipow<3>(z)*ipow<4>(x + y - 1) + 7560*ipow<2>(z)*ipow<5>(x + y - 1) + 504*z*ipow<6>(x + y - 1) - 8,
            102960*ipow<7>(z) + 7*ipow<6>(z)*(51480*x + 51480*y - 51480) + 504504*ipow<5>(z)*ipow<2>(x + y - 1) + 360360*ipow<4>(z)*ipow<3>(x + y - 1) + 138600*ipow<3>(z)*ipow<4>(x + y - 1) + 27720*ipow<2>(z)*ipow<5>(x + y - 1) + 2520*z*ipow<6>(x + y - 1) + 72*ipow<7>(x + y - 1)
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 165
template<>
struct DGBasis<165> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return 2*x*(13*x*(x*(x*(x*(17*x*(x*(19*x*(20*x - 81) + 2592) - 2352) + 21168) - 6615) + 1176) - 108) + 54) - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            26*x*(x*(x*(x*(17*x*(x*(19*x*(20*x - 81) + 2592) - 2352) + 21168) - 6615) + 1176) - 108) + 2*x*(13*x*(x*(x*(17*x*(x*(19*x*(20*x - 81) + 2592) - 2352) + 21168) - 6615) + 1176) + 13*x*(x*(x*(17*x*(x*(19*x*(20*x - 81) + 2592) - 2352) + 21168) - 6615) + x*(x*(17*x*(x*(19*x*(20*x - 81) + 2592) - 2352) + 21168) + x*(17*x*(x*(19*x*(20*x - 81) + 2592) - 2352) + x*(17*x*(19*x*(20*x - 81) + 2592) + 17*x*(19*x*(20*x - 81) + x*(760*x - 1539) + 2592) - 39984) + 21168) - 6615) + 1176) - 1404) + 108,
            0,
            0
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 166
template<>
struct DGBasis<166> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (26*x*(x*(x*(x*(17*x*(3*x*(19*x*(5*x - 16) + 392) - 784) + 4900) - 980) + 98) - 4) + 1)*(x + 3*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            26*x*(x*(x*(x*(17*x*(3*x*(19*x*(5*x - 16) + 392) - 784) + 4900) - 980) + 98) - 4) + (x + 3*y - 1)*(26*x*(x*(x*(17*x*(3*x*(19*x*(5*x - 16) + 392) - 784) + 4900) - 980) + 98) + 26*x*(x*(x*(17*x*(3*x*(19*x*(5*x - 16) + 392) - 784) + 4900) - 980) + x*(x*(17*x*(3*x*(19*x*(5*x - 16) + 392) - 784) + 4900) + x*(17*x*(3*x*(19*x*(5*x - 16) + 392) - 784) + x*(51*x*(19*x*(5*x - 16) + 392) + 17*x*(57*x*(5*x - 16) + 3*x*(190*x - 304) + 1176) - 13328) + 4900) - 980) + 98) - 104) + 1,
            78*x*(x*(x*(x*(17*x*(3*x*(19*x*(5*x - 16) + 392) - 784) + 4900) - 980) + 98) - 4) + 3,
            0
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 167
template<>
struct DGBasis<167> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x*(x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) - 2205) + 98) - 1)*(10*ipow<2>(y) + y*(8*x - 8) + ipow<2>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (x*(x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) - 2205) + 98) - 1)*(2*x + 8*y - 2) + (10*ipow<2>(y) + y*(8*x - 8) + ipow<2>(x - 1))*(x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) - 2205) + x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) + x*(68*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4*x*(51*x*(19*x*(20*x - 49) + 882) + 17*x*(57*x*(20*x - 49) + 3*x*(760*x - 931) + 2646) - 20825) + 19600) - 2205) + 98),
            (x*(x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) - 2205) + 98) - 1)*(8*x + 20*y - 8),
            0
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 168
template<>
struct DGBasis<168> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(35*ipow<3>(y) + ipow<2>(y)*(45*x - 45) + 15*y*ipow<2>(x - 1) + ipow<3>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(45*ipow<2>(y) + 15*y*(2*x - 2) + 3*ipow<2>(x - 1)) + (4*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) + 2*x*(34*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 2*x*(51*x*(38*x*(5*x - 9) + 225) + 17*x*(114*x*(5*x - 9) + 3*x*(380*x - 342) + 675) - 3400) + 900) - 90)*(35*ipow<3>(y) + ipow<2>(y)*(45*x - 45) + 15*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(105*ipow<2>(y) + 2*y*(45*x - 45) + 15*ipow<2>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 169
template<>
struct DGBasis<169> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(126*ipow<4>(y) + ipow<3>(y)*(224*x - 224) + 126*ipow<2>(y)*ipow<2>(x - 1) + 24*y*ipow<3>(x - 1) + ipow<4>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(224*ipow<3>(y) + 126*ipow<2>(y)*(2*x - 2) + 72*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1)) + (68*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 4*x*(51*x*(19*x*(4*x - 5) + 40) + 17*x*(57*x*(4*x - 5) + 3*x*(152*x - 95) + 120) - 340) + 80)*(126*ipow<4>(y) + ipow<3>(y)*(224*x - 224) + 126*ipow<2>(y)*ipow<2>(x - 1) + 24*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(504*ipow<3>(y) + 3*ipow<2>(y)*(224*x - 224) + 252*y*ipow<2>(x - 1) + 24*ipow<3>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 170
template<>
struct DGBasis<170> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(462*ipow<5>(y) + ipow<4>(y)*(1050*x - 1050) + 840*ipow<3>(y)*ipow<2>(x - 1) + 280*ipow<2>(y)*ipow<3>(x - 1) + 35*y*ipow<4>(x - 1) + ipow<5>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(1050*ipow<4>(y) + 840*ipow<3>(y)*(2*x - 2) + 840*ipow<2>(y)*ipow<2>(x - 1) + 140*y*ipow<3>(x - 1) + 5*ipow<4>(x - 1)) + (51*x*(19*x*(5*x - 4) + 18) + 17*x*(57*x*(5*x - 4) + 3*x*(190*x - 76) + 54) - 68)*(462*ipow<5>(y) + ipow<4>(y)*(1050*x - 1050) + 840*ipow<3>(y)*ipow<2>(x - 1) + 280*ipow<2>(y)*ipow<3>(x - 1) + 35*y*ipow<4>(x - 1) + ipow<5>(x - 1)),
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(2310*ipow<4>(y) + 4*ipow<3>(y)*(1050*x - 1050) + 2520*ipow<2>(y)*ipow<2>(x - 1) + 560*y*ipow<3>(x - 1) + 35*ipow<4>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 171
template<>
struct DGBasis<171> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(20*x - 9) + 18) - 1)*(1716*ipow<6>(y) + ipow<5>(y)*(4752*x - 4752) + 4950*ipow<4>(y)*ipow<2>(x - 1) + 2400*ipow<3>(y)*ipow<3>(x - 1) + 540*ipow<2>(y)*ipow<4>(x - 1) + 48*y*ipow<5>(x - 1) + ipow<6>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(4752*ipow<5>(y) + 4950*ipow<4>(y)*(2*x - 2) + 7200*ipow<3>(y)*ipow<2>(x - 1) + 2160*ipow<2>(y)*ipow<3>(x - 1) + 240*y*ipow<4>(x - 1) + 6*ipow<5>(x - 1)) + (57*x*(20*x - 9) + 3*x*(760*x - 171) + 54)*(1716*ipow<6>(y) + ipow<5>(y)*(4752*x - 4752) + 4950*ipow<4>(y)*ipow<2>(x - 1) + 2400*ipow<3>(y)*ipow<3>(x - 1) + 540*ipow<2>(y)*ipow<4>(x - 1) + 48*y*ipow<5>(x - 1) + ipow<6>(x - 1)),
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(10296*ipow<5>(y) + 5*ipow<4>(y)*(4752*x - 4752) + 19800*ipow<3>(y)*ipow<2>(x - 1) + 7200*ipow<2>(y)*ipow<3>(x - 1) + 1080*y*ipow<4>(x - 1) + 48*ipow<5>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 172
template<>
struct DGBasis<172> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (38*x*(5*x - 1) + 1)*(6435*ipow<7>(y) + ipow<6>(y)*(21021*x - 21021) + 27027*ipow<5>(y)*ipow<2>(x - 1) + 17325*ipow<4>(y)*ipow<3>(x - 1) + 5775*ipow<3>(y)*ipow<4>(x - 1) + 945*ipow<2>(y)*ipow<5>(x - 1) + 63*y*ipow<6>(x - 1) + ipow<7>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (380*x - 38)*(6435*ipow<7>(y) + ipow<6>(y)*(21021*x - 21021) + 27027*ipow<5>(y)*ipow<2>(x - 1) + 17325*ipow<4>(y)*ipow<3>(x - 1) + 5775*ipow<3>(y)*ipow<4>(x - 1) + 945*ipow<2>(y)*ipow<5>(x - 1) + 63*y*ipow<6>(x - 1) + ipow<7>(x - 1)) + (38*x*(5*x - 1) + 1)*(21021*ipow<6>(y) + 27027*ipow<5>(y)*(2*x - 2) + 51975*ipow<4>(y)*ipow<2>(x - 1) + 23100*ipow<3>(y)*ipow<3>(x - 1) + 4725*ipow<2>(y)*ipow<4>(x - 1) + 378*y*ipow<5>(x - 1) + 7*ipow<6>(x - 1)),
            (38*x*(5*x - 1) + 1)*(45045*ipow<6>(y) + 6*ipow<5>(y)*(21021*x - 21021) + 135135*ipow<4>(y)*ipow<2>(x - 1) + 69300*ipow<3>(y)*ipow<3>(x - 1) + 17325*ipow<2>(y)*ipow<4>(x - 1) + 1890*y*ipow<5>(x - 1) + 63*ipow<6>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 173
template<>
struct DGBasis<173> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (20*x - 1)*(24310*ipow<8>(y) + ipow<7>(y)*(91520*x - 91520) + 140140*ipow<6>(y)*ipow<2>(x - 1) + 112112*ipow<5>(y)*ipow<3>(x - 1) + 50050*ipow<4>(y)*ipow<4>(x - 1) + 12320*ipow<3>(y)*ipow<5>(x - 1) + 1540*ipow<2>(y)*ipow<6>(x - 1) + 80*y*ipow<7>(x - 1) + ipow<8>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            486200*ipow<8>(y) + 20*ipow<7>(y)*(91520*x - 91520) + 2802800*ipow<6>(y)*ipow<2>(x - 1) + 2242240*ipow<5>(y)*ipow<3>(x - 1) + 1001000*ipow<4>(y)*ipow<4>(x - 1) + 246400*ipow<3>(y)*ipow<5>(x - 1) + 30800*ipow<2>(y)*ipow<6>(x - 1) + 1600*y*ipow<7>(x - 1) + 20*ipow<8>(x - 1) + (20*x - 1)*(91520*ipow<7>(y) + 140140*ipow<6>(y)*(2*x - 2) + 336336*ipow<5>(y)*ipow<2>(x - 1) + 200200*ipow<4>(y)*ipow<3>(x - 1) + 61600*ipow<3>(y)*ipow<4>(x - 1) + 9240*ipow<2>(y)*ipow<5>(x - 1) + 560*y*ipow<6>(x - 1) + 8*ipow<7>(x - 1)),
            (20*x - 1)*(194480*ipow<7>(y) + 7*ipow<6>(y)*(91520*x - 91520) + 840840*ipow<5>(y)*ipow<2>(x - 1) + 560560*ipow<4>(y)*ipow<3>(x - 1) + 200200*ipow<3>(y)*ipow<4>(x - 1) + 36960*ipow<2>(y)*ipow<5>(x - 1) + 3080*y*ipow<6>(x - 1) + 80*ipow<7>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 174
template<>
struct DGBasis<174> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return 92378*ipow<9>(y) + ipow<8>(y)*(393822*x - 393822) + 700128*ipow<7>(y)*ipow<2>(x - 1) + 672672*ipow<6>(y)*ipow<3>(x - 1) + 378378*ipow<5>(y)*ipow<4>(x - 1) + 126126*ipow<4>(y)*ipow<5>(x - 1) + 24024*ipow<3>(y)*ipow<6>(x - 1) + 2376*ipow<2>(y)*ipow<7>(x - 1) + 99*y*ipow<8>(x - 1) + ipow<9>(x - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            393822*ipow<8>(y) + 700128*ipow<7>(y)*(2*x - 2) + 2018016*ipow<6>(y)*ipow<2>(x - 1) + 1513512*ipow<5>(y)*ipow<3>(x - 1) + 630630*ipow<4>(y)*ipow<4>(x - 1) + 144144*ipow<3>(y)*ipow<5>(x - 1) + 16632*ipow<2>(y)*ipow<6>(x - 1) + 792*y*ipow<7>(x - 1) + 9*ipow<8>(x - 1),
            831402*ipow<8>(y) + 8*ipow<7>(y)*(393822*x - 393822) + 4900896*ipow<6>(y)*ipow<2>(x - 1) + 4036032*ipow<5>(y)*ipow<3>(x - 1) + 1891890*ipow<4>(y)*ipow<4>(x - 1) + 504504*ipow<3>(y)*ipow<5>(x - 1) + 72072*ipow<2>(y)*ipow<6>(x - 1) + 4752*y*ipow<7>(x - 1) + 99*ipow<8>(x - 1),
            0
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 175
template<>
struct DGBasis<175> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (26*x*(x*(x*(x*(17*x*(3*x*(19*x*(5*x - 16) + 392) - 784) + 4900) - 980) + 98) - 4) + 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            26*x*(x*(x*(x*(17*x*(3*x*(19*x*(5*x - 16) + 392) - 784) + 4900) - 980) + 98) - 4) + (26*x*(x*(x*(17*x*(3*x*(19*x*(5*x - 16) + 392) - 784) + 4900) - 980) + 98) + 26*x*(x*(x*(17*x*(3*x*(19*x*(5*x - 16) + 392) - 784) + 4900) - 980) + x*(x*(17*x*(3*x*(19*x*(5*x - 16) + 392) - 784) + 4900) + x*(17*x*(3*x*(19*x*(5*x - 16) + 392) - 784) + x*(51*x*(19*x*(5*x - 16) + 392) + 17*x*(57*x*(5*x - 16) + 3*x*(190*x - 304) + 1176) - 13328) + 4900) - 980) + 98) - 104)*(x + y + 2*z - 1) + 1,
            26*x*(x*(x*(x*(17*x*(3*x*(19*x*(5*x - 16) + 392) - 784) + 4900) - 980) + 98) - 4) + 1,
            52*x*(x*(x*(x*(17*x*(3*x*(19*x*(5*x - 16) + 392) - 784) + 4900) - 980) + 98) - 4) + 2
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 176
template<>
struct DGBasis<176> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x*(x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) - 2205) + 98) - 1)*(x + 5*y - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (x*(x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) - 2205) + 98) - 1)*(x + 5*y - 1) + (x*(x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) - 2205) + 98) - 1)*(x + y + 2*z - 1) + (x + 5*y - 1)*(x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) - 2205) + x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) + x*(68*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4*x*(51*x*(19*x*(20*x - 49) + 882) + 17*x*(57*x*(20*x - 49) + 3*x*(760*x - 931) + 2646) - 20825) + 19600) - 2205) + 98)*(x + y + 2*z - 1),
            (x*(x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) - 2205) + 98) - 1)*(x + 5*y - 1) + 5*(x*(x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) - 2205) + 98) - 1)*(x + y + 2*z - 1),
            2*(x*(x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) - 2205) + 98) - 1)*(x + 5*y - 1)
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 177
template<>
struct DGBasis<177> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1))*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(2*x + 12*y - 2)*(x + y + 2*z - 1) + (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1)) + (21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1))*(4*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) + 2*x*(34*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 2*x*(51*x*(38*x*(5*x - 9) + 225) + 17*x*(114*x*(5*x - 9) + 3*x*(380*x - 342) + 675) - 3400) + 900) - 90)*(x + y + 2*z - 1),
            (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(12*x + 42*y - 12)*(x + y + 2*z - 1) + (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1)),
            2*(2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 178
template<>
struct DGBasis<178> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(x + y + 2*z - 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(84*ipow<2>(y) + 21*y*(2*x - 2) + 3*ipow<2>(x - 1))*(x + y + 2*z - 1) + (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1)) + (68*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 4*x*(51*x*(19*x*(4*x - 5) + 40) + 17*x*(57*x*(4*x - 5) + 3*x*(152*x - 95) + 120) - 340) + 80)*(x + y + 2*z - 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(252*ipow<2>(y) + 2*y*(84*x - 84) + 21*ipow<2>(x - 1))*(x + y + 2*z - 1) + (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            2*(4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 179
template<>
struct DGBasis<179> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(x + y + 2*z - 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(x + y + 2*z - 1)*(480*ipow<3>(y) + 216*ipow<2>(y)*(2*x - 2) + 96*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1)) + (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1)) + (51*x*(19*x*(5*x - 4) + 18) + 17*x*(57*x*(5*x - 4) + 3*x*(190*x - 76) + 54) - 68)*(x + y + 2*z - 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(x + y + 2*z - 1)*(1320*ipow<3>(y) + 3*ipow<2>(y)*(480*x - 480) + 432*y*ipow<2>(x - 1) + 32*ipow<3>(x - 1)) + (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            2*(17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 180
template<>
struct DGBasis<180> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(20*x - 9) + 18) - 1)*(x + y + 2*z - 1)*(1287*ipow<5>(y) + ipow<4>(y)*(2475*x - 2475) + 1650*ipow<3>(y)*ipow<2>(x - 1) + 450*ipow<2>(y)*ipow<3>(x - 1) + 45*y*ipow<4>(x - 1) + ipow<5>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(x + y + 2*z - 1)*(2475*ipow<4>(y) + 1650*ipow<3>(y)*(2*x - 2) + 1350*ipow<2>(y)*ipow<2>(x - 1) + 180*y*ipow<3>(x - 1) + 5*ipow<4>(x - 1)) + (3*x*(19*x*(20*x - 9) + 18) - 1)*(1287*ipow<5>(y) + ipow<4>(y)*(2475*x - 2475) + 1650*ipow<3>(y)*ipow<2>(x - 1) + 450*ipow<2>(y)*ipow<3>(x - 1) + 45*y*ipow<4>(x - 1) + ipow<5>(x - 1)) + (57*x*(20*x - 9) + 3*x*(760*x - 171) + 54)*(x + y + 2*z - 1)*(1287*ipow<5>(y) + ipow<4>(y)*(2475*x - 2475) + 1650*ipow<3>(y)*ipow<2>(x - 1) + 450*ipow<2>(y)*ipow<3>(x - 1) + 45*y*ipow<4>(x - 1) + ipow<5>(x - 1)),
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(x + y + 2*z - 1)*(6435*ipow<4>(y) + 4*ipow<3>(y)*(2475*x - 2475) + 4950*ipow<2>(y)*ipow<2>(x - 1) + 900*y*ipow<3>(x - 1) + 45*ipow<4>(x - 1)) + (3*x*(19*x*(20*x - 9) + 18) - 1)*(1287*ipow<5>(y) + ipow<4>(y)*(2475*x - 2475) + 1650*ipow<3>(y)*ipow<2>(x - 1) + 450*ipow<2>(y)*ipow<3>(x - 1) + 45*y*ipow<4>(x - 1) + ipow<5>(x - 1)),
            2*(3*x*(19*x*(20*x - 9) + 18) - 1)*(1287*ipow<5>(y) + ipow<4>(y)*(2475*x - 2475) + 1650*ipow<3>(y)*ipow<2>(x - 1) + 450*ipow<2>(y)*ipow<3>(x - 1) + 45*y*ipow<4>(x - 1) + ipow<5>(x - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 181
template<>
struct DGBasis<181> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (38*x*(5*x - 1) + 1)*(x + y + 2*z - 1)*(5005*ipow<6>(y) + ipow<5>(y)*(12012*x - 12012) + 10725*ipow<4>(y)*ipow<2>(x - 1) + 4400*ipow<3>(y)*ipow<3>(x - 1) + 825*ipow<2>(y)*ipow<4>(x - 1) + 60*y*ipow<5>(x - 1) + ipow<6>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (380*x - 38)*(x + y + 2*z - 1)*(5005*ipow<6>(y) + ipow<5>(y)*(12012*x - 12012) + 10725*ipow<4>(y)*ipow<2>(x - 1) + 4400*ipow<3>(y)*ipow<3>(x - 1) + 825*ipow<2>(y)*ipow<4>(x - 1) + 60*y*ipow<5>(x - 1) + ipow<6>(x - 1)) + (38*x*(5*x - 1) + 1)*(x + y + 2*z - 1)*(12012*ipow<5>(y) + 10725*ipow<4>(y)*(2*x - 2) + 13200*ipow<3>(y)*ipow<2>(x - 1) + 3300*ipow<2>(y)*ipow<3>(x - 1) + 300*y*ipow<4>(x - 1) + 6*ipow<5>(x - 1)) + (38*x*(5*x - 1) + 1)*(5005*ipow<6>(y) + ipow<5>(y)*(12012*x - 12012) + 10725*ipow<4>(y)*ipow<2>(x - 1) + 4400*ipow<3>(y)*ipow<3>(x - 1) + 825*ipow<2>(y)*ipow<4>(x - 1) + 60*y*ipow<5>(x - 1) + ipow<6>(x - 1)),
            (38*x*(5*x - 1) + 1)*(x + y + 2*z - 1)*(30030*ipow<5>(y) + 5*ipow<4>(y)*(12012*x - 12012) + 42900*ipow<3>(y)*ipow<2>(x - 1) + 13200*ipow<2>(y)*ipow<3>(x - 1) + 1650*y*ipow<4>(x - 1) + 60*ipow<5>(x - 1)) + (38*x*(5*x - 1) + 1)*(5005*ipow<6>(y) + ipow<5>(y)*(12012*x - 12012) + 10725*ipow<4>(y)*ipow<2>(x - 1) + 4400*ipow<3>(y)*ipow<3>(x - 1) + 825*ipow<2>(y)*ipow<4>(x - 1) + 60*y*ipow<5>(x - 1) + ipow<6>(x - 1)),
            2*(38*x*(5*x - 1) + 1)*(5005*ipow<6>(y) + ipow<5>(y)*(12012*x - 12012) + 10725*ipow<4>(y)*ipow<2>(x - 1) + 4400*ipow<3>(y)*ipow<3>(x - 1) + 825*ipow<2>(y)*ipow<4>(x - 1) + 60*y*ipow<5>(x - 1) + ipow<6>(x - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 182
template<>
struct DGBasis<182> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (20*x - 1)*(x + y + 2*z - 1)*(19448*ipow<7>(y) + ipow<6>(y)*(56056*x - 56056) + 63063*ipow<5>(y)*ipow<2>(x - 1) + 35035*ipow<4>(y)*ipow<3>(x - 1) + 10010*ipow<3>(y)*ipow<4>(x - 1) + 1386*ipow<2>(y)*ipow<5>(x - 1) + 77*y*ipow<6>(x - 1) + ipow<7>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (20*x - 1)*(x + y + 2*z - 1)*(56056*ipow<6>(y) + 63063*ipow<5>(y)*(2*x - 2) + 105105*ipow<4>(y)*ipow<2>(x - 1) + 40040*ipow<3>(y)*ipow<3>(x - 1) + 6930*ipow<2>(y)*ipow<4>(x - 1) + 462*y*ipow<5>(x - 1) + 7*ipow<6>(x - 1)) + (20*x - 1)*(19448*ipow<7>(y) + ipow<6>(y)*(56056*x - 56056) + 63063*ipow<5>(y)*ipow<2>(x - 1) + 35035*ipow<4>(y)*ipow<3>(x - 1) + 10010*ipow<3>(y)*ipow<4>(x - 1) + 1386*ipow<2>(y)*ipow<5>(x - 1) + 77*y*ipow<6>(x - 1) + ipow<7>(x - 1)) + 20*(x + y + 2*z - 1)*(19448*ipow<7>(y) + ipow<6>(y)*(56056*x - 56056) + 63063*ipow<5>(y)*ipow<2>(x - 1) + 35035*ipow<4>(y)*ipow<3>(x - 1) + 10010*ipow<3>(y)*ipow<4>(x - 1) + 1386*ipow<2>(y)*ipow<5>(x - 1) + 77*y*ipow<6>(x - 1) + ipow<7>(x - 1)),
            (20*x - 1)*(x + y + 2*z - 1)*(136136*ipow<6>(y) + 6*ipow<5>(y)*(56056*x - 56056) + 315315*ipow<4>(y)*ipow<2>(x - 1) + 140140*ipow<3>(y)*ipow<3>(x - 1) + 30030*ipow<2>(y)*ipow<4>(x - 1) + 2772*y*ipow<5>(x - 1) + 77*ipow<6>(x - 1)) + (20*x - 1)*(19448*ipow<7>(y) + ipow<6>(y)*(56056*x - 56056) + 63063*ipow<5>(y)*ipow<2>(x - 1) + 35035*ipow<4>(y)*ipow<3>(x - 1) + 10010*ipow<3>(y)*ipow<4>(x - 1) + 1386*ipow<2>(y)*ipow<5>(x - 1) + 77*y*ipow<6>(x - 1) + ipow<7>(x - 1)),
            2*(20*x - 1)*(19448*ipow<7>(y) + ipow<6>(y)*(56056*x - 56056) + 63063*ipow<5>(y)*ipow<2>(x - 1) + 35035*ipow<4>(y)*ipow<3>(x - 1) + 10010*ipow<3>(y)*ipow<4>(x - 1) + 1386*ipow<2>(y)*ipow<5>(x - 1) + 77*y*ipow<6>(x - 1) + ipow<7>(x - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 183
template<>
struct DGBasis<183> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x + y + 2*z - 1)*(75582*ipow<8>(y) + ipow<7>(y)*(254592*x - 254592) + 346528*ipow<6>(y)*ipow<2>(x - 1) + 244608*ipow<5>(y)*ipow<3>(x - 1) + 95550*ipow<4>(y)*ipow<4>(x - 1) + 20384*ipow<3>(y)*ipow<5>(x - 1) + 2184*ipow<2>(y)*ipow<6>(x - 1) + 96*y*ipow<7>(x - 1) + ipow<8>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            75582*ipow<8>(y) + ipow<7>(y)*(254592*x - 254592) + 346528*ipow<6>(y)*ipow<2>(x - 1) + 244608*ipow<5>(y)*ipow<3>(x - 1) + 95550*ipow<4>(y)*ipow<4>(x - 1) + 20384*ipow<3>(y)*ipow<5>(x - 1) + 2184*ipow<2>(y)*ipow<6>(x - 1) + 96*y*ipow<7>(x - 1) + ipow<8>(x - 1) + (x + y + 2*z - 1)*(254592*ipow<7>(y) + 346528*ipow<6>(y)*(2*x - 2) + 733824*ipow<5>(y)*ipow<2>(x - 1) + 382200*ipow<4>(y)*ipow<3>(x - 1) + 101920*ipow<3>(y)*ipow<4>(x - 1) + 13104*ipow<2>(y)*ipow<5>(x - 1) + 672*y*ipow<6>(x - 1) + 8*ipow<7>(x - 1)),
            75582*ipow<8>(y) + ipow<7>(y)*(254592*x - 254592) + 346528*ipow<6>(y)*ipow<2>(x - 1) + 244608*ipow<5>(y)*ipow<3>(x - 1) + 95550*ipow<4>(y)*ipow<4>(x - 1) + 20384*ipow<3>(y)*ipow<5>(x - 1) + 2184*ipow<2>(y)*ipow<6>(x - 1) + 96*y*ipow<7>(x - 1) + ipow<8>(x - 1) + (x + y + 2*z - 1)*(604656*ipow<7>(y) + 7*ipow<6>(y)*(254592*x - 254592) + 2079168*ipow<5>(y)*ipow<2>(x - 1) + 1223040*ipow<4>(y)*ipow<3>(x - 1) + 382200*ipow<3>(y)*ipow<4>(x - 1) + 61152*ipow<2>(y)*ipow<5>(x - 1) + 4368*y*ipow<6>(x - 1) + 96*ipow<7>(x - 1)),
            151164*ipow<8>(y) + 2*ipow<7>(y)*(254592*x - 254592) + 693056*ipow<6>(y)*ipow<2>(x - 1) + 489216*ipow<5>(y)*ipow<3>(x - 1) + 191100*ipow<4>(y)*ipow<4>(x - 1) + 40768*ipow<3>(y)*ipow<5>(x - 1) + 4368*ipow<2>(y)*ipow<6>(x - 1) + 192*y*ipow<7>(x - 1) + 2*ipow<8>(x - 1)
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 184
template<>
struct DGBasis<184> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x*(x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) - 2205) + 98) - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (x*(x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) - 2205) + 98) - 1)*(2*x + 2*y + 6*z - 2) + (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) - 2205) + x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) + x*(68*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4*x*(51*x*(19*x*(20*x - 49) + 882) + 17*x*(57*x*(20*x - 49) + 3*x*(760*x - 931) + 2646) - 20825) + 19600) - 2205) + 98),
            (x*(x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) - 2205) + 98) - 1)*(2*x + 2*y + 6*z - 2),
            (x*(x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) - 2205) + 98) - 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 185
template<>
struct DGBasis<185> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(x + 7*y - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (x + 7*y - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(4*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) + 2*x*(34*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 2*x*(51*x*(38*x*(5*x - 9) + 225) + 17*x*(114*x*(5*x - 9) + 3*x*(380*x - 342) + 675) - 3400) + 900) - 90),
            (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + 7*(2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)),
            (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(x + 7*y - 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 186
template<>
struct DGBasis<186> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(2*x + 16*y - 2)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(2*x + 2*y + 6*z - 2) + (36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(68*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 4*x*(51*x*(19*x*(4*x - 5) + 40) + 17*x*(57*x*(4*x - 5) + 3*x*(152*x - 95) + 120) - 340) + 80),
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(16*x + 72*y - 16)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(2*x + 2*y + 6*z - 2),
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 187
template<>
struct DGBasis<187> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(135*ipow<2>(y) + 27*y*(2*x - 2) + 3*ipow<2>(x - 1))*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(2*x + 2*y + 6*z - 2)*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1)) + (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(51*x*(19*x*(5*x - 4) + 18) + 17*x*(57*x*(5*x - 4) + 3*x*(190*x - 76) + 54) - 68)*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(495*ipow<2>(y) + 2*y*(135*x - 135) + 27*ipow<2>(x - 1))*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(2*x + 2*y + 6*z - 2)*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(6*x + 6*y + 12*z - 6)*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 188
template<>
struct DGBasis<188> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(20*x - 9) + 18) - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(715*ipow<4>(y) + ipow<3>(y)*(880*x - 880) + 330*ipow<2>(y)*ipow<2>(x - 1) + 40*y*ipow<3>(x - 1) + ipow<4>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(880*ipow<3>(y) + 330*ipow<2>(y)*(2*x - 2) + 120*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1)) + (3*x*(19*x*(20*x - 9) + 18) - 1)*(2*x + 2*y + 6*z - 2)*(715*ipow<4>(y) + ipow<3>(y)*(880*x - 880) + 330*ipow<2>(y)*ipow<2>(x - 1) + 40*y*ipow<3>(x - 1) + ipow<4>(x - 1)) + (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(57*x*(20*x - 9) + 3*x*(760*x - 171) + 54)*(715*ipow<4>(y) + ipow<3>(y)*(880*x - 880) + 330*ipow<2>(y)*ipow<2>(x - 1) + 40*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(2860*ipow<3>(y) + 3*ipow<2>(y)*(880*x - 880) + 660*y*ipow<2>(x - 1) + 40*ipow<3>(x - 1)) + (3*x*(19*x*(20*x - 9) + 18) - 1)*(2*x + 2*y + 6*z - 2)*(715*ipow<4>(y) + ipow<3>(y)*(880*x - 880) + 330*ipow<2>(y)*ipow<2>(x - 1) + 40*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(6*x + 6*y + 12*z - 6)*(715*ipow<4>(y) + ipow<3>(y)*(880*x - 880) + 330*ipow<2>(y)*ipow<2>(x - 1) + 40*y*ipow<3>(x - 1) + ipow<4>(x - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 189
template<>
struct DGBasis<189> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (38*x*(5*x - 1) + 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(3003*ipow<5>(y) + ipow<4>(y)*(5005*x - 5005) + 2860*ipow<3>(y)*ipow<2>(x - 1) + 660*ipow<2>(y)*ipow<3>(x - 1) + 55*y*ipow<4>(x - 1) + ipow<5>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (380*x - 38)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(3003*ipow<5>(y) + ipow<4>(y)*(5005*x - 5005) + 2860*ipow<3>(y)*ipow<2>(x - 1) + 660*ipow<2>(y)*ipow<3>(x - 1) + 55*y*ipow<4>(x - 1) + ipow<5>(x - 1)) + (38*x*(5*x - 1) + 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(5005*ipow<4>(y) + 2860*ipow<3>(y)*(2*x - 2) + 1980*ipow<2>(y)*ipow<2>(x - 1) + 220*y*ipow<3>(x - 1) + 5*ipow<4>(x - 1)) + (38*x*(5*x - 1) + 1)*(2*x + 2*y + 6*z - 2)*(3003*ipow<5>(y) + ipow<4>(y)*(5005*x - 5005) + 2860*ipow<3>(y)*ipow<2>(x - 1) + 660*ipow<2>(y)*ipow<3>(x - 1) + 55*y*ipow<4>(x - 1) + ipow<5>(x - 1)),
            (38*x*(5*x - 1) + 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(15015*ipow<4>(y) + 4*ipow<3>(y)*(5005*x - 5005) + 8580*ipow<2>(y)*ipow<2>(x - 1) + 1320*y*ipow<3>(x - 1) + 55*ipow<4>(x - 1)) + (38*x*(5*x - 1) + 1)*(2*x + 2*y + 6*z - 2)*(3003*ipow<5>(y) + ipow<4>(y)*(5005*x - 5005) + 2860*ipow<3>(y)*ipow<2>(x - 1) + 660*ipow<2>(y)*ipow<3>(x - 1) + 55*y*ipow<4>(x - 1) + ipow<5>(x - 1)),
            (38*x*(5*x - 1) + 1)*(6*x + 6*y + 12*z - 6)*(3003*ipow<5>(y) + ipow<4>(y)*(5005*x - 5005) + 2860*ipow<3>(y)*ipow<2>(x - 1) + 660*ipow<2>(y)*ipow<3>(x - 1) + 55*y*ipow<4>(x - 1) + ipow<5>(x - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 190
template<>
struct DGBasis<190> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (20*x - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(12376*ipow<6>(y) + ipow<5>(y)*(26208*x - 26208) + 20475*ipow<4>(y)*ipow<2>(x - 1) + 7280*ipow<3>(y)*ipow<3>(x - 1) + 1170*ipow<2>(y)*ipow<4>(x - 1) + 72*y*ipow<5>(x - 1) + ipow<6>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (20*x - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(26208*ipow<5>(y) + 20475*ipow<4>(y)*(2*x - 2) + 21840*ipow<3>(y)*ipow<2>(x - 1) + 4680*ipow<2>(y)*ipow<3>(x - 1) + 360*y*ipow<4>(x - 1) + 6*ipow<5>(x - 1)) + (20*x - 1)*(2*x + 2*y + 6*z - 2)*(12376*ipow<6>(y) + ipow<5>(y)*(26208*x - 26208) + 20475*ipow<4>(y)*ipow<2>(x - 1) + 7280*ipow<3>(y)*ipow<3>(x - 1) + 1170*ipow<2>(y)*ipow<4>(x - 1) + 72*y*ipow<5>(x - 1) + ipow<6>(x - 1)) + 20*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(12376*ipow<6>(y) + ipow<5>(y)*(26208*x - 26208) + 20475*ipow<4>(y)*ipow<2>(x - 1) + 7280*ipow<3>(y)*ipow<3>(x - 1) + 1170*ipow<2>(y)*ipow<4>(x - 1) + 72*y*ipow<5>(x - 1) + ipow<6>(x - 1)),
            (20*x - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(74256*ipow<5>(y) + 5*ipow<4>(y)*(26208*x - 26208) + 81900*ipow<3>(y)*ipow<2>(x - 1) + 21840*ipow<2>(y)*ipow<3>(x - 1) + 2340*y*ipow<4>(x - 1) + 72*ipow<5>(x - 1)) + (20*x - 1)*(2*x + 2*y + 6*z - 2)*(12376*ipow<6>(y) + ipow<5>(y)*(26208*x - 26208) + 20475*ipow<4>(y)*ipow<2>(x - 1) + 7280*ipow<3>(y)*ipow<3>(x - 1) + 1170*ipow<2>(y)*ipow<4>(x - 1) + 72*y*ipow<5>(x - 1) + ipow<6>(x - 1)),
            (20*x - 1)*(6*x + 6*y + 12*z - 6)*(12376*ipow<6>(y) + ipow<5>(y)*(26208*x - 26208) + 20475*ipow<4>(y)*ipow<2>(x - 1) + 7280*ipow<3>(y)*ipow<3>(x - 1) + 1170*ipow<2>(y)*ipow<4>(x - 1) + 72*y*ipow<5>(x - 1) + ipow<6>(x - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 191
template<>
struct DGBasis<191> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(50388*ipow<7>(y) + ipow<6>(y)*(129948*x - 129948) + 129948*ipow<5>(y)*ipow<2>(x - 1) + 63700*ipow<4>(y)*ipow<3>(x - 1) + 15925*ipow<3>(y)*ipow<4>(x - 1) + 1911*ipow<2>(y)*ipow<5>(x - 1) + 91*y*ipow<6>(x - 1) + ipow<7>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(129948*ipow<6>(y) + 129948*ipow<5>(y)*(2*x - 2) + 191100*ipow<4>(y)*ipow<2>(x - 1) + 63700*ipow<3>(y)*ipow<3>(x - 1) + 9555*ipow<2>(y)*ipow<4>(x - 1) + 546*y*ipow<5>(x - 1) + 7*ipow<6>(x - 1)) + (2*x + 2*y + 6*z - 2)*(50388*ipow<7>(y) + ipow<6>(y)*(129948*x - 129948) + 129948*ipow<5>(y)*ipow<2>(x - 1) + 63700*ipow<4>(y)*ipow<3>(x - 1) + 15925*ipow<3>(y)*ipow<4>(x - 1) + 1911*ipow<2>(y)*ipow<5>(x - 1) + 91*y*ipow<6>(x - 1) + ipow<7>(x - 1)),
            (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(352716*ipow<6>(y) + 6*ipow<5>(y)*(129948*x - 129948) + 649740*ipow<4>(y)*ipow<2>(x - 1) + 254800*ipow<3>(y)*ipow<3>(x - 1) + 47775*ipow<2>(y)*ipow<4>(x - 1) + 3822*y*ipow<5>(x - 1) + 91*ipow<6>(x - 1)) + (2*x + 2*y + 6*z - 2)*(50388*ipow<7>(y) + ipow<6>(y)*(129948*x - 129948) + 129948*ipow<5>(y)*ipow<2>(x - 1) + 63700*ipow<4>(y)*ipow<3>(x - 1) + 15925*ipow<3>(y)*ipow<4>(x - 1) + 1911*ipow<2>(y)*ipow<5>(x - 1) + 91*y*ipow<6>(x - 1) + ipow<7>(x - 1)),
            (6*x + 6*y + 12*z - 6)*(50388*ipow<7>(y) + ipow<6>(y)*(129948*x - 129948) + 129948*ipow<5>(y)*ipow<2>(x - 1) + 63700*ipow<4>(y)*ipow<3>(x - 1) + 15925*ipow<3>(y)*ipow<4>(x - 1) + 1911*ipow<2>(y)*ipow<5>(x - 1) + 91*y*ipow<6>(x - 1) + ipow<7>(x - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 192
template<>
struct DGBasis<192> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)) + (4*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) + 2*x*(34*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 2*x*(51*x*(38*x*(5*x - 9) + 225) + 17*x*(114*x*(5*x - 9) + 3*x*(380*x - 342) + 675) - 3400) + 900) - 90)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)),
            (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 193
template<>
struct DGBasis<193> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(x + 9*y - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(x + 9*y - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)) + (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (x + 9*y - 1)*(68*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 4*x*(51*x*(19*x*(4*x - 5) + 40) + 17*x*(57*x*(4*x - 5) + 3*x*(152*x - 95) + 120) - 340) + 80)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(x + 9*y - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)) + 9*(4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(x + 9*y - 1)*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 194
template<>
struct DGBasis<194> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(2*x + 20*y - 2)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)) + (55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(51*x*(19*x*(5*x - 4) + 18) + 17*x*(57*x*(5*x - 4) + 3*x*(190*x - 76) + 54) - 68)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(20*x + 110*y - 20)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)),
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 195
template<>
struct DGBasis<195> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(20*x - 9) + 18) - 1)*(286*ipow<3>(y) + ipow<2>(y)*(198*x - 198) + 33*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(198*ipow<2>(y) + 33*y*(2*x - 2) + 3*ipow<2>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (3*x*(19*x*(20*x - 9) + 18) - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(286*ipow<3>(y) + ipow<2>(y)*(198*x - 198) + 33*y*ipow<2>(x - 1) + ipow<3>(x - 1)) + (57*x*(20*x - 9) + 3*x*(760*x - 171) + 54)*(286*ipow<3>(y) + ipow<2>(y)*(198*x - 198) + 33*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(858*ipow<2>(y) + 2*y*(198*x - 198) + 33*ipow<2>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (3*x*(19*x*(20*x - 9) + 18) - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(286*ipow<3>(y) + ipow<2>(y)*(198*x - 198) + 33*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))*(286*ipow<3>(y) + ipow<2>(y)*(198*x - 198) + 33*y*ipow<2>(x - 1) + ipow<3>(x - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 196
template<>
struct DGBasis<196> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (38*x*(5*x - 1) + 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(1365*ipow<4>(y) + ipow<3>(y)*(1456*x - 1456) + 468*ipow<2>(y)*ipow<2>(x - 1) + 48*y*ipow<3>(x - 1) + ipow<4>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (380*x - 38)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(1365*ipow<4>(y) + ipow<3>(y)*(1456*x - 1456) + 468*ipow<2>(y)*ipow<2>(x - 1) + 48*y*ipow<3>(x - 1) + ipow<4>(x - 1)) + (38*x*(5*x - 1) + 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(1365*ipow<4>(y) + ipow<3>(y)*(1456*x - 1456) + 468*ipow<2>(y)*ipow<2>(x - 1) + 48*y*ipow<3>(x - 1) + ipow<4>(x - 1)) + (38*x*(5*x - 1) + 1)*(1456*ipow<3>(y) + 468*ipow<2>(y)*(2*x - 2) + 144*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (38*x*(5*x - 1) + 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(1365*ipow<4>(y) + ipow<3>(y)*(1456*x - 1456) + 468*ipow<2>(y)*ipow<2>(x - 1) + 48*y*ipow<3>(x - 1) + ipow<4>(x - 1)) + (38*x*(5*x - 1) + 1)*(5460*ipow<3>(y) + 3*ipow<2>(y)*(1456*x - 1456) + 936*y*ipow<2>(x - 1) + 48*ipow<3>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (38*x*(5*x - 1) + 1)*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))*(1365*ipow<4>(y) + ipow<3>(y)*(1456*x - 1456) + 468*ipow<2>(y)*ipow<2>(x - 1) + 48*y*ipow<3>(x - 1) + ipow<4>(x - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 197
template<>
struct DGBasis<197> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (20*x - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(6188*ipow<5>(y) + ipow<4>(y)*(9100*x - 9100) + 4550*ipow<3>(y)*ipow<2>(x - 1) + 910*ipow<2>(y)*ipow<3>(x - 1) + 65*y*ipow<4>(x - 1) + ipow<5>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (20*x - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(6188*ipow<5>(y) + ipow<4>(y)*(9100*x - 9100) + 4550*ipow<3>(y)*ipow<2>(x - 1) + 910*ipow<2>(y)*ipow<3>(x - 1) + 65*y*ipow<4>(x - 1) + ipow<5>(x - 1)) + (20*x - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(9100*ipow<4>(y) + 4550*ipow<3>(y)*(2*x - 2) + 2730*ipow<2>(y)*ipow<2>(x - 1) + 260*y*ipow<3>(x - 1) + 5*ipow<4>(x - 1)) + 20*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(6188*ipow<5>(y) + ipow<4>(y)*(9100*x - 9100) + 4550*ipow<3>(y)*ipow<2>(x - 1) + 910*ipow<2>(y)*ipow<3>(x - 1) + 65*y*ipow<4>(x - 1) + ipow<5>(x - 1)),
            (20*x - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(6188*ipow<5>(y) + ipow<4>(y)*(9100*x - 9100) + 4550*ipow<3>(y)*ipow<2>(x - 1) + 910*ipow<2>(y)*ipow<3>(x - 1) + 65*y*ipow<4>(x - 1) + ipow<5>(x - 1)) + (20*x - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(30940*ipow<4>(y) + 4*ipow<3>(y)*(9100*x - 9100) + 13650*ipow<2>(y)*ipow<2>(x - 1) + 1820*y*ipow<3>(x - 1) + 65*ipow<4>(x - 1)),
            (20*x - 1)*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))*(6188*ipow<5>(y) + ipow<4>(y)*(9100*x - 9100) + 4550*ipow<3>(y)*ipow<2>(x - 1) + 910*ipow<2>(y)*ipow<3>(x - 1) + 65*y*ipow<4>(x - 1) + ipow<5>(x - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 198
template<>
struct DGBasis<198> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(27132*ipow<6>(y) + ipow<5>(y)*(51408*x - 51408) + 35700*ipow<4>(y)*ipow<2>(x - 1) + 11200*ipow<3>(y)*ipow<3>(x - 1) + 1575*ipow<2>(y)*ipow<4>(x - 1) + 84*y*ipow<5>(x - 1) + ipow<6>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(27132*ipow<6>(y) + ipow<5>(y)*(51408*x - 51408) + 35700*ipow<4>(y)*ipow<2>(x - 1) + 11200*ipow<3>(y)*ipow<3>(x - 1) + 1575*ipow<2>(y)*ipow<4>(x - 1) + 84*y*ipow<5>(x - 1) + ipow<6>(x - 1)) + (20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(51408*ipow<5>(y) + 35700*ipow<4>(y)*(2*x - 2) + 33600*ipow<3>(y)*ipow<2>(x - 1) + 6300*ipow<2>(y)*ipow<3>(x - 1) + 420*y*ipow<4>(x - 1) + 6*ipow<5>(x - 1)),
            (30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(27132*ipow<6>(y) + ipow<5>(y)*(51408*x - 51408) + 35700*ipow<4>(y)*ipow<2>(x - 1) + 11200*ipow<3>(y)*ipow<3>(x - 1) + 1575*ipow<2>(y)*ipow<4>(x - 1) + 84*y*ipow<5>(x - 1) + ipow<6>(x - 1)) + (20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(162792*ipow<5>(y) + 5*ipow<4>(y)*(51408*x - 51408) + 142800*ipow<3>(y)*ipow<2>(x - 1) + 33600*ipow<2>(y)*ipow<3>(x - 1) + 3150*y*ipow<4>(x - 1) + 84*ipow<5>(x - 1)),
            (60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))*(27132*ipow<6>(y) + ipow<5>(y)*(51408*x - 51408) + 35700*ipow<4>(y)*ipow<2>(x - 1) + 11200*ipow<3>(y)*ipow<3>(x - 1) + 1575*ipow<2>(y)*ipow<4>(x - 1) + 84*y*ipow<5>(x - 1) + ipow<6>(x - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 199
template<>
struct DGBasis<199> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)) + (68*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 4*x*(51*x*(19*x*(4*x - 5) + 40) + 17*x*(57*x*(4*x - 5) + 3*x*(152*x - 95) + 120) - 340) + 80)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)),
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)),
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 200
template<>
struct DGBasis<200> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(x + 11*y - 1)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(x + 11*y - 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)) + (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (x + 11*y - 1)*(51*x*(19*x*(5*x - 4) + 18) + 17*x*(57*x*(5*x - 4) + 3*x*(190*x - 76) + 54) - 68)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)),
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(x + 11*y - 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)) + 11*(17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)),
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(x + 11*y - 1)*(280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 201
template<>
struct DGBasis<201> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(20*x - 9) + 18) - 1)*(78*ipow<2>(y) + y*(24*x - 24) + ipow<2>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(2*x + 24*y - 2)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (3*x*(19*x*(20*x - 9) + 18) - 1)*(78*ipow<2>(y) + y*(24*x - 24) + ipow<2>(x - 1))*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)) + (78*ipow<2>(y) + y*(24*x - 24) + ipow<2>(x - 1))*(57*x*(20*x - 9) + 3*x*(760*x - 171) + 54)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)),
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(24*x + 156*y - 24)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (3*x*(19*x*(20*x - 9) + 18) - 1)*(78*ipow<2>(y) + y*(24*x - 24) + ipow<2>(x - 1))*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)),
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(78*ipow<2>(y) + y*(24*x - 24) + ipow<2>(x - 1))*(280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 202
template<>
struct DGBasis<202> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (38*x*(5*x - 1) + 1)*(455*ipow<3>(y) + ipow<2>(y)*(273*x - 273) + 39*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (380*x - 38)*(455*ipow<3>(y) + ipow<2>(y)*(273*x - 273) + 39*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (38*x*(5*x - 1) + 1)*(273*ipow<2>(y) + 39*y*(2*x - 2) + 3*ipow<2>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (38*x*(5*x - 1) + 1)*(455*ipow<3>(y) + ipow<2>(y)*(273*x - 273) + 39*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)),
            (38*x*(5*x - 1) + 1)*(1365*ipow<2>(y) + 2*y*(273*x - 273) + 39*ipow<2>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (38*x*(5*x - 1) + 1)*(455*ipow<3>(y) + ipow<2>(y)*(273*x - 273) + 39*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)),
            (38*x*(5*x - 1) + 1)*(455*ipow<3>(y) + ipow<2>(y)*(273*x - 273) + 39*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 203
template<>
struct DGBasis<203> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (20*x - 1)*(2380*ipow<4>(y) + ipow<3>(y)*(2240*x - 2240) + 630*ipow<2>(y)*ipow<2>(x - 1) + 56*y*ipow<3>(x - 1) + ipow<4>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (20*x - 1)*(2240*ipow<3>(y) + 630*ipow<2>(y)*(2*x - 2) + 168*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (20*x - 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1))*(2380*ipow<4>(y) + ipow<3>(y)*(2240*x - 2240) + 630*ipow<2>(y)*ipow<2>(x - 1) + 56*y*ipow<3>(x - 1) + ipow<4>(x - 1)) + 20*(2380*ipow<4>(y) + ipow<3>(y)*(2240*x - 2240) + 630*ipow<2>(y)*ipow<2>(x - 1) + 56*y*ipow<3>(x - 1) + ipow<4>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)),
            (20*x - 1)*(9520*ipow<3>(y) + 3*ipow<2>(y)*(2240*x - 2240) + 1260*y*ipow<2>(x - 1) + 56*ipow<3>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (20*x - 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1))*(2380*ipow<4>(y) + ipow<3>(y)*(2240*x - 2240) + 630*ipow<2>(y)*ipow<2>(x - 1) + 56*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (20*x - 1)*(280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))*(2380*ipow<4>(y) + ipow<3>(y)*(2240*x - 2240) + 630*ipow<2>(y)*ipow<2>(x - 1) + 56*y*ipow<3>(x - 1) + ipow<4>(x - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 204
template<>
struct DGBasis<204> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1))*(11628*ipow<5>(y) + ipow<4>(y)*(15300*x - 15300) + 6800*ipow<3>(y)*ipow<2>(x - 1) + 1200*ipow<2>(y)*ipow<3>(x - 1) + 75*y*ipow<4>(x - 1) + ipow<5>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1))*(11628*ipow<5>(y) + ipow<4>(y)*(15300*x - 15300) + 6800*ipow<3>(y)*ipow<2>(x - 1) + 1200*ipow<2>(y)*ipow<3>(x - 1) + 75*y*ipow<4>(x - 1) + ipow<5>(x - 1)) + (15300*ipow<4>(y) + 6800*ipow<3>(y)*(2*x - 2) + 3600*ipow<2>(y)*ipow<2>(x - 1) + 300*y*ipow<3>(x - 1) + 5*ipow<4>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)),
            (140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1))*(11628*ipow<5>(y) + ipow<4>(y)*(15300*x - 15300) + 6800*ipow<3>(y)*ipow<2>(x - 1) + 1200*ipow<2>(y)*ipow<3>(x - 1) + 75*y*ipow<4>(x - 1) + ipow<5>(x - 1)) + (58140*ipow<4>(y) + 4*ipow<3>(y)*(15300*x - 15300) + 20400*ipow<2>(y)*ipow<2>(x - 1) + 2400*y*ipow<3>(x - 1) + 75*ipow<4>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)),
            (280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))*(11628*ipow<5>(y) + ipow<4>(y)*(15300*x - 15300) + 6800*ipow<3>(y)*ipow<2>(x - 1) + 1200*ipow<2>(y)*ipow<3>(x - 1) + 75*y*ipow<4>(x - 1) + ipow<5>(x - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 205
template<>
struct DGBasis<205> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)) + (51*x*(19*x*(5*x - 4) + 18) + 17*x*(57*x*(5*x - 4) + 3*x*(190*x - 76) + 54) - 68)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)),
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)),
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(1260*ipow<4>(z) + 4*ipow<3>(z)*(630*x + 630*y - 630) + 1680*ipow<2>(z)*ipow<2>(x + y - 1) + 420*z*ipow<3>(x + y - 1) + 30*ipow<4>(x + y - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 206
template<>
struct DGBasis<206> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(20*x - 9) + 18) - 1)*(x + 13*y - 1)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(x + 13*y - 1)*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)) + (3*x*(19*x*(20*x - 9) + 18) - 1)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (x + 13*y - 1)*(57*x*(20*x - 9) + 3*x*(760*x - 171) + 54)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)),
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(x + 13*y - 1)*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)) + 13*(3*x*(19*x*(20*x - 9) + 18) - 1)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)),
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(x + 13*y - 1)*(1260*ipow<4>(z) + 4*ipow<3>(z)*(630*x + 630*y - 630) + 1680*ipow<2>(z)*ipow<2>(x + y - 1) + 420*z*ipow<3>(x + y - 1) + 30*ipow<4>(x + y - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 207
template<>
struct DGBasis<207> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (38*x*(5*x - 1) + 1)*(105*ipow<2>(y) + y*(28*x - 28) + ipow<2>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (380*x - 38)*(105*ipow<2>(y) + y*(28*x - 28) + ipow<2>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (38*x*(5*x - 1) + 1)*(2*x + 28*y - 2)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (38*x*(5*x - 1) + 1)*(105*ipow<2>(y) + y*(28*x - 28) + ipow<2>(x - 1))*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)),
            (38*x*(5*x - 1) + 1)*(28*x + 210*y - 28)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (38*x*(5*x - 1) + 1)*(105*ipow<2>(y) + y*(28*x - 28) + ipow<2>(x - 1))*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)),
            (38*x*(5*x - 1) + 1)*(105*ipow<2>(y) + y*(28*x - 28) + ipow<2>(x - 1))*(1260*ipow<4>(z) + 4*ipow<3>(z)*(630*x + 630*y - 630) + 1680*ipow<2>(z)*ipow<2>(x + y - 1) + 420*z*ipow<3>(x + y - 1) + 30*ipow<4>(x + y - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 208
template<>
struct DGBasis<208> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (20*x - 1)*(680*ipow<3>(y) + ipow<2>(y)*(360*x - 360) + 45*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (20*x - 1)*(360*ipow<2>(y) + 45*y*(2*x - 2) + 3*ipow<2>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (20*x - 1)*(680*ipow<3>(y) + ipow<2>(y)*(360*x - 360) + 45*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)) + 20*(680*ipow<3>(y) + ipow<2>(y)*(360*x - 360) + 45*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)),
            (20*x - 1)*(2040*ipow<2>(y) + 2*y*(360*x - 360) + 45*ipow<2>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (20*x - 1)*(680*ipow<3>(y) + ipow<2>(y)*(360*x - 360) + 45*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)),
            (20*x - 1)*(680*ipow<3>(y) + ipow<2>(y)*(360*x - 360) + 45*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(1260*ipow<4>(z) + 4*ipow<3>(z)*(630*x + 630*y - 630) + 1680*ipow<2>(z)*ipow<2>(x + y - 1) + 420*z*ipow<3>(x + y - 1) + 30*ipow<4>(x + y - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 209
template<>
struct DGBasis<209> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (3876*ipow<4>(y) + ipow<3>(y)*(3264*x - 3264) + 816*ipow<2>(y)*ipow<2>(x - 1) + 64*y*ipow<3>(x - 1) + ipow<4>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3264*ipow<3>(y) + 816*ipow<2>(y)*(2*x - 2) + 192*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (3876*ipow<4>(y) + ipow<3>(y)*(3264*x - 3264) + 816*ipow<2>(y)*ipow<2>(x - 1) + 64*y*ipow<3>(x - 1) + ipow<4>(x - 1))*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)),
            (15504*ipow<3>(y) + 3*ipow<2>(y)*(3264*x - 3264) + 1632*y*ipow<2>(x - 1) + 64*ipow<3>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (3876*ipow<4>(y) + ipow<3>(y)*(3264*x - 3264) + 816*ipow<2>(y)*ipow<2>(x - 1) + 64*y*ipow<3>(x - 1) + ipow<4>(x - 1))*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)),
            (3876*ipow<4>(y) + ipow<3>(y)*(3264*x - 3264) + 816*ipow<2>(y)*ipow<2>(x - 1) + 64*y*ipow<3>(x - 1) + ipow<4>(x - 1))*(1260*ipow<4>(z) + 4*ipow<3>(z)*(630*x + 630*y - 630) + 1680*ipow<2>(z)*ipow<2>(x + y - 1) + 420*z*ipow<3>(x + y - 1) + 30*ipow<4>(x + y - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 210
template<>
struct DGBasis<210> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(20*x - 9) + 18) - 1)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)) + (57*x*(20*x - 9) + 3*x*(760*x - 171) + 54)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)),
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)),
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(5544*ipow<5>(z) + 5*ipow<4>(z)*(2772*x + 2772*y - 2772) + 12600*ipow<3>(z)*ipow<2>(x + y - 1) + 5040*ipow<2>(z)*ipow<3>(x + y - 1) + 840*z*ipow<4>(x + y - 1) + 42*ipow<5>(x + y - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 211
template<>
struct DGBasis<211> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (38*x*(5*x - 1) + 1)*(x + 15*y - 1)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (380*x - 38)*(x + 15*y - 1)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)) + (38*x*(5*x - 1) + 1)*(x + 15*y - 1)*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)) + (38*x*(5*x - 1) + 1)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)),
            (38*x*(5*x - 1) + 1)*(x + 15*y - 1)*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)) + 15*(38*x*(5*x - 1) + 1)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)),
            (38*x*(5*x - 1) + 1)*(x + 15*y - 1)*(5544*ipow<5>(z) + 5*ipow<4>(z)*(2772*x + 2772*y - 2772) + 12600*ipow<3>(z)*ipow<2>(x + y - 1) + 5040*ipow<2>(z)*ipow<3>(x + y - 1) + 840*z*ipow<4>(x + y - 1) + 42*ipow<5>(x + y - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 212
template<>
struct DGBasis<212> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (20*x - 1)*(136*ipow<2>(y) + y*(32*x - 32) + ipow<2>(x - 1))*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (20*x - 1)*(2*x + 32*y - 2)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)) + (20*x - 1)*(136*ipow<2>(y) + y*(32*x - 32) + ipow<2>(x - 1))*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)) + 20*(136*ipow<2>(y) + y*(32*x - 32) + ipow<2>(x - 1))*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)),
            (20*x - 1)*(32*x + 272*y - 32)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)) + (20*x - 1)*(136*ipow<2>(y) + y*(32*x - 32) + ipow<2>(x - 1))*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)),
            (20*x - 1)*(136*ipow<2>(y) + y*(32*x - 32) + ipow<2>(x - 1))*(5544*ipow<5>(z) + 5*ipow<4>(z)*(2772*x + 2772*y - 2772) + 12600*ipow<3>(z)*ipow<2>(x + y - 1) + 5040*ipow<2>(z)*ipow<3>(x + y - 1) + 840*z*ipow<4>(x + y - 1) + 42*ipow<5>(x + y - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 213
template<>
struct DGBasis<213> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (969*ipow<3>(y) + ipow<2>(y)*(459*x - 459) + 51*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (459*ipow<2>(y) + 51*y*(2*x - 2) + 3*ipow<2>(x - 1))*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)) + (969*ipow<3>(y) + ipow<2>(y)*(459*x - 459) + 51*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)),
            (2907*ipow<2>(y) + 2*y*(459*x - 459) + 51*ipow<2>(x - 1))*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)) + (969*ipow<3>(y) + ipow<2>(y)*(459*x - 459) + 51*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)),
            (969*ipow<3>(y) + ipow<2>(y)*(459*x - 459) + 51*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(5544*ipow<5>(z) + 5*ipow<4>(z)*(2772*x + 2772*y - 2772) + 12600*ipow<3>(z)*ipow<2>(x + y - 1) + 5040*ipow<2>(z)*ipow<3>(x + y - 1) + 840*z*ipow<4>(x + y - 1) + 42*ipow<5>(x + y - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 214
template<>
struct DGBasis<214> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (38*x*(5*x - 1) + 1)*(ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (380*x - 38)*(ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) - 1) + (38*x*(5*x - 1) + 1)*(7*ipow<6>(x) + 42*ipow<5>(x)*y - 42*ipow<5>(x) + 105*ipow<4>(x)*ipow<2>(y) - 210*ipow<4>(x)*y + 105*ipow<4>(x) + 140*ipow<3>(x)*ipow<3>(y) - 420*ipow<3>(x)*ipow<2>(y) + 420*ipow<3>(x)*y - 140*ipow<3>(x) + 105*ipow<2>(x)*ipow<4>(y) - 420*ipow<2>(x)*ipow<3>(y) + 630*ipow<2>(x)*ipow<2>(y) - 420*ipow<2>(x)*y + 105*ipow<2>(x) + 42*x*ipow<5>(y) - 210*x*ipow<4>(y) + 420*x*ipow<3>(y) - 420*x*ipow<2>(y) + 210*x*y - 42*x + 7*ipow<6>(y) - 42*ipow<5>(y) + 105*ipow<4>(y) - 140*ipow<3>(y) + 105*ipow<2>(y) - 42*y + 12012*ipow<6>(z) + 16632*ipow<5>(z)*(2*x + 2*y - 2) + 34650*ipow<4>(z)*ipow<2>(x + y - 1) + 16800*ipow<3>(z)*ipow<3>(x + y - 1) + 3780*ipow<2>(z)*ipow<4>(x + y - 1) + 336*z*ipow<5>(x + y - 1) + 7),
            (38*x*(5*x - 1) + 1)*(7*ipow<6>(x) + 42*ipow<5>(x)*y - 42*ipow<5>(x) + 105*ipow<4>(x)*ipow<2>(y) - 210*ipow<4>(x)*y + 105*ipow<4>(x) + 140*ipow<3>(x)*ipow<3>(y) - 420*ipow<3>(x)*ipow<2>(y) + 420*ipow<3>(x)*y - 140*ipow<3>(x) + 105*ipow<2>(x)*ipow<4>(y) - 420*ipow<2>(x)*ipow<3>(y) + 630*ipow<2>(x)*ipow<2>(y) - 420*ipow<2>(x)*y + 105*ipow<2>(x) + 42*x*ipow<5>(y) - 210*x*ipow<4>(y) + 420*x*ipow<3>(y) - 420*x*ipow<2>(y) + 210*x*y - 42*x + 7*ipow<6>(y) - 42*ipow<5>(y) + 105*ipow<4>(y) - 140*ipow<3>(y) + 105*ipow<2>(y) - 42*y + 12012*ipow<6>(z) + 16632*ipow<5>(z)*(2*x + 2*y - 2) + 34650*ipow<4>(z)*ipow<2>(x + y - 1) + 16800*ipow<3>(z)*ipow<3>(x + y - 1) + 3780*ipow<2>(z)*ipow<4>(x + y - 1) + 336*z*ipow<5>(x + y - 1) + 7),
            (38*x*(5*x - 1) + 1)*(24024*ipow<6>(z) + 6*ipow<5>(z)*(12012*x + 12012*y - 12012) + 83160*ipow<4>(z)*ipow<2>(x + y - 1) + 46200*ipow<3>(z)*ipow<3>(x + y - 1) + 12600*ipow<2>(z)*ipow<4>(x + y - 1) + 1512*z*ipow<5>(x + y - 1) + 56*ipow<6>(x + y - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 215
template<>
struct DGBasis<215> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (20*x - 1)*(x + 17*y - 1)*(ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (20*x - 1)*(x + 17*y - 1)*(7*ipow<6>(x) + 42*ipow<5>(x)*y - 42*ipow<5>(x) + 105*ipow<4>(x)*ipow<2>(y) - 210*ipow<4>(x)*y + 105*ipow<4>(x) + 140*ipow<3>(x)*ipow<3>(y) - 420*ipow<3>(x)*ipow<2>(y) + 420*ipow<3>(x)*y - 140*ipow<3>(x) + 105*ipow<2>(x)*ipow<4>(y) - 420*ipow<2>(x)*ipow<3>(y) + 630*ipow<2>(x)*ipow<2>(y) - 420*ipow<2>(x)*y + 105*ipow<2>(x) + 42*x*ipow<5>(y) - 210*x*ipow<4>(y) + 420*x*ipow<3>(y) - 420*x*ipow<2>(y) + 210*x*y - 42*x + 7*ipow<6>(y) - 42*ipow<5>(y) + 105*ipow<4>(y) - 140*ipow<3>(y) + 105*ipow<2>(y) - 42*y + 12012*ipow<6>(z) + 16632*ipow<5>(z)*(2*x + 2*y - 2) + 34650*ipow<4>(z)*ipow<2>(x + y - 1) + 16800*ipow<3>(z)*ipow<3>(x + y - 1) + 3780*ipow<2>(z)*ipow<4>(x + y - 1) + 336*z*ipow<5>(x + y - 1) + 7) + (20*x - 1)*(ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) - 1) + 20*(x + 17*y - 1)*(ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) - 1),
            (20*x - 1)*(x + 17*y - 1)*(7*ipow<6>(x) + 42*ipow<5>(x)*y - 42*ipow<5>(x) + 105*ipow<4>(x)*ipow<2>(y) - 210*ipow<4>(x)*y + 105*ipow<4>(x) + 140*ipow<3>(x)*ipow<3>(y) - 420*ipow<3>(x)*ipow<2>(y) + 420*ipow<3>(x)*y - 140*ipow<3>(x) + 105*ipow<2>(x)*ipow<4>(y) - 420*ipow<2>(x)*ipow<3>(y) + 630*ipow<2>(x)*ipow<2>(y) - 420*ipow<2>(x)*y + 105*ipow<2>(x) + 42*x*ipow<5>(y) - 210*x*ipow<4>(y) + 420*x*ipow<3>(y) - 420*x*ipow<2>(y) + 210*x*y - 42*x + 7*ipow<6>(y) - 42*ipow<5>(y) + 105*ipow<4>(y) - 140*ipow<3>(y) + 105*ipow<2>(y) - 42*y + 12012*ipow<6>(z) + 16632*ipow<5>(z)*(2*x + 2*y - 2) + 34650*ipow<4>(z)*ipow<2>(x + y - 1) + 16800*ipow<3>(z)*ipow<3>(x + y - 1) + 3780*ipow<2>(z)*ipow<4>(x + y - 1) + 336*z*ipow<5>(x + y - 1) + 7) + 17*(20*x - 1)*(ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) - 1),
            (20*x - 1)*(x + 17*y - 1)*(24024*ipow<6>(z) + 6*ipow<5>(z)*(12012*x + 12012*y - 12012) + 83160*ipow<4>(z)*ipow<2>(x + y - 1) + 46200*ipow<3>(z)*ipow<3>(x + y - 1) + 12600*ipow<2>(z)*ipow<4>(x + y - 1) + 1512*z*ipow<5>(x + y - 1) + 56*ipow<6>(x + y - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 216
template<>
struct DGBasis<216> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (171*ipow<2>(y) + y*(36*x - 36) + ipow<2>(x - 1))*(ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x + 36*y - 2)*(ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) - 1) + (171*ipow<2>(y) + y*(36*x - 36) + ipow<2>(x - 1))*(7*ipow<6>(x) + 42*ipow<5>(x)*y - 42*ipow<5>(x) + 105*ipow<4>(x)*ipow<2>(y) - 210*ipow<4>(x)*y + 105*ipow<4>(x) + 140*ipow<3>(x)*ipow<3>(y) - 420*ipow<3>(x)*ipow<2>(y) + 420*ipow<3>(x)*y - 140*ipow<3>(x) + 105*ipow<2>(x)*ipow<4>(y) - 420*ipow<2>(x)*ipow<3>(y) + 630*ipow<2>(x)*ipow<2>(y) - 420*ipow<2>(x)*y + 105*ipow<2>(x) + 42*x*ipow<5>(y) - 210*x*ipow<4>(y) + 420*x*ipow<3>(y) - 420*x*ipow<2>(y) + 210*x*y - 42*x + 7*ipow<6>(y) - 42*ipow<5>(y) + 105*ipow<4>(y) - 140*ipow<3>(y) + 105*ipow<2>(y) - 42*y + 12012*ipow<6>(z) + 16632*ipow<5>(z)*(2*x + 2*y - 2) + 34650*ipow<4>(z)*ipow<2>(x + y - 1) + 16800*ipow<3>(z)*ipow<3>(x + y - 1) + 3780*ipow<2>(z)*ipow<4>(x + y - 1) + 336*z*ipow<5>(x + y - 1) + 7),
            (36*x + 342*y - 36)*(ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) - 1) + (171*ipow<2>(y) + y*(36*x - 36) + ipow<2>(x - 1))*(7*ipow<6>(x) + 42*ipow<5>(x)*y - 42*ipow<5>(x) + 105*ipow<4>(x)*ipow<2>(y) - 210*ipow<4>(x)*y + 105*ipow<4>(x) + 140*ipow<3>(x)*ipow<3>(y) - 420*ipow<3>(x)*ipow<2>(y) + 420*ipow<3>(x)*y - 140*ipow<3>(x) + 105*ipow<2>(x)*ipow<4>(y) - 420*ipow<2>(x)*ipow<3>(y) + 630*ipow<2>(x)*ipow<2>(y) - 420*ipow<2>(x)*y + 105*ipow<2>(x) + 42*x*ipow<5>(y) - 210*x*ipow<4>(y) + 420*x*ipow<3>(y) - 420*x*ipow<2>(y) + 210*x*y - 42*x + 7*ipow<6>(y) - 42*ipow<5>(y) + 105*ipow<4>(y) - 140*ipow<3>(y) + 105*ipow<2>(y) - 42*y + 12012*ipow<6>(z) + 16632*ipow<5>(z)*(2*x + 2*y - 2) + 34650*ipow<4>(z)*ipow<2>(x + y - 1) + 16800*ipow<3>(z)*ipow<3>(x + y - 1) + 3780*ipow<2>(z)*ipow<4>(x + y - 1) + 336*z*ipow<5>(x + y - 1) + 7),
            (171*ipow<2>(y) + y*(36*x - 36) + ipow<2>(x - 1))*(24024*ipow<6>(z) + 6*ipow<5>(z)*(12012*x + 12012*y - 12012) + 83160*ipow<4>(z)*ipow<2>(x + y - 1) + 46200*ipow<3>(z)*ipow<3>(x + y - 1) + 12600*ipow<2>(z)*ipow<4>(x + y - 1) + 1512*z*ipow<5>(x + y - 1) + 56*ipow<6>(x + y - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 217
template<>
struct DGBasis<217> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (20*x - 1)*(ipow<8>(x) + 8*ipow<7>(x)*y - 8*ipow<7>(x) + 28*ipow<6>(x)*ipow<2>(y) - 56*ipow<6>(x)*y + 28*ipow<6>(x) + 56*ipow<5>(x)*ipow<3>(y) - 168*ipow<5>(x)*ipow<2>(y) + 168*ipow<5>(x)*y - 56*ipow<5>(x) + 70*ipow<4>(x)*ipow<4>(y) - 280*ipow<4>(x)*ipow<3>(y) + 420*ipow<4>(x)*ipow<2>(y) - 280*ipow<4>(x)*y + 70*ipow<4>(x) + 56*ipow<3>(x)*ipow<5>(y) - 280*ipow<3>(x)*ipow<4>(y) + 560*ipow<3>(x)*ipow<3>(y) - 560*ipow<3>(x)*ipow<2>(y) + 280*ipow<3>(x)*y - 56*ipow<3>(x) + 28*ipow<2>(x)*ipow<6>(y) - 168*ipow<2>(x)*ipow<5>(y) + 420*ipow<2>(x)*ipow<4>(y) - 560*ipow<2>(x)*ipow<3>(y) + 420*ipow<2>(x)*ipow<2>(y) - 168*ipow<2>(x)*y + 28*ipow<2>(x) + 8*x*ipow<7>(y) - 56*x*ipow<6>(y) + 168*x*ipow<5>(y) - 280*x*ipow<4>(y) + 280*x*ipow<3>(y) - 168*x*ipow<2>(y) + 56*x*y - 8*x + ipow<8>(y) - 8*ipow<7>(y) + 28*ipow<6>(y) - 56*ipow<5>(y) + 70*ipow<4>(y) - 56*ipow<3>(y) + 28*ipow<2>(y) - 8*y + 12870*ipow<8>(z) + ipow<7>(z)*(51480*x + 51480*y - 51480) + 84084*ipow<6>(z)*ipow<2>(x + y - 1) + 72072*ipow<5>(z)*ipow<3>(x + y - 1) + 34650*ipow<4>(z)*ipow<4>(x + y - 1) + 9240*ipow<3>(z)*ipow<5>(x + y - 1) + 1260*ipow<2>(z)*ipow<6>(x + y - 1) + 72*z*ipow<7>(x + y - 1) + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            20*ipow<8>(x) + 160*ipow<7>(x)*y - 160*ipow<7>(x) + 560*ipow<6>(x)*ipow<2>(y) - 1120*ipow<6>(x)*y + 560*ipow<6>(x) + 1120*ipow<5>(x)*ipow<3>(y) - 3360*ipow<5>(x)*ipow<2>(y) + 3360*ipow<5>(x)*y - 1120*ipow<5>(x) + 1400*ipow<4>(x)*ipow<4>(y) - 5600*ipow<4>(x)*ipow<3>(y) + 8400*ipow<4>(x)*ipow<2>(y) - 5600*ipow<4>(x)*y + 1400*ipow<4>(x) + 1120*ipow<3>(x)*ipow<5>(y) - 5600*ipow<3>(x)*ipow<4>(y) + 11200*ipow<3>(x)*ipow<3>(y) - 11200*ipow<3>(x)*ipow<2>(y) + 5600*ipow<3>(x)*y - 1120*ipow<3>(x) + 560*ipow<2>(x)*ipow<6>(y) - 3360*ipow<2>(x)*ipow<5>(y) + 8400*ipow<2>(x)*ipow<4>(y) - 11200*ipow<2>(x)*ipow<3>(y) + 8400*ipow<2>(x)*ipow<2>(y) - 3360*ipow<2>(x)*y + 560*ipow<2>(x) + 160*x*ipow<7>(y) - 1120*x*ipow<6>(y) + 3360*x*ipow<5>(y) - 5600*x*ipow<4>(y) + 5600*x*ipow<3>(y) - 3360*x*ipow<2>(y) + 1120*x*y - 160*x + 20*ipow<8>(y) - 160*ipow<7>(y) + 560*ipow<6>(y) - 1120*ipow<5>(y) + 1400*ipow<4>(y) - 1120*ipow<3>(y) + 560*ipow<2>(y) - 160*y + 257400*ipow<8>(z) + 20*ipow<7>(z)*(51480*x + 51480*y - 51480) + 1681680*ipow<6>(z)*ipow<2>(x + y - 1) + 1441440*ipow<5>(z)*ipow<3>(x + y - 1) + 693000*ipow<4>(z)*ipow<4>(x + y - 1) + 184800*ipow<3>(z)*ipow<5>(x + y - 1) + 25200*ipow<2>(z)*ipow<6>(x + y - 1) + 1440*z*ipow<7>(x + y - 1) + (20*x - 1)*(8*ipow<7>(x) + 56*ipow<6>(x)*y - 56*ipow<6>(x) + 168*ipow<5>(x)*ipow<2>(y) - 336*ipow<5>(x)*y + 168*ipow<5>(x) + 280*ipow<4>(x)*ipow<3>(y) - 840*ipow<4>(x)*ipow<2>(y) + 840*ipow<4>(x)*y - 280*ipow<4>(x) + 280*ipow<3>(x)*ipow<4>(y) - 1120*ipow<3>(x)*ipow<3>(y) + 1680*ipow<3>(x)*ipow<2>(y) - 1120*ipow<3>(x)*y + 280*ipow<3>(x) + 168*ipow<2>(x)*ipow<5>(y) - 840*ipow<2>(x)*ipow<4>(y) + 1680*ipow<2>(x)*ipow<3>(y) - 1680*ipow<2>(x)*ipow<2>(y) + 840*ipow<2>(x)*y - 168*ipow<2>(x) + 56*x*ipow<6>(y) - 336*x*ipow<5>(y) + 840*x*ipow<4>(y) - 1120*x*ipow<3>(y) + 840*x*ipow<2>(y) - 336*x*y + 56*x + 8*ipow<7>(y) - 56*ipow<6>(y) + 168*ipow<5>(y) - 280*ipow<4>(y) + 280*ipow<3>(y) - 168*ipow<2>(y) + 56*y + 51480*ipow<7>(z) + 84084*ipow<6>(z)*(2*x + 2*y - 2) + 216216*ipow<5>(z)*ipow<2>(x + y - 1) + 138600*ipow<4>(z)*ipow<3>(x + y - 1) + 46200*ipow<3>(z)*ipow<4>(x + y - 1) + 7560*ipow<2>(z)*ipow<5>(x + y - 1) + 504*z*ipow<6>(x + y - 1) - 8) + 20,
            (20*x - 1)*(8*ipow<7>(x) + 56*ipow<6>(x)*y - 56*ipow<6>(x) + 168*ipow<5>(x)*ipow<2>(y) - 336*ipow<5>(x)*y + 168*ipow<5>(x) + 280*ipow<4>(x)*ipow<3>(y) - 840*ipow<4>(x)*ipow<2>(y) + 840*ipow<4>(x)*y - 280*ipow<4>(x) + 280*ipow<3>(x)*ipow<4>(y) - 1120*ipow<3>(x)*ipow<3>(y) + 1680*ipow<3>(x)*ipow<2>(y) - 1120*ipow<3>(x)*y + 280*ipow<3>(x) + 168*ipow<2>(x)*ipow<5>(y) - 840*ipow<2>(x)*ipow<4>(y) + 1680*ipow<2>(x)*ipow<3>(y) - 1680*ipow<2>(x)*ipow<2>(y) + 840*ipow<2>(x)*y - 168*ipow<2>(x) + 56*x*ipow<6>(y) - 336*x*ipow<5>(y) + 840*x*ipow<4>(y) - 1120*x*ipow<3>(y) + 840*x*ipow<2>(y) - 336*x*y + 56*x + 8*ipow<7>(y) - 56*ipow<6>(y) + 168*ipow<5>(y) - 280*ipow<4>(y) + 280*ipow<3>(y) - 168*ipow<2>(y) + 56*y + 51480*ipow<7>(z) + 84084*ipow<6>(z)*(2*x + 2*y - 2) + 216216*ipow<5>(z)*ipow<2>(x + y - 1) + 138600*ipow<4>(z)*ipow<3>(x + y - 1) + 46200*ipow<3>(z)*ipow<4>(x + y - 1) + 7560*ipow<2>(z)*ipow<5>(x + y - 1) + 504*z*ipow<6>(x + y - 1) - 8),
            (20*x - 1)*(102960*ipow<7>(z) + 7*ipow<6>(z)*(51480*x + 51480*y - 51480) + 504504*ipow<5>(z)*ipow<2>(x + y - 1) + 360360*ipow<4>(z)*ipow<3>(x + y - 1) + 138600*ipow<3>(z)*ipow<4>(x + y - 1) + 27720*ipow<2>(z)*ipow<5>(x + y - 1) + 2520*z*ipow<6>(x + y - 1) + 72*ipow<7>(x + y - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 218
template<>
struct DGBasis<218> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x + 19*y - 1)*(ipow<8>(x) + 8*ipow<7>(x)*y - 8*ipow<7>(x) + 28*ipow<6>(x)*ipow<2>(y) - 56*ipow<6>(x)*y + 28*ipow<6>(x) + 56*ipow<5>(x)*ipow<3>(y) - 168*ipow<5>(x)*ipow<2>(y) + 168*ipow<5>(x)*y - 56*ipow<5>(x) + 70*ipow<4>(x)*ipow<4>(y) - 280*ipow<4>(x)*ipow<3>(y) + 420*ipow<4>(x)*ipow<2>(y) - 280*ipow<4>(x)*y + 70*ipow<4>(x) + 56*ipow<3>(x)*ipow<5>(y) - 280*ipow<3>(x)*ipow<4>(y) + 560*ipow<3>(x)*ipow<3>(y) - 560*ipow<3>(x)*ipow<2>(y) + 280*ipow<3>(x)*y - 56*ipow<3>(x) + 28*ipow<2>(x)*ipow<6>(y) - 168*ipow<2>(x)*ipow<5>(y) + 420*ipow<2>(x)*ipow<4>(y) - 560*ipow<2>(x)*ipow<3>(y) + 420*ipow<2>(x)*ipow<2>(y) - 168*ipow<2>(x)*y + 28*ipow<2>(x) + 8*x*ipow<7>(y) - 56*x*ipow<6>(y) + 168*x*ipow<5>(y) - 280*x*ipow<4>(y) + 280*x*ipow<3>(y) - 168*x*ipow<2>(y) + 56*x*y - 8*x + ipow<8>(y) - 8*ipow<7>(y) + 28*ipow<6>(y) - 56*ipow<5>(y) + 70*ipow<4>(y) - 56*ipow<3>(y) + 28*ipow<2>(y) - 8*y + 12870*ipow<8>(z) + ipow<7>(z)*(51480*x + 51480*y - 51480) + 84084*ipow<6>(z)*ipow<2>(x + y - 1) + 72072*ipow<5>(z)*ipow<3>(x + y - 1) + 34650*ipow<4>(z)*ipow<4>(x + y - 1) + 9240*ipow<3>(z)*ipow<5>(x + y - 1) + 1260*ipow<2>(z)*ipow<6>(x + y - 1) + 72*z*ipow<7>(x + y - 1) + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            ipow<8>(x) + 8*ipow<7>(x)*y - 8*ipow<7>(x) + 28*ipow<6>(x)*ipow<2>(y) - 56*ipow<6>(x)*y + 28*ipow<6>(x) + 56*ipow<5>(x)*ipow<3>(y) - 168*ipow<5>(x)*ipow<2>(y) + 168*ipow<5>(x)*y - 56*ipow<5>(x) + 70*ipow<4>(x)*ipow<4>(y) - 280*ipow<4>(x)*ipow<3>(y) + 420*ipow<4>(x)*ipow<2>(y) - 280*ipow<4>(x)*y + 70*ipow<4>(x) + 56*ipow<3>(x)*ipow<5>(y) - 280*ipow<3>(x)*ipow<4>(y) + 560*ipow<3>(x)*ipow<3>(y) - 560*ipow<3>(x)*ipow<2>(y) + 280*ipow<3>(x)*y - 56*ipow<3>(x) + 28*ipow<2>(x)*ipow<6>(y) - 168*ipow<2>(x)*ipow<5>(y) + 420*ipow<2>(x)*ipow<4>(y) - 560*ipow<2>(x)*ipow<3>(y) + 420*ipow<2>(x)*ipow<2>(y) - 168*ipow<2>(x)*y + 28*ipow<2>(x) + 8*x*ipow<7>(y) - 56*x*ipow<6>(y) + 168*x*ipow<5>(y) - 280*x*ipow<4>(y) + 280*x*ipow<3>(y) - 168*x*ipow<2>(y) + 56*x*y - 8*x + ipow<8>(y) - 8*ipow<7>(y) + 28*ipow<6>(y) - 56*ipow<5>(y) + 70*ipow<4>(y) - 56*ipow<3>(y) + 28*ipow<2>(y) - 8*y + 12870*ipow<8>(z) + ipow<7>(z)*(51480*x + 51480*y - 51480) + 84084*ipow<6>(z)*ipow<2>(x + y - 1) + 72072*ipow<5>(z)*ipow<3>(x + y - 1) + 34650*ipow<4>(z)*ipow<4>(x + y - 1) + 9240*ipow<3>(z)*ipow<5>(x + y - 1) + 1260*ipow<2>(z)*ipow<6>(x + y - 1) + 72*z*ipow<7>(x + y - 1) + (x + 19*y - 1)*(8*ipow<7>(x) + 56*ipow<6>(x)*y - 56*ipow<6>(x) + 168*ipow<5>(x)*ipow<2>(y) - 336*ipow<5>(x)*y + 168*ipow<5>(x) + 280*ipow<4>(x)*ipow<3>(y) - 840*ipow<4>(x)*ipow<2>(y) + 840*ipow<4>(x)*y - 280*ipow<4>(x) + 280*ipow<3>(x)*ipow<4>(y) - 1120*ipow<3>(x)*ipow<3>(y) + 1680*ipow<3>(x)*ipow<2>(y) - 1120*ipow<3>(x)*y + 280*ipow<3>(x) + 168*ipow<2>(x)*ipow<5>(y) - 840*ipow<2>(x)*ipow<4>(y) + 1680*ipow<2>(x)*ipow<3>(y) - 1680*ipow<2>(x)*ipow<2>(y) + 840*ipow<2>(x)*y - 168*ipow<2>(x) + 56*x*ipow<6>(y) - 336*x*ipow<5>(y) + 840*x*ipow<4>(y) - 1120*x*ipow<3>(y) + 840*x*ipow<2>(y) - 336*x*y + 56*x + 8*ipow<7>(y) - 56*ipow<6>(y) + 168*ipow<5>(y) - 280*ipow<4>(y) + 280*ipow<3>(y) - 168*ipow<2>(y) + 56*y + 51480*ipow<7>(z) + 84084*ipow<6>(z)*(2*x + 2*y - 2) + 216216*ipow<5>(z)*ipow<2>(x + y - 1) + 138600*ipow<4>(z)*ipow<3>(x + y - 1) + 46200*ipow<3>(z)*ipow<4>(x + y - 1) + 7560*ipow<2>(z)*ipow<5>(x + y - 1) + 504*z*ipow<6>(x + y - 1) - 8) + 1,
            19*ipow<8>(x) + 152*ipow<7>(x)*y - 152*ipow<7>(x) + 532*ipow<6>(x)*ipow<2>(y) - 1064*ipow<6>(x)*y + 532*ipow<6>(x) + 1064*ipow<5>(x)*ipow<3>(y) - 3192*ipow<5>(x)*ipow<2>(y) + 3192*ipow<5>(x)*y - 1064*ipow<5>(x) + 1330*ipow<4>(x)*ipow<4>(y) - 5320*ipow<4>(x)*ipow<3>(y) + 7980*ipow<4>(x)*ipow<2>(y) - 5320*ipow<4>(x)*y + 1330*ipow<4>(x) + 1064*ipow<3>(x)*ipow<5>(y) - 5320*ipow<3>(x)*ipow<4>(y) + 10640*ipow<3>(x)*ipow<3>(y) - 10640*ipow<3>(x)*ipow<2>(y) + 5320*ipow<3>(x)*y - 1064*ipow<3>(x) + 532*ipow<2>(x)*ipow<6>(y) - 3192*ipow<2>(x)*ipow<5>(y) + 7980*ipow<2>(x)*ipow<4>(y) - 10640*ipow<2>(x)*ipow<3>(y) + 7980*ipow<2>(x)*ipow<2>(y) - 3192*ipow<2>(x)*y + 532*ipow<2>(x) + 152*x*ipow<7>(y) - 1064*x*ipow<6>(y) + 3192*x*ipow<5>(y) - 5320*x*ipow<4>(y) + 5320*x*ipow<3>(y) - 3192*x*ipow<2>(y) + 1064*x*y - 152*x + 19*ipow<8>(y) - 152*ipow<7>(y) + 532*ipow<6>(y) - 1064*ipow<5>(y) + 1330*ipow<4>(y) - 1064*ipow<3>(y) + 532*ipow<2>(y) - 152*y + 244530*ipow<8>(z) + 19*ipow<7>(z)*(51480*x + 51480*y - 51480) + 1597596*ipow<6>(z)*ipow<2>(x + y - 1) + 1369368*ipow<5>(z)*ipow<3>(x + y - 1) + 658350*ipow<4>(z)*ipow<4>(x + y - 1) + 175560*ipow<3>(z)*ipow<5>(x + y - 1) + 23940*ipow<2>(z)*ipow<6>(x + y - 1) + 1368*z*ipow<7>(x + y - 1) + (x + 19*y - 1)*(8*ipow<7>(x) + 56*ipow<6>(x)*y - 56*ipow<6>(x) + 168*ipow<5>(x)*ipow<2>(y) - 336*ipow<5>(x)*y + 168*ipow<5>(x) + 280*ipow<4>(x)*ipow<3>(y) - 840*ipow<4>(x)*ipow<2>(y) + 840*ipow<4>(x)*y - 280*ipow<4>(x) + 280*ipow<3>(x)*ipow<4>(y) - 1120*ipow<3>(x)*ipow<3>(y) + 1680*ipow<3>(x)*ipow<2>(y) - 1120*ipow<3>(x)*y + 280*ipow<3>(x) + 168*ipow<2>(x)*ipow<5>(y) - 840*ipow<2>(x)*ipow<4>(y) + 1680*ipow<2>(x)*ipow<3>(y) - 1680*ipow<2>(x)*ipow<2>(y) + 840*ipow<2>(x)*y - 168*ipow<2>(x) + 56*x*ipow<6>(y) - 336*x*ipow<5>(y) + 840*x*ipow<4>(y) - 1120*x*ipow<3>(y) + 840*x*ipow<2>(y) - 336*x*y + 56*x + 8*ipow<7>(y) - 56*ipow<6>(y) + 168*ipow<5>(y) - 280*ipow<4>(y) + 280*ipow<3>(y) - 168*ipow<2>(y) + 56*y + 51480*ipow<7>(z) + 84084*ipow<6>(z)*(2*x + 2*y - 2) + 216216*ipow<5>(z)*ipow<2>(x + y - 1) + 138600*ipow<4>(z)*ipow<3>(x + y - 1) + 46200*ipow<3>(z)*ipow<4>(x + y - 1) + 7560*ipow<2>(z)*ipow<5>(x + y - 1) + 504*z*ipow<6>(x + y - 1) - 8) + 19,
            (x + 19*y - 1)*(102960*ipow<7>(z) + 7*ipow<6>(z)*(51480*x + 51480*y - 51480) + 504504*ipow<5>(z)*ipow<2>(x + y - 1) + 360360*ipow<4>(z)*ipow<3>(x + y - 1) + 138600*ipow<3>(z)*ipow<4>(x + y - 1) + 27720*ipow<2>(z)*ipow<5>(x + y - 1) + 2520*z*ipow<6>(x + y - 1) + 72*ipow<7>(x + y - 1))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 219
template<>
struct DGBasis<219> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return ipow<9>(x) + 9*ipow<8>(x)*y - 9*ipow<8>(x) + 36*ipow<7>(x)*ipow<2>(y) - 72*ipow<7>(x)*y + 36*ipow<7>(x) + 84*ipow<6>(x)*ipow<3>(y) - 252*ipow<6>(x)*ipow<2>(y) + 252*ipow<6>(x)*y - 84*ipow<6>(x) + 126*ipow<5>(x)*ipow<4>(y) - 504*ipow<5>(x)*ipow<3>(y) + 756*ipow<5>(x)*ipow<2>(y) - 504*ipow<5>(x)*y + 126*ipow<5>(x) + 126*ipow<4>(x)*ipow<5>(y) - 630*ipow<4>(x)*ipow<4>(y) + 1260*ipow<4>(x)*ipow<3>(y) - 1260*ipow<4>(x)*ipow<2>(y) + 630*ipow<4>(x)*y - 126*ipow<4>(x) + 84*ipow<3>(x)*ipow<6>(y) - 504*ipow<3>(x)*ipow<5>(y) + 1260*ipow<3>(x)*ipow<4>(y) - 1680*ipow<3>(x)*ipow<3>(y) + 1260*ipow<3>(x)*ipow<2>(y) - 504*ipow<3>(x)*y + 84*ipow<3>(x) + 36*ipow<2>(x)*ipow<7>(y) - 252*ipow<2>(x)*ipow<6>(y) + 756*ipow<2>(x)*ipow<5>(y) - 1260*ipow<2>(x)*ipow<4>(y) + 1260*ipow<2>(x)*ipow<3>(y) - 756*ipow<2>(x)*ipow<2>(y) + 252*ipow<2>(x)*y - 36*ipow<2>(x) + 9*x*ipow<8>(y) - 72*x*ipow<7>(y) + 252*x*ipow<6>(y) - 504*x*ipow<5>(y) + 630*x*ipow<4>(y) - 504*x*ipow<3>(y) + 252*x*ipow<2>(y) - 72*x*y + 9*x + ipow<9>(y) - 9*ipow<8>(y) + 36*ipow<7>(y) - 84*ipow<6>(y) + 126*ipow<5>(y) - 126*ipow<4>(y) + 84*ipow<3>(y) - 36*ipow<2>(y) + 9*y + 48620*ipow<9>(z) + ipow<8>(z)*(218790*x + 218790*y - 218790) + 411840*ipow<7>(z)*ipow<2>(x + y - 1) + 420420*ipow<6>(z)*ipow<3>(x + y - 1) + 252252*ipow<5>(z)*ipow<4>(x + y - 1) + 90090*ipow<4>(z)*ipow<5>(x + y - 1) + 18480*ipow<3>(z)*ipow<6>(x + y - 1) + 1980*ipow<2>(z)*ipow<7>(x + y - 1) + 90*z*ipow<8>(x + y - 1) - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            9*ipow<8>(x) + 72*ipow<7>(x)*y - 72*ipow<7>(x) + 252*ipow<6>(x)*ipow<2>(y) - 504*ipow<6>(x)*y + 252*ipow<6>(x) + 504*ipow<5>(x)*ipow<3>(y) - 1512*ipow<5>(x)*ipow<2>(y) + 1512*ipow<5>(x)*y - 504*ipow<5>(x) + 630*ipow<4>(x)*ipow<4>(y) - 2520*ipow<4>(x)*ipow<3>(y) + 3780*ipow<4>(x)*ipow<2>(y) - 2520*ipow<4>(x)*y + 630*ipow<4>(x) + 504*ipow<3>(x)*ipow<5>(y) - 2520*ipow<3>(x)*ipow<4>(y) + 5040*ipow<3>(x)*ipow<3>(y) - 5040*ipow<3>(x)*ipow<2>(y) + 2520*ipow<3>(x)*y - 504*ipow<3>(x) + 252*ipow<2>(x)*ipow<6>(y) - 1512*ipow<2>(x)*ipow<5>(y) + 3780*ipow<2>(x)*ipow<4>(y) - 5040*ipow<2>(x)*ipow<3>(y) + 3780*ipow<2>(x)*ipow<2>(y) - 1512*ipow<2>(x)*y + 252*ipow<2>(x) + 72*x*ipow<7>(y) - 504*x*ipow<6>(y) + 1512*x*ipow<5>(y) - 2520*x*ipow<4>(y) + 2520*x*ipow<3>(y) - 1512*x*ipow<2>(y) + 504*x*y - 72*x + 9*ipow<8>(y) - 72*ipow<7>(y) + 252*ipow<6>(y) - 504*ipow<5>(y) + 630*ipow<4>(y) - 504*ipow<3>(y) + 252*ipow<2>(y) - 72*y + 218790*ipow<8>(z) + 411840*ipow<7>(z)*(2*x + 2*y - 2) + 1261260*ipow<6>(z)*ipow<2>(x + y - 1) + 1009008*ipow<5>(z)*ipow<3>(x + y - 1) + 450450*ipow<4>(z)*ipow<4>(x + y - 1) + 110880*ipow<3>(z)*ipow<5>(x + y - 1) + 13860*ipow<2>(z)*ipow<6>(x + y - 1) + 720*z*ipow<7>(x + y - 1) + 9,
            9*ipow<8>(x) + 72*ipow<7>(x)*y - 72*ipow<7>(x) + 252*ipow<6>(x)*ipow<2>(y) - 504*ipow<6>(x)*y + 252*ipow<6>(x) + 504*ipow<5>(x)*ipow<3>(y) - 1512*ipow<5>(x)*ipow<2>(y) + 1512*ipow<5>(x)*y - 504*ipow<5>(x) + 630*ipow<4>(x)*ipow<4>(y) - 2520*ipow<4>(x)*ipow<3>(y) + 3780*ipow<4>(x)*ipow<2>(y) - 2520*ipow<4>(x)*y + 630*ipow<4>(x) + 504*ipow<3>(x)*ipow<5>(y) - 2520*ipow<3>(x)*ipow<4>(y) + 5040*ipow<3>(x)*ipow<3>(y) - 5040*ipow<3>(x)*ipow<2>(y) + 2520*ipow<3>(x)*y - 504*ipow<3>(x) + 252*ipow<2>(x)*ipow<6>(y) - 1512*ipow<2>(x)*ipow<5>(y) + 3780*ipow<2>(x)*ipow<4>(y) - 5040*ipow<2>(x)*ipow<3>(y) + 3780*ipow<2>(x)*ipow<2>(y) - 1512*ipow<2>(x)*y + 252*ipow<2>(x) + 72*x*ipow<7>(y) - 504*x*ipow<6>(y) + 1512*x*ipow<5>(y) - 2520*x*ipow<4>(y) + 2520*x*ipow<3>(y) - 1512*x*ipow<2>(y) + 504*x*y - 72*x + 9*ipow<8>(y) - 72*ipow<7>(y) + 252*ipow<6>(y) - 504*ipow<5>(y) + 630*ipow<4>(y) - 504*ipow<3>(y) + 252*ipow<2>(y) - 72*y + 218790*ipow<8>(z) + 411840*ipow<7>(z)*(2*x + 2*y - 2) + 1261260*ipow<6>(z)*ipow<2>(x + y - 1) + 1009008*ipow<5>(z)*ipow<3>(x + y - 1) + 450450*ipow<4>(z)*ipow<4>(x + y - 1) + 110880*ipow<3>(z)*ipow<5>(x + y - 1) + 13860*ipow<2>(z)*ipow<6>(x + y - 1) + 720*z*ipow<7>(x + y - 1) + 9,
            437580*ipow<8>(z) + 8*ipow<7>(z)*(218790*x + 218790*y - 218790) + 2882880*ipow<6>(z)*ipow<2>(x + y - 1) + 2522520*ipow<5>(z)*ipow<3>(x + y - 1) + 1261260*ipow<4>(z)*ipow<4>(x + y - 1) + 360360*ipow<3>(z)*ipow<5>(x + y - 1) + 55440*ipow<2>(z)*ipow<6>(x + y - 1) + 3960*z*ipow<7>(x + y - 1) + 90*ipow<8>(x + y - 1)
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 220
template<>
struct DGBasis<220> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return 13*x*(x*(2*x*(x*(17*x*(x*(19*x*(x*(7*x*(11*x - 50) + 675) - 720) + 8820) - 3528) + 14700) - 2100) + 315) - 10) + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            13*x*(2*x*(x*(17*x*(x*(19*x*(x*(7*x*(11*x - 50) + 675) - 720) + 8820) - 3528) + 14700) - 2100) + 315) + 13*x*(2*x*(x*(17*x*(x*(19*x*(x*(7*x*(11*x - 50) + 675) - 720) + 8820) - 3528) + 14700) - 2100) + x*(2*x*(17*x*(x*(19*x*(x*(7*x*(11*x - 50) + 675) - 720) + 8820) - 3528) + 14700) + 2*x*(17*x*(x*(19*x*(x*(7*x*(11*x - 50) + 675) - 720) + 8820) - 3528) + x*(17*x*(19*x*(x*(7*x*(11*x - 50) + 675) - 720) + 8820) + 17*x*(19*x*(x*(7*x*(11*x - 50) + 675) - 720) + x*(19*x*(7*x*(11*x - 50) + 675) + 19*x*(7*x*(11*x - 50) + x*(154*x - 350) + 675) - 13680) + 8820) - 59976) + 14700) - 4200) + 315) - 130,
            0,
            0
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 221
template<>
struct DGBasis<221> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (2*x*(x*(x*(17*x*(x*(19*x*(5*x*(7*x*(22*x - 81) + 864) - 3528) + 31752) - 8820) + 23520) - 1890) + 63) - 1)*(x + 3*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            2*x*(x*(x*(17*x*(x*(19*x*(5*x*(7*x*(22*x - 81) + 864) - 3528) + 31752) - 8820) + 23520) - 1890) + 63) + (x + 3*y - 1)*(2*x*(x*(17*x*(x*(19*x*(5*x*(7*x*(22*x - 81) + 864) - 3528) + 31752) - 8820) + 23520) - 1890) + 2*x*(x*(17*x*(x*(19*x*(5*x*(7*x*(22*x - 81) + 864) - 3528) + 31752) - 8820) + 23520) + x*(17*x*(x*(19*x*(5*x*(7*x*(22*x - 81) + 864) - 3528) + 31752) - 8820) + x*(17*x*(19*x*(5*x*(7*x*(22*x - 81) + 864) - 3528) + 31752) + 17*x*(19*x*(5*x*(7*x*(22*x - 81) + 864) - 3528) + x*(95*x*(7*x*(22*x - 81) + 864) + 19*x*(35*x*(22*x - 81) + 5*x*(308*x - 567) + 4320) - 67032) + 31752) - 149940) + 23520) - 1890) + 126) - 1,
            6*x*(x*(x*(17*x*(x*(19*x*(5*x*(7*x*(22*x - 81) + 864) - 3528) + 31752) - 8820) + 23520) - 1890) + 63) - 3,
            0
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 222
template<>
struct DGBasis<222> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*ipow<2>(x) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(10*ipow<2>(y) + y*(8*x - 8) + ipow<2>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*ipow<2>(x) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(2*x + 8*y - 2) + (10*ipow<2>(y) + y*(8*x - 8) + ipow<2>(x - 1))*(2*x*(17*x*(3*x*(19*x*(5*x*(33*ipow<2>(x) - 96*x + 112) - 336) + 2100) - 1120) + 1680) + 2*x*(17*x*(3*x*(19*x*(5*x*(33*ipow<2>(x) - 96*x + 112) - 336) + 2100) - 1120) + x*(51*x*(19*x*(5*x*(33*ipow<2>(x) - 96*x + 112) - 336) + 2100) + 17*x*(57*x*(5*x*(33*ipow<2>(x) - 96*x + 112) - 336) + 3*x*(95*x*(33*ipow<2>(x) - 96*x + 112) + 19*x*(165*ipow<2>(x) + 5*x*(66*x - 96) - 480*x + 560) - 6384) + 6300) - 19040) + 1680) - 120),
            (2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*ipow<2>(x) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(8*x + 20*y - 8),
            0
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 223
template<>
struct DGBasis<223> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(35*ipow<3>(y) + ipow<2>(y)*(45*x - 45) + 15*y*ipow<2>(x - 1) + ipow<3>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(45*ipow<2>(y) + 15*y*(2*x - 2) + 3*ipow<2>(x - 1)) + (204*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 4*x*(51*x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) + 51*x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + x*(38*x*(x*(22*x - 49) + 42) + 19*x*(2*x*(22*x - 49) + 2*x*(44*x - 49) + 84) - 665) + 140) - 714) + 112)*(35*ipow<3>(y) + ipow<2>(y)*(45*x - 45) + 15*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(105*ipow<2>(y) + 2*y*(45*x - 45) + 15*ipow<2>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 224
template<>
struct DGBasis<224> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(126*ipow<4>(y) + ipow<3>(y)*(224*x - 224) + 126*ipow<2>(y)*ipow<2>(x - 1) + 24*y*ipow<3>(x - 1) + ipow<4>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(224*ipow<3>(y) + 126*ipow<2>(y)*(2*x - 2) + 72*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1)) + (51*x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) + 51*x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + x*(19*x*(7*x*(11*x - 18) + 75) + 19*x*(7*x*(11*x - 18) + x*(154*x - 126) + 75) - 380) + 45) - 102)*(126*ipow<4>(y) + ipow<3>(y)*(224*x - 224) + 126*ipow<2>(y)*ipow<2>(x - 1) + 24*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(504*ipow<3>(y) + 3*ipow<2>(y)*(224*x - 224) + 252*y*ipow<2>(x - 1) + 24*ipow<3>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 225
template<>
struct DGBasis<225> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(462*ipow<5>(y) + ipow<4>(y)*(1050*x - 1050) + 840*ipow<3>(y)*ipow<2>(x - 1) + 280*ipow<2>(y)*ipow<3>(x - 1) + 35*y*ipow<4>(x - 1) + ipow<5>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(1050*ipow<4>(y) + 840*ipow<3>(y)*(2*x - 2) + 840*ipow<2>(y)*ipow<2>(x - 1) + 140*y*ipow<3>(x - 1) + 5*ipow<4>(x - 1)) + (57*x*(x*(21*x*(22*x - 25) + 200) - 30) + 3*x*(19*x*(21*x*(22*x - 25) + 200) + 19*x*(21*x*(22*x - 25) + x*(924*x - 525) + 200) - 570) + 90)*(462*ipow<5>(y) + ipow<4>(y)*(1050*x - 1050) + 840*ipow<3>(y)*ipow<2>(x - 1) + 280*ipow<2>(y)*ipow<3>(x - 1) + 35*y*ipow<4>(x - 1) + ipow<5>(x - 1)),
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(2310*ipow<4>(y) + 4*ipow<3>(y)*(1050*x - 1050) + 2520*ipow<2>(y)*ipow<2>(x - 1) + 560*y*ipow<3>(x - 1) + 35*ipow<4>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 226
template<>
struct DGBasis<226> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(1716*ipow<6>(y) + ipow<5>(y)*(4752*x - 4752) + 4950*ipow<4>(y)*ipow<2>(x - 1) + 2400*ipow<3>(y)*ipow<3>(x - 1) + 540*ipow<2>(y)*ipow<4>(x - 1) + 48*y*ipow<5>(x - 1) + ipow<6>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(4752*ipow<5>(y) + 4950*ipow<4>(y)*(2*x - 2) + 7200*ipow<3>(y)*ipow<2>(x - 1) + 2160*ipow<2>(y)*ipow<3>(x - 1) + 240*y*ipow<4>(x - 1) + 6*ipow<5>(x - 1)) + (95*x*(7*x*(11*x - 8) + 12) + 19*x*(35*x*(11*x - 8) + 5*x*(154*x - 56) + 60) - 76)*(1716*ipow<6>(y) + ipow<5>(y)*(4752*x - 4752) + 4950*ipow<4>(y)*ipow<2>(x - 1) + 2400*ipow<3>(y)*ipow<3>(x - 1) + 540*ipow<2>(y)*ipow<4>(x - 1) + 48*y*ipow<5>(x - 1) + ipow<6>(x - 1)),
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(10296*ipow<5>(y) + 5*ipow<4>(y)*(4752*x - 4752) + 19800*ipow<3>(y)*ipow<2>(x - 1) + 7200*ipow<2>(y)*ipow<3>(x - 1) + 1080*y*ipow<4>(x - 1) + 48*ipow<5>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 227
template<>
struct DGBasis<227> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (10*x*(7*x*(22*x - 9) + 6) - 1)*(6435*ipow<7>(y) + ipow<6>(y)*(21021*x - 21021) + 27027*ipow<5>(y)*ipow<2>(x - 1) + 17325*ipow<4>(y)*ipow<3>(x - 1) + 5775*ipow<3>(y)*ipow<4>(x - 1) + 945*ipow<2>(y)*ipow<5>(x - 1) + 63*y*ipow<6>(x - 1) + ipow<7>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(21021*ipow<6>(y) + 27027*ipow<5>(y)*(2*x - 2) + 51975*ipow<4>(y)*ipow<2>(x - 1) + 23100*ipow<3>(y)*ipow<3>(x - 1) + 4725*ipow<2>(y)*ipow<4>(x - 1) + 378*y*ipow<5>(x - 1) + 7*ipow<6>(x - 1)) + (70*x*(22*x - 9) + 10*x*(308*x - 63) + 60)*(6435*ipow<7>(y) + ipow<6>(y)*(21021*x - 21021) + 27027*ipow<5>(y)*ipow<2>(x - 1) + 17325*ipow<4>(y)*ipow<3>(x - 1) + 5775*ipow<3>(y)*ipow<4>(x - 1) + 945*ipow<2>(y)*ipow<5>(x - 1) + 63*y*ipow<6>(x - 1) + ipow<7>(x - 1)),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(45045*ipow<6>(y) + 6*ipow<5>(y)*(21021*x - 21021) + 135135*ipow<4>(y)*ipow<2>(x - 1) + 69300*ipow<3>(y)*ipow<3>(x - 1) + 17325*ipow<2>(y)*ipow<4>(x - 1) + 1890*y*ipow<5>(x - 1) + 63*ipow<6>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 228
template<>
struct DGBasis<228> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (21*x*(11*x - 2) + 1)*(24310*ipow<8>(y) + ipow<7>(y)*(91520*x - 91520) + 140140*ipow<6>(y)*ipow<2>(x - 1) + 112112*ipow<5>(y)*ipow<3>(x - 1) + 50050*ipow<4>(y)*ipow<4>(x - 1) + 12320*ipow<3>(y)*ipow<5>(x - 1) + 1540*ipow<2>(y)*ipow<6>(x - 1) + 80*y*ipow<7>(x - 1) + ipow<8>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (462*x - 42)*(24310*ipow<8>(y) + ipow<7>(y)*(91520*x - 91520) + 140140*ipow<6>(y)*ipow<2>(x - 1) + 112112*ipow<5>(y)*ipow<3>(x - 1) + 50050*ipow<4>(y)*ipow<4>(x - 1) + 12320*ipow<3>(y)*ipow<5>(x - 1) + 1540*ipow<2>(y)*ipow<6>(x - 1) + 80*y*ipow<7>(x - 1) + ipow<8>(x - 1)) + (21*x*(11*x - 2) + 1)*(91520*ipow<7>(y) + 140140*ipow<6>(y)*(2*x - 2) + 336336*ipow<5>(y)*ipow<2>(x - 1) + 200200*ipow<4>(y)*ipow<3>(x - 1) + 61600*ipow<3>(y)*ipow<4>(x - 1) + 9240*ipow<2>(y)*ipow<5>(x - 1) + 560*y*ipow<6>(x - 1) + 8*ipow<7>(x - 1)),
            (21*x*(11*x - 2) + 1)*(194480*ipow<7>(y) + 7*ipow<6>(y)*(91520*x - 91520) + 840840*ipow<5>(y)*ipow<2>(x - 1) + 560560*ipow<4>(y)*ipow<3>(x - 1) + 200200*ipow<3>(y)*ipow<4>(x - 1) + 36960*ipow<2>(y)*ipow<5>(x - 1) + 3080*y*ipow<6>(x - 1) + 80*ipow<7>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 229
template<>
struct DGBasis<229> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (22*x - 1)*(92378*ipow<9>(y) + ipow<8>(y)*(393822*x - 393822) + 700128*ipow<7>(y)*ipow<2>(x - 1) + 672672*ipow<6>(y)*ipow<3>(x - 1) + 378378*ipow<5>(y)*ipow<4>(x - 1) + 126126*ipow<4>(y)*ipow<5>(x - 1) + 24024*ipow<3>(y)*ipow<6>(x - 1) + 2376*ipow<2>(y)*ipow<7>(x - 1) + 99*y*ipow<8>(x - 1) + ipow<9>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            2032316*ipow<9>(y) + 22*ipow<8>(y)*(393822*x - 393822) + 15402816*ipow<7>(y)*ipow<2>(x - 1) + 14798784*ipow<6>(y)*ipow<3>(x - 1) + 8324316*ipow<5>(y)*ipow<4>(x - 1) + 2774772*ipow<4>(y)*ipow<5>(x - 1) + 528528*ipow<3>(y)*ipow<6>(x - 1) + 52272*ipow<2>(y)*ipow<7>(x - 1) + 2178*y*ipow<8>(x - 1) + 22*ipow<9>(x - 1) + (22*x - 1)*(393822*ipow<8>(y) + 700128*ipow<7>(y)*(2*x - 2) + 2018016*ipow<6>(y)*ipow<2>(x - 1) + 1513512*ipow<5>(y)*ipow<3>(x - 1) + 630630*ipow<4>(y)*ipow<4>(x - 1) + 144144*ipow<3>(y)*ipow<5>(x - 1) + 16632*ipow<2>(y)*ipow<6>(x - 1) + 792*y*ipow<7>(x - 1) + 9*ipow<8>(x - 1)),
            (22*x - 1)*(831402*ipow<8>(y) + 8*ipow<7>(y)*(393822*x - 393822) + 4900896*ipow<6>(y)*ipow<2>(x - 1) + 4036032*ipow<5>(y)*ipow<3>(x - 1) + 1891890*ipow<4>(y)*ipow<4>(x - 1) + 504504*ipow<3>(y)*ipow<5>(x - 1) + 72072*ipow<2>(y)*ipow<6>(x - 1) + 4752*y*ipow<7>(x - 1) + 99*ipow<8>(x - 1)),
            0
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 230
template<>
struct DGBasis<230> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return 352716*ipow<10>(y) + ipow<9>(y)*(1679600*x - 1679600) + 3401190*ipow<8>(y)*ipow<2>(x - 1) + 3818880*ipow<7>(y)*ipow<3>(x - 1) + 2598960*ipow<6>(y)*ipow<4>(x - 1) + 1100736*ipow<5>(y)*ipow<5>(x - 1) + 286650*ipow<4>(y)*ipow<6>(x - 1) + 43680*ipow<3>(y)*ipow<7>(x - 1) + 3510*ipow<2>(y)*ipow<8>(x - 1) + 120*y*ipow<9>(x - 1) + ipow<10>(x - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            1679600*ipow<9>(y) + 3401190*ipow<8>(y)*(2*x - 2) + 11456640*ipow<7>(y)*ipow<2>(x - 1) + 10395840*ipow<6>(y)*ipow<3>(x - 1) + 5503680*ipow<5>(y)*ipow<4>(x - 1) + 1719900*ipow<4>(y)*ipow<5>(x - 1) + 305760*ipow<3>(y)*ipow<6>(x - 1) + 28080*ipow<2>(y)*ipow<7>(x - 1) + 1080*y*ipow<8>(x - 1) + 10*ipow<9>(x - 1),
            3527160*ipow<9>(y) + 9*ipow<8>(y)*(1679600*x - 1679600) + 27209520*ipow<7>(y)*ipow<2>(x - 1) + 26732160*ipow<6>(y)*ipow<3>(x - 1) + 15593760*ipow<5>(y)*ipow<4>(x - 1) + 5503680*ipow<4>(y)*ipow<5>(x - 1) + 1146600*ipow<3>(y)*ipow<6>(x - 1) + 131040*ipow<2>(y)*ipow<7>(x - 1) + 7020*y*ipow<8>(x - 1) + 120*ipow<9>(x - 1),
            0
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 231
template<>
struct DGBasis<231> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (2*x*(x*(x*(17*x*(x*(19*x*(5*x*(7*x*(22*x - 81) + 864) - 3528) + 31752) - 8820) + 23520) - 1890) + 63) - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            2*x*(x*(x*(17*x*(x*(19*x*(5*x*(7*x*(22*x - 81) + 864) - 3528) + 31752) - 8820) + 23520) - 1890) + 63) + (2*x*(x*(17*x*(x*(19*x*(5*x*(7*x*(22*x - 81) + 864) - 3528) + 31752) - 8820) + 23520) - 1890) + 2*x*(x*(17*x*(x*(19*x*(5*x*(7*x*(22*x - 81) + 864) - 3528) + 31752) - 8820) + 23520) + x*(17*x*(x*(19*x*(5*x*(7*x*(22*x - 81) + 864) - 3528) + 31752) - 8820) + x*(17*x*(19*x*(5*x*(7*x*(22*x - 81) + 864) - 3528) + 31752) + 17*x*(19*x*(5*x*(7*x*(22*x - 81) + 864) - 3528) + x*(95*x*(7*x*(22*x - 81) + 864) + 19*x*(35*x*(22*x - 81) + 5*x*(308*x - 567) + 4320) - 67032) + 31752) - 149940) + 23520) - 1890) + 126)*(x + y + 2*z - 1) - 1,
            2*x*(x*(x*(17*x*(x*(19*x*(5*x*(7*x*(22*x - 81) + 864) - 3528) + 31752) - 8820) + 23520) - 1890) + 63) - 1,
            4*x*(x*(x*(17*x*(x*(19*x*(5*x*(7*x*(22*x - 81) + 864) - 3528) + 31752) - 8820) + 23520) - 1890) + 63) - 2
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 232
template<>
struct DGBasis<232> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*ipow<2>(x) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(x + 5*y - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*ipow<2>(x) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(x + 5*y - 1) + (2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*ipow<2>(x) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(x + y + 2*z - 1) + (x + 5*y - 1)*(2*x*(17*x*(3*x*(19*x*(5*x*(33*ipow<2>(x) - 96*x + 112) - 336) + 2100) - 1120) + 1680) + 2*x*(17*x*(3*x*(19*x*(5*x*(33*ipow<2>(x) - 96*x + 112) - 336) + 2100) - 1120) + x*(51*x*(19*x*(5*x*(33*ipow<2>(x) - 96*x + 112) - 336) + 2100) + 17*x*(57*x*(5*x*(33*ipow<2>(x) - 96*x + 112) - 336) + 3*x*(95*x*(33*ipow<2>(x) - 96*x + 112) + 19*x*(165*ipow<2>(x) + 5*x*(66*x - 96) - 480*x + 560) - 6384) + 6300) - 19040) + 1680) - 120)*(x + y + 2*z - 1),
            (2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*ipow<2>(x) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(x + 5*y - 1) + 5*(2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*ipow<2>(x) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(x + y + 2*z - 1),
            2*(2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*ipow<2>(x) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(x + 5*y - 1)
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 233
template<>
struct DGBasis<233> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1))*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(2*x + 12*y - 2)*(x + y + 2*z - 1) + (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1)) + (21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1))*(204*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 4*x*(51*x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) + 51*x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + x*(38*x*(x*(22*x - 49) + 42) + 19*x*(2*x*(22*x - 49) + 2*x*(44*x - 49) + 84) - 665) + 140) - 714) + 112)*(x + y + 2*z - 1),
            (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(12*x + 42*y - 12)*(x + y + 2*z - 1) + (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1)),
            2*(4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(21*ipow<2>(y) + y*(12*x - 12) + ipow<2>(x - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 234
template<>
struct DGBasis<234> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(x + y + 2*z - 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(84*ipow<2>(y) + 21*y*(2*x - 2) + 3*ipow<2>(x - 1))*(x + y + 2*z - 1) + (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1)) + (51*x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) + 51*x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + x*(19*x*(7*x*(11*x - 18) + 75) + 19*x*(7*x*(11*x - 18) + x*(154*x - 126) + 75) - 380) + 45) - 102)*(x + y + 2*z - 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(252*ipow<2>(y) + 2*y*(84*x - 84) + 21*ipow<2>(x - 1))*(x + y + 2*z - 1) + (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            2*(51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(84*ipow<3>(y) + ipow<2>(y)*(84*x - 84) + 21*y*ipow<2>(x - 1) + ipow<3>(x - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 235
template<>
struct DGBasis<235> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(x + y + 2*z - 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(x + y + 2*z - 1)*(480*ipow<3>(y) + 216*ipow<2>(y)*(2*x - 2) + 96*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1)) + (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1)) + (57*x*(x*(21*x*(22*x - 25) + 200) - 30) + 3*x*(19*x*(21*x*(22*x - 25) + 200) + 19*x*(21*x*(22*x - 25) + x*(924*x - 525) + 200) - 570) + 90)*(x + y + 2*z - 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(x + y + 2*z - 1)*(1320*ipow<3>(y) + 3*ipow<2>(y)*(480*x - 480) + 432*y*ipow<2>(x - 1) + 32*ipow<3>(x - 1)) + (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            2*(3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(330*ipow<4>(y) + ipow<3>(y)*(480*x - 480) + 216*ipow<2>(y)*ipow<2>(x - 1) + 32*y*ipow<3>(x - 1) + ipow<4>(x - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 236
template<>
struct DGBasis<236> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(x + y + 2*z - 1)*(1287*ipow<5>(y) + ipow<4>(y)*(2475*x - 2475) + 1650*ipow<3>(y)*ipow<2>(x - 1) + 450*ipow<2>(y)*ipow<3>(x - 1) + 45*y*ipow<4>(x - 1) + ipow<5>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(x + y + 2*z - 1)*(2475*ipow<4>(y) + 1650*ipow<3>(y)*(2*x - 2) + 1350*ipow<2>(y)*ipow<2>(x - 1) + 180*y*ipow<3>(x - 1) + 5*ipow<4>(x - 1)) + (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(1287*ipow<5>(y) + ipow<4>(y)*(2475*x - 2475) + 1650*ipow<3>(y)*ipow<2>(x - 1) + 450*ipow<2>(y)*ipow<3>(x - 1) + 45*y*ipow<4>(x - 1) + ipow<5>(x - 1)) + (95*x*(7*x*(11*x - 8) + 12) + 19*x*(35*x*(11*x - 8) + 5*x*(154*x - 56) + 60) - 76)*(x + y + 2*z - 1)*(1287*ipow<5>(y) + ipow<4>(y)*(2475*x - 2475) + 1650*ipow<3>(y)*ipow<2>(x - 1) + 450*ipow<2>(y)*ipow<3>(x - 1) + 45*y*ipow<4>(x - 1) + ipow<5>(x - 1)),
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(x + y + 2*z - 1)*(6435*ipow<4>(y) + 4*ipow<3>(y)*(2475*x - 2475) + 4950*ipow<2>(y)*ipow<2>(x - 1) + 900*y*ipow<3>(x - 1) + 45*ipow<4>(x - 1)) + (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(1287*ipow<5>(y) + ipow<4>(y)*(2475*x - 2475) + 1650*ipow<3>(y)*ipow<2>(x - 1) + 450*ipow<2>(y)*ipow<3>(x - 1) + 45*y*ipow<4>(x - 1) + ipow<5>(x - 1)),
            2*(19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(1287*ipow<5>(y) + ipow<4>(y)*(2475*x - 2475) + 1650*ipow<3>(y)*ipow<2>(x - 1) + 450*ipow<2>(y)*ipow<3>(x - 1) + 45*y*ipow<4>(x - 1) + ipow<5>(x - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 237
template<>
struct DGBasis<237> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (10*x*(7*x*(22*x - 9) + 6) - 1)*(x + y + 2*z - 1)*(5005*ipow<6>(y) + ipow<5>(y)*(12012*x - 12012) + 10725*ipow<4>(y)*ipow<2>(x - 1) + 4400*ipow<3>(y)*ipow<3>(x - 1) + 825*ipow<2>(y)*ipow<4>(x - 1) + 60*y*ipow<5>(x - 1) + ipow<6>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(x + y + 2*z - 1)*(12012*ipow<5>(y) + 10725*ipow<4>(y)*(2*x - 2) + 13200*ipow<3>(y)*ipow<2>(x - 1) + 3300*ipow<2>(y)*ipow<3>(x - 1) + 300*y*ipow<4>(x - 1) + 6*ipow<5>(x - 1)) + (10*x*(7*x*(22*x - 9) + 6) - 1)*(5005*ipow<6>(y) + ipow<5>(y)*(12012*x - 12012) + 10725*ipow<4>(y)*ipow<2>(x - 1) + 4400*ipow<3>(y)*ipow<3>(x - 1) + 825*ipow<2>(y)*ipow<4>(x - 1) + 60*y*ipow<5>(x - 1) + ipow<6>(x - 1)) + (70*x*(22*x - 9) + 10*x*(308*x - 63) + 60)*(x + y + 2*z - 1)*(5005*ipow<6>(y) + ipow<5>(y)*(12012*x - 12012) + 10725*ipow<4>(y)*ipow<2>(x - 1) + 4400*ipow<3>(y)*ipow<3>(x - 1) + 825*ipow<2>(y)*ipow<4>(x - 1) + 60*y*ipow<5>(x - 1) + ipow<6>(x - 1)),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(x + y + 2*z - 1)*(30030*ipow<5>(y) + 5*ipow<4>(y)*(12012*x - 12012) + 42900*ipow<3>(y)*ipow<2>(x - 1) + 13200*ipow<2>(y)*ipow<3>(x - 1) + 1650*y*ipow<4>(x - 1) + 60*ipow<5>(x - 1)) + (10*x*(7*x*(22*x - 9) + 6) - 1)*(5005*ipow<6>(y) + ipow<5>(y)*(12012*x - 12012) + 10725*ipow<4>(y)*ipow<2>(x - 1) + 4400*ipow<3>(y)*ipow<3>(x - 1) + 825*ipow<2>(y)*ipow<4>(x - 1) + 60*y*ipow<5>(x - 1) + ipow<6>(x - 1)),
            2*(10*x*(7*x*(22*x - 9) + 6) - 1)*(5005*ipow<6>(y) + ipow<5>(y)*(12012*x - 12012) + 10725*ipow<4>(y)*ipow<2>(x - 1) + 4400*ipow<3>(y)*ipow<3>(x - 1) + 825*ipow<2>(y)*ipow<4>(x - 1) + 60*y*ipow<5>(x - 1) + ipow<6>(x - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 238
template<>
struct DGBasis<238> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (21*x*(11*x - 2) + 1)*(x + y + 2*z - 1)*(19448*ipow<7>(y) + ipow<6>(y)*(56056*x - 56056) + 63063*ipow<5>(y)*ipow<2>(x - 1) + 35035*ipow<4>(y)*ipow<3>(x - 1) + 10010*ipow<3>(y)*ipow<4>(x - 1) + 1386*ipow<2>(y)*ipow<5>(x - 1) + 77*y*ipow<6>(x - 1) + ipow<7>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (462*x - 42)*(x + y + 2*z - 1)*(19448*ipow<7>(y) + ipow<6>(y)*(56056*x - 56056) + 63063*ipow<5>(y)*ipow<2>(x - 1) + 35035*ipow<4>(y)*ipow<3>(x - 1) + 10010*ipow<3>(y)*ipow<4>(x - 1) + 1386*ipow<2>(y)*ipow<5>(x - 1) + 77*y*ipow<6>(x - 1) + ipow<7>(x - 1)) + (21*x*(11*x - 2) + 1)*(x + y + 2*z - 1)*(56056*ipow<6>(y) + 63063*ipow<5>(y)*(2*x - 2) + 105105*ipow<4>(y)*ipow<2>(x - 1) + 40040*ipow<3>(y)*ipow<3>(x - 1) + 6930*ipow<2>(y)*ipow<4>(x - 1) + 462*y*ipow<5>(x - 1) + 7*ipow<6>(x - 1)) + (21*x*(11*x - 2) + 1)*(19448*ipow<7>(y) + ipow<6>(y)*(56056*x - 56056) + 63063*ipow<5>(y)*ipow<2>(x - 1) + 35035*ipow<4>(y)*ipow<3>(x - 1) + 10010*ipow<3>(y)*ipow<4>(x - 1) + 1386*ipow<2>(y)*ipow<5>(x - 1) + 77*y*ipow<6>(x - 1) + ipow<7>(x - 1)),
            (21*x*(11*x - 2) + 1)*(x + y + 2*z - 1)*(136136*ipow<6>(y) + 6*ipow<5>(y)*(56056*x - 56056) + 315315*ipow<4>(y)*ipow<2>(x - 1) + 140140*ipow<3>(y)*ipow<3>(x - 1) + 30030*ipow<2>(y)*ipow<4>(x - 1) + 2772*y*ipow<5>(x - 1) + 77*ipow<6>(x - 1)) + (21*x*(11*x - 2) + 1)*(19448*ipow<7>(y) + ipow<6>(y)*(56056*x - 56056) + 63063*ipow<5>(y)*ipow<2>(x - 1) + 35035*ipow<4>(y)*ipow<3>(x - 1) + 10010*ipow<3>(y)*ipow<4>(x - 1) + 1386*ipow<2>(y)*ipow<5>(x - 1) + 77*y*ipow<6>(x - 1) + ipow<7>(x - 1)),
            2*(21*x*(11*x - 2) + 1)*(19448*ipow<7>(y) + ipow<6>(y)*(56056*x - 56056) + 63063*ipow<5>(y)*ipow<2>(x - 1) + 35035*ipow<4>(y)*ipow<3>(x - 1) + 10010*ipow<3>(y)*ipow<4>(x - 1) + 1386*ipow<2>(y)*ipow<5>(x - 1) + 77*y*ipow<6>(x - 1) + ipow<7>(x - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 239
template<>
struct DGBasis<239> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (22*x - 1)*(x + y + 2*z - 1)*(75582*ipow<8>(y) + ipow<7>(y)*(254592*x - 254592) + 346528*ipow<6>(y)*ipow<2>(x - 1) + 244608*ipow<5>(y)*ipow<3>(x - 1) + 95550*ipow<4>(y)*ipow<4>(x - 1) + 20384*ipow<3>(y)*ipow<5>(x - 1) + 2184*ipow<2>(y)*ipow<6>(x - 1) + 96*y*ipow<7>(x - 1) + ipow<8>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (22*x - 1)*(x + y + 2*z - 1)*(254592*ipow<7>(y) + 346528*ipow<6>(y)*(2*x - 2) + 733824*ipow<5>(y)*ipow<2>(x - 1) + 382200*ipow<4>(y)*ipow<3>(x - 1) + 101920*ipow<3>(y)*ipow<4>(x - 1) + 13104*ipow<2>(y)*ipow<5>(x - 1) + 672*y*ipow<6>(x - 1) + 8*ipow<7>(x - 1)) + (22*x - 1)*(75582*ipow<8>(y) + ipow<7>(y)*(254592*x - 254592) + 346528*ipow<6>(y)*ipow<2>(x - 1) + 244608*ipow<5>(y)*ipow<3>(x - 1) + 95550*ipow<4>(y)*ipow<4>(x - 1) + 20384*ipow<3>(y)*ipow<5>(x - 1) + 2184*ipow<2>(y)*ipow<6>(x - 1) + 96*y*ipow<7>(x - 1) + ipow<8>(x - 1)) + 22*(x + y + 2*z - 1)*(75582*ipow<8>(y) + ipow<7>(y)*(254592*x - 254592) + 346528*ipow<6>(y)*ipow<2>(x - 1) + 244608*ipow<5>(y)*ipow<3>(x - 1) + 95550*ipow<4>(y)*ipow<4>(x - 1) + 20384*ipow<3>(y)*ipow<5>(x - 1) + 2184*ipow<2>(y)*ipow<6>(x - 1) + 96*y*ipow<7>(x - 1) + ipow<8>(x - 1)),
            (22*x - 1)*(x + y + 2*z - 1)*(604656*ipow<7>(y) + 7*ipow<6>(y)*(254592*x - 254592) + 2079168*ipow<5>(y)*ipow<2>(x - 1) + 1223040*ipow<4>(y)*ipow<3>(x - 1) + 382200*ipow<3>(y)*ipow<4>(x - 1) + 61152*ipow<2>(y)*ipow<5>(x - 1) + 4368*y*ipow<6>(x - 1) + 96*ipow<7>(x - 1)) + (22*x - 1)*(75582*ipow<8>(y) + ipow<7>(y)*(254592*x - 254592) + 346528*ipow<6>(y)*ipow<2>(x - 1) + 244608*ipow<5>(y)*ipow<3>(x - 1) + 95550*ipow<4>(y)*ipow<4>(x - 1) + 20384*ipow<3>(y)*ipow<5>(x - 1) + 2184*ipow<2>(y)*ipow<6>(x - 1) + 96*y*ipow<7>(x - 1) + ipow<8>(x - 1)),
            2*(22*x - 1)*(75582*ipow<8>(y) + ipow<7>(y)*(254592*x - 254592) + 346528*ipow<6>(y)*ipow<2>(x - 1) + 244608*ipow<5>(y)*ipow<3>(x - 1) + 95550*ipow<4>(y)*ipow<4>(x - 1) + 20384*ipow<3>(y)*ipow<5>(x - 1) + 2184*ipow<2>(y)*ipow<6>(x - 1) + 96*y*ipow<7>(x - 1) + ipow<8>(x - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 240
template<>
struct DGBasis<240> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x + y + 2*z - 1)*(293930*ipow<9>(y) + ipow<8>(y)*(1133730*x - 1133730) + 1813968*ipow<7>(y)*ipow<2>(x - 1) + 1559376*ipow<6>(y)*ipow<3>(x - 1) + 779688*ipow<5>(y)*ipow<4>(x - 1) + 229320*ipow<4>(y)*ipow<5>(x - 1) + 38220*ipow<3>(y)*ipow<6>(x - 1) + 3276*ipow<2>(y)*ipow<7>(x - 1) + 117*y*ipow<8>(x - 1) + ipow<9>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            293930*ipow<9>(y) + ipow<8>(y)*(1133730*x - 1133730) + 1813968*ipow<7>(y)*ipow<2>(x - 1) + 1559376*ipow<6>(y)*ipow<3>(x - 1) + 779688*ipow<5>(y)*ipow<4>(x - 1) + 229320*ipow<4>(y)*ipow<5>(x - 1) + 38220*ipow<3>(y)*ipow<6>(x - 1) + 3276*ipow<2>(y)*ipow<7>(x - 1) + 117*y*ipow<8>(x - 1) + ipow<9>(x - 1) + (x + y + 2*z - 1)*(1133730*ipow<8>(y) + 1813968*ipow<7>(y)*(2*x - 2) + 4678128*ipow<6>(y)*ipow<2>(x - 1) + 3118752*ipow<5>(y)*ipow<3>(x - 1) + 1146600*ipow<4>(y)*ipow<4>(x - 1) + 229320*ipow<3>(y)*ipow<5>(x - 1) + 22932*ipow<2>(y)*ipow<6>(x - 1) + 936*y*ipow<7>(x - 1) + 9*ipow<8>(x - 1)),
            293930*ipow<9>(y) + ipow<8>(y)*(1133730*x - 1133730) + 1813968*ipow<7>(y)*ipow<2>(x - 1) + 1559376*ipow<6>(y)*ipow<3>(x - 1) + 779688*ipow<5>(y)*ipow<4>(x - 1) + 229320*ipow<4>(y)*ipow<5>(x - 1) + 38220*ipow<3>(y)*ipow<6>(x - 1) + 3276*ipow<2>(y)*ipow<7>(x - 1) + 117*y*ipow<8>(x - 1) + ipow<9>(x - 1) + (x + y + 2*z - 1)*(2645370*ipow<8>(y) + 8*ipow<7>(y)*(1133730*x - 1133730) + 12697776*ipow<6>(y)*ipow<2>(x - 1) + 9356256*ipow<5>(y)*ipow<3>(x - 1) + 3898440*ipow<4>(y)*ipow<4>(x - 1) + 917280*ipow<3>(y)*ipow<5>(x - 1) + 114660*ipow<2>(y)*ipow<6>(x - 1) + 6552*y*ipow<7>(x - 1) + 117*ipow<8>(x - 1)),
            587860*ipow<9>(y) + 2*ipow<8>(y)*(1133730*x - 1133730) + 3627936*ipow<7>(y)*ipow<2>(x - 1) + 3118752*ipow<6>(y)*ipow<3>(x - 1) + 1559376*ipow<5>(y)*ipow<4>(x - 1) + 458640*ipow<4>(y)*ipow<5>(x - 1) + 76440*ipow<3>(y)*ipow<6>(x - 1) + 6552*ipow<2>(y)*ipow<7>(x - 1) + 234*y*ipow<8>(x - 1) + 2*ipow<9>(x - 1)
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 241
template<>
struct DGBasis<241> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*ipow<2>(x) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*ipow<2>(x) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(2*x + 2*y + 6*z - 2) + (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(2*x*(17*x*(3*x*(19*x*(5*x*(33*ipow<2>(x) - 96*x + 112) - 336) + 2100) - 1120) + 1680) + 2*x*(17*x*(3*x*(19*x*(5*x*(33*ipow<2>(x) - 96*x + 112) - 336) + 2100) - 1120) + x*(51*x*(19*x*(5*x*(33*ipow<2>(x) - 96*x + 112) - 336) + 2100) + 17*x*(57*x*(5*x*(33*ipow<2>(x) - 96*x + 112) - 336) + 3*x*(95*x*(33*ipow<2>(x) - 96*x + 112) + 19*x*(165*ipow<2>(x) + 5*x*(66*x - 96) - 480*x + 560) - 6384) + 6300) - 19040) + 1680) - 120),
            (2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*ipow<2>(x) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(2*x + 2*y + 6*z - 2),
            (2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*ipow<2>(x) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 242
template<>
struct DGBasis<242> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(x + 7*y - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (x + 7*y - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(204*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 4*x*(51*x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) + 51*x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + x*(38*x*(x*(22*x - 49) + 42) + 19*x*(2*x*(22*x - 49) + 2*x*(44*x - 49) + 84) - 665) + 140) - 714) + 112),
            (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + 7*(4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)),
            (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(x + 7*y - 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 243
template<>
struct DGBasis<243> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(2*x + 16*y - 2)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(2*x + 2*y + 6*z - 2) + (36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(51*x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) + 51*x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + x*(19*x*(7*x*(11*x - 18) + 75) + 19*x*(7*x*(11*x - 18) + x*(154*x - 126) + 75) - 380) + 45) - 102),
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(16*x + 72*y - 16)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(2*x + 2*y + 6*z - 2),
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(36*ipow<2>(y) + y*(16*x - 16) + ipow<2>(x - 1))*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 244
template<>
struct DGBasis<244> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(135*ipow<2>(y) + 27*y*(2*x - 2) + 3*ipow<2>(x - 1))*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(2*x + 2*y + 6*z - 2)*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1)) + (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(57*x*(x*(21*x*(22*x - 25) + 200) - 30) + 3*x*(19*x*(21*x*(22*x - 25) + 200) + 19*x*(21*x*(22*x - 25) + x*(924*x - 525) + 200) - 570) + 90)*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(495*ipow<2>(y) + 2*y*(135*x - 135) + 27*ipow<2>(x - 1))*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1)) + (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(2*x + 2*y + 6*z - 2)*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(6*x + 6*y + 12*z - 6)*(165*ipow<3>(y) + ipow<2>(y)*(135*x - 135) + 27*y*ipow<2>(x - 1) + ipow<3>(x - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 245
template<>
struct DGBasis<245> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(715*ipow<4>(y) + ipow<3>(y)*(880*x - 880) + 330*ipow<2>(y)*ipow<2>(x - 1) + 40*y*ipow<3>(x - 1) + ipow<4>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(880*ipow<3>(y) + 330*ipow<2>(y)*(2*x - 2) + 120*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1)) + (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(2*x + 2*y + 6*z - 2)*(715*ipow<4>(y) + ipow<3>(y)*(880*x - 880) + 330*ipow<2>(y)*ipow<2>(x - 1) + 40*y*ipow<3>(x - 1) + ipow<4>(x - 1)) + (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(95*x*(7*x*(11*x - 8) + 12) + 19*x*(35*x*(11*x - 8) + 5*x*(154*x - 56) + 60) - 76)*(715*ipow<4>(y) + ipow<3>(y)*(880*x - 880) + 330*ipow<2>(y)*ipow<2>(x - 1) + 40*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(2860*ipow<3>(y) + 3*ipow<2>(y)*(880*x - 880) + 660*y*ipow<2>(x - 1) + 40*ipow<3>(x - 1)) + (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(2*x + 2*y + 6*z - 2)*(715*ipow<4>(y) + ipow<3>(y)*(880*x - 880) + 330*ipow<2>(y)*ipow<2>(x - 1) + 40*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(6*x + 6*y + 12*z - 6)*(715*ipow<4>(y) + ipow<3>(y)*(880*x - 880) + 330*ipow<2>(y)*ipow<2>(x - 1) + 40*y*ipow<3>(x - 1) + ipow<4>(x - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 246
template<>
struct DGBasis<246> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (10*x*(7*x*(22*x - 9) + 6) - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(3003*ipow<5>(y) + ipow<4>(y)*(5005*x - 5005) + 2860*ipow<3>(y)*ipow<2>(x - 1) + 660*ipow<2>(y)*ipow<3>(x - 1) + 55*y*ipow<4>(x - 1) + ipow<5>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(5005*ipow<4>(y) + 2860*ipow<3>(y)*(2*x - 2) + 1980*ipow<2>(y)*ipow<2>(x - 1) + 220*y*ipow<3>(x - 1) + 5*ipow<4>(x - 1)) + (10*x*(7*x*(22*x - 9) + 6) - 1)*(2*x + 2*y + 6*z - 2)*(3003*ipow<5>(y) + ipow<4>(y)*(5005*x - 5005) + 2860*ipow<3>(y)*ipow<2>(x - 1) + 660*ipow<2>(y)*ipow<3>(x - 1) + 55*y*ipow<4>(x - 1) + ipow<5>(x - 1)) + (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(70*x*(22*x - 9) + 10*x*(308*x - 63) + 60)*(3003*ipow<5>(y) + ipow<4>(y)*(5005*x - 5005) + 2860*ipow<3>(y)*ipow<2>(x - 1) + 660*ipow<2>(y)*ipow<3>(x - 1) + 55*y*ipow<4>(x - 1) + ipow<5>(x - 1)),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(15015*ipow<4>(y) + 4*ipow<3>(y)*(5005*x - 5005) + 8580*ipow<2>(y)*ipow<2>(x - 1) + 1320*y*ipow<3>(x - 1) + 55*ipow<4>(x - 1)) + (10*x*(7*x*(22*x - 9) + 6) - 1)*(2*x + 2*y + 6*z - 2)*(3003*ipow<5>(y) + ipow<4>(y)*(5005*x - 5005) + 2860*ipow<3>(y)*ipow<2>(x - 1) + 660*ipow<2>(y)*ipow<3>(x - 1) + 55*y*ipow<4>(x - 1) + ipow<5>(x - 1)),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(6*x + 6*y + 12*z - 6)*(3003*ipow<5>(y) + ipow<4>(y)*(5005*x - 5005) + 2860*ipow<3>(y)*ipow<2>(x - 1) + 660*ipow<2>(y)*ipow<3>(x - 1) + 55*y*ipow<4>(x - 1) + ipow<5>(x - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 247
template<>
struct DGBasis<247> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (21*x*(11*x - 2) + 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(12376*ipow<6>(y) + ipow<5>(y)*(26208*x - 26208) + 20475*ipow<4>(y)*ipow<2>(x - 1) + 7280*ipow<3>(y)*ipow<3>(x - 1) + 1170*ipow<2>(y)*ipow<4>(x - 1) + 72*y*ipow<5>(x - 1) + ipow<6>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (462*x - 42)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(12376*ipow<6>(y) + ipow<5>(y)*(26208*x - 26208) + 20475*ipow<4>(y)*ipow<2>(x - 1) + 7280*ipow<3>(y)*ipow<3>(x - 1) + 1170*ipow<2>(y)*ipow<4>(x - 1) + 72*y*ipow<5>(x - 1) + ipow<6>(x - 1)) + (21*x*(11*x - 2) + 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(26208*ipow<5>(y) + 20475*ipow<4>(y)*(2*x - 2) + 21840*ipow<3>(y)*ipow<2>(x - 1) + 4680*ipow<2>(y)*ipow<3>(x - 1) + 360*y*ipow<4>(x - 1) + 6*ipow<5>(x - 1)) + (21*x*(11*x - 2) + 1)*(2*x + 2*y + 6*z - 2)*(12376*ipow<6>(y) + ipow<5>(y)*(26208*x - 26208) + 20475*ipow<4>(y)*ipow<2>(x - 1) + 7280*ipow<3>(y)*ipow<3>(x - 1) + 1170*ipow<2>(y)*ipow<4>(x - 1) + 72*y*ipow<5>(x - 1) + ipow<6>(x - 1)),
            (21*x*(11*x - 2) + 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(74256*ipow<5>(y) + 5*ipow<4>(y)*(26208*x - 26208) + 81900*ipow<3>(y)*ipow<2>(x - 1) + 21840*ipow<2>(y)*ipow<3>(x - 1) + 2340*y*ipow<4>(x - 1) + 72*ipow<5>(x - 1)) + (21*x*(11*x - 2) + 1)*(2*x + 2*y + 6*z - 2)*(12376*ipow<6>(y) + ipow<5>(y)*(26208*x - 26208) + 20475*ipow<4>(y)*ipow<2>(x - 1) + 7280*ipow<3>(y)*ipow<3>(x - 1) + 1170*ipow<2>(y)*ipow<4>(x - 1) + 72*y*ipow<5>(x - 1) + ipow<6>(x - 1)),
            (21*x*(11*x - 2) + 1)*(6*x + 6*y + 12*z - 6)*(12376*ipow<6>(y) + ipow<5>(y)*(26208*x - 26208) + 20475*ipow<4>(y)*ipow<2>(x - 1) + 7280*ipow<3>(y)*ipow<3>(x - 1) + 1170*ipow<2>(y)*ipow<4>(x - 1) + 72*y*ipow<5>(x - 1) + ipow<6>(x - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 248
template<>
struct DGBasis<248> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (22*x - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(50388*ipow<7>(y) + ipow<6>(y)*(129948*x - 129948) + 129948*ipow<5>(y)*ipow<2>(x - 1) + 63700*ipow<4>(y)*ipow<3>(x - 1) + 15925*ipow<3>(y)*ipow<4>(x - 1) + 1911*ipow<2>(y)*ipow<5>(x - 1) + 91*y*ipow<6>(x - 1) + ipow<7>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (22*x - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(129948*ipow<6>(y) + 129948*ipow<5>(y)*(2*x - 2) + 191100*ipow<4>(y)*ipow<2>(x - 1) + 63700*ipow<3>(y)*ipow<3>(x - 1) + 9555*ipow<2>(y)*ipow<4>(x - 1) + 546*y*ipow<5>(x - 1) + 7*ipow<6>(x - 1)) + (22*x - 1)*(2*x + 2*y + 6*z - 2)*(50388*ipow<7>(y) + ipow<6>(y)*(129948*x - 129948) + 129948*ipow<5>(y)*ipow<2>(x - 1) + 63700*ipow<4>(y)*ipow<3>(x - 1) + 15925*ipow<3>(y)*ipow<4>(x - 1) + 1911*ipow<2>(y)*ipow<5>(x - 1) + 91*y*ipow<6>(x - 1) + ipow<7>(x - 1)) + 22*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(50388*ipow<7>(y) + ipow<6>(y)*(129948*x - 129948) + 129948*ipow<5>(y)*ipow<2>(x - 1) + 63700*ipow<4>(y)*ipow<3>(x - 1) + 15925*ipow<3>(y)*ipow<4>(x - 1) + 1911*ipow<2>(y)*ipow<5>(x - 1) + 91*y*ipow<6>(x - 1) + ipow<7>(x - 1)),
            (22*x - 1)*(6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(352716*ipow<6>(y) + 6*ipow<5>(y)*(129948*x - 129948) + 649740*ipow<4>(y)*ipow<2>(x - 1) + 254800*ipow<3>(y)*ipow<3>(x - 1) + 47775*ipow<2>(y)*ipow<4>(x - 1) + 3822*y*ipow<5>(x - 1) + 91*ipow<6>(x - 1)) + (22*x - 1)*(2*x + 2*y + 6*z - 2)*(50388*ipow<7>(y) + ipow<6>(y)*(129948*x - 129948) + 129948*ipow<5>(y)*ipow<2>(x - 1) + 63700*ipow<4>(y)*ipow<3>(x - 1) + 15925*ipow<3>(y)*ipow<4>(x - 1) + 1911*ipow<2>(y)*ipow<5>(x - 1) + 91*y*ipow<6>(x - 1) + ipow<7>(x - 1)),
            (22*x - 1)*(6*x + 6*y + 12*z - 6)*(50388*ipow<7>(y) + ipow<6>(y)*(129948*x - 129948) + 129948*ipow<5>(y)*ipow<2>(x - 1) + 63700*ipow<4>(y)*ipow<3>(x - 1) + 15925*ipow<3>(y)*ipow<4>(x - 1) + 1911*ipow<2>(y)*ipow<5>(x - 1) + 91*y*ipow<6>(x - 1) + ipow<7>(x - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 249
template<>
struct DGBasis<249> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(203490*ipow<8>(y) + ipow<7>(y)*(620160*x - 620160) + 759696*ipow<6>(y)*ipow<2>(x - 1) + 479808*ipow<5>(y)*ipow<3>(x - 1) + 166600*ipow<4>(y)*ipow<4>(x - 1) + 31360*ipow<3>(y)*ipow<5>(x - 1) + 2940*ipow<2>(y)*ipow<6>(x - 1) + 112*y*ipow<7>(x - 1) + ipow<8>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(620160*ipow<7>(y) + 759696*ipow<6>(y)*(2*x - 2) + 1439424*ipow<5>(y)*ipow<2>(x - 1) + 666400*ipow<4>(y)*ipow<3>(x - 1) + 156800*ipow<3>(y)*ipow<4>(x - 1) + 17640*ipow<2>(y)*ipow<5>(x - 1) + 784*y*ipow<6>(x - 1) + 8*ipow<7>(x - 1)) + (2*x + 2*y + 6*z - 2)*(203490*ipow<8>(y) + ipow<7>(y)*(620160*x - 620160) + 759696*ipow<6>(y)*ipow<2>(x - 1) + 479808*ipow<5>(y)*ipow<3>(x - 1) + 166600*ipow<4>(y)*ipow<4>(x - 1) + 31360*ipow<3>(y)*ipow<5>(x - 1) + 2940*ipow<2>(y)*ipow<6>(x - 1) + 112*y*ipow<7>(x - 1) + ipow<8>(x - 1)),
            (6*ipow<2>(z) + z*(6*x + 6*y - 6) + ipow<2>(x + y - 1))*(1627920*ipow<7>(y) + 7*ipow<6>(y)*(620160*x - 620160) + 4558176*ipow<5>(y)*ipow<2>(x - 1) + 2399040*ipow<4>(y)*ipow<3>(x - 1) + 666400*ipow<3>(y)*ipow<4>(x - 1) + 94080*ipow<2>(y)*ipow<5>(x - 1) + 5880*y*ipow<6>(x - 1) + 112*ipow<7>(x - 1)) + (2*x + 2*y + 6*z - 2)*(203490*ipow<8>(y) + ipow<7>(y)*(620160*x - 620160) + 759696*ipow<6>(y)*ipow<2>(x - 1) + 479808*ipow<5>(y)*ipow<3>(x - 1) + 166600*ipow<4>(y)*ipow<4>(x - 1) + 31360*ipow<3>(y)*ipow<5>(x - 1) + 2940*ipow<2>(y)*ipow<6>(x - 1) + 112*y*ipow<7>(x - 1) + ipow<8>(x - 1)),
            (6*x + 6*y + 12*z - 6)*(203490*ipow<8>(y) + ipow<7>(y)*(620160*x - 620160) + 759696*ipow<6>(y)*ipow<2>(x - 1) + 479808*ipow<5>(y)*ipow<3>(x - 1) + 166600*ipow<4>(y)*ipow<4>(x - 1) + 31360*ipow<3>(y)*ipow<5>(x - 1) + 2940*ipow<2>(y)*ipow<6>(x - 1) + 112*y*ipow<7>(x - 1) + ipow<8>(x - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 250
template<>
struct DGBasis<250> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)) + (204*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 4*x*(51*x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) + 51*x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + x*(38*x*(x*(22*x - 49) + 42) + 19*x*(2*x*(22*x - 49) + 2*x*(44*x - 49) + 84) - 665) + 140) - 714) + 112)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)),
            (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 251
template<>
struct DGBasis<251> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(x + 9*y - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(x + 9*y - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)) + (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (x + 9*y - 1)*(51*x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) + 51*x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + x*(19*x*(7*x*(11*x - 18) + 75) + 19*x*(7*x*(11*x - 18) + x*(154*x - 126) + 75) - 380) + 45) - 102)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(x + 9*y - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)) + 9*(51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(x + 9*y - 1)*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 252
template<>
struct DGBasis<252> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(2*x + 20*y - 2)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)) + (55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(57*x*(x*(21*x*(22*x - 25) + 200) - 30) + 3*x*(19*x*(21*x*(22*x - 25) + 200) + 19*x*(21*x*(22*x - 25) + x*(924*x - 525) + 200) - 570) + 90)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(20*x + 110*y - 20)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1)),
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(55*ipow<2>(y) + y*(20*x - 20) + ipow<2>(x - 1))*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 253
template<>
struct DGBasis<253> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(286*ipow<3>(y) + ipow<2>(y)*(198*x - 198) + 33*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(198*ipow<2>(y) + 33*y*(2*x - 2) + 3*ipow<2>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(286*ipow<3>(y) + ipow<2>(y)*(198*x - 198) + 33*y*ipow<2>(x - 1) + ipow<3>(x - 1)) + (95*x*(7*x*(11*x - 8) + 12) + 19*x*(35*x*(11*x - 8) + 5*x*(154*x - 56) + 60) - 76)*(286*ipow<3>(y) + ipow<2>(y)*(198*x - 198) + 33*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(858*ipow<2>(y) + 2*y*(198*x - 198) + 33*ipow<2>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(286*ipow<3>(y) + ipow<2>(y)*(198*x - 198) + 33*y*ipow<2>(x - 1) + ipow<3>(x - 1)),
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))*(286*ipow<3>(y) + ipow<2>(y)*(198*x - 198) + 33*y*ipow<2>(x - 1) + ipow<3>(x - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 254
template<>
struct DGBasis<254> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (10*x*(7*x*(22*x - 9) + 6) - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(1365*ipow<4>(y) + ipow<3>(y)*(1456*x - 1456) + 468*ipow<2>(y)*ipow<2>(x - 1) + 48*y*ipow<3>(x - 1) + ipow<4>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(1365*ipow<4>(y) + ipow<3>(y)*(1456*x - 1456) + 468*ipow<2>(y)*ipow<2>(x - 1) + 48*y*ipow<3>(x - 1) + ipow<4>(x - 1)) + (10*x*(7*x*(22*x - 9) + 6) - 1)*(1456*ipow<3>(y) + 468*ipow<2>(y)*(2*x - 2) + 144*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)) + (70*x*(22*x - 9) + 10*x*(308*x - 63) + 60)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(1365*ipow<4>(y) + ipow<3>(y)*(1456*x - 1456) + 468*ipow<2>(y)*ipow<2>(x - 1) + 48*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(1365*ipow<4>(y) + ipow<3>(y)*(1456*x - 1456) + 468*ipow<2>(y)*ipow<2>(x - 1) + 48*y*ipow<3>(x - 1) + ipow<4>(x - 1)) + (10*x*(7*x*(22*x - 9) + 6) - 1)*(5460*ipow<3>(y) + 3*ipow<2>(y)*(1456*x - 1456) + 936*y*ipow<2>(x - 1) + 48*ipow<3>(x - 1))*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1)),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))*(1365*ipow<4>(y) + ipow<3>(y)*(1456*x - 1456) + 468*ipow<2>(y)*ipow<2>(x - 1) + 48*y*ipow<3>(x - 1) + ipow<4>(x - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 255
template<>
struct DGBasis<255> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (21*x*(11*x - 2) + 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(6188*ipow<5>(y) + ipow<4>(y)*(9100*x - 9100) + 4550*ipow<3>(y)*ipow<2>(x - 1) + 910*ipow<2>(y)*ipow<3>(x - 1) + 65*y*ipow<4>(x - 1) + ipow<5>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (462*x - 42)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(6188*ipow<5>(y) + ipow<4>(y)*(9100*x - 9100) + 4550*ipow<3>(y)*ipow<2>(x - 1) + 910*ipow<2>(y)*ipow<3>(x - 1) + 65*y*ipow<4>(x - 1) + ipow<5>(x - 1)) + (21*x*(11*x - 2) + 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(6188*ipow<5>(y) + ipow<4>(y)*(9100*x - 9100) + 4550*ipow<3>(y)*ipow<2>(x - 1) + 910*ipow<2>(y)*ipow<3>(x - 1) + 65*y*ipow<4>(x - 1) + ipow<5>(x - 1)) + (21*x*(11*x - 2) + 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(9100*ipow<4>(y) + 4550*ipow<3>(y)*(2*x - 2) + 2730*ipow<2>(y)*ipow<2>(x - 1) + 260*y*ipow<3>(x - 1) + 5*ipow<4>(x - 1)),
            (21*x*(11*x - 2) + 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(6188*ipow<5>(y) + ipow<4>(y)*(9100*x - 9100) + 4550*ipow<3>(y)*ipow<2>(x - 1) + 910*ipow<2>(y)*ipow<3>(x - 1) + 65*y*ipow<4>(x - 1) + ipow<5>(x - 1)) + (21*x*(11*x - 2) + 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(30940*ipow<4>(y) + 4*ipow<3>(y)*(9100*x - 9100) + 13650*ipow<2>(y)*ipow<2>(x - 1) + 1820*y*ipow<3>(x - 1) + 65*ipow<4>(x - 1)),
            (21*x*(11*x - 2) + 1)*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))*(6188*ipow<5>(y) + ipow<4>(y)*(9100*x - 9100) + 4550*ipow<3>(y)*ipow<2>(x - 1) + 910*ipow<2>(y)*ipow<3>(x - 1) + 65*y*ipow<4>(x - 1) + ipow<5>(x - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 256
template<>
struct DGBasis<256> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (22*x - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(27132*ipow<6>(y) + ipow<5>(y)*(51408*x - 51408) + 35700*ipow<4>(y)*ipow<2>(x - 1) + 11200*ipow<3>(y)*ipow<3>(x - 1) + 1575*ipow<2>(y)*ipow<4>(x - 1) + 84*y*ipow<5>(x - 1) + ipow<6>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (22*x - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(27132*ipow<6>(y) + ipow<5>(y)*(51408*x - 51408) + 35700*ipow<4>(y)*ipow<2>(x - 1) + 11200*ipow<3>(y)*ipow<3>(x - 1) + 1575*ipow<2>(y)*ipow<4>(x - 1) + 84*y*ipow<5>(x - 1) + ipow<6>(x - 1)) + (22*x - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(51408*ipow<5>(y) + 35700*ipow<4>(y)*(2*x - 2) + 33600*ipow<3>(y)*ipow<2>(x - 1) + 6300*ipow<2>(y)*ipow<3>(x - 1) + 420*y*ipow<4>(x - 1) + 6*ipow<5>(x - 1)) + 22*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(27132*ipow<6>(y) + ipow<5>(y)*(51408*x - 51408) + 35700*ipow<4>(y)*ipow<2>(x - 1) + 11200*ipow<3>(y)*ipow<3>(x - 1) + 1575*ipow<2>(y)*ipow<4>(x - 1) + 84*y*ipow<5>(x - 1) + ipow<6>(x - 1)),
            (22*x - 1)*(30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(27132*ipow<6>(y) + ipow<5>(y)*(51408*x - 51408) + 35700*ipow<4>(y)*ipow<2>(x - 1) + 11200*ipow<3>(y)*ipow<3>(x - 1) + 1575*ipow<2>(y)*ipow<4>(x - 1) + 84*y*ipow<5>(x - 1) + ipow<6>(x - 1)) + (22*x - 1)*(20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(162792*ipow<5>(y) + 5*ipow<4>(y)*(51408*x - 51408) + 142800*ipow<3>(y)*ipow<2>(x - 1) + 33600*ipow<2>(y)*ipow<3>(x - 1) + 3150*y*ipow<4>(x - 1) + 84*ipow<5>(x - 1)),
            (22*x - 1)*(60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))*(27132*ipow<6>(y) + ipow<5>(y)*(51408*x - 51408) + 35700*ipow<4>(y)*ipow<2>(x - 1) + 11200*ipow<3>(y)*ipow<3>(x - 1) + 1575*ipow<2>(y)*ipow<4>(x - 1) + 84*y*ipow<5>(x - 1) + ipow<6>(x - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 257
template<>
struct DGBasis<257> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(116280*ipow<7>(y) + ipow<6>(y)*(271320*x - 271320) + 244188*ipow<5>(y)*ipow<2>(x - 1) + 107100*ipow<4>(y)*ipow<3>(x - 1) + 23800*ipow<3>(y)*ipow<4>(x - 1) + 2520*ipow<2>(y)*ipow<5>(x - 1) + 105*y*ipow<6>(x - 1) + ipow<7>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(116280*ipow<7>(y) + ipow<6>(y)*(271320*x - 271320) + 244188*ipow<5>(y)*ipow<2>(x - 1) + 107100*ipow<4>(y)*ipow<3>(x - 1) + 23800*ipow<3>(y)*ipow<4>(x - 1) + 2520*ipow<2>(y)*ipow<5>(x - 1) + 105*y*ipow<6>(x - 1) + ipow<7>(x - 1)) + (20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(271320*ipow<6>(y) + 244188*ipow<5>(y)*(2*x - 2) + 321300*ipow<4>(y)*ipow<2>(x - 1) + 95200*ipow<3>(y)*ipow<3>(x - 1) + 12600*ipow<2>(y)*ipow<4>(x - 1) + 630*y*ipow<5>(x - 1) + 7*ipow<6>(x - 1)),
            (30*ipow<2>(z) + 12*z*(2*x + 2*y - 2) + 3*ipow<2>(x + y - 1))*(116280*ipow<7>(y) + ipow<6>(y)*(271320*x - 271320) + 244188*ipow<5>(y)*ipow<2>(x - 1) + 107100*ipow<4>(y)*ipow<3>(x - 1) + 23800*ipow<3>(y)*ipow<4>(x - 1) + 2520*ipow<2>(y)*ipow<5>(x - 1) + 105*y*ipow<6>(x - 1) + ipow<7>(x - 1)) + (20*ipow<3>(z) + ipow<2>(z)*(30*x + 30*y - 30) + 12*z*ipow<2>(x + y - 1) + ipow<3>(x + y - 1))*(813960*ipow<6>(y) + 6*ipow<5>(y)*(271320*x - 271320) + 1220940*ipow<4>(y)*ipow<2>(x - 1) + 428400*ipow<3>(y)*ipow<3>(x - 1) + 71400*ipow<2>(y)*ipow<4>(x - 1) + 5040*y*ipow<5>(x - 1) + 105*ipow<6>(x - 1)),
            (60*ipow<2>(z) + 2*z*(30*x + 30*y - 30) + 12*ipow<2>(x + y - 1))*(116280*ipow<7>(y) + ipow<6>(y)*(271320*x - 271320) + 244188*ipow<5>(y)*ipow<2>(x - 1) + 107100*ipow<4>(y)*ipow<3>(x - 1) + 23800*ipow<3>(y)*ipow<4>(x - 1) + 2520*ipow<2>(y)*ipow<5>(x - 1) + 105*y*ipow<6>(x - 1) + ipow<7>(x - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 258
template<>
struct DGBasis<258> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)) + (51*x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) + 51*x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + x*(19*x*(7*x*(11*x - 18) + 75) + 19*x*(7*x*(11*x - 18) + x*(154*x - 126) + 75) - 380) + 45) - 102)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)),
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)),
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 259
template<>
struct DGBasis<259> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(x + 11*y - 1)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(x + 11*y - 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)) + (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (x + 11*y - 1)*(57*x*(x*(21*x*(22*x - 25) + 200) - 30) + 3*x*(19*x*(21*x*(22*x - 25) + 200) + 19*x*(21*x*(22*x - 25) + x*(924*x - 525) + 200) - 570) + 90)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)),
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(x + 11*y - 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)) + 11*(3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)),
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(x + 11*y - 1)*(280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 260
template<>
struct DGBasis<260> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(78*ipow<2>(y) + y*(24*x - 24) + ipow<2>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(2*x + 24*y - 2)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(78*ipow<2>(y) + y*(24*x - 24) + ipow<2>(x - 1))*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)) + (78*ipow<2>(y) + y*(24*x - 24) + ipow<2>(x - 1))*(95*x*(7*x*(11*x - 8) + 12) + 19*x*(35*x*(11*x - 8) + 5*x*(154*x - 56) + 60) - 76)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)),
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(24*x + 156*y - 24)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(78*ipow<2>(y) + y*(24*x - 24) + ipow<2>(x - 1))*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)),
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(78*ipow<2>(y) + y*(24*x - 24) + ipow<2>(x - 1))*(280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 261
template<>
struct DGBasis<261> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (10*x*(7*x*(22*x - 9) + 6) - 1)*(455*ipow<3>(y) + ipow<2>(y)*(273*x - 273) + 39*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(273*ipow<2>(y) + 39*y*(2*x - 2) + 3*ipow<2>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (10*x*(7*x*(22*x - 9) + 6) - 1)*(455*ipow<3>(y) + ipow<2>(y)*(273*x - 273) + 39*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)) + (70*x*(22*x - 9) + 10*x*(308*x - 63) + 60)*(455*ipow<3>(y) + ipow<2>(y)*(273*x - 273) + 39*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(1365*ipow<2>(y) + 2*y*(273*x - 273) + 39*ipow<2>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (10*x*(7*x*(22*x - 9) + 6) - 1)*(455*ipow<3>(y) + ipow<2>(y)*(273*x - 273) + 39*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1)),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(455*ipow<3>(y) + ipow<2>(y)*(273*x - 273) + 39*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 262
template<>
struct DGBasis<262> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (21*x*(11*x - 2) + 1)*(2380*ipow<4>(y) + ipow<3>(y)*(2240*x - 2240) + 630*ipow<2>(y)*ipow<2>(x - 1) + 56*y*ipow<3>(x - 1) + ipow<4>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (462*x - 42)*(2380*ipow<4>(y) + ipow<3>(y)*(2240*x - 2240) + 630*ipow<2>(y)*ipow<2>(x - 1) + 56*y*ipow<3>(x - 1) + ipow<4>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (21*x*(11*x - 2) + 1)*(2240*ipow<3>(y) + 630*ipow<2>(y)*(2*x - 2) + 168*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (21*x*(11*x - 2) + 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1))*(2380*ipow<4>(y) + ipow<3>(y)*(2240*x - 2240) + 630*ipow<2>(y)*ipow<2>(x - 1) + 56*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (21*x*(11*x - 2) + 1)*(9520*ipow<3>(y) + 3*ipow<2>(y)*(2240*x - 2240) + 1260*y*ipow<2>(x - 1) + 56*ipow<3>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + (21*x*(11*x - 2) + 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1))*(2380*ipow<4>(y) + ipow<3>(y)*(2240*x - 2240) + 630*ipow<2>(y)*ipow<2>(x - 1) + 56*y*ipow<3>(x - 1) + ipow<4>(x - 1)),
            (21*x*(11*x - 2) + 1)*(280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))*(2380*ipow<4>(y) + ipow<3>(y)*(2240*x - 2240) + 630*ipow<2>(y)*ipow<2>(x - 1) + 56*y*ipow<3>(x - 1) + ipow<4>(x - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 263
template<>
struct DGBasis<263> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (22*x - 1)*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1))*(11628*ipow<5>(y) + ipow<4>(y)*(15300*x - 15300) + 6800*ipow<3>(y)*ipow<2>(x - 1) + 1200*ipow<2>(y)*ipow<3>(x - 1) + 75*y*ipow<4>(x - 1) + ipow<5>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (22*x - 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1))*(11628*ipow<5>(y) + ipow<4>(y)*(15300*x - 15300) + 6800*ipow<3>(y)*ipow<2>(x - 1) + 1200*ipow<2>(y)*ipow<3>(x - 1) + 75*y*ipow<4>(x - 1) + ipow<5>(x - 1)) + (22*x - 1)*(15300*ipow<4>(y) + 6800*ipow<3>(y)*(2*x - 2) + 3600*ipow<2>(y)*ipow<2>(x - 1) + 300*y*ipow<3>(x - 1) + 5*ipow<4>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)) + 22*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1))*(11628*ipow<5>(y) + ipow<4>(y)*(15300*x - 15300) + 6800*ipow<3>(y)*ipow<2>(x - 1) + 1200*ipow<2>(y)*ipow<3>(x - 1) + 75*y*ipow<4>(x - 1) + ipow<5>(x - 1)),
            (22*x - 1)*(140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1))*(11628*ipow<5>(y) + ipow<4>(y)*(15300*x - 15300) + 6800*ipow<3>(y)*ipow<2>(x - 1) + 1200*ipow<2>(y)*ipow<3>(x - 1) + 75*y*ipow<4>(x - 1) + ipow<5>(x - 1)) + (22*x - 1)*(58140*ipow<4>(y) + 4*ipow<3>(y)*(15300*x - 15300) + 20400*ipow<2>(y)*ipow<2>(x - 1) + 2400*y*ipow<3>(x - 1) + 75*ipow<4>(x - 1))*(70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1)),
            (22*x - 1)*(280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))*(11628*ipow<5>(y) + ipow<4>(y)*(15300*x - 15300) + 6800*ipow<3>(y)*ipow<2>(x - 1) + 1200*ipow<2>(y)*ipow<3>(x - 1) + 75*y*ipow<4>(x - 1) + ipow<5>(x - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 264
template<>
struct DGBasis<264> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1))*(54264*ipow<6>(y) + ipow<5>(y)*(93024*x - 93024) + 58140*ipow<4>(y)*ipow<2>(x - 1) + 16320*ipow<3>(y)*ipow<3>(x - 1) + 2040*ipow<2>(y)*ipow<4>(x - 1) + 96*y*ipow<5>(x - 1) + ipow<6>(x - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1))*(54264*ipow<6>(y) + ipow<5>(y)*(93024*x - 93024) + 58140*ipow<4>(y)*ipow<2>(x - 1) + 16320*ipow<3>(y)*ipow<3>(x - 1) + 2040*ipow<2>(y)*ipow<4>(x - 1) + 96*y*ipow<5>(x - 1) + ipow<6>(x - 1)) + (70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1))*(93024*ipow<5>(y) + 58140*ipow<4>(y)*(2*x - 2) + 48960*ipow<3>(y)*ipow<2>(x - 1) + 8160*ipow<2>(y)*ipow<3>(x - 1) + 480*y*ipow<4>(x - 1) + 6*ipow<5>(x - 1)),
            (140*ipow<3>(z) + 90*ipow<2>(z)*(2*x + 2*y - 2) + 60*z*ipow<2>(x + y - 1) + 4*ipow<3>(x + y - 1))*(54264*ipow<6>(y) + ipow<5>(y)*(93024*x - 93024) + 58140*ipow<4>(y)*ipow<2>(x - 1) + 16320*ipow<3>(y)*ipow<3>(x - 1) + 2040*ipow<2>(y)*ipow<4>(x - 1) + 96*y*ipow<5>(x - 1) + ipow<6>(x - 1)) + (70*ipow<4>(z) + ipow<3>(z)*(140*x + 140*y - 140) + 90*ipow<2>(z)*ipow<2>(x + y - 1) + 20*z*ipow<3>(x + y - 1) + ipow<4>(x + y - 1))*(325584*ipow<5>(y) + 5*ipow<4>(y)*(93024*x - 93024) + 232560*ipow<3>(y)*ipow<2>(x - 1) + 48960*ipow<2>(y)*ipow<3>(x - 1) + 4080*y*ipow<4>(x - 1) + 96*ipow<5>(x - 1)),
            (280*ipow<3>(z) + 3*ipow<2>(z)*(140*x + 140*y - 140) + 180*z*ipow<2>(x + y - 1) + 20*ipow<3>(x + y - 1))*(54264*ipow<6>(y) + ipow<5>(y)*(93024*x - 93024) + 58140*ipow<4>(y)*ipow<2>(x - 1) + 16320*ipow<3>(y)*ipow<3>(x - 1) + 2040*ipow<2>(y)*ipow<4>(x - 1) + 96*y*ipow<5>(x - 1) + ipow<6>(x - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 265
template<>
struct DGBasis<265> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)) + (57*x*(x*(21*x*(22*x - 25) + 200) - 30) + 3*x*(19*x*(21*x*(22*x - 25) + 200) + 19*x*(21*x*(22*x - 25) + x*(924*x - 525) + 200) - 570) + 90)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)),
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)),
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(1260*ipow<4>(z) + 4*ipow<3>(z)*(630*x + 630*y - 630) + 1680*ipow<2>(z)*ipow<2>(x + y - 1) + 420*z*ipow<3>(x + y - 1) + 30*ipow<4>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 266
template<>
struct DGBasis<266> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(x + 13*y - 1)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(x + 13*y - 1)*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)) + (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (x + 13*y - 1)*(95*x*(7*x*(11*x - 8) + 12) + 19*x*(35*x*(11*x - 8) + 5*x*(154*x - 56) + 60) - 76)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)),
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(x + 13*y - 1)*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)) + 13*(19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)),
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(x + 13*y - 1)*(1260*ipow<4>(z) + 4*ipow<3>(z)*(630*x + 630*y - 630) + 1680*ipow<2>(z)*ipow<2>(x + y - 1) + 420*z*ipow<3>(x + y - 1) + 30*ipow<4>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 267
template<>
struct DGBasis<267> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (10*x*(7*x*(22*x - 9) + 6) - 1)*(105*ipow<2>(y) + y*(28*x - 28) + ipow<2>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(2*x + 28*y - 2)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (10*x*(7*x*(22*x - 9) + 6) - 1)*(105*ipow<2>(y) + y*(28*x - 28) + ipow<2>(x - 1))*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)) + (105*ipow<2>(y) + y*(28*x - 28) + ipow<2>(x - 1))*(70*x*(22*x - 9) + 10*x*(308*x - 63) + 60)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(28*x + 210*y - 28)*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (10*x*(7*x*(22*x - 9) + 6) - 1)*(105*ipow<2>(y) + y*(28*x - 28) + ipow<2>(x - 1))*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(105*ipow<2>(y) + y*(28*x - 28) + ipow<2>(x - 1))*(1260*ipow<4>(z) + 4*ipow<3>(z)*(630*x + 630*y - 630) + 1680*ipow<2>(z)*ipow<2>(x + y - 1) + 420*z*ipow<3>(x + y - 1) + 30*ipow<4>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 268
template<>
struct DGBasis<268> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (21*x*(11*x - 2) + 1)*(680*ipow<3>(y) + ipow<2>(y)*(360*x - 360) + 45*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (462*x - 42)*(680*ipow<3>(y) + ipow<2>(y)*(360*x - 360) + 45*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (21*x*(11*x - 2) + 1)*(360*ipow<2>(y) + 45*y*(2*x - 2) + 3*ipow<2>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (21*x*(11*x - 2) + 1)*(680*ipow<3>(y) + ipow<2>(y)*(360*x - 360) + 45*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)),
            (21*x*(11*x - 2) + 1)*(2040*ipow<2>(y) + 2*y*(360*x - 360) + 45*ipow<2>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (21*x*(11*x - 2) + 1)*(680*ipow<3>(y) + ipow<2>(y)*(360*x - 360) + 45*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)),
            (21*x*(11*x - 2) + 1)*(680*ipow<3>(y) + ipow<2>(y)*(360*x - 360) + 45*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(1260*ipow<4>(z) + 4*ipow<3>(z)*(630*x + 630*y - 630) + 1680*ipow<2>(z)*ipow<2>(x + y - 1) + 420*z*ipow<3>(x + y - 1) + 30*ipow<4>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 269
template<>
struct DGBasis<269> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (22*x - 1)*(3876*ipow<4>(y) + ipow<3>(y)*(3264*x - 3264) + 816*ipow<2>(y)*ipow<2>(x - 1) + 64*y*ipow<3>(x - 1) + ipow<4>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (22*x - 1)*(3264*ipow<3>(y) + 816*ipow<2>(y)*(2*x - 2) + 192*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (22*x - 1)*(3876*ipow<4>(y) + ipow<3>(y)*(3264*x - 3264) + 816*ipow<2>(y)*ipow<2>(x - 1) + 64*y*ipow<3>(x - 1) + ipow<4>(x - 1))*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)) + 22*(3876*ipow<4>(y) + ipow<3>(y)*(3264*x - 3264) + 816*ipow<2>(y)*ipow<2>(x - 1) + 64*y*ipow<3>(x - 1) + ipow<4>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)),
            (22*x - 1)*(15504*ipow<3>(y) + 3*ipow<2>(y)*(3264*x - 3264) + 1632*y*ipow<2>(x - 1) + 64*ipow<3>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (22*x - 1)*(3876*ipow<4>(y) + ipow<3>(y)*(3264*x - 3264) + 816*ipow<2>(y)*ipow<2>(x - 1) + 64*y*ipow<3>(x - 1) + ipow<4>(x - 1))*(630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1)),
            (22*x - 1)*(3876*ipow<4>(y) + ipow<3>(y)*(3264*x - 3264) + 816*ipow<2>(y)*ipow<2>(x - 1) + 64*y*ipow<3>(x - 1) + ipow<4>(x - 1))*(1260*ipow<4>(z) + 4*ipow<3>(z)*(630*x + 630*y - 630) + 1680*ipow<2>(z)*ipow<2>(x + y - 1) + 420*z*ipow<3>(x + y - 1) + 30*ipow<4>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 270
template<>
struct DGBasis<270> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (20349*ipow<5>(y) + ipow<4>(y)*(24225*x - 24225) + 9690*ipow<3>(y)*ipow<2>(x - 1) + 1530*ipow<2>(y)*ipow<3>(x - 1) + 85*y*ipow<4>(x - 1) + ipow<5>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (24225*ipow<4>(y) + 9690*ipow<3>(y)*(2*x - 2) + 4590*ipow<2>(y)*ipow<2>(x - 1) + 340*y*ipow<3>(x - 1) + 5*ipow<4>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1))*(20349*ipow<5>(y) + ipow<4>(y)*(24225*x - 24225) + 9690*ipow<3>(y)*ipow<2>(x - 1) + 1530*ipow<2>(y)*ipow<3>(x - 1) + 85*y*ipow<4>(x - 1) + ipow<5>(x - 1)),
            (101745*ipow<4>(y) + 4*ipow<3>(y)*(24225*x - 24225) + 29070*ipow<2>(y)*ipow<2>(x - 1) + 3060*y*ipow<3>(x - 1) + 85*ipow<4>(x - 1))*(252*ipow<5>(z) + ipow<4>(z)*(630*x + 630*y - 630) + 560*ipow<3>(z)*ipow<2>(x + y - 1) + 210*ipow<2>(z)*ipow<3>(x + y - 1) + 30*z*ipow<4>(x + y - 1) + ipow<5>(x + y - 1)) + (630*ipow<4>(z) + 560*ipow<3>(z)*(2*x + 2*y - 2) + 630*ipow<2>(z)*ipow<2>(x + y - 1) + 120*z*ipow<3>(x + y - 1) + 5*ipow<4>(x + y - 1))*(20349*ipow<5>(y) + ipow<4>(y)*(24225*x - 24225) + 9690*ipow<3>(y)*ipow<2>(x - 1) + 1530*ipow<2>(y)*ipow<3>(x - 1) + 85*y*ipow<4>(x - 1) + ipow<5>(x - 1)),
            (1260*ipow<4>(z) + 4*ipow<3>(z)*(630*x + 630*y - 630) + 1680*ipow<2>(z)*ipow<2>(x + y - 1) + 420*z*ipow<3>(x + y - 1) + 30*ipow<4>(x + y - 1))*(20349*ipow<5>(y) + ipow<4>(y)*(24225*x - 24225) + 9690*ipow<3>(y)*ipow<2>(x - 1) + 1530*ipow<2>(y)*ipow<3>(x - 1) + 85*y*ipow<4>(x - 1) + ipow<5>(x - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 271
template<>
struct DGBasis<271> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)) + (95*x*(7*x*(11*x - 8) + 12) + 19*x*(35*x*(11*x - 8) + 5*x*(154*x - 56) + 60) - 76)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)),
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)),
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(5544*ipow<5>(z) + 5*ipow<4>(z)*(2772*x + 2772*y - 2772) + 12600*ipow<3>(z)*ipow<2>(x + y - 1) + 5040*ipow<2>(z)*ipow<3>(x + y - 1) + 840*z*ipow<4>(x + y - 1) + 42*ipow<5>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 272
template<>
struct DGBasis<272> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (10*x*(7*x*(22*x - 9) + 6) - 1)*(x + 15*y - 1)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(x + 15*y - 1)*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)) + (10*x*(7*x*(22*x - 9) + 6) - 1)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)) + (x + 15*y - 1)*(70*x*(22*x - 9) + 10*x*(308*x - 63) + 60)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(x + 15*y - 1)*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)) + 15*(10*x*(7*x*(22*x - 9) + 6) - 1)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(x + 15*y - 1)*(5544*ipow<5>(z) + 5*ipow<4>(z)*(2772*x + 2772*y - 2772) + 12600*ipow<3>(z)*ipow<2>(x + y - 1) + 5040*ipow<2>(z)*ipow<3>(x + y - 1) + 840*z*ipow<4>(x + y - 1) + 42*ipow<5>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 273
template<>
struct DGBasis<273> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (21*x*(11*x - 2) + 1)*(136*ipow<2>(y) + y*(32*x - 32) + ipow<2>(x - 1))*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (462*x - 42)*(136*ipow<2>(y) + y*(32*x - 32) + ipow<2>(x - 1))*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)) + (21*x*(11*x - 2) + 1)*(2*x + 32*y - 2)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)) + (21*x*(11*x - 2) + 1)*(136*ipow<2>(y) + y*(32*x - 32) + ipow<2>(x - 1))*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)),
            (21*x*(11*x - 2) + 1)*(32*x + 272*y - 32)*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)) + (21*x*(11*x - 2) + 1)*(136*ipow<2>(y) + y*(32*x - 32) + ipow<2>(x - 1))*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)),
            (21*x*(11*x - 2) + 1)*(136*ipow<2>(y) + y*(32*x - 32) + ipow<2>(x - 1))*(5544*ipow<5>(z) + 5*ipow<4>(z)*(2772*x + 2772*y - 2772) + 12600*ipow<3>(z)*ipow<2>(x + y - 1) + 5040*ipow<2>(z)*ipow<3>(x + y - 1) + 840*z*ipow<4>(x + y - 1) + 42*ipow<5>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 274
template<>
struct DGBasis<274> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (22*x - 1)*(969*ipow<3>(y) + ipow<2>(y)*(459*x - 459) + 51*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (22*x - 1)*(459*ipow<2>(y) + 51*y*(2*x - 2) + 3*ipow<2>(x - 1))*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)) + (22*x - 1)*(969*ipow<3>(y) + ipow<2>(y)*(459*x - 459) + 51*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)) + 22*(969*ipow<3>(y) + ipow<2>(y)*(459*x - 459) + 51*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)),
            (22*x - 1)*(2907*ipow<2>(y) + 2*y*(459*x - 459) + 51*ipow<2>(x - 1))*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)) + (22*x - 1)*(969*ipow<3>(y) + ipow<2>(y)*(459*x - 459) + 51*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)),
            (22*x - 1)*(969*ipow<3>(y) + ipow<2>(y)*(459*x - 459) + 51*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(5544*ipow<5>(z) + 5*ipow<4>(z)*(2772*x + 2772*y - 2772) + 12600*ipow<3>(z)*ipow<2>(x + y - 1) + 5040*ipow<2>(z)*ipow<3>(x + y - 1) + 840*z*ipow<4>(x + y - 1) + 42*ipow<5>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 275
template<>
struct DGBasis<275> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (5985*ipow<4>(y) + ipow<3>(y)*(4560*x - 4560) + 1026*ipow<2>(y)*ipow<2>(x - 1) + 72*y*ipow<3>(x - 1) + ipow<4>(x - 1))*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1));
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (4560*ipow<3>(y) + 1026*ipow<2>(y)*(2*x - 2) + 216*y*ipow<2>(x - 1) + 4*ipow<3>(x - 1))*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)) + (5985*ipow<4>(y) + ipow<3>(y)*(4560*x - 4560) + 1026*ipow<2>(y)*ipow<2>(x - 1) + 72*y*ipow<3>(x - 1) + ipow<4>(x - 1))*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)),
            (23940*ipow<3>(y) + 3*ipow<2>(y)*(4560*x - 4560) + 2052*y*ipow<2>(x - 1) + 72*ipow<3>(x - 1))*(924*ipow<6>(z) + ipow<5>(z)*(2772*x + 2772*y - 2772) + 3150*ipow<4>(z)*ipow<2>(x + y - 1) + 1680*ipow<3>(z)*ipow<3>(x + y - 1) + 420*ipow<2>(z)*ipow<4>(x + y - 1) + 42*z*ipow<5>(x + y - 1) + ipow<6>(x + y - 1)) + (5985*ipow<4>(y) + ipow<3>(y)*(4560*x - 4560) + 1026*ipow<2>(y)*ipow<2>(x - 1) + 72*y*ipow<3>(x - 1) + ipow<4>(x - 1))*(2772*ipow<5>(z) + 3150*ipow<4>(z)*(2*x + 2*y - 2) + 5040*ipow<3>(z)*ipow<2>(x + y - 1) + 1680*ipow<2>(z)*ipow<3>(x + y - 1) + 210*z*ipow<4>(x + y - 1) + 6*ipow<5>(x + y - 1)),
            (5985*ipow<4>(y) + ipow<3>(y)*(4560*x - 4560) + 1026*ipow<2>(y)*ipow<2>(x - 1) + 72*y*ipow<3>(x - 1) + ipow<4>(x - 1))*(5544*ipow<5>(z) + 5*ipow<4>(z)*(2772*x + 2772*y - 2772) + 12600*ipow<3>(z)*ipow<2>(x + y - 1) + 5040*ipow<2>(z)*ipow<3>(x + y - 1) + 840*z*ipow<4>(x + y - 1) + 42*ipow<5>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 276
template<>
struct DGBasis<276> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (10*x*(7*x*(22*x - 9) + 6) - 1)*(ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(7*ipow<6>(x) + 42*ipow<5>(x)*y - 42*ipow<5>(x) + 105*ipow<4>(x)*ipow<2>(y) - 210*ipow<4>(x)*y + 105*ipow<4>(x) + 140*ipow<3>(x)*ipow<3>(y) - 420*ipow<3>(x)*ipow<2>(y) + 420*ipow<3>(x)*y - 140*ipow<3>(x) + 105*ipow<2>(x)*ipow<4>(y) - 420*ipow<2>(x)*ipow<3>(y) + 630*ipow<2>(x)*ipow<2>(y) - 420*ipow<2>(x)*y + 105*ipow<2>(x) + 42*x*ipow<5>(y) - 210*x*ipow<4>(y) + 420*x*ipow<3>(y) - 420*x*ipow<2>(y) + 210*x*y - 42*x + 7*ipow<6>(y) - 42*ipow<5>(y) + 105*ipow<4>(y) - 140*ipow<3>(y) + 105*ipow<2>(y) - 42*y + 12012*ipow<6>(z) + 16632*ipow<5>(z)*(2*x + 2*y - 2) + 34650*ipow<4>(z)*ipow<2>(x + y - 1) + 16800*ipow<3>(z)*ipow<3>(x + y - 1) + 3780*ipow<2>(z)*ipow<4>(x + y - 1) + 336*z*ipow<5>(x + y - 1) + 7) + (70*x*(22*x - 9) + 10*x*(308*x - 63) + 60)*(ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) - 1),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(7*ipow<6>(x) + 42*ipow<5>(x)*y - 42*ipow<5>(x) + 105*ipow<4>(x)*ipow<2>(y) - 210*ipow<4>(x)*y + 105*ipow<4>(x) + 140*ipow<3>(x)*ipow<3>(y) - 420*ipow<3>(x)*ipow<2>(y) + 420*ipow<3>(x)*y - 140*ipow<3>(x) + 105*ipow<2>(x)*ipow<4>(y) - 420*ipow<2>(x)*ipow<3>(y) + 630*ipow<2>(x)*ipow<2>(y) - 420*ipow<2>(x)*y + 105*ipow<2>(x) + 42*x*ipow<5>(y) - 210*x*ipow<4>(y) + 420*x*ipow<3>(y) - 420*x*ipow<2>(y) + 210*x*y - 42*x + 7*ipow<6>(y) - 42*ipow<5>(y) + 105*ipow<4>(y) - 140*ipow<3>(y) + 105*ipow<2>(y) - 42*y + 12012*ipow<6>(z) + 16632*ipow<5>(z)*(2*x + 2*y - 2) + 34650*ipow<4>(z)*ipow<2>(x + y - 1) + 16800*ipow<3>(z)*ipow<3>(x + y - 1) + 3780*ipow<2>(z)*ipow<4>(x + y - 1) + 336*z*ipow<5>(x + y - 1) + 7),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(24024*ipow<6>(z) + 6*ipow<5>(z)*(12012*x + 12012*y - 12012) + 83160*ipow<4>(z)*ipow<2>(x + y - 1) + 46200*ipow<3>(z)*ipow<3>(x + y - 1) + 12600*ipow<2>(z)*ipow<4>(x + y - 1) + 1512*z*ipow<5>(x + y - 1) + 56*ipow<6>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 277
template<>
struct DGBasis<277> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (21*x*(11*x - 2) + 1)*(x + 17*y - 1)*(ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (462*x - 42)*(x + 17*y - 1)*(ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) - 1) + (21*x*(11*x - 2) + 1)*(x + 17*y - 1)*(7*ipow<6>(x) + 42*ipow<5>(x)*y - 42*ipow<5>(x) + 105*ipow<4>(x)*ipow<2>(y) - 210*ipow<4>(x)*y + 105*ipow<4>(x) + 140*ipow<3>(x)*ipow<3>(y) - 420*ipow<3>(x)*ipow<2>(y) + 420*ipow<3>(x)*y - 140*ipow<3>(x) + 105*ipow<2>(x)*ipow<4>(y) - 420*ipow<2>(x)*ipow<3>(y) + 630*ipow<2>(x)*ipow<2>(y) - 420*ipow<2>(x)*y + 105*ipow<2>(x) + 42*x*ipow<5>(y) - 210*x*ipow<4>(y) + 420*x*ipow<3>(y) - 420*x*ipow<2>(y) + 210*x*y - 42*x + 7*ipow<6>(y) - 42*ipow<5>(y) + 105*ipow<4>(y) - 140*ipow<3>(y) + 105*ipow<2>(y) - 42*y + 12012*ipow<6>(z) + 16632*ipow<5>(z)*(2*x + 2*y - 2) + 34650*ipow<4>(z)*ipow<2>(x + y - 1) + 16800*ipow<3>(z)*ipow<3>(x + y - 1) + 3780*ipow<2>(z)*ipow<4>(x + y - 1) + 336*z*ipow<5>(x + y - 1) + 7) + (21*x*(11*x - 2) + 1)*(ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) - 1),
            (21*x*(11*x - 2) + 1)*(x + 17*y - 1)*(7*ipow<6>(x) + 42*ipow<5>(x)*y - 42*ipow<5>(x) + 105*ipow<4>(x)*ipow<2>(y) - 210*ipow<4>(x)*y + 105*ipow<4>(x) + 140*ipow<3>(x)*ipow<3>(y) - 420*ipow<3>(x)*ipow<2>(y) + 420*ipow<3>(x)*y - 140*ipow<3>(x) + 105*ipow<2>(x)*ipow<4>(y) - 420*ipow<2>(x)*ipow<3>(y) + 630*ipow<2>(x)*ipow<2>(y) - 420*ipow<2>(x)*y + 105*ipow<2>(x) + 42*x*ipow<5>(y) - 210*x*ipow<4>(y) + 420*x*ipow<3>(y) - 420*x*ipow<2>(y) + 210*x*y - 42*x + 7*ipow<6>(y) - 42*ipow<5>(y) + 105*ipow<4>(y) - 140*ipow<3>(y) + 105*ipow<2>(y) - 42*y + 12012*ipow<6>(z) + 16632*ipow<5>(z)*(2*x + 2*y - 2) + 34650*ipow<4>(z)*ipow<2>(x + y - 1) + 16800*ipow<3>(z)*ipow<3>(x + y - 1) + 3780*ipow<2>(z)*ipow<4>(x + y - 1) + 336*z*ipow<5>(x + y - 1) + 7) + 17*(21*x*(11*x - 2) + 1)*(ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) - 1),
            (21*x*(11*x - 2) + 1)*(x + 17*y - 1)*(24024*ipow<6>(z) + 6*ipow<5>(z)*(12012*x + 12012*y - 12012) + 83160*ipow<4>(z)*ipow<2>(x + y - 1) + 46200*ipow<3>(z)*ipow<3>(x + y - 1) + 12600*ipow<2>(z)*ipow<4>(x + y - 1) + 1512*z*ipow<5>(x + y - 1) + 56*ipow<6>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 278
template<>
struct DGBasis<278> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (22*x - 1)*(171*ipow<2>(y) + y*(36*x - 36) + ipow<2>(x - 1))*(ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (22*x - 1)*(2*x + 36*y - 2)*(ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) - 1) + (22*x - 1)*(171*ipow<2>(y) + y*(36*x - 36) + ipow<2>(x - 1))*(7*ipow<6>(x) + 42*ipow<5>(x)*y - 42*ipow<5>(x) + 105*ipow<4>(x)*ipow<2>(y) - 210*ipow<4>(x)*y + 105*ipow<4>(x) + 140*ipow<3>(x)*ipow<3>(y) - 420*ipow<3>(x)*ipow<2>(y) + 420*ipow<3>(x)*y - 140*ipow<3>(x) + 105*ipow<2>(x)*ipow<4>(y) - 420*ipow<2>(x)*ipow<3>(y) + 630*ipow<2>(x)*ipow<2>(y) - 420*ipow<2>(x)*y + 105*ipow<2>(x) + 42*x*ipow<5>(y) - 210*x*ipow<4>(y) + 420*x*ipow<3>(y) - 420*x*ipow<2>(y) + 210*x*y - 42*x + 7*ipow<6>(y) - 42*ipow<5>(y) + 105*ipow<4>(y) - 140*ipow<3>(y) + 105*ipow<2>(y) - 42*y + 12012*ipow<6>(z) + 16632*ipow<5>(z)*(2*x + 2*y - 2) + 34650*ipow<4>(z)*ipow<2>(x + y - 1) + 16800*ipow<3>(z)*ipow<3>(x + y - 1) + 3780*ipow<2>(z)*ipow<4>(x + y - 1) + 336*z*ipow<5>(x + y - 1) + 7) + 22*(171*ipow<2>(y) + y*(36*x - 36) + ipow<2>(x - 1))*(ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) - 1),
            (22*x - 1)*(36*x + 342*y - 36)*(ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) - 1) + (22*x - 1)*(171*ipow<2>(y) + y*(36*x - 36) + ipow<2>(x - 1))*(7*ipow<6>(x) + 42*ipow<5>(x)*y - 42*ipow<5>(x) + 105*ipow<4>(x)*ipow<2>(y) - 210*ipow<4>(x)*y + 105*ipow<4>(x) + 140*ipow<3>(x)*ipow<3>(y) - 420*ipow<3>(x)*ipow<2>(y) + 420*ipow<3>(x)*y - 140*ipow<3>(x) + 105*ipow<2>(x)*ipow<4>(y) - 420*ipow<2>(x)*ipow<3>(y) + 630*ipow<2>(x)*ipow<2>(y) - 420*ipow<2>(x)*y + 105*ipow<2>(x) + 42*x*ipow<5>(y) - 210*x*ipow<4>(y) + 420*x*ipow<3>(y) - 420*x*ipow<2>(y) + 210*x*y - 42*x + 7*ipow<6>(y) - 42*ipow<5>(y) + 105*ipow<4>(y) - 140*ipow<3>(y) + 105*ipow<2>(y) - 42*y + 12012*ipow<6>(z) + 16632*ipow<5>(z)*(2*x + 2*y - 2) + 34650*ipow<4>(z)*ipow<2>(x + y - 1) + 16800*ipow<3>(z)*ipow<3>(x + y - 1) + 3780*ipow<2>(z)*ipow<4>(x + y - 1) + 336*z*ipow<5>(x + y - 1) + 7),
            (22*x - 1)*(171*ipow<2>(y) + y*(36*x - 36) + ipow<2>(x - 1))*(24024*ipow<6>(z) + 6*ipow<5>(z)*(12012*x + 12012*y - 12012) + 83160*ipow<4>(z)*ipow<2>(x + y - 1) + 46200*ipow<3>(z)*ipow<3>(x + y - 1) + 12600*ipow<2>(z)*ipow<4>(x + y - 1) + 1512*z*ipow<5>(x + y - 1) + 56*ipow<6>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 279
template<>
struct DGBasis<279> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (1330*ipow<3>(y) + ipow<2>(y)*(570*x - 570) + 57*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (570*ipow<2>(y) + 57*y*(2*x - 2) + 3*ipow<2>(x - 1))*(ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) - 1) + (1330*ipow<3>(y) + ipow<2>(y)*(570*x - 570) + 57*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(7*ipow<6>(x) + 42*ipow<5>(x)*y - 42*ipow<5>(x) + 105*ipow<4>(x)*ipow<2>(y) - 210*ipow<4>(x)*y + 105*ipow<4>(x) + 140*ipow<3>(x)*ipow<3>(y) - 420*ipow<3>(x)*ipow<2>(y) + 420*ipow<3>(x)*y - 140*ipow<3>(x) + 105*ipow<2>(x)*ipow<4>(y) - 420*ipow<2>(x)*ipow<3>(y) + 630*ipow<2>(x)*ipow<2>(y) - 420*ipow<2>(x)*y + 105*ipow<2>(x) + 42*x*ipow<5>(y) - 210*x*ipow<4>(y) + 420*x*ipow<3>(y) - 420*x*ipow<2>(y) + 210*x*y - 42*x + 7*ipow<6>(y) - 42*ipow<5>(y) + 105*ipow<4>(y) - 140*ipow<3>(y) + 105*ipow<2>(y) - 42*y + 12012*ipow<6>(z) + 16632*ipow<5>(z)*(2*x + 2*y - 2) + 34650*ipow<4>(z)*ipow<2>(x + y - 1) + 16800*ipow<3>(z)*ipow<3>(x + y - 1) + 3780*ipow<2>(z)*ipow<4>(x + y - 1) + 336*z*ipow<5>(x + y - 1) + 7),
            (3990*ipow<2>(y) + 2*y*(570*x - 570) + 57*ipow<2>(x - 1))*(ipow<7>(x) + 7*ipow<6>(x)*y - 7*ipow<6>(x) + 21*ipow<5>(x)*ipow<2>(y) - 42*ipow<5>(x)*y + 21*ipow<5>(x) + 35*ipow<4>(x)*ipow<3>(y) - 105*ipow<4>(x)*ipow<2>(y) + 105*ipow<4>(x)*y - 35*ipow<4>(x) + 35*ipow<3>(x)*ipow<4>(y) - 140*ipow<3>(x)*ipow<3>(y) + 210*ipow<3>(x)*ipow<2>(y) - 140*ipow<3>(x)*y + 35*ipow<3>(x) + 21*ipow<2>(x)*ipow<5>(y) - 105*ipow<2>(x)*ipow<4>(y) + 210*ipow<2>(x)*ipow<3>(y) - 210*ipow<2>(x)*ipow<2>(y) + 105*ipow<2>(x)*y - 21*ipow<2>(x) + 7*x*ipow<6>(y) - 42*x*ipow<5>(y) + 105*x*ipow<4>(y) - 140*x*ipow<3>(y) + 105*x*ipow<2>(y) - 42*x*y + 7*x + ipow<7>(y) - 7*ipow<6>(y) + 21*ipow<5>(y) - 35*ipow<4>(y) + 35*ipow<3>(y) - 21*ipow<2>(y) + 7*y + 3432*ipow<7>(z) + ipow<6>(z)*(12012*x + 12012*y - 12012) + 16632*ipow<5>(z)*ipow<2>(x + y - 1) + 11550*ipow<4>(z)*ipow<3>(x + y - 1) + 4200*ipow<3>(z)*ipow<4>(x + y - 1) + 756*ipow<2>(z)*ipow<5>(x + y - 1) + 56*z*ipow<6>(x + y - 1) - 1) + (1330*ipow<3>(y) + ipow<2>(y)*(570*x - 570) + 57*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(7*ipow<6>(x) + 42*ipow<5>(x)*y - 42*ipow<5>(x) + 105*ipow<4>(x)*ipow<2>(y) - 210*ipow<4>(x)*y + 105*ipow<4>(x) + 140*ipow<3>(x)*ipow<3>(y) - 420*ipow<3>(x)*ipow<2>(y) + 420*ipow<3>(x)*y - 140*ipow<3>(x) + 105*ipow<2>(x)*ipow<4>(y) - 420*ipow<2>(x)*ipow<3>(y) + 630*ipow<2>(x)*ipow<2>(y) - 420*ipow<2>(x)*y + 105*ipow<2>(x) + 42*x*ipow<5>(y) - 210*x*ipow<4>(y) + 420*x*ipow<3>(y) - 420*x*ipow<2>(y) + 210*x*y - 42*x + 7*ipow<6>(y) - 42*ipow<5>(y) + 105*ipow<4>(y) - 140*ipow<3>(y) + 105*ipow<2>(y) - 42*y + 12012*ipow<6>(z) + 16632*ipow<5>(z)*(2*x + 2*y - 2) + 34650*ipow<4>(z)*ipow<2>(x + y - 1) + 16800*ipow<3>(z)*ipow<3>(x + y - 1) + 3780*ipow<2>(z)*ipow<4>(x + y - 1) + 336*z*ipow<5>(x + y - 1) + 7),
            (1330*ipow<3>(y) + ipow<2>(y)*(570*x - 570) + 57*y*ipow<2>(x - 1) + ipow<3>(x - 1))*(24024*ipow<6>(z) + 6*ipow<5>(z)*(12012*x + 12012*y - 12012) + 83160*ipow<4>(z)*ipow<2>(x + y - 1) + 46200*ipow<3>(z)*ipow<3>(x + y - 1) + 12600*ipow<2>(z)*ipow<4>(x + y - 1) + 1512*z*ipow<5>(x + y - 1) + 56*ipow<6>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 280
template<>
struct DGBasis<280> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (21*x*(11*x - 2) + 1)*(ipow<8>(x) + 8*ipow<7>(x)*y - 8*ipow<7>(x) + 28*ipow<6>(x)*ipow<2>(y) - 56*ipow<6>(x)*y + 28*ipow<6>(x) + 56*ipow<5>(x)*ipow<3>(y) - 168*ipow<5>(x)*ipow<2>(y) + 168*ipow<5>(x)*y - 56*ipow<5>(x) + 70*ipow<4>(x)*ipow<4>(y) - 280*ipow<4>(x)*ipow<3>(y) + 420*ipow<4>(x)*ipow<2>(y) - 280*ipow<4>(x)*y + 70*ipow<4>(x) + 56*ipow<3>(x)*ipow<5>(y) - 280*ipow<3>(x)*ipow<4>(y) + 560*ipow<3>(x)*ipow<3>(y) - 560*ipow<3>(x)*ipow<2>(y) + 280*ipow<3>(x)*y - 56*ipow<3>(x) + 28*ipow<2>(x)*ipow<6>(y) - 168*ipow<2>(x)*ipow<5>(y) + 420*ipow<2>(x)*ipow<4>(y) - 560*ipow<2>(x)*ipow<3>(y) + 420*ipow<2>(x)*ipow<2>(y) - 168*ipow<2>(x)*y + 28*ipow<2>(x) + 8*x*ipow<7>(y) - 56*x*ipow<6>(y) + 168*x*ipow<5>(y) - 280*x*ipow<4>(y) + 280*x*ipow<3>(y) - 168*x*ipow<2>(y) + 56*x*y - 8*x + ipow<8>(y) - 8*ipow<7>(y) + 28*ipow<6>(y) - 56*ipow<5>(y) + 70*ipow<4>(y) - 56*ipow<3>(y) + 28*ipow<2>(y) - 8*y + 12870*ipow<8>(z) + ipow<7>(z)*(51480*x + 51480*y - 51480) + 84084*ipow<6>(z)*ipow<2>(x + y - 1) + 72072*ipow<5>(z)*ipow<3>(x + y - 1) + 34650*ipow<4>(z)*ipow<4>(x + y - 1) + 9240*ipow<3>(z)*ipow<5>(x + y - 1) + 1260*ipow<2>(z)*ipow<6>(x + y - 1) + 72*z*ipow<7>(x + y - 1) + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (462*x - 42)*(ipow<8>(x) + 8*ipow<7>(x)*y - 8*ipow<7>(x) + 28*ipow<6>(x)*ipow<2>(y) - 56*ipow<6>(x)*y + 28*ipow<6>(x) + 56*ipow<5>(x)*ipow<3>(y) - 168*ipow<5>(x)*ipow<2>(y) + 168*ipow<5>(x)*y - 56*ipow<5>(x) + 70*ipow<4>(x)*ipow<4>(y) - 280*ipow<4>(x)*ipow<3>(y) + 420*ipow<4>(x)*ipow<2>(y) - 280*ipow<4>(x)*y + 70*ipow<4>(x) + 56*ipow<3>(x)*ipow<5>(y) - 280*ipow<3>(x)*ipow<4>(y) + 560*ipow<3>(x)*ipow<3>(y) - 560*ipow<3>(x)*ipow<2>(y) + 280*ipow<3>(x)*y - 56*ipow<3>(x) + 28*ipow<2>(x)*ipow<6>(y) - 168*ipow<2>(x)*ipow<5>(y) + 420*ipow<2>(x)*ipow<4>(y) - 560*ipow<2>(x)*ipow<3>(y) + 420*ipow<2>(x)*ipow<2>(y) - 168*ipow<2>(x)*y + 28*ipow<2>(x) + 8*x*ipow<7>(y) - 56*x*ipow<6>(y) + 168*x*ipow<5>(y) - 280*x*ipow<4>(y) + 280*x*ipow<3>(y) - 168*x*ipow<2>(y) + 56*x*y - 8*x + ipow<8>(y) - 8*ipow<7>(y) + 28*ipow<6>(y) - 56*ipow<5>(y) + 70*ipow<4>(y) - 56*ipow<3>(y) + 28*ipow<2>(y) - 8*y + 12870*ipow<8>(z) + ipow<7>(z)*(51480*x + 51480*y - 51480) + 84084*ipow<6>(z)*ipow<2>(x + y - 1) + 72072*ipow<5>(z)*ipow<3>(x + y - 1) + 34650*ipow<4>(z)*ipow<4>(x + y - 1) + 9240*ipow<3>(z)*ipow<5>(x + y - 1) + 1260*ipow<2>(z)*ipow<6>(x + y - 1) + 72*z*ipow<7>(x + y - 1) + 1) + (21*x*(11*x - 2) + 1)*(8*ipow<7>(x) + 56*ipow<6>(x)*y - 56*ipow<6>(x) + 168*ipow<5>(x)*ipow<2>(y) - 336*ipow<5>(x)*y + 168*ipow<5>(x) + 280*ipow<4>(x)*ipow<3>(y) - 840*ipow<4>(x)*ipow<2>(y) + 840*ipow<4>(x)*y - 280*ipow<4>(x) + 280*ipow<3>(x)*ipow<4>(y) - 1120*ipow<3>(x)*ipow<3>(y) + 1680*ipow<3>(x)*ipow<2>(y) - 1120*ipow<3>(x)*y + 280*ipow<3>(x) + 168*ipow<2>(x)*ipow<5>(y) - 840*ipow<2>(x)*ipow<4>(y) + 1680*ipow<2>(x)*ipow<3>(y) - 1680*ipow<2>(x)*ipow<2>(y) + 840*ipow<2>(x)*y - 168*ipow<2>(x) + 56*x*ipow<6>(y) - 336*x*ipow<5>(y) + 840*x*ipow<4>(y) - 1120*x*ipow<3>(y) + 840*x*ipow<2>(y) - 336*x*y + 56*x + 8*ipow<7>(y) - 56*ipow<6>(y) + 168*ipow<5>(y) - 280*ipow<4>(y) + 280*ipow<3>(y) - 168*ipow<2>(y) + 56*y + 51480*ipow<7>(z) + 84084*ipow<6>(z)*(2*x + 2*y - 2) + 216216*ipow<5>(z)*ipow<2>(x + y - 1) + 138600*ipow<4>(z)*ipow<3>(x + y - 1) + 46200*ipow<3>(z)*ipow<4>(x + y - 1) + 7560*ipow<2>(z)*ipow<5>(x + y - 1) + 504*z*ipow<6>(x + y - 1) - 8),
            (21*x*(11*x - 2) + 1)*(8*ipow<7>(x) + 56*ipow<6>(x)*y - 56*ipow<6>(x) + 168*ipow<5>(x)*ipow<2>(y) - 336*ipow<5>(x)*y + 168*ipow<5>(x) + 280*ipow<4>(x)*ipow<3>(y) - 840*ipow<4>(x)*ipow<2>(y) + 840*ipow<4>(x)*y - 280*ipow<4>(x) + 280*ipow<3>(x)*ipow<4>(y) - 1120*ipow<3>(x)*ipow<3>(y) + 1680*ipow<3>(x)*ipow<2>(y) - 1120*ipow<3>(x)*y + 280*ipow<3>(x) + 168*ipow<2>(x)*ipow<5>(y) - 840*ipow<2>(x)*ipow<4>(y) + 1680*ipow<2>(x)*ipow<3>(y) - 1680*ipow<2>(x)*ipow<2>(y) + 840*ipow<2>(x)*y - 168*ipow<2>(x) + 56*x*ipow<6>(y) - 336*x*ipow<5>(y) + 840*x*ipow<4>(y) - 1120*x*ipow<3>(y) + 840*x*ipow<2>(y) - 336*x*y + 56*x + 8*ipow<7>(y) - 56*ipow<6>(y) + 168*ipow<5>(y) - 280*ipow<4>(y) + 280*ipow<3>(y) - 168*ipow<2>(y) + 56*y + 51480*ipow<7>(z) + 84084*ipow<6>(z)*(2*x + 2*y - 2) + 216216*ipow<5>(z)*ipow<2>(x + y - 1) + 138600*ipow<4>(z)*ipow<3>(x + y - 1) + 46200*ipow<3>(z)*ipow<4>(x + y - 1) + 7560*ipow<2>(z)*ipow<5>(x + y - 1) + 504*z*ipow<6>(x + y - 1) - 8),
            (21*x*(11*x - 2) + 1)*(102960*ipow<7>(z) + 7*ipow<6>(z)*(51480*x + 51480*y - 51480) + 504504*ipow<5>(z)*ipow<2>(x + y - 1) + 360360*ipow<4>(z)*ipow<3>(x + y - 1) + 138600*ipow<3>(z)*ipow<4>(x + y - 1) + 27720*ipow<2>(z)*ipow<5>(x + y - 1) + 2520*z*ipow<6>(x + y - 1) + 72*ipow<7>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 281
template<>
struct DGBasis<281> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (22*x - 1)*(x + 19*y - 1)*(ipow<8>(x) + 8*ipow<7>(x)*y - 8*ipow<7>(x) + 28*ipow<6>(x)*ipow<2>(y) - 56*ipow<6>(x)*y + 28*ipow<6>(x) + 56*ipow<5>(x)*ipow<3>(y) - 168*ipow<5>(x)*ipow<2>(y) + 168*ipow<5>(x)*y - 56*ipow<5>(x) + 70*ipow<4>(x)*ipow<4>(y) - 280*ipow<4>(x)*ipow<3>(y) + 420*ipow<4>(x)*ipow<2>(y) - 280*ipow<4>(x)*y + 70*ipow<4>(x) + 56*ipow<3>(x)*ipow<5>(y) - 280*ipow<3>(x)*ipow<4>(y) + 560*ipow<3>(x)*ipow<3>(y) - 560*ipow<3>(x)*ipow<2>(y) + 280*ipow<3>(x)*y - 56*ipow<3>(x) + 28*ipow<2>(x)*ipow<6>(y) - 168*ipow<2>(x)*ipow<5>(y) + 420*ipow<2>(x)*ipow<4>(y) - 560*ipow<2>(x)*ipow<3>(y) + 420*ipow<2>(x)*ipow<2>(y) - 168*ipow<2>(x)*y + 28*ipow<2>(x) + 8*x*ipow<7>(y) - 56*x*ipow<6>(y) + 168*x*ipow<5>(y) - 280*x*ipow<4>(y) + 280*x*ipow<3>(y) - 168*x*ipow<2>(y) + 56*x*y - 8*x + ipow<8>(y) - 8*ipow<7>(y) + 28*ipow<6>(y) - 56*ipow<5>(y) + 70*ipow<4>(y) - 56*ipow<3>(y) + 28*ipow<2>(y) - 8*y + 12870*ipow<8>(z) + ipow<7>(z)*(51480*x + 51480*y - 51480) + 84084*ipow<6>(z)*ipow<2>(x + y - 1) + 72072*ipow<5>(z)*ipow<3>(x + y - 1) + 34650*ipow<4>(z)*ipow<4>(x + y - 1) + 9240*ipow<3>(z)*ipow<5>(x + y - 1) + 1260*ipow<2>(z)*ipow<6>(x + y - 1) + 72*z*ipow<7>(x + y - 1) + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (22*x - 1)*(x + 19*y - 1)*(8*ipow<7>(x) + 56*ipow<6>(x)*y - 56*ipow<6>(x) + 168*ipow<5>(x)*ipow<2>(y) - 336*ipow<5>(x)*y + 168*ipow<5>(x) + 280*ipow<4>(x)*ipow<3>(y) - 840*ipow<4>(x)*ipow<2>(y) + 840*ipow<4>(x)*y - 280*ipow<4>(x) + 280*ipow<3>(x)*ipow<4>(y) - 1120*ipow<3>(x)*ipow<3>(y) + 1680*ipow<3>(x)*ipow<2>(y) - 1120*ipow<3>(x)*y + 280*ipow<3>(x) + 168*ipow<2>(x)*ipow<5>(y) - 840*ipow<2>(x)*ipow<4>(y) + 1680*ipow<2>(x)*ipow<3>(y) - 1680*ipow<2>(x)*ipow<2>(y) + 840*ipow<2>(x)*y - 168*ipow<2>(x) + 56*x*ipow<6>(y) - 336*x*ipow<5>(y) + 840*x*ipow<4>(y) - 1120*x*ipow<3>(y) + 840*x*ipow<2>(y) - 336*x*y + 56*x + 8*ipow<7>(y) - 56*ipow<6>(y) + 168*ipow<5>(y) - 280*ipow<4>(y) + 280*ipow<3>(y) - 168*ipow<2>(y) + 56*y + 51480*ipow<7>(z) + 84084*ipow<6>(z)*(2*x + 2*y - 2) + 216216*ipow<5>(z)*ipow<2>(x + y - 1) + 138600*ipow<4>(z)*ipow<3>(x + y - 1) + 46200*ipow<3>(z)*ipow<4>(x + y - 1) + 7560*ipow<2>(z)*ipow<5>(x + y - 1) + 504*z*ipow<6>(x + y - 1) - 8) + (22*x - 1)*(ipow<8>(x) + 8*ipow<7>(x)*y - 8*ipow<7>(x) + 28*ipow<6>(x)*ipow<2>(y) - 56*ipow<6>(x)*y + 28*ipow<6>(x) + 56*ipow<5>(x)*ipow<3>(y) - 168*ipow<5>(x)*ipow<2>(y) + 168*ipow<5>(x)*y - 56*ipow<5>(x) + 70*ipow<4>(x)*ipow<4>(y) - 280*ipow<4>(x)*ipow<3>(y) + 420*ipow<4>(x)*ipow<2>(y) - 280*ipow<4>(x)*y + 70*ipow<4>(x) + 56*ipow<3>(x)*ipow<5>(y) - 280*ipow<3>(x)*ipow<4>(y) + 560*ipow<3>(x)*ipow<3>(y) - 560*ipow<3>(x)*ipow<2>(y) + 280*ipow<3>(x)*y - 56*ipow<3>(x) + 28*ipow<2>(x)*ipow<6>(y) - 168*ipow<2>(x)*ipow<5>(y) + 420*ipow<2>(x)*ipow<4>(y) - 560*ipow<2>(x)*ipow<3>(y) + 420*ipow<2>(x)*ipow<2>(y) - 168*ipow<2>(x)*y + 28*ipow<2>(x) + 8*x*ipow<7>(y) - 56*x*ipow<6>(y) + 168*x*ipow<5>(y) - 280*x*ipow<4>(y) + 280*x*ipow<3>(y) - 168*x*ipow<2>(y) + 56*x*y - 8*x + ipow<8>(y) - 8*ipow<7>(y) + 28*ipow<6>(y) - 56*ipow<5>(y) + 70*ipow<4>(y) - 56*ipow<3>(y) + 28*ipow<2>(y) - 8*y + 12870*ipow<8>(z) + ipow<7>(z)*(51480*x + 51480*y - 51480) + 84084*ipow<6>(z)*ipow<2>(x + y - 1) + 72072*ipow<5>(z)*ipow<3>(x + y - 1) + 34650*ipow<4>(z)*ipow<4>(x + y - 1) + 9240*ipow<3>(z)*ipow<5>(x + y - 1) + 1260*ipow<2>(z)*ipow<6>(x + y - 1) + 72*z*ipow<7>(x + y - 1) + 1) + 22*(x + 19*y - 1)*(ipow<8>(x) + 8*ipow<7>(x)*y - 8*ipow<7>(x) + 28*ipow<6>(x)*ipow<2>(y) - 56*ipow<6>(x)*y + 28*ipow<6>(x) + 56*ipow<5>(x)*ipow<3>(y) - 168*ipow<5>(x)*ipow<2>(y) + 168*ipow<5>(x)*y - 56*ipow<5>(x) + 70*ipow<4>(x)*ipow<4>(y) - 280*ipow<4>(x)*ipow<3>(y) + 420*ipow<4>(x)*ipow<2>(y) - 280*ipow<4>(x)*y + 70*ipow<4>(x) + 56*ipow<3>(x)*ipow<5>(y) - 280*ipow<3>(x)*ipow<4>(y) + 560*ipow<3>(x)*ipow<3>(y) - 560*ipow<3>(x)*ipow<2>(y) + 280*ipow<3>(x)*y - 56*ipow<3>(x) + 28*ipow<2>(x)*ipow<6>(y) - 168*ipow<2>(x)*ipow<5>(y) + 420*ipow<2>(x)*ipow<4>(y) - 560*ipow<2>(x)*ipow<3>(y) + 420*ipow<2>(x)*ipow<2>(y) - 168*ipow<2>(x)*y + 28*ipow<2>(x) + 8*x*ipow<7>(y) - 56*x*ipow<6>(y) + 168*x*ipow<5>(y) - 280*x*ipow<4>(y) + 280*x*ipow<3>(y) - 168*x*ipow<2>(y) + 56*x*y - 8*x + ipow<8>(y) - 8*ipow<7>(y) + 28*ipow<6>(y) - 56*ipow<5>(y) + 70*ipow<4>(y) - 56*ipow<3>(y) + 28*ipow<2>(y) - 8*y + 12870*ipow<8>(z) + ipow<7>(z)*(51480*x + 51480*y - 51480) + 84084*ipow<6>(z)*ipow<2>(x + y - 1) + 72072*ipow<5>(z)*ipow<3>(x + y - 1) + 34650*ipow<4>(z)*ipow<4>(x + y - 1) + 9240*ipow<3>(z)*ipow<5>(x + y - 1) + 1260*ipow<2>(z)*ipow<6>(x + y - 1) + 72*z*ipow<7>(x + y - 1) + 1),
            (22*x - 1)*(x + 19*y - 1)*(8*ipow<7>(x) + 56*ipow<6>(x)*y - 56*ipow<6>(x) + 168*ipow<5>(x)*ipow<2>(y) - 336*ipow<5>(x)*y + 168*ipow<5>(x) + 280*ipow<4>(x)*ipow<3>(y) - 840*ipow<4>(x)*ipow<2>(y) + 840*ipow<4>(x)*y - 280*ipow<4>(x) + 280*ipow<3>(x)*ipow<4>(y) - 1120*ipow<3>(x)*ipow<3>(y) + 1680*ipow<3>(x)*ipow<2>(y) - 1120*ipow<3>(x)*y + 280*ipow<3>(x) + 168*ipow<2>(x)*ipow<5>(y) - 840*ipow<2>(x)*ipow<4>(y) + 1680*ipow<2>(x)*ipow<3>(y) - 1680*ipow<2>(x)*ipow<2>(y) + 840*ipow<2>(x)*y - 168*ipow<2>(x) + 56*x*ipow<6>(y) - 336*x*ipow<5>(y) + 840*x*ipow<4>(y) - 1120*x*ipow<3>(y) + 840*x*ipow<2>(y) - 336*x*y + 56*x + 8*ipow<7>(y) - 56*ipow<6>(y) + 168*ipow<5>(y) - 280*ipow<4>(y) + 280*ipow<3>(y) - 168*ipow<2>(y) + 56*y + 51480*ipow<7>(z) + 84084*ipow<6>(z)*(2*x + 2*y - 2) + 216216*ipow<5>(z)*ipow<2>(x + y - 1) + 138600*ipow<4>(z)*ipow<3>(x + y - 1) + 46200*ipow<3>(z)*ipow<4>(x + y - 1) + 7560*ipow<2>(z)*ipow<5>(x + y - 1) + 504*z*ipow<6>(x + y - 1) - 8) + 19*(22*x - 1)*(ipow<8>(x) + 8*ipow<7>(x)*y - 8*ipow<7>(x) + 28*ipow<6>(x)*ipow<2>(y) - 56*ipow<6>(x)*y + 28*ipow<6>(x) + 56*ipow<5>(x)*ipow<3>(y) - 168*ipow<5>(x)*ipow<2>(y) + 168*ipow<5>(x)*y - 56*ipow<5>(x) + 70*ipow<4>(x)*ipow<4>(y) - 280*ipow<4>(x)*ipow<3>(y) + 420*ipow<4>(x)*ipow<2>(y) - 280*ipow<4>(x)*y + 70*ipow<4>(x) + 56*ipow<3>(x)*ipow<5>(y) - 280*ipow<3>(x)*ipow<4>(y) + 560*ipow<3>(x)*ipow<3>(y) - 560*ipow<3>(x)*ipow<2>(y) + 280*ipow<3>(x)*y - 56*ipow<3>(x) + 28*ipow<2>(x)*ipow<6>(y) - 168*ipow<2>(x)*ipow<5>(y) + 420*ipow<2>(x)*ipow<4>(y) - 560*ipow<2>(x)*ipow<3>(y) + 420*ipow<2>(x)*ipow<2>(y) - 168*ipow<2>(x)*y + 28*ipow<2>(x) + 8*x*ipow<7>(y) - 56*x*ipow<6>(y) + 168*x*ipow<5>(y) - 280*x*ipow<4>(y) + 280*x*ipow<3>(y) - 168*x*ipow<2>(y) + 56*x*y - 8*x + ipow<8>(y) - 8*ipow<7>(y) + 28*ipow<6>(y) - 56*ipow<5>(y) + 70*ipow<4>(y) - 56*ipow<3>(y) + 28*ipow<2>(y) - 8*y + 12870*ipow<8>(z) + ipow<7>(z)*(51480*x + 51480*y - 51480) + 84084*ipow<6>(z)*ipow<2>(x + y - 1) + 72072*ipow<5>(z)*ipow<3>(x + y - 1) + 34650*ipow<4>(z)*ipow<4>(x + y - 1) + 9240*ipow<3>(z)*ipow<5>(x + y - 1) + 1260*ipow<2>(z)*ipow<6>(x + y - 1) + 72*z*ipow<7>(x + y - 1) + 1),
            (22*x - 1)*(x + 19*y - 1)*(102960*ipow<7>(z) + 7*ipow<6>(z)*(51480*x + 51480*y - 51480) + 504504*ipow<5>(z)*ipow<2>(x + y - 1) + 360360*ipow<4>(z)*ipow<3>(x + y - 1) + 138600*ipow<3>(z)*ipow<4>(x + y - 1) + 27720*ipow<2>(z)*ipow<5>(x + y - 1) + 2520*z*ipow<6>(x + y - 1) + 72*ipow<7>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 282
template<>
struct DGBasis<282> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (210*ipow<2>(y) + y*(40*x - 40) + ipow<2>(x - 1))*(ipow<8>(x) + 8*ipow<7>(x)*y - 8*ipow<7>(x) + 28*ipow<6>(x)*ipow<2>(y) - 56*ipow<6>(x)*y + 28*ipow<6>(x) + 56*ipow<5>(x)*ipow<3>(y) - 168*ipow<5>(x)*ipow<2>(y) + 168*ipow<5>(x)*y - 56*ipow<5>(x) + 70*ipow<4>(x)*ipow<4>(y) - 280*ipow<4>(x)*ipow<3>(y) + 420*ipow<4>(x)*ipow<2>(y) - 280*ipow<4>(x)*y + 70*ipow<4>(x) + 56*ipow<3>(x)*ipow<5>(y) - 280*ipow<3>(x)*ipow<4>(y) + 560*ipow<3>(x)*ipow<3>(y) - 560*ipow<3>(x)*ipow<2>(y) + 280*ipow<3>(x)*y - 56*ipow<3>(x) + 28*ipow<2>(x)*ipow<6>(y) - 168*ipow<2>(x)*ipow<5>(y) + 420*ipow<2>(x)*ipow<4>(y) - 560*ipow<2>(x)*ipow<3>(y) + 420*ipow<2>(x)*ipow<2>(y) - 168*ipow<2>(x)*y + 28*ipow<2>(x) + 8*x*ipow<7>(y) - 56*x*ipow<6>(y) + 168*x*ipow<5>(y) - 280*x*ipow<4>(y) + 280*x*ipow<3>(y) - 168*x*ipow<2>(y) + 56*x*y - 8*x + ipow<8>(y) - 8*ipow<7>(y) + 28*ipow<6>(y) - 56*ipow<5>(y) + 70*ipow<4>(y) - 56*ipow<3>(y) + 28*ipow<2>(y) - 8*y + 12870*ipow<8>(z) + ipow<7>(z)*(51480*x + 51480*y - 51480) + 84084*ipow<6>(z)*ipow<2>(x + y - 1) + 72072*ipow<5>(z)*ipow<3>(x + y - 1) + 34650*ipow<4>(z)*ipow<4>(x + y - 1) + 9240*ipow<3>(z)*ipow<5>(x + y - 1) + 1260*ipow<2>(z)*ipow<6>(x + y - 1) + 72*z*ipow<7>(x + y - 1) + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x + 40*y - 2)*(ipow<8>(x) + 8*ipow<7>(x)*y - 8*ipow<7>(x) + 28*ipow<6>(x)*ipow<2>(y) - 56*ipow<6>(x)*y + 28*ipow<6>(x) + 56*ipow<5>(x)*ipow<3>(y) - 168*ipow<5>(x)*ipow<2>(y) + 168*ipow<5>(x)*y - 56*ipow<5>(x) + 70*ipow<4>(x)*ipow<4>(y) - 280*ipow<4>(x)*ipow<3>(y) + 420*ipow<4>(x)*ipow<2>(y) - 280*ipow<4>(x)*y + 70*ipow<4>(x) + 56*ipow<3>(x)*ipow<5>(y) - 280*ipow<3>(x)*ipow<4>(y) + 560*ipow<3>(x)*ipow<3>(y) - 560*ipow<3>(x)*ipow<2>(y) + 280*ipow<3>(x)*y - 56*ipow<3>(x) + 28*ipow<2>(x)*ipow<6>(y) - 168*ipow<2>(x)*ipow<5>(y) + 420*ipow<2>(x)*ipow<4>(y) - 560*ipow<2>(x)*ipow<3>(y) + 420*ipow<2>(x)*ipow<2>(y) - 168*ipow<2>(x)*y + 28*ipow<2>(x) + 8*x*ipow<7>(y) - 56*x*ipow<6>(y) + 168*x*ipow<5>(y) - 280*x*ipow<4>(y) + 280*x*ipow<3>(y) - 168*x*ipow<2>(y) + 56*x*y - 8*x + ipow<8>(y) - 8*ipow<7>(y) + 28*ipow<6>(y) - 56*ipow<5>(y) + 70*ipow<4>(y) - 56*ipow<3>(y) + 28*ipow<2>(y) - 8*y + 12870*ipow<8>(z) + ipow<7>(z)*(51480*x + 51480*y - 51480) + 84084*ipow<6>(z)*ipow<2>(x + y - 1) + 72072*ipow<5>(z)*ipow<3>(x + y - 1) + 34650*ipow<4>(z)*ipow<4>(x + y - 1) + 9240*ipow<3>(z)*ipow<5>(x + y - 1) + 1260*ipow<2>(z)*ipow<6>(x + y - 1) + 72*z*ipow<7>(x + y - 1) + 1) + (210*ipow<2>(y) + y*(40*x - 40) + ipow<2>(x - 1))*(8*ipow<7>(x) + 56*ipow<6>(x)*y - 56*ipow<6>(x) + 168*ipow<5>(x)*ipow<2>(y) - 336*ipow<5>(x)*y + 168*ipow<5>(x) + 280*ipow<4>(x)*ipow<3>(y) - 840*ipow<4>(x)*ipow<2>(y) + 840*ipow<4>(x)*y - 280*ipow<4>(x) + 280*ipow<3>(x)*ipow<4>(y) - 1120*ipow<3>(x)*ipow<3>(y) + 1680*ipow<3>(x)*ipow<2>(y) - 1120*ipow<3>(x)*y + 280*ipow<3>(x) + 168*ipow<2>(x)*ipow<5>(y) - 840*ipow<2>(x)*ipow<4>(y) + 1680*ipow<2>(x)*ipow<3>(y) - 1680*ipow<2>(x)*ipow<2>(y) + 840*ipow<2>(x)*y - 168*ipow<2>(x) + 56*x*ipow<6>(y) - 336*x*ipow<5>(y) + 840*x*ipow<4>(y) - 1120*x*ipow<3>(y) + 840*x*ipow<2>(y) - 336*x*y + 56*x + 8*ipow<7>(y) - 56*ipow<6>(y) + 168*ipow<5>(y) - 280*ipow<4>(y) + 280*ipow<3>(y) - 168*ipow<2>(y) + 56*y + 51480*ipow<7>(z) + 84084*ipow<6>(z)*(2*x + 2*y - 2) + 216216*ipow<5>(z)*ipow<2>(x + y - 1) + 138600*ipow<4>(z)*ipow<3>(x + y - 1) + 46200*ipow<3>(z)*ipow<4>(x + y - 1) + 7560*ipow<2>(z)*ipow<5>(x + y - 1) + 504*z*ipow<6>(x + y - 1) - 8),
            (40*x + 420*y - 40)*(ipow<8>(x) + 8*ipow<7>(x)*y - 8*ipow<7>(x) + 28*ipow<6>(x)*ipow<2>(y) - 56*ipow<6>(x)*y + 28*ipow<6>(x) + 56*ipow<5>(x)*ipow<3>(y) - 168*ipow<5>(x)*ipow<2>(y) + 168*ipow<5>(x)*y - 56*ipow<5>(x) + 70*ipow<4>(x)*ipow<4>(y) - 280*ipow<4>(x)*ipow<3>(y) + 420*ipow<4>(x)*ipow<2>(y) - 280*ipow<4>(x)*y + 70*ipow<4>(x) + 56*ipow<3>(x)*ipow<5>(y) - 280*ipow<3>(x)*ipow<4>(y) + 560*ipow<3>(x)*ipow<3>(y) - 560*ipow<3>(x)*ipow<2>(y) + 280*ipow<3>(x)*y - 56*ipow<3>(x) + 28*ipow<2>(x)*ipow<6>(y) - 168*ipow<2>(x)*ipow<5>(y) + 420*ipow<2>(x)*ipow<4>(y) - 560*ipow<2>(x)*ipow<3>(y) + 420*ipow<2>(x)*ipow<2>(y) - 168*ipow<2>(x)*y + 28*ipow<2>(x) + 8*x*ipow<7>(y) - 56*x*ipow<6>(y) + 168*x*ipow<5>(y) - 280*x*ipow<4>(y) + 280*x*ipow<3>(y) - 168*x*ipow<2>(y) + 56*x*y - 8*x + ipow<8>(y) - 8*ipow<7>(y) + 28*ipow<6>(y) - 56*ipow<5>(y) + 70*ipow<4>(y) - 56*ipow<3>(y) + 28*ipow<2>(y) - 8*y + 12870*ipow<8>(z) + ipow<7>(z)*(51480*x + 51480*y - 51480) + 84084*ipow<6>(z)*ipow<2>(x + y - 1) + 72072*ipow<5>(z)*ipow<3>(x + y - 1) + 34650*ipow<4>(z)*ipow<4>(x + y - 1) + 9240*ipow<3>(z)*ipow<5>(x + y - 1) + 1260*ipow<2>(z)*ipow<6>(x + y - 1) + 72*z*ipow<7>(x + y - 1) + 1) + (210*ipow<2>(y) + y*(40*x - 40) + ipow<2>(x - 1))*(8*ipow<7>(x) + 56*ipow<6>(x)*y - 56*ipow<6>(x) + 168*ipow<5>(x)*ipow<2>(y) - 336*ipow<5>(x)*y + 168*ipow<5>(x) + 280*ipow<4>(x)*ipow<3>(y) - 840*ipow<4>(x)*ipow<2>(y) + 840*ipow<4>(x)*y - 280*ipow<4>(x) + 280*ipow<3>(x)*ipow<4>(y) - 1120*ipow<3>(x)*ipow<3>(y) + 1680*ipow<3>(x)*ipow<2>(y) - 1120*ipow<3>(x)*y + 280*ipow<3>(x) + 168*ipow<2>(x)*ipow<5>(y) - 840*ipow<2>(x)*ipow<4>(y) + 1680*ipow<2>(x)*ipow<3>(y) - 1680*ipow<2>(x)*ipow<2>(y) + 840*ipow<2>(x)*y - 168*ipow<2>(x) + 56*x*ipow<6>(y) - 336*x*ipow<5>(y) + 840*x*ipow<4>(y) - 1120*x*ipow<3>(y) + 840*x*ipow<2>(y) - 336*x*y + 56*x + 8*ipow<7>(y) - 56*ipow<6>(y) + 168*ipow<5>(y) - 280*ipow<4>(y) + 280*ipow<3>(y) - 168*ipow<2>(y) + 56*y + 51480*ipow<7>(z) + 84084*ipow<6>(z)*(2*x + 2*y - 2) + 216216*ipow<5>(z)*ipow<2>(x + y - 1) + 138600*ipow<4>(z)*ipow<3>(x + y - 1) + 46200*ipow<3>(z)*ipow<4>(x + y - 1) + 7560*ipow<2>(z)*ipow<5>(x + y - 1) + 504*z*ipow<6>(x + y - 1) - 8),
            (210*ipow<2>(y) + y*(40*x - 40) + ipow<2>(x - 1))*(102960*ipow<7>(z) + 7*ipow<6>(z)*(51480*x + 51480*y - 51480) + 504504*ipow<5>(z)*ipow<2>(x + y - 1) + 360360*ipow<4>(z)*ipow<3>(x + y - 1) + 138600*ipow<3>(z)*ipow<4>(x + y - 1) + 27720*ipow<2>(z)*ipow<5>(x + y - 1) + 2520*z*ipow<6>(x + y - 1) + 72*ipow<7>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 283
template<>
struct DGBasis<283> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (22*x - 1)*(ipow<9>(x) + 9*ipow<8>(x)*y - 9*ipow<8>(x) + 36*ipow<7>(x)*ipow<2>(y) - 72*ipow<7>(x)*y + 36*ipow<7>(x) + 84*ipow<6>(x)*ipow<3>(y) - 252*ipow<6>(x)*ipow<2>(y) + 252*ipow<6>(x)*y - 84*ipow<6>(x) + 126*ipow<5>(x)*ipow<4>(y) - 504*ipow<5>(x)*ipow<3>(y) + 756*ipow<5>(x)*ipow<2>(y) - 504*ipow<5>(x)*y + 126*ipow<5>(x) + 126*ipow<4>(x)*ipow<5>(y) - 630*ipow<4>(x)*ipow<4>(y) + 1260*ipow<4>(x)*ipow<3>(y) - 1260*ipow<4>(x)*ipow<2>(y) + 630*ipow<4>(x)*y - 126*ipow<4>(x) + 84*ipow<3>(x)*ipow<6>(y) - 504*ipow<3>(x)*ipow<5>(y) + 1260*ipow<3>(x)*ipow<4>(y) - 1680*ipow<3>(x)*ipow<3>(y) + 1260*ipow<3>(x)*ipow<2>(y) - 504*ipow<3>(x)*y + 84*ipow<3>(x) + 36*ipow<2>(x)*ipow<7>(y) - 252*ipow<2>(x)*ipow<6>(y) + 756*ipow<2>(x)*ipow<5>(y) - 1260*ipow<2>(x)*ipow<4>(y) + 1260*ipow<2>(x)*ipow<3>(y) - 756*ipow<2>(x)*ipow<2>(y) + 252*ipow<2>(x)*y - 36*ipow<2>(x) + 9*x*ipow<8>(y) - 72*x*ipow<7>(y) + 252*x*ipow<6>(y) - 504*x*ipow<5>(y) + 630*x*ipow<4>(y) - 504*x*ipow<3>(y) + 252*x*ipow<2>(y) - 72*x*y + 9*x + ipow<9>(y) - 9*ipow<8>(y) + 36*ipow<7>(y) - 84*ipow<6>(y) + 126*ipow<5>(y) - 126*ipow<4>(y) + 84*ipow<3>(y) - 36*ipow<2>(y) + 9*y + 48620*ipow<9>(z) + ipow<8>(z)*(218790*x + 218790*y - 218790) + 411840*ipow<7>(z)*ipow<2>(x + y - 1) + 420420*ipow<6>(z)*ipow<3>(x + y - 1) + 252252*ipow<5>(z)*ipow<4>(x + y - 1) + 90090*ipow<4>(z)*ipow<5>(x + y - 1) + 18480*ipow<3>(z)*ipow<6>(x + y - 1) + 1980*ipow<2>(z)*ipow<7>(x + y - 1) + 90*z*ipow<8>(x + y - 1) - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            22*ipow<9>(x) + 198*ipow<8>(x)*y - 198*ipow<8>(x) + 792*ipow<7>(x)*ipow<2>(y) - 1584*ipow<7>(x)*y + 792*ipow<7>(x) + 1848*ipow<6>(x)*ipow<3>(y) - 5544*ipow<6>(x)*ipow<2>(y) + 5544*ipow<6>(x)*y - 1848*ipow<6>(x) + 2772*ipow<5>(x)*ipow<4>(y) - 11088*ipow<5>(x)*ipow<3>(y) + 16632*ipow<5>(x)*ipow<2>(y) - 11088*ipow<5>(x)*y + 2772*ipow<5>(x) + 2772*ipow<4>(x)*ipow<5>(y) - 13860*ipow<4>(x)*ipow<4>(y) + 27720*ipow<4>(x)*ipow<3>(y) - 27720*ipow<4>(x)*ipow<2>(y) + 13860*ipow<4>(x)*y - 2772*ipow<4>(x) + 1848*ipow<3>(x)*ipow<6>(y) - 11088*ipow<3>(x)*ipow<5>(y) + 27720*ipow<3>(x)*ipow<4>(y) - 36960*ipow<3>(x)*ipow<3>(y) + 27720*ipow<3>(x)*ipow<2>(y) - 11088*ipow<3>(x)*y + 1848*ipow<3>(x) + 792*ipow<2>(x)*ipow<7>(y) - 5544*ipow<2>(x)*ipow<6>(y) + 16632*ipow<2>(x)*ipow<5>(y) - 27720*ipow<2>(x)*ipow<4>(y) + 27720*ipow<2>(x)*ipow<3>(y) - 16632*ipow<2>(x)*ipow<2>(y) + 5544*ipow<2>(x)*y - 792*ipow<2>(x) + 198*x*ipow<8>(y) - 1584*x*ipow<7>(y) + 5544*x*ipow<6>(y) - 11088*x*ipow<5>(y) + 13860*x*ipow<4>(y) - 11088*x*ipow<3>(y) + 5544*x*ipow<2>(y) - 1584*x*y + 198*x + 22*ipow<9>(y) - 198*ipow<8>(y) + 792*ipow<7>(y) - 1848*ipow<6>(y) + 2772*ipow<5>(y) - 2772*ipow<4>(y) + 1848*ipow<3>(y) - 792*ipow<2>(y) + 198*y + 1069640*ipow<9>(z) + 22*ipow<8>(z)*(218790*x + 218790*y - 218790) + 9060480*ipow<7>(z)*ipow<2>(x + y - 1) + 9249240*ipow<6>(z)*ipow<3>(x + y - 1) + 5549544*ipow<5>(z)*ipow<4>(x + y - 1) + 1981980*ipow<4>(z)*ipow<5>(x + y - 1) + 406560*ipow<3>(z)*ipow<6>(x + y - 1) + 43560*ipow<2>(z)*ipow<7>(x + y - 1) + 1980*z*ipow<8>(x + y - 1) + (22*x - 1)*(9*ipow<8>(x) + 72*ipow<7>(x)*y - 72*ipow<7>(x) + 252*ipow<6>(x)*ipow<2>(y) - 504*ipow<6>(x)*y + 252*ipow<6>(x) + 504*ipow<5>(x)*ipow<3>(y) - 1512*ipow<5>(x)*ipow<2>(y) + 1512*ipow<5>(x)*y - 504*ipow<5>(x) + 630*ipow<4>(x)*ipow<4>(y) - 2520*ipow<4>(x)*ipow<3>(y) + 3780*ipow<4>(x)*ipow<2>(y) - 2520*ipow<4>(x)*y + 630*ipow<4>(x) + 504*ipow<3>(x)*ipow<5>(y) - 2520*ipow<3>(x)*ipow<4>(y) + 5040*ipow<3>(x)*ipow<3>(y) - 5040*ipow<3>(x)*ipow<2>(y) + 2520*ipow<3>(x)*y - 504*ipow<3>(x) + 252*ipow<2>(x)*ipow<6>(y) - 1512*ipow<2>(x)*ipow<5>(y) + 3780*ipow<2>(x)*ipow<4>(y) - 5040*ipow<2>(x)*ipow<3>(y) + 3780*ipow<2>(x)*ipow<2>(y) - 1512*ipow<2>(x)*y + 252*ipow<2>(x) + 72*x*ipow<7>(y) - 504*x*ipow<6>(y) + 1512*x*ipow<5>(y) - 2520*x*ipow<4>(y) + 2520*x*ipow<3>(y) - 1512*x*ipow<2>(y) + 504*x*y - 72*x + 9*ipow<8>(y) - 72*ipow<7>(y) + 252*ipow<6>(y) - 504*ipow<5>(y) + 630*ipow<4>(y) - 504*ipow<3>(y) + 252*ipow<2>(y) - 72*y + 218790*ipow<8>(z) + 411840*ipow<7>(z)*(2*x + 2*y - 2) + 1261260*ipow<6>(z)*ipow<2>(x + y - 1) + 1009008*ipow<5>(z)*ipow<3>(x + y - 1) + 450450*ipow<4>(z)*ipow<4>(x + y - 1) + 110880*ipow<3>(z)*ipow<5>(x + y - 1) + 13860*ipow<2>(z)*ipow<6>(x + y - 1) + 720*z*ipow<7>(x + y - 1) + 9) - 22,
            (22*x - 1)*(9*ipow<8>(x) + 72*ipow<7>(x)*y - 72*ipow<7>(x) + 252*ipow<6>(x)*ipow<2>(y) - 504*ipow<6>(x)*y + 252*ipow<6>(x) + 504*ipow<5>(x)*ipow<3>(y) - 1512*ipow<5>(x)*ipow<2>(y) + 1512*ipow<5>(x)*y - 504*ipow<5>(x) + 630*ipow<4>(x)*ipow<4>(y) - 2520*ipow<4>(x)*ipow<3>(y) + 3780*ipow<4>(x)*ipow<2>(y) - 2520*ipow<4>(x)*y + 630*ipow<4>(x) + 504*ipow<3>(x)*ipow<5>(y) - 2520*ipow<3>(x)*ipow<4>(y) + 5040*ipow<3>(x)*ipow<3>(y) - 5040*ipow<3>(x)*ipow<2>(y) + 2520*ipow<3>(x)*y - 504*ipow<3>(x) + 252*ipow<2>(x)*ipow<6>(y) - 1512*ipow<2>(x)*ipow<5>(y) + 3780*ipow<2>(x)*ipow<4>(y) - 5040*ipow<2>(x)*ipow<3>(y) + 3780*ipow<2>(x)*ipow<2>(y) - 1512*ipow<2>(x)*y + 252*ipow<2>(x) + 72*x*ipow<7>(y) - 504*x*ipow<6>(y) + 1512*x*ipow<5>(y) - 2520*x*ipow<4>(y) + 2520*x*ipow<3>(y) - 1512*x*ipow<2>(y) + 504*x*y - 72*x + 9*ipow<8>(y) - 72*ipow<7>(y) + 252*ipow<6>(y) - 504*ipow<5>(y) + 630*ipow<4>(y) - 504*ipow<3>(y) + 252*ipow<2>(y) - 72*y + 218790*ipow<8>(z) + 411840*ipow<7>(z)*(2*x + 2*y - 2) + 1261260*ipow<6>(z)*ipow<2>(x + y - 1) + 1009008*ipow<5>(z)*ipow<3>(x + y - 1) + 450450*ipow<4>(z)*ipow<4>(x + y - 1) + 110880*ipow<3>(z)*ipow<5>(x + y - 1) + 13860*ipow<2>(z)*ipow<6>(x + y - 1) + 720*z*ipow<7>(x + y - 1) + 9),
            (22*x - 1)*(437580*ipow<8>(z) + 8*ipow<7>(z)*(218790*x + 218790*y - 218790) + 2882880*ipow<6>(z)*ipow<2>(x + y - 1) + 2522520*ipow<5>(z)*ipow<3>(x + y - 1) + 1261260*ipow<4>(z)*ipow<4>(x + y - 1) + 360360*ipow<3>(z)*ipow<5>(x + y - 1) + 55440*ipow<2>(z)*ipow<6>(x + y - 1) + 3960*z*ipow<7>(x + y - 1) + 90*ipow<8>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 284
template<>
struct DGBasis<284> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return (x + 21*y - 1)*(ipow<9>(x) + 9*ipow<8>(x)*y - 9*ipow<8>(x) + 36*ipow<7>(x)*ipow<2>(y) - 72*ipow<7>(x)*y + 36*ipow<7>(x) + 84*ipow<6>(x)*ipow<3>(y) - 252*ipow<6>(x)*ipow<2>(y) + 252*ipow<6>(x)*y - 84*ipow<6>(x) + 126*ipow<5>(x)*ipow<4>(y) - 504*ipow<5>(x)*ipow<3>(y) + 756*ipow<5>(x)*ipow<2>(y) - 504*ipow<5>(x)*y + 126*ipow<5>(x) + 126*ipow<4>(x)*ipow<5>(y) - 630*ipow<4>(x)*ipow<4>(y) + 1260*ipow<4>(x)*ipow<3>(y) - 1260*ipow<4>(x)*ipow<2>(y) + 630*ipow<4>(x)*y - 126*ipow<4>(x) + 84*ipow<3>(x)*ipow<6>(y) - 504*ipow<3>(x)*ipow<5>(y) + 1260*ipow<3>(x)*ipow<4>(y) - 1680*ipow<3>(x)*ipow<3>(y) + 1260*ipow<3>(x)*ipow<2>(y) - 504*ipow<3>(x)*y + 84*ipow<3>(x) + 36*ipow<2>(x)*ipow<7>(y) - 252*ipow<2>(x)*ipow<6>(y) + 756*ipow<2>(x)*ipow<5>(y) - 1260*ipow<2>(x)*ipow<4>(y) + 1260*ipow<2>(x)*ipow<3>(y) - 756*ipow<2>(x)*ipow<2>(y) + 252*ipow<2>(x)*y - 36*ipow<2>(x) + 9*x*ipow<8>(y) - 72*x*ipow<7>(y) + 252*x*ipow<6>(y) - 504*x*ipow<5>(y) + 630*x*ipow<4>(y) - 504*x*ipow<3>(y) + 252*x*ipow<2>(y) - 72*x*y + 9*x + ipow<9>(y) - 9*ipow<8>(y) + 36*ipow<7>(y) - 84*ipow<6>(y) + 126*ipow<5>(y) - 126*ipow<4>(y) + 84*ipow<3>(y) - 36*ipow<2>(y) + 9*y + 48620*ipow<9>(z) + ipow<8>(z)*(218790*x + 218790*y - 218790) + 411840*ipow<7>(z)*ipow<2>(x + y - 1) + 420420*ipow<6>(z)*ipow<3>(x + y - 1) + 252252*ipow<5>(z)*ipow<4>(x + y - 1) + 90090*ipow<4>(z)*ipow<5>(x + y - 1) + 18480*ipow<3>(z)*ipow<6>(x + y - 1) + 1980*ipow<2>(z)*ipow<7>(x + y - 1) + 90*z*ipow<8>(x + y - 1) - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            ipow<9>(x) + 9*ipow<8>(x)*y - 9*ipow<8>(x) + 36*ipow<7>(x)*ipow<2>(y) - 72*ipow<7>(x)*y + 36*ipow<7>(x) + 84*ipow<6>(x)*ipow<3>(y) - 252*ipow<6>(x)*ipow<2>(y) + 252*ipow<6>(x)*y - 84*ipow<6>(x) + 126*ipow<5>(x)*ipow<4>(y) - 504*ipow<5>(x)*ipow<3>(y) + 756*ipow<5>(x)*ipow<2>(y) - 504*ipow<5>(x)*y + 126*ipow<5>(x) + 126*ipow<4>(x)*ipow<5>(y) - 630*ipow<4>(x)*ipow<4>(y) + 1260*ipow<4>(x)*ipow<3>(y) - 1260*ipow<4>(x)*ipow<2>(y) + 630*ipow<4>(x)*y - 126*ipow<4>(x) + 84*ipow<3>(x)*ipow<6>(y) - 504*ipow<3>(x)*ipow<5>(y) + 1260*ipow<3>(x)*ipow<4>(y) - 1680*ipow<3>(x)*ipow<3>(y) + 1260*ipow<3>(x)*ipow<2>(y) - 504*ipow<3>(x)*y + 84*ipow<3>(x) + 36*ipow<2>(x)*ipow<7>(y) - 252*ipow<2>(x)*ipow<6>(y) + 756*ipow<2>(x)*ipow<5>(y) - 1260*ipow<2>(x)*ipow<4>(y) + 1260*ipow<2>(x)*ipow<3>(y) - 756*ipow<2>(x)*ipow<2>(y) + 252*ipow<2>(x)*y - 36*ipow<2>(x) + 9*x*ipow<8>(y) - 72*x*ipow<7>(y) + 252*x*ipow<6>(y) - 504*x*ipow<5>(y) + 630*x*ipow<4>(y) - 504*x*ipow<3>(y) + 252*x*ipow<2>(y) - 72*x*y + 9*x + ipow<9>(y) - 9*ipow<8>(y) + 36*ipow<7>(y) - 84*ipow<6>(y) + 126*ipow<5>(y) - 126*ipow<4>(y) + 84*ipow<3>(y) - 36*ipow<2>(y) + 9*y + 48620*ipow<9>(z) + ipow<8>(z)*(218790*x + 218790*y - 218790) + 411840*ipow<7>(z)*ipow<2>(x + y - 1) + 420420*ipow<6>(z)*ipow<3>(x + y - 1) + 252252*ipow<5>(z)*ipow<4>(x + y - 1) + 90090*ipow<4>(z)*ipow<5>(x + y - 1) + 18480*ipow<3>(z)*ipow<6>(x + y - 1) + 1980*ipow<2>(z)*ipow<7>(x + y - 1) + 90*z*ipow<8>(x + y - 1) + (x + 21*y - 1)*(9*ipow<8>(x) + 72*ipow<7>(x)*y - 72*ipow<7>(x) + 252*ipow<6>(x)*ipow<2>(y) - 504*ipow<6>(x)*y + 252*ipow<6>(x) + 504*ipow<5>(x)*ipow<3>(y) - 1512*ipow<5>(x)*ipow<2>(y) + 1512*ipow<5>(x)*y - 504*ipow<5>(x) + 630*ipow<4>(x)*ipow<4>(y) - 2520*ipow<4>(x)*ipow<3>(y) + 3780*ipow<4>(x)*ipow<2>(y) - 2520*ipow<4>(x)*y + 630*ipow<4>(x) + 504*ipow<3>(x)*ipow<5>(y) - 2520*ipow<3>(x)*ipow<4>(y) + 5040*ipow<3>(x)*ipow<3>(y) - 5040*ipow<3>(x)*ipow<2>(y) + 2520*ipow<3>(x)*y - 504*ipow<3>(x) + 252*ipow<2>(x)*ipow<6>(y) - 1512*ipow<2>(x)*ipow<5>(y) + 3780*ipow<2>(x)*ipow<4>(y) - 5040*ipow<2>(x)*ipow<3>(y) + 3780*ipow<2>(x)*ipow<2>(y) - 1512*ipow<2>(x)*y + 252*ipow<2>(x) + 72*x*ipow<7>(y) - 504*x*ipow<6>(y) + 1512*x*ipow<5>(y) - 2520*x*ipow<4>(y) + 2520*x*ipow<3>(y) - 1512*x*ipow<2>(y) + 504*x*y - 72*x + 9*ipow<8>(y) - 72*ipow<7>(y) + 252*ipow<6>(y) - 504*ipow<5>(y) + 630*ipow<4>(y) - 504*ipow<3>(y) + 252*ipow<2>(y) - 72*y + 218790*ipow<8>(z) + 411840*ipow<7>(z)*(2*x + 2*y - 2) + 1261260*ipow<6>(z)*ipow<2>(x + y - 1) + 1009008*ipow<5>(z)*ipow<3>(x + y - 1) + 450450*ipow<4>(z)*ipow<4>(x + y - 1) + 110880*ipow<3>(z)*ipow<5>(x + y - 1) + 13860*ipow<2>(z)*ipow<6>(x + y - 1) + 720*z*ipow<7>(x + y - 1) + 9) - 1,
            21*ipow<9>(x) + 189*ipow<8>(x)*y - 189*ipow<8>(x) + 756*ipow<7>(x)*ipow<2>(y) - 1512*ipow<7>(x)*y + 756*ipow<7>(x) + 1764*ipow<6>(x)*ipow<3>(y) - 5292*ipow<6>(x)*ipow<2>(y) + 5292*ipow<6>(x)*y - 1764*ipow<6>(x) + 2646*ipow<5>(x)*ipow<4>(y) - 10584*ipow<5>(x)*ipow<3>(y) + 15876*ipow<5>(x)*ipow<2>(y) - 10584*ipow<5>(x)*y + 2646*ipow<5>(x) + 2646*ipow<4>(x)*ipow<5>(y) - 13230*ipow<4>(x)*ipow<4>(y) + 26460*ipow<4>(x)*ipow<3>(y) - 26460*ipow<4>(x)*ipow<2>(y) + 13230*ipow<4>(x)*y - 2646*ipow<4>(x) + 1764*ipow<3>(x)*ipow<6>(y) - 10584*ipow<3>(x)*ipow<5>(y) + 26460*ipow<3>(x)*ipow<4>(y) - 35280*ipow<3>(x)*ipow<3>(y) + 26460*ipow<3>(x)*ipow<2>(y) - 10584*ipow<3>(x)*y + 1764*ipow<3>(x) + 756*ipow<2>(x)*ipow<7>(y) - 5292*ipow<2>(x)*ipow<6>(y) + 15876*ipow<2>(x)*ipow<5>(y) - 26460*ipow<2>(x)*ipow<4>(y) + 26460*ipow<2>(x)*ipow<3>(y) - 15876*ipow<2>(x)*ipow<2>(y) + 5292*ipow<2>(x)*y - 756*ipow<2>(x) + 189*x*ipow<8>(y) - 1512*x*ipow<7>(y) + 5292*x*ipow<6>(y) - 10584*x*ipow<5>(y) + 13230*x*ipow<4>(y) - 10584*x*ipow<3>(y) + 5292*x*ipow<2>(y) - 1512*x*y + 189*x + 21*ipow<9>(y) - 189*ipow<8>(y) + 756*ipow<7>(y) - 1764*ipow<6>(y) + 2646*ipow<5>(y) - 2646*ipow<4>(y) + 1764*ipow<3>(y) - 756*ipow<2>(y) + 189*y + 1021020*ipow<9>(z) + 21*ipow<8>(z)*(218790*x + 218790*y - 218790) + 8648640*ipow<7>(z)*ipow<2>(x + y - 1) + 8828820*ipow<6>(z)*ipow<3>(x + y - 1) + 5297292*ipow<5>(z)*ipow<4>(x + y - 1) + 1891890*ipow<4>(z)*ipow<5>(x + y - 1) + 388080*ipow<3>(z)*ipow<6>(x + y - 1) + 41580*ipow<2>(z)*ipow<7>(x + y - 1) + 1890*z*ipow<8>(x + y - 1) + (x + 21*y - 1)*(9*ipow<8>(x) + 72*ipow<7>(x)*y - 72*ipow<7>(x) + 252*ipow<6>(x)*ipow<2>(y) - 504*ipow<6>(x)*y + 252*ipow<6>(x) + 504*ipow<5>(x)*ipow<3>(y) - 1512*ipow<5>(x)*ipow<2>(y) + 1512*ipow<5>(x)*y - 504*ipow<5>(x) + 630*ipow<4>(x)*ipow<4>(y) - 2520*ipow<4>(x)*ipow<3>(y) + 3780*ipow<4>(x)*ipow<2>(y) - 2520*ipow<4>(x)*y + 630*ipow<4>(x) + 504*ipow<3>(x)*ipow<5>(y) - 2520*ipow<3>(x)*ipow<4>(y) + 5040*ipow<3>(x)*ipow<3>(y) - 5040*ipow<3>(x)*ipow<2>(y) + 2520*ipow<3>(x)*y - 504*ipow<3>(x) + 252*ipow<2>(x)*ipow<6>(y) - 1512*ipow<2>(x)*ipow<5>(y) + 3780*ipow<2>(x)*ipow<4>(y) - 5040*ipow<2>(x)*ipow<3>(y) + 3780*ipow<2>(x)*ipow<2>(y) - 1512*ipow<2>(x)*y + 252*ipow<2>(x) + 72*x*ipow<7>(y) - 504*x*ipow<6>(y) + 1512*x*ipow<5>(y) - 2520*x*ipow<4>(y) + 2520*x*ipow<3>(y) - 1512*x*ipow<2>(y) + 504*x*y - 72*x + 9*ipow<8>(y) - 72*ipow<7>(y) + 252*ipow<6>(y) - 504*ipow<5>(y) + 630*ipow<4>(y) - 504*ipow<3>(y) + 252*ipow<2>(y) - 72*y + 218790*ipow<8>(z) + 411840*ipow<7>(z)*(2*x + 2*y - 2) + 1261260*ipow<6>(z)*ipow<2>(x + y - 1) + 1009008*ipow<5>(z)*ipow<3>(x + y - 1) + 450450*ipow<4>(z)*ipow<4>(x + y - 1) + 110880*ipow<3>(z)*ipow<5>(x + y - 1) + 13860*ipow<2>(z)*ipow<6>(x + y - 1) + 720*z*ipow<7>(x + y - 1) + 9) - 21,
            (x + 21*y - 1)*(437580*ipow<8>(z) + 8*ipow<7>(z)*(218790*x + 218790*y - 218790) + 2882880*ipow<6>(z)*ipow<2>(x + y - 1) + 2522520*ipow<5>(z)*ipow<3>(x + y - 1) + 1261260*ipow<4>(z)*ipow<4>(x + y - 1) + 360360*ipow<3>(z)*ipow<5>(x + y - 1) + 55440*ipow<2>(z)*ipow<6>(x + y - 1) + 3960*z*ipow<7>(x + y - 1) + 90*ipow<8>(x + y - 1))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 285
template<>
struct DGBasis<285> {
    template<typename Type>
    HostDevice constexpr static ForceInline Type eval(Type x, Type y, Type z) {
        return 184756*ipow<10>(z) + ipow<9>(z)*(923780*x + 923780*y - 923780) + 1969110*ipow<8>(z)*ipow<2>(x + y - 1) + 2333760*ipow<7>(z)*ipow<3>(x + y - 1) + 1681680*ipow<6>(z)*ipow<4>(x + y - 1) + 756756*ipow<5>(z)*ipow<5>(x + y - 1) + 210210*ipow<4>(z)*ipow<6>(x + y - 1) + 34320*ipow<3>(z)*ipow<7>(x + y - 1) + 2970*ipow<2>(z)*ipow<8>(x + y - 1) + 110*z*ipow<9>(x + y - 1) + ipow<10>(x + y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            923780*ipow<9>(z) + 1969110*ipow<8>(z)*(2*x + 2*y - 2) + 7001280*ipow<7>(z)*ipow<2>(x + y - 1) + 6726720*ipow<6>(z)*ipow<3>(x + y - 1) + 3783780*ipow<5>(z)*ipow<4>(x + y - 1) + 1261260*ipow<4>(z)*ipow<5>(x + y - 1) + 240240*ipow<3>(z)*ipow<6>(x + y - 1) + 23760*ipow<2>(z)*ipow<7>(x + y - 1) + 990*z*ipow<8>(x + y - 1) + 10*ipow<9>(x + y - 1),
            923780*ipow<9>(z) + 1969110*ipow<8>(z)*(2*x + 2*y - 2) + 7001280*ipow<7>(z)*ipow<2>(x + y - 1) + 6726720*ipow<6>(z)*ipow<3>(x + y - 1) + 3783780*ipow<5>(z)*ipow<4>(x + y - 1) + 1261260*ipow<4>(z)*ipow<5>(x + y - 1) + 240240*ipow<3>(z)*ipow<6>(x + y - 1) + 23760*ipow<2>(z)*ipow<7>(x + y - 1) + 990*z*ipow<8>(x + y - 1) + 10*ipow<9>(x + y - 1),
            1847560*ipow<9>(z) + 9*ipow<8>(z)*(923780*x + 923780*y - 923780) + 15752880*ipow<7>(z)*ipow<2>(x + y - 1) + 16336320*ipow<6>(z)*ipow<3>(x + y - 1) + 10090080*ipow<5>(z)*ipow<4>(x + y - 1) + 3783780*ipow<4>(z)*ipow<5>(x + y - 1) + 840840*ipow<3>(z)*ipow<6>(x + y - 1) + 102960*ipow<2>(z)*ipow<7>(x + y - 1) + 5940*z*ipow<8>(x + y - 1) + 110*ipow<9>(x + y - 1)
        };
    }
    static constexpr uInt Order = 10;
};