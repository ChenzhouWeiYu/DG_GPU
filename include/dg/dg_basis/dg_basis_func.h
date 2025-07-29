#pragma once
#include "base/type.h"
#include "dg/dg_basis/dg_basis.h"


// Basis 0
template<>
struct DGBasis<0> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return 4*x - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return x + 3*y - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return x + y + 2*z - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return 5*x*(3*x - 2) + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (6*x - 1)*(x + 3*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return 10*std::pow(y, 2) + y*(8*x - 8) + std::pow(x - 1, 2);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (6*x - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x + 5*y - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return 6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return x*(7*x*(8*x - 9) + 18) - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (14*x*(2*x - 1) + 1)*(x + 3*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (8*x - 1)*(10*std::pow(y, 2) + y*(8*x - 8) + std::pow(x - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            80*std::pow(y, 2) + 8*y*(8*x - 8) + 8*std::pow(x - 1, 2) + (8*x - 1)*(2*x + 8*y - 2),
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return 35*std::pow(y, 3) + std::pow(y, 2)*(45*x - 45) + 15*y*std::pow(x - 1, 2) + std::pow(x - 1, 3);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            45*std::pow(y, 2) + 15*y*(2*x - 2) + 3*std::pow(x - 1, 2),
            105*std::pow(y, 2) + 2*y*(45*x - 45) + 15*std::pow(x - 1, 2),
            0
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 14
template<>
struct DGBasis<14> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (14*x*(2*x - 1) + 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (8*x - 1)*(x + 5*y - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2))*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2) + (2*x + 12*y - 2)*(x + y + 2*z - 1),
            21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2) + (12*x + 42*y - 12)*(x + y + 2*z - 1),
            42*std::pow(y, 2) + 2*y*(12*x - 12) + 2*std::pow(x - 1, 2)
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 17
template<>
struct DGBasis<17> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (8*x - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            48*std::pow(z, 2) + 8*z*(6*x + 6*y - 6) + (8*x - 1)*(2*x + 2*y + 6*z - 2) + 8*std::pow(x + y - 1, 2),
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x + 7*y - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2) + (x + 7*y - 1)*(2*x + 2*y + 6*z - 2),
            42*std::pow(z, 2) + 7*z*(6*x + 6*y - 6) + 7*std::pow(x + y - 1, 2) + (x + 7*y - 1)*(2*x + 2*y + 6*z - 2),
            (x + 7*y - 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 19
template<>
struct DGBasis<19> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return 20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2),
            30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2),
            60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2)
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 20
template<>
struct DGBasis<20> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return 14*x*(3*x*(x*(5*x - 8) + 4) - 2) + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (12*x*(x*(10*x - 9) + 2) - 1)*(x + 3*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (9*x*(5*x - 2) + 1)*(10*std::pow(y, 2) + y*(8*x - 8) + std::pow(x - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (90*x - 18)*(10*std::pow(y, 2) + y*(8*x - 8) + std::pow(x - 1, 2)) + (9*x*(5*x - 2) + 1)*(2*x + 8*y - 2),
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (10*x - 1)*(35*std::pow(y, 3) + std::pow(y, 2)*(45*x - 45) + 15*y*std::pow(x - 1, 2) + std::pow(x - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            350*std::pow(y, 3) + 10*std::pow(y, 2)*(45*x - 45) + 150*y*std::pow(x - 1, 2) + 10*std::pow(x - 1, 3) + (10*x - 1)*(45*std::pow(y, 2) + 15*y*(2*x - 2) + 3*std::pow(x - 1, 2)),
            (10*x - 1)*(105*std::pow(y, 2) + 2*y*(45*x - 45) + 15*std::pow(x - 1, 2)),
            0
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 24
template<>
struct DGBasis<24> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return 126*std::pow(y, 4) + std::pow(y, 3)*(224*x - 224) + 126*std::pow(y, 2)*std::pow(x - 1, 2) + 24*y*std::pow(x - 1, 3) + std::pow(x - 1, 4);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            224*std::pow(y, 3) + 126*std::pow(y, 2)*(2*x - 2) + 72*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3),
            504*std::pow(y, 3) + 3*std::pow(y, 2)*(224*x - 224) + 252*y*std::pow(x - 1, 2) + 24*std::pow(x - 1, 3),
            0
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 25
template<>
struct DGBasis<25> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (12*x*(x*(10*x - 9) + 2) - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (9*x*(5*x - 2) + 1)*(x + 5*y - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (10*x - 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2))*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (10*x - 1)*(2*x + 12*y - 2)*(x + y + 2*z - 1) + (10*x - 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2)) + 10*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2))*(x + y + 2*z - 1),
            (10*x - 1)*(12*x + 42*y - 12)*(x + y + 2*z - 1) + (10*x - 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2)),
            2*(10*x - 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2))
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 28
template<>
struct DGBasis<28> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x + y + 2*z - 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3) + (84*std::pow(y, 2) + 21*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(x + y + 2*z - 1),
            84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3) + (252*std::pow(y, 2) + 2*y*(84*x - 84) + 21*std::pow(x - 1, 2))*(x + y + 2*z - 1),
            168*std::pow(y, 3) + 2*std::pow(y, 2)*(84*x - 84) + 42*y*std::pow(x - 1, 2) + 2*std::pow(x - 1, 3)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 29
template<>
struct DGBasis<29> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (9*x*(5*x - 2) + 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (90*x - 18)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (9*x*(5*x - 2) + 1)*(2*x + 2*y + 6*z - 2),
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (10*x - 1)*(x + 7*y - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (10*x - 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + (10*x - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + 10*(x + 7*y - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)),
            (10*x - 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + 7*(10*x - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)),
            (10*x - 1)*(x + 7*y - 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 31
template<>
struct DGBasis<31> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x + 16*y - 2)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(2*x + 2*y + 6*z - 2),
            (16*x + 72*y - 16)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(2*x + 2*y + 6*z - 2),
            (36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 32
template<>
struct DGBasis<32> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (10*x - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            200*std::pow(z, 3) + 10*std::pow(z, 2)*(30*x + 30*y - 30) + 120*z*std::pow(x + y - 1, 2) + (10*x - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)) + 10*std::pow(x + y - 1, 3),
            (10*x - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)),
            (10*x - 1)*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 33
template<>
struct DGBasis<33> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x + 9*y - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3) + (x + 9*y - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)),
            180*std::pow(z, 3) + 9*std::pow(z, 2)*(30*x + 30*y - 30) + 108*z*std::pow(x + y - 1, 2) + 9*std::pow(x + y - 1, 3) + (x + 9*y - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)),
            (x + 9*y - 1)*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 34
template<>
struct DGBasis<34> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return 70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3),
            140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3),
            280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 35
template<>
struct DGBasis<35> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return 2*x*(3*x*(x*(11*x*(12*x - 25) + 200) - 60) + 20) - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (3*x*(5*x*(11*x*(3*x - 4) + 18) - 12) + 1)*(x + 3*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (5*x*(11*x*(4*x - 3) + 6) - 1)*(10*std::pow(y, 2) + y*(8*x - 8) + std::pow(x - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (5*x*(11*x*(4*x - 3) + 6) - 1)*(2*x + 8*y - 2) + (10*std::pow(y, 2) + y*(8*x - 8) + std::pow(x - 1, 2))*(55*x*(4*x - 3) + 5*x*(88*x - 33) + 30),
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (22*x*(3*x - 1) + 1)*(35*std::pow(y, 3) + std::pow(y, 2)*(45*x - 45) + 15*y*std::pow(x - 1, 2) + std::pow(x - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (132*x - 22)*(35*std::pow(y, 3) + std::pow(y, 2)*(45*x - 45) + 15*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)) + (22*x*(3*x - 1) + 1)*(45*std::pow(y, 2) + 15*y*(2*x - 2) + 3*std::pow(x - 1, 2)),
            (22*x*(3*x - 1) + 1)*(105*std::pow(y, 2) + 2*y*(45*x - 45) + 15*std::pow(x - 1, 2)),
            0
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 39
template<>
struct DGBasis<39> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (12*x - 1)*(126*std::pow(y, 4) + std::pow(y, 3)*(224*x - 224) + 126*std::pow(y, 2)*std::pow(x - 1, 2) + 24*y*std::pow(x - 1, 3) + std::pow(x - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            1512*std::pow(y, 4) + 12*std::pow(y, 3)*(224*x - 224) + 1512*std::pow(y, 2)*std::pow(x - 1, 2) + 288*y*std::pow(x - 1, 3) + 12*std::pow(x - 1, 4) + (12*x - 1)*(224*std::pow(y, 3) + 126*std::pow(y, 2)*(2*x - 2) + 72*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3)),
            (12*x - 1)*(504*std::pow(y, 3) + 3*std::pow(y, 2)*(224*x - 224) + 252*y*std::pow(x - 1, 2) + 24*std::pow(x - 1, 3)),
            0
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 40
template<>
struct DGBasis<40> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return 462*std::pow(y, 5) + std::pow(y, 4)*(1050*x - 1050) + 840*std::pow(y, 3)*std::pow(x - 1, 2) + 280*std::pow(y, 2)*std::pow(x - 1, 3) + 35*y*std::pow(x - 1, 4) + std::pow(x - 1, 5);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            1050*std::pow(y, 4) + 840*std::pow(y, 3)*(2*x - 2) + 840*std::pow(y, 2)*std::pow(x - 1, 2) + 140*y*std::pow(x - 1, 3) + 5*std::pow(x - 1, 4),
            2310*std::pow(y, 4) + 4*std::pow(y, 3)*(1050*x - 1050) + 2520*std::pow(y, 2)*std::pow(x - 1, 2) + 560*y*std::pow(x - 1, 3) + 35*std::pow(x - 1, 4),
            0
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 41
template<>
struct DGBasis<41> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (3*x*(5*x*(11*x*(3*x - 4) + 18) - 12) + 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (5*x*(11*x*(4*x - 3) + 6) - 1)*(x + 5*y - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (22*x*(3*x - 1) + 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2))*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (132*x - 22)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2))*(x + y + 2*z - 1) + (22*x*(3*x - 1) + 1)*(2*x + 12*y - 2)*(x + y + 2*z - 1) + (22*x*(3*x - 1) + 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2)),
            (22*x*(3*x - 1) + 1)*(12*x + 42*y - 12)*(x + y + 2*z - 1) + (22*x*(3*x - 1) + 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2)),
            2*(22*x*(3*x - 1) + 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2))
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 44
template<>
struct DGBasis<44> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (12*x - 1)*(x + y + 2*z - 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (12*x - 1)*(84*std::pow(y, 2) + 21*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(x + y + 2*z - 1) + (12*x - 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)) + 12*(x + y + 2*z - 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (12*x - 1)*(252*std::pow(y, 2) + 2*y*(84*x - 84) + 21*std::pow(x - 1, 2))*(x + y + 2*z - 1) + (12*x - 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            2*(12*x - 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 45
template<>
struct DGBasis<45> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x + y + 2*z - 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4) + (x + y + 2*z - 1)*(480*std::pow(y, 3) + 216*std::pow(y, 2)*(2*x - 2) + 96*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3)),
            330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4) + (x + y + 2*z - 1)*(1320*std::pow(y, 3) + 3*std::pow(y, 2)*(480*x - 480) + 432*y*std::pow(x - 1, 2) + 32*std::pow(x - 1, 3)),
            660*std::pow(y, 4) + 2*std::pow(y, 3)*(480*x - 480) + 432*std::pow(y, 2)*std::pow(x - 1, 2) + 64*y*std::pow(x - 1, 3) + 2*std::pow(x - 1, 4)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 46
template<>
struct DGBasis<46> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (5*x*(11*x*(4*x - 3) + 6) - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (5*x*(11*x*(4*x - 3) + 6) - 1)*(2*x + 2*y + 6*z - 2) + (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(55*x*(4*x - 3) + 5*x*(88*x - 33) + 30),
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (22*x*(3*x - 1) + 1)*(x + 7*y - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (132*x - 22)*(x + 7*y - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (22*x*(3*x - 1) + 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + (22*x*(3*x - 1) + 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)),
            (22*x*(3*x - 1) + 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + 7*(22*x*(3*x - 1) + 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)),
            (22*x*(3*x - 1) + 1)*(x + 7*y - 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 48
template<>
struct DGBasis<48> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (12*x - 1)*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (12*x - 1)*(2*x + 16*y - 2)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (12*x - 1)*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(2*x + 2*y + 6*z - 2) + 12*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)),
            (12*x - 1)*(16*x + 72*y - 16)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (12*x - 1)*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(2*x + 2*y + 6*z - 2),
            (12*x - 1)*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 49
template<>
struct DGBasis<49> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (135*std::pow(y, 2) + 27*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (2*x + 2*y + 6*z - 2)*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (495*std::pow(y, 2) + 2*y*(135*x - 135) + 27*std::pow(x - 1, 2))*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (2*x + 2*y + 6*z - 2)*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (6*x + 6*y + 12*z - 6)*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 50
template<>
struct DGBasis<50> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (22*x*(3*x - 1) + 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (132*x - 22)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (22*x*(3*x - 1) + 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)),
            (22*x*(3*x - 1) + 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)),
            (22*x*(3*x - 1) + 1)*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 51
template<>
struct DGBasis<51> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (12*x - 1)*(x + 9*y - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (12*x - 1)*(x + 9*y - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)) + (12*x - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + 12*(x + 9*y - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (12*x - 1)*(x + 9*y - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)) + 9*(12*x - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (12*x - 1)*(x + 9*y - 1)*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 52
template<>
struct DGBasis<52> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x + 20*y - 2)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)),
            (20*x + 110*y - 20)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)),
            (55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 53
template<>
struct DGBasis<53> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (12*x - 1)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            840*std::pow(z, 4) + 12*std::pow(z, 3)*(140*x + 140*y - 140) + 1080*std::pow(z, 2)*std::pow(x + y - 1, 2) + 240*z*std::pow(x + y - 1, 3) + (12*x - 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)) + 12*std::pow(x + y - 1, 4),
            (12*x - 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)),
            (12*x - 1)*(280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 54
template<>
struct DGBasis<54> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x + 11*y - 1)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4) + (x + 11*y - 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)),
            770*std::pow(z, 4) + 11*std::pow(z, 3)*(140*x + 140*y - 140) + 990*std::pow(z, 2)*std::pow(x + y - 1, 2) + 220*z*std::pow(x + y - 1, 3) + 11*std::pow(x + y - 1, 4) + (x + 11*y - 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)),
            (x + 11*y - 1)*(280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 55
template<>
struct DGBasis<55> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return 252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4),
            630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4),
            1260*std::pow(z, 4) + 4*std::pow(z, 3)*(630*x + 630*y - 630) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 2) + 420*z*std::pow(x + y - 1, 3) + 30*std::pow(x + y - 1, 4)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 56
template<>
struct DGBasis<56> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return 3*x*(x*(11*x*(x*(13*x*(7*x - 18) + 225) - 100) + 225) - 18) + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x*(11*x*(x*(13*x*(14*x - 25) + 200) - 50) + 50) - 1)*(x + 3*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (11*x*(x*(13*x*(7*x - 8) + 36) - 4) + 1)*(10*std::pow(y, 2) + y*(8*x - 8) + std::pow(x - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (11*x*(x*(13*x*(7*x - 8) + 36) - 4) + 1)*(2*x + 8*y - 2) + (10*std::pow(y, 2) + y*(8*x - 8) + std::pow(x - 1, 2))*(11*x*(13*x*(7*x - 8) + 36) + 11*x*(13*x*(7*x - 8) + x*(182*x - 104) + 36) - 44),
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (2*x*(13*x*(14*x - 9) + 18) - 1)*(35*std::pow(y, 3) + std::pow(y, 2)*(45*x - 45) + 15*y*std::pow(x - 1, 2) + std::pow(x - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x*(13*x*(14*x - 9) + 18) - 1)*(45*std::pow(y, 2) + 15*y*(2*x - 2) + 3*std::pow(x - 1, 2)) + (26*x*(14*x - 9) + 2*x*(364*x - 117) + 36)*(35*std::pow(y, 3) + std::pow(y, 2)*(45*x - 45) + 15*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (2*x*(13*x*(14*x - 9) + 18) - 1)*(105*std::pow(y, 2) + 2*y*(45*x - 45) + 15*std::pow(x - 1, 2)),
            0
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 60
template<>
struct DGBasis<60> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (13*x*(7*x - 2) + 1)*(126*std::pow(y, 4) + std::pow(y, 3)*(224*x - 224) + 126*std::pow(y, 2)*std::pow(x - 1, 2) + 24*y*std::pow(x - 1, 3) + std::pow(x - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (182*x - 26)*(126*std::pow(y, 4) + std::pow(y, 3)*(224*x - 224) + 126*std::pow(y, 2)*std::pow(x - 1, 2) + 24*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)) + (13*x*(7*x - 2) + 1)*(224*std::pow(y, 3) + 126*std::pow(y, 2)*(2*x - 2) + 72*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3)),
            (13*x*(7*x - 2) + 1)*(504*std::pow(y, 3) + 3*std::pow(y, 2)*(224*x - 224) + 252*y*std::pow(x - 1, 2) + 24*std::pow(x - 1, 3)),
            0
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 61
template<>
struct DGBasis<61> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (14*x - 1)*(462*std::pow(y, 5) + std::pow(y, 4)*(1050*x - 1050) + 840*std::pow(y, 3)*std::pow(x - 1, 2) + 280*std::pow(y, 2)*std::pow(x - 1, 3) + 35*y*std::pow(x - 1, 4) + std::pow(x - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            6468*std::pow(y, 5) + 14*std::pow(y, 4)*(1050*x - 1050) + 11760*std::pow(y, 3)*std::pow(x - 1, 2) + 3920*std::pow(y, 2)*std::pow(x - 1, 3) + 490*y*std::pow(x - 1, 4) + 14*std::pow(x - 1, 5) + (14*x - 1)*(1050*std::pow(y, 4) + 840*std::pow(y, 3)*(2*x - 2) + 840*std::pow(y, 2)*std::pow(x - 1, 2) + 140*y*std::pow(x - 1, 3) + 5*std::pow(x - 1, 4)),
            (14*x - 1)*(2310*std::pow(y, 4) + 4*std::pow(y, 3)*(1050*x - 1050) + 2520*std::pow(y, 2)*std::pow(x - 1, 2) + 560*y*std::pow(x - 1, 3) + 35*std::pow(x - 1, 4)),
            0
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 62
template<>
struct DGBasis<62> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return 1716*std::pow(y, 6) + std::pow(y, 5)*(4752*x - 4752) + 4950*std::pow(y, 4)*std::pow(x - 1, 2) + 2400*std::pow(y, 3)*std::pow(x - 1, 3) + 540*std::pow(y, 2)*std::pow(x - 1, 4) + 48*y*std::pow(x - 1, 5) + std::pow(x - 1, 6);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            4752*std::pow(y, 5) + 4950*std::pow(y, 4)*(2*x - 2) + 7200*std::pow(y, 3)*std::pow(x - 1, 2) + 2160*std::pow(y, 2)*std::pow(x - 1, 3) + 240*y*std::pow(x - 1, 4) + 6*std::pow(x - 1, 5),
            10296*std::pow(y, 5) + 5*std::pow(y, 4)*(4752*x - 4752) + 19800*std::pow(y, 3)*std::pow(x - 1, 2) + 7200*std::pow(y, 2)*std::pow(x - 1, 3) + 1080*y*std::pow(x - 1, 4) + 48*std::pow(x - 1, 5),
            0
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 63
template<>
struct DGBasis<63> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x*(11*x*(x*(13*x*(14*x - 25) + 200) - 50) + 50) - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (11*x*(x*(13*x*(7*x - 8) + 36) - 4) + 1)*(x + 5*y - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (2*x*(13*x*(14*x - 9) + 18) - 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2))*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x*(13*x*(14*x - 9) + 18) - 1)*(2*x + 12*y - 2)*(x + y + 2*z - 1) + (2*x*(13*x*(14*x - 9) + 18) - 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2)) + (21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2))*(26*x*(14*x - 9) + 2*x*(364*x - 117) + 36)*(x + y + 2*z - 1),
            (2*x*(13*x*(14*x - 9) + 18) - 1)*(12*x + 42*y - 12)*(x + y + 2*z - 1) + (2*x*(13*x*(14*x - 9) + 18) - 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2)),
            2*(2*x*(13*x*(14*x - 9) + 18) - 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 66
template<>
struct DGBasis<66> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (13*x*(7*x - 2) + 1)*(x + y + 2*z - 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (182*x - 26)*(x + y + 2*z - 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)) + (13*x*(7*x - 2) + 1)*(84*std::pow(y, 2) + 21*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(x + y + 2*z - 1) + (13*x*(7*x - 2) + 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (13*x*(7*x - 2) + 1)*(252*std::pow(y, 2) + 2*y*(84*x - 84) + 21*std::pow(x - 1, 2))*(x + y + 2*z - 1) + (13*x*(7*x - 2) + 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            2*(13*x*(7*x - 2) + 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 67
template<>
struct DGBasis<67> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (14*x - 1)*(x + y + 2*z - 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (14*x - 1)*(x + y + 2*z - 1)*(480*std::pow(y, 3) + 216*std::pow(y, 2)*(2*x - 2) + 96*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3)) + (14*x - 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)) + 14*(x + y + 2*z - 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (14*x - 1)*(x + y + 2*z - 1)*(1320*std::pow(y, 3) + 3*std::pow(y, 2)*(480*x - 480) + 432*y*std::pow(x - 1, 2) + 32*std::pow(x - 1, 3)) + (14*x - 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            2*(14*x - 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 68
template<>
struct DGBasis<68> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x + y + 2*z - 1)*(1287*std::pow(y, 5) + std::pow(y, 4)*(2475*x - 2475) + 1650*std::pow(y, 3)*std::pow(x - 1, 2) + 450*std::pow(y, 2)*std::pow(x - 1, 3) + 45*y*std::pow(x - 1, 4) + std::pow(x - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            1287*std::pow(y, 5) + std::pow(y, 4)*(2475*x - 2475) + 1650*std::pow(y, 3)*std::pow(x - 1, 2) + 450*std::pow(y, 2)*std::pow(x - 1, 3) + 45*y*std::pow(x - 1, 4) + std::pow(x - 1, 5) + (x + y + 2*z - 1)*(2475*std::pow(y, 4) + 1650*std::pow(y, 3)*(2*x - 2) + 1350*std::pow(y, 2)*std::pow(x - 1, 2) + 180*y*std::pow(x - 1, 3) + 5*std::pow(x - 1, 4)),
            1287*std::pow(y, 5) + std::pow(y, 4)*(2475*x - 2475) + 1650*std::pow(y, 3)*std::pow(x - 1, 2) + 450*std::pow(y, 2)*std::pow(x - 1, 3) + 45*y*std::pow(x - 1, 4) + std::pow(x - 1, 5) + (x + y + 2*z - 1)*(6435*std::pow(y, 4) + 4*std::pow(y, 3)*(2475*x - 2475) + 4950*std::pow(y, 2)*std::pow(x - 1, 2) + 900*y*std::pow(x - 1, 3) + 45*std::pow(x - 1, 4)),
            2574*std::pow(y, 5) + 2*std::pow(y, 4)*(2475*x - 2475) + 3300*std::pow(y, 3)*std::pow(x - 1, 2) + 900*std::pow(y, 2)*std::pow(x - 1, 3) + 90*y*std::pow(x - 1, 4) + 2*std::pow(x - 1, 5)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 69
template<>
struct DGBasis<69> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (11*x*(x*(13*x*(7*x - 8) + 36) - 4) + 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (11*x*(x*(13*x*(7*x - 8) + 36) - 4) + 1)*(2*x + 2*y + 6*z - 2) + (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(11*x*(13*x*(7*x - 8) + 36) + 11*x*(13*x*(7*x - 8) + x*(182*x - 104) + 36) - 44),
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (2*x*(13*x*(14*x - 9) + 18) - 1)*(x + 7*y - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x*(13*x*(14*x - 9) + 18) - 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + (2*x*(13*x*(14*x - 9) + 18) - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (x + 7*y - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(26*x*(14*x - 9) + 2*x*(364*x - 117) + 36),
            (2*x*(13*x*(14*x - 9) + 18) - 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + 7*(2*x*(13*x*(14*x - 9) + 18) - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)),
            (2*x*(13*x*(14*x - 9) + 18) - 1)*(x + 7*y - 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 71
template<>
struct DGBasis<71> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (13*x*(7*x - 2) + 1)*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (182*x - 26)*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (13*x*(7*x - 2) + 1)*(2*x + 16*y - 2)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (13*x*(7*x - 2) + 1)*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(2*x + 2*y + 6*z - 2),
            (13*x*(7*x - 2) + 1)*(16*x + 72*y - 16)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (13*x*(7*x - 2) + 1)*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(2*x + 2*y + 6*z - 2),
            (13*x*(7*x - 2) + 1)*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 72
template<>
struct DGBasis<72> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (14*x - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (14*x - 1)*(135*std::pow(y, 2) + 27*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (14*x - 1)*(2*x + 2*y + 6*z - 2)*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)) + 14*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (14*x - 1)*(495*std::pow(y, 2) + 2*y*(135*x - 135) + 27*std::pow(x - 1, 2))*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (14*x - 1)*(2*x + 2*y + 6*z - 2)*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (14*x - 1)*(6*x + 6*y + 12*z - 6)*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 73
template<>
struct DGBasis<73> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(715*std::pow(y, 4) + std::pow(y, 3)*(880*x - 880) + 330*std::pow(y, 2)*std::pow(x - 1, 2) + 40*y*std::pow(x - 1, 3) + std::pow(x - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(880*std::pow(y, 3) + 330*std::pow(y, 2)*(2*x - 2) + 120*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3)) + (2*x + 2*y + 6*z - 2)*(715*std::pow(y, 4) + std::pow(y, 3)*(880*x - 880) + 330*std::pow(y, 2)*std::pow(x - 1, 2) + 40*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(2860*std::pow(y, 3) + 3*std::pow(y, 2)*(880*x - 880) + 660*y*std::pow(x - 1, 2) + 40*std::pow(x - 1, 3)) + (2*x + 2*y + 6*z - 2)*(715*std::pow(y, 4) + std::pow(y, 3)*(880*x - 880) + 330*std::pow(y, 2)*std::pow(x - 1, 2) + 40*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (6*x + 6*y + 12*z - 6)*(715*std::pow(y, 4) + std::pow(y, 3)*(880*x - 880) + 330*std::pow(y, 2)*std::pow(x - 1, 2) + 40*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 74
template<>
struct DGBasis<74> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (2*x*(13*x*(14*x - 9) + 18) - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x*(13*x*(14*x - 9) + 18) - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)) + (26*x*(14*x - 9) + 2*x*(364*x - 117) + 36)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (2*x*(13*x*(14*x - 9) + 18) - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)),
            (2*x*(13*x*(14*x - 9) + 18) - 1)*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 75
template<>
struct DGBasis<75> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (13*x*(7*x - 2) + 1)*(x + 9*y - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (182*x - 26)*(x + 9*y - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (13*x*(7*x - 2) + 1)*(x + 9*y - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)) + (13*x*(7*x - 2) + 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (13*x*(7*x - 2) + 1)*(x + 9*y - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)) + 9*(13*x*(7*x - 2) + 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (13*x*(7*x - 2) + 1)*(x + 9*y - 1)*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 76
template<>
struct DGBasis<76> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (14*x - 1)*(55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (14*x - 1)*(2*x + 20*y - 2)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (14*x - 1)*(55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)) + 14*(55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (14*x - 1)*(20*x + 110*y - 20)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (14*x - 1)*(55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)),
            (14*x - 1)*(55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 77
template<>
struct DGBasis<77> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (286*std::pow(y, 3) + std::pow(y, 2)*(198*x - 198) + 33*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (198*std::pow(y, 2) + 33*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(286*std::pow(y, 3) + std::pow(y, 2)*(198*x - 198) + 33*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (858*std::pow(y, 2) + 2*y*(198*x - 198) + 33*std::pow(x - 1, 2))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(286*std::pow(y, 3) + std::pow(y, 2)*(198*x - 198) + 33*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))*(286*std::pow(y, 3) + std::pow(y, 2)*(198*x - 198) + 33*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 78
template<>
struct DGBasis<78> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (13*x*(7*x - 2) + 1)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (182*x - 26)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (13*x*(7*x - 2) + 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)),
            (13*x*(7*x - 2) + 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)),
            (13*x*(7*x - 2) + 1)*(280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 79
template<>
struct DGBasis<79> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (14*x - 1)*(x + 11*y - 1)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (14*x - 1)*(x + 11*y - 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)) + (14*x - 1)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + 14*(x + 11*y - 1)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)),
            (14*x - 1)*(x + 11*y - 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)) + 11*(14*x - 1)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)),
            (14*x - 1)*(x + 11*y - 1)*(280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 80
template<>
struct DGBasis<80> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (78*std::pow(y, 2) + y*(24*x - 24) + std::pow(x - 1, 2))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x + 24*y - 2)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (78*std::pow(y, 2) + y*(24*x - 24) + std::pow(x - 1, 2))*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)),
            (24*x + 156*y - 24)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (78*std::pow(y, 2) + y*(24*x - 24) + std::pow(x - 1, 2))*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)),
            (78*std::pow(y, 2) + y*(24*x - 24) + std::pow(x - 1, 2))*(280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 81
template<>
struct DGBasis<81> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (14*x - 1)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            3528*std::pow(z, 5) + 14*std::pow(z, 4)*(630*x + 630*y - 630) + 7840*std::pow(z, 3)*std::pow(x + y - 1, 2) + 2940*std::pow(z, 2)*std::pow(x + y - 1, 3) + 420*z*std::pow(x + y - 1, 4) + (14*x - 1)*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)) + 14*std::pow(x + y - 1, 5),
            (14*x - 1)*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)),
            (14*x - 1)*(1260*std::pow(z, 4) + 4*std::pow(z, 3)*(630*x + 630*y - 630) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 2) + 420*z*std::pow(x + y - 1, 3) + 30*std::pow(x + y - 1, 4))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 82
template<>
struct DGBasis<82> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x + 13*y - 1)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5) + (x + 13*y - 1)*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)),
            3276*std::pow(z, 5) + 13*std::pow(z, 4)*(630*x + 630*y - 630) + 7280*std::pow(z, 3)*std::pow(x + y - 1, 2) + 2730*std::pow(z, 2)*std::pow(x + y - 1, 3) + 390*z*std::pow(x + y - 1, 4) + 13*std::pow(x + y - 1, 5) + (x + 13*y - 1)*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)),
            (x + 13*y - 1)*(1260*std::pow(z, 4) + 4*std::pow(z, 3)*(630*x + 630*y - 630) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 2) + 420*z*std::pow(x + y - 1, 3) + 30*std::pow(x + y - 1, 4))
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 83
template<>
struct DGBasis<83> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return 924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5),
            2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5),
            5544*std::pow(z, 5) + 5*std::pow(z, 4)*(2772*x + 2772*y - 2772) + 12600*std::pow(z, 3)*std::pow(x + y - 1, 2) + 5040*std::pow(z, 2)*std::pow(x + y - 1, 3) + 840*z*std::pow(x + y - 1, 4) + 42*std::pow(x + y - 1, 5)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 84
template<>
struct DGBasis<84> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return x*(11*x*(x*(13*x*(x*(5*x*(16*x - 49) + 294) - 175) + 700) - 105) + 70) - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (11*x*(2*x - 1)*(13*x*(7*x*(x*(4*x - 7) + 4) - 6) + 6) + 1)*(x + 3*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x*(13*x*(7*x*(48*std::pow(x, 2) - 75*x + 40) - 60) + 60) - 1)*(10*std::pow(y, 2) + y*(8*x - 8) + std::pow(x - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (x*(13*x*(7*x*(48*std::pow(x, 2) - 75*x + 40) - 60) + 60) - 1)*(2*x + 8*y - 2) + (10*std::pow(y, 2) + y*(8*x - 8) + std::pow(x - 1, 2))*(13*x*(7*x*(48*std::pow(x, 2) - 75*x + 40) - 60) + x*(91*x*(48*std::pow(x, 2) - 75*x + 40) + 13*x*(336*std::pow(x, 2) + 7*x*(96*x - 75) - 525*x + 280) - 780) + 60),
            (x*(13*x*(7*x*(48*std::pow(x, 2) - 75*x + 40) - 60) + 60) - 1)*(8*x + 20*y - 8),
            0
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 87
template<>
struct DGBasis<87> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(35*std::pow(y, 3) + std::pow(y, 2)*(45*x - 45) + 15*y*std::pow(x - 1, 2) + std::pow(x - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(45*std::pow(y, 2) + 15*y*(2*x - 2) + 3*std::pow(x - 1, 2)) + (182*x*(x*(10*x - 10) + 3) + 26*x*(7*x*(10*x - 10) + 7*x*(20*x - 10) + 21) - 52)*(35*std::pow(y, 3) + std::pow(y, 2)*(45*x - 45) + 15*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(105*std::pow(y, 2) + 2*y*(45*x - 45) + 15*std::pow(x - 1, 2)),
            0
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 88
template<>
struct DGBasis<88> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (7*x*(5*x*(16*x - 9) + 6) - 1)*(126*std::pow(y, 4) + std::pow(y, 3)*(224*x - 224) + 126*std::pow(y, 2)*std::pow(x - 1, 2) + 24*y*std::pow(x - 1, 3) + std::pow(x - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(224*std::pow(y, 3) + 126*std::pow(y, 2)*(2*x - 2) + 72*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3)) + (35*x*(16*x - 9) + 7*x*(160*x - 45) + 42)*(126*std::pow(y, 4) + std::pow(y, 3)*(224*x - 224) + 126*std::pow(y, 2)*std::pow(x - 1, 2) + 24*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(504*std::pow(y, 3) + 3*std::pow(y, 2)*(224*x - 224) + 252*y*std::pow(x - 1, 2) + 24*std::pow(x - 1, 3)),
            0
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 89
template<>
struct DGBasis<89> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (30*x*(4*x - 1) + 1)*(462*std::pow(y, 5) + std::pow(y, 4)*(1050*x - 1050) + 840*std::pow(y, 3)*std::pow(x - 1, 2) + 280*std::pow(y, 2)*std::pow(x - 1, 3) + 35*y*std::pow(x - 1, 4) + std::pow(x - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (240*x - 30)*(462*std::pow(y, 5) + std::pow(y, 4)*(1050*x - 1050) + 840*std::pow(y, 3)*std::pow(x - 1, 2) + 280*std::pow(y, 2)*std::pow(x - 1, 3) + 35*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)) + (30*x*(4*x - 1) + 1)*(1050*std::pow(y, 4) + 840*std::pow(y, 3)*(2*x - 2) + 840*std::pow(y, 2)*std::pow(x - 1, 2) + 140*y*std::pow(x - 1, 3) + 5*std::pow(x - 1, 4)),
            (30*x*(4*x - 1) + 1)*(2310*std::pow(y, 4) + 4*std::pow(y, 3)*(1050*x - 1050) + 2520*std::pow(y, 2)*std::pow(x - 1, 2) + 560*y*std::pow(x - 1, 3) + 35*std::pow(x - 1, 4)),
            0
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 90
template<>
struct DGBasis<90> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (16*x - 1)*(1716*std::pow(y, 6) + std::pow(y, 5)*(4752*x - 4752) + 4950*std::pow(y, 4)*std::pow(x - 1, 2) + 2400*std::pow(y, 3)*std::pow(x - 1, 3) + 540*std::pow(y, 2)*std::pow(x - 1, 4) + 48*y*std::pow(x - 1, 5) + std::pow(x - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            27456*std::pow(y, 6) + 16*std::pow(y, 5)*(4752*x - 4752) + 79200*std::pow(y, 4)*std::pow(x - 1, 2) + 38400*std::pow(y, 3)*std::pow(x - 1, 3) + 8640*std::pow(y, 2)*std::pow(x - 1, 4) + 768*y*std::pow(x - 1, 5) + 16*std::pow(x - 1, 6) + (16*x - 1)*(4752*std::pow(y, 5) + 4950*std::pow(y, 4)*(2*x - 2) + 7200*std::pow(y, 3)*std::pow(x - 1, 2) + 2160*std::pow(y, 2)*std::pow(x - 1, 3) + 240*y*std::pow(x - 1, 4) + 6*std::pow(x - 1, 5)),
            (16*x - 1)*(10296*std::pow(y, 5) + 5*std::pow(y, 4)*(4752*x - 4752) + 19800*std::pow(y, 3)*std::pow(x - 1, 2) + 7200*std::pow(y, 2)*std::pow(x - 1, 3) + 1080*y*std::pow(x - 1, 4) + 48*std::pow(x - 1, 5)),
            0
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 91
template<>
struct DGBasis<91> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return 6435*std::pow(y, 7) + std::pow(y, 6)*(21021*x - 21021) + 27027*std::pow(y, 5)*std::pow(x - 1, 2) + 17325*std::pow(y, 4)*std::pow(x - 1, 3) + 5775*std::pow(y, 3)*std::pow(x - 1, 4) + 945*std::pow(y, 2)*std::pow(x - 1, 5) + 63*y*std::pow(x - 1, 6) + std::pow(x - 1, 7);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            21021*std::pow(y, 6) + 27027*std::pow(y, 5)*(2*x - 2) + 51975*std::pow(y, 4)*std::pow(x - 1, 2) + 23100*std::pow(y, 3)*std::pow(x - 1, 3) + 4725*std::pow(y, 2)*std::pow(x - 1, 4) + 378*y*std::pow(x - 1, 5) + 7*std::pow(x - 1, 6),
            45045*std::pow(y, 6) + 6*std::pow(y, 5)*(21021*x - 21021) + 135135*std::pow(y, 4)*std::pow(x - 1, 2) + 69300*std::pow(y, 3)*std::pow(x - 1, 3) + 17325*std::pow(y, 2)*std::pow(x - 1, 4) + 1890*y*std::pow(x - 1, 5) + 63*std::pow(x - 1, 6),
            0
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 92
template<>
struct DGBasis<92> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (11*x*(2*x - 1)*(13*x*(7*x*(x*(4*x - 7) + 4) - 6) + 6) + 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x*(13*x*(7*x*(48*std::pow(x, 2) - 75*x + 40) - 60) + 60) - 1)*(x + 5*y - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (x*(13*x*(7*x*(48*std::pow(x, 2) - 75*x + 40) - 60) + 60) - 1)*(x + 5*y - 1) + (x*(13*x*(7*x*(48*std::pow(x, 2) - 75*x + 40) - 60) + 60) - 1)*(x + y + 2*z - 1) + (x + 5*y - 1)*(13*x*(7*x*(48*std::pow(x, 2) - 75*x + 40) - 60) + x*(91*x*(48*std::pow(x, 2) - 75*x + 40) + 13*x*(336*std::pow(x, 2) + 7*x*(96*x - 75) - 525*x + 280) - 780) + 60)*(x + y + 2*z - 1),
            (x*(13*x*(7*x*(48*std::pow(x, 2) - 75*x + 40) - 60) + 60) - 1)*(x + 5*y - 1) + 5*(x*(13*x*(7*x*(48*std::pow(x, 2) - 75*x + 40) - 60) + 60) - 1)*(x + y + 2*z - 1),
            2*(x*(13*x*(7*x*(48*std::pow(x, 2) - 75*x + 40) - 60) + 60) - 1)*(x + 5*y - 1)
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 94
template<>
struct DGBasis<94> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2))*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(2*x + 12*y - 2)*(x + y + 2*z - 1) + (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2)) + (21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2))*(182*x*(x*(10*x - 10) + 3) + 26*x*(7*x*(10*x - 10) + 7*x*(20*x - 10) + 21) - 52)*(x + y + 2*z - 1),
            (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(12*x + 42*y - 12)*(x + y + 2*z - 1) + (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2)),
            2*(26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 95
template<>
struct DGBasis<95> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (7*x*(5*x*(16*x - 9) + 6) - 1)*(x + y + 2*z - 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(84*std::pow(y, 2) + 21*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(x + y + 2*z - 1) + (7*x*(5*x*(16*x - 9) + 6) - 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)) + (35*x*(16*x - 9) + 7*x*(160*x - 45) + 42)*(x + y + 2*z - 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(252*std::pow(y, 2) + 2*y*(84*x - 84) + 21*std::pow(x - 1, 2))*(x + y + 2*z - 1) + (7*x*(5*x*(16*x - 9) + 6) - 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            2*(7*x*(5*x*(16*x - 9) + 6) - 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 96
template<>
struct DGBasis<96> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (30*x*(4*x - 1) + 1)*(x + y + 2*z - 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (240*x - 30)*(x + y + 2*z - 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)) + (30*x*(4*x - 1) + 1)*(x + y + 2*z - 1)*(480*std::pow(y, 3) + 216*std::pow(y, 2)*(2*x - 2) + 96*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3)) + (30*x*(4*x - 1) + 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (30*x*(4*x - 1) + 1)*(x + y + 2*z - 1)*(1320*std::pow(y, 3) + 3*std::pow(y, 2)*(480*x - 480) + 432*y*std::pow(x - 1, 2) + 32*std::pow(x - 1, 3)) + (30*x*(4*x - 1) + 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            2*(30*x*(4*x - 1) + 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 97
template<>
struct DGBasis<97> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (16*x - 1)*(x + y + 2*z - 1)*(1287*std::pow(y, 5) + std::pow(y, 4)*(2475*x - 2475) + 1650*std::pow(y, 3)*std::pow(x - 1, 2) + 450*std::pow(y, 2)*std::pow(x - 1, 3) + 45*y*std::pow(x - 1, 4) + std::pow(x - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (16*x - 1)*(x + y + 2*z - 1)*(2475*std::pow(y, 4) + 1650*std::pow(y, 3)*(2*x - 2) + 1350*std::pow(y, 2)*std::pow(x - 1, 2) + 180*y*std::pow(x - 1, 3) + 5*std::pow(x - 1, 4)) + (16*x - 1)*(1287*std::pow(y, 5) + std::pow(y, 4)*(2475*x - 2475) + 1650*std::pow(y, 3)*std::pow(x - 1, 2) + 450*std::pow(y, 2)*std::pow(x - 1, 3) + 45*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)) + 16*(x + y + 2*z - 1)*(1287*std::pow(y, 5) + std::pow(y, 4)*(2475*x - 2475) + 1650*std::pow(y, 3)*std::pow(x - 1, 2) + 450*std::pow(y, 2)*std::pow(x - 1, 3) + 45*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)),
            (16*x - 1)*(x + y + 2*z - 1)*(6435*std::pow(y, 4) + 4*std::pow(y, 3)*(2475*x - 2475) + 4950*std::pow(y, 2)*std::pow(x - 1, 2) + 900*y*std::pow(x - 1, 3) + 45*std::pow(x - 1, 4)) + (16*x - 1)*(1287*std::pow(y, 5) + std::pow(y, 4)*(2475*x - 2475) + 1650*std::pow(y, 3)*std::pow(x - 1, 2) + 450*std::pow(y, 2)*std::pow(x - 1, 3) + 45*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)),
            2*(16*x - 1)*(1287*std::pow(y, 5) + std::pow(y, 4)*(2475*x - 2475) + 1650*std::pow(y, 3)*std::pow(x - 1, 2) + 450*std::pow(y, 2)*std::pow(x - 1, 3) + 45*y*std::pow(x - 1, 4) + std::pow(x - 1, 5))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 98
template<>
struct DGBasis<98> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x + y + 2*z - 1)*(5005*std::pow(y, 6) + std::pow(y, 5)*(12012*x - 12012) + 10725*std::pow(y, 4)*std::pow(x - 1, 2) + 4400*std::pow(y, 3)*std::pow(x - 1, 3) + 825*std::pow(y, 2)*std::pow(x - 1, 4) + 60*y*std::pow(x - 1, 5) + std::pow(x - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            5005*std::pow(y, 6) + std::pow(y, 5)*(12012*x - 12012) + 10725*std::pow(y, 4)*std::pow(x - 1, 2) + 4400*std::pow(y, 3)*std::pow(x - 1, 3) + 825*std::pow(y, 2)*std::pow(x - 1, 4) + 60*y*std::pow(x - 1, 5) + std::pow(x - 1, 6) + (x + y + 2*z - 1)*(12012*std::pow(y, 5) + 10725*std::pow(y, 4)*(2*x - 2) + 13200*std::pow(y, 3)*std::pow(x - 1, 2) + 3300*std::pow(y, 2)*std::pow(x - 1, 3) + 300*y*std::pow(x - 1, 4) + 6*std::pow(x - 1, 5)),
            5005*std::pow(y, 6) + std::pow(y, 5)*(12012*x - 12012) + 10725*std::pow(y, 4)*std::pow(x - 1, 2) + 4400*std::pow(y, 3)*std::pow(x - 1, 3) + 825*std::pow(y, 2)*std::pow(x - 1, 4) + 60*y*std::pow(x - 1, 5) + std::pow(x - 1, 6) + (x + y + 2*z - 1)*(30030*std::pow(y, 5) + 5*std::pow(y, 4)*(12012*x - 12012) + 42900*std::pow(y, 3)*std::pow(x - 1, 2) + 13200*std::pow(y, 2)*std::pow(x - 1, 3) + 1650*y*std::pow(x - 1, 4) + 60*std::pow(x - 1, 5)),
            10010*std::pow(y, 6) + 2*std::pow(y, 5)*(12012*x - 12012) + 21450*std::pow(y, 4)*std::pow(x - 1, 2) + 8800*std::pow(y, 3)*std::pow(x - 1, 3) + 1650*std::pow(y, 2)*std::pow(x - 1, 4) + 120*y*std::pow(x - 1, 5) + 2*std::pow(x - 1, 6)
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 99
template<>
struct DGBasis<99> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x*(13*x*(7*x*(48*std::pow(x, 2) - 75*x + 40) - 60) + 60) - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (x*(13*x*(7*x*(48*std::pow(x, 2) - 75*x + 40) - 60) + 60) - 1)*(2*x + 2*y + 6*z - 2) + (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(13*x*(7*x*(48*std::pow(x, 2) - 75*x + 40) - 60) + x*(91*x*(48*std::pow(x, 2) - 75*x + 40) + 13*x*(336*std::pow(x, 2) + 7*x*(96*x - 75) - 525*x + 280) - 780) + 60),
            (x*(13*x*(7*x*(48*std::pow(x, 2) - 75*x + 40) - 60) + 60) - 1)*(2*x + 2*y + 6*z - 2),
            (x*(13*x*(7*x*(48*std::pow(x, 2) - 75*x + 40) - 60) + 60) - 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 100
template<>
struct DGBasis<100> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(x + 7*y - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (x + 7*y - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(182*x*(x*(10*x - 10) + 3) + 26*x*(7*x*(10*x - 10) + 7*x*(20*x - 10) + 21) - 52),
            (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + 7*(26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)),
            (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(x + 7*y - 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 101
template<>
struct DGBasis<101> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (7*x*(5*x*(16*x - 9) + 6) - 1)*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(2*x + 16*y - 2)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (7*x*(5*x*(16*x - 9) + 6) - 1)*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(2*x + 2*y + 6*z - 2) + (36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(35*x*(16*x - 9) + 7*x*(160*x - 45) + 42),
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(16*x + 72*y - 16)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (7*x*(5*x*(16*x - 9) + 6) - 1)*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(2*x + 2*y + 6*z - 2),
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 102
template<>
struct DGBasis<102> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (30*x*(4*x - 1) + 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (240*x - 30)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)) + (30*x*(4*x - 1) + 1)*(135*std::pow(y, 2) + 27*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (30*x*(4*x - 1) + 1)*(2*x + 2*y + 6*z - 2)*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (30*x*(4*x - 1) + 1)*(495*std::pow(y, 2) + 2*y*(135*x - 135) + 27*std::pow(x - 1, 2))*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (30*x*(4*x - 1) + 1)*(2*x + 2*y + 6*z - 2)*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (30*x*(4*x - 1) + 1)*(6*x + 6*y + 12*z - 6)*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 103
template<>
struct DGBasis<103> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (16*x - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(715*std::pow(y, 4) + std::pow(y, 3)*(880*x - 880) + 330*std::pow(y, 2)*std::pow(x - 1, 2) + 40*y*std::pow(x - 1, 3) + std::pow(x - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (16*x - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(880*std::pow(y, 3) + 330*std::pow(y, 2)*(2*x - 2) + 120*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3)) + (16*x - 1)*(2*x + 2*y + 6*z - 2)*(715*std::pow(y, 4) + std::pow(y, 3)*(880*x - 880) + 330*std::pow(y, 2)*std::pow(x - 1, 2) + 40*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)) + 16*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(715*std::pow(y, 4) + std::pow(y, 3)*(880*x - 880) + 330*std::pow(y, 2)*std::pow(x - 1, 2) + 40*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (16*x - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(2860*std::pow(y, 3) + 3*std::pow(y, 2)*(880*x - 880) + 660*y*std::pow(x - 1, 2) + 40*std::pow(x - 1, 3)) + (16*x - 1)*(2*x + 2*y + 6*z - 2)*(715*std::pow(y, 4) + std::pow(y, 3)*(880*x - 880) + 330*std::pow(y, 2)*std::pow(x - 1, 2) + 40*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (16*x - 1)*(6*x + 6*y + 12*z - 6)*(715*std::pow(y, 4) + std::pow(y, 3)*(880*x - 880) + 330*std::pow(y, 2)*std::pow(x - 1, 2) + 40*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 104
template<>
struct DGBasis<104> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(3003*std::pow(y, 5) + std::pow(y, 4)*(5005*x - 5005) + 2860*std::pow(y, 3)*std::pow(x - 1, 2) + 660*std::pow(y, 2)*std::pow(x - 1, 3) + 55*y*std::pow(x - 1, 4) + std::pow(x - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(5005*std::pow(y, 4) + 2860*std::pow(y, 3)*(2*x - 2) + 1980*std::pow(y, 2)*std::pow(x - 1, 2) + 220*y*std::pow(x - 1, 3) + 5*std::pow(x - 1, 4)) + (2*x + 2*y + 6*z - 2)*(3003*std::pow(y, 5) + std::pow(y, 4)*(5005*x - 5005) + 2860*std::pow(y, 3)*std::pow(x - 1, 2) + 660*std::pow(y, 2)*std::pow(x - 1, 3) + 55*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)),
            (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(15015*std::pow(y, 4) + 4*std::pow(y, 3)*(5005*x - 5005) + 8580*std::pow(y, 2)*std::pow(x - 1, 2) + 1320*y*std::pow(x - 1, 3) + 55*std::pow(x - 1, 4)) + (2*x + 2*y + 6*z - 2)*(3003*std::pow(y, 5) + std::pow(y, 4)*(5005*x - 5005) + 2860*std::pow(y, 3)*std::pow(x - 1, 2) + 660*std::pow(y, 2)*std::pow(x - 1, 3) + 55*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)),
            (6*x + 6*y + 12*z - 6)*(3003*std::pow(y, 5) + std::pow(y, 4)*(5005*x - 5005) + 2860*std::pow(y, 3)*std::pow(x - 1, 2) + 660*std::pow(y, 2)*std::pow(x - 1, 3) + 55*y*std::pow(x - 1, 4) + std::pow(x - 1, 5))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 105
template<>
struct DGBasis<105> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)) + (182*x*(x*(10*x - 10) + 3) + 26*x*(7*x*(10*x - 10) + 7*x*(20*x - 10) + 21) - 52)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)),
            (26*x*(7*x*(x*(10*x - 10) + 3) - 2) + 1)*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 106
template<>
struct DGBasis<106> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (7*x*(5*x*(16*x - 9) + 6) - 1)*(x + 9*y - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(x + 9*y - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)) + (7*x*(5*x*(16*x - 9) + 6) - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (x + 9*y - 1)*(35*x*(16*x - 9) + 7*x*(160*x - 45) + 42)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(x + 9*y - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)) + 9*(7*x*(5*x*(16*x - 9) + 6) - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(x + 9*y - 1)*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 107
template<>
struct DGBasis<107> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (30*x*(4*x - 1) + 1)*(55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (240*x - 30)*(55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (30*x*(4*x - 1) + 1)*(2*x + 20*y - 2)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (30*x*(4*x - 1) + 1)*(55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)),
            (30*x*(4*x - 1) + 1)*(20*x + 110*y - 20)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (30*x*(4*x - 1) + 1)*(55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)),
            (30*x*(4*x - 1) + 1)*(55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 108
template<>
struct DGBasis<108> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (16*x - 1)*(286*std::pow(y, 3) + std::pow(y, 2)*(198*x - 198) + 33*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (16*x - 1)*(198*std::pow(y, 2) + 33*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (16*x - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(286*std::pow(y, 3) + std::pow(y, 2)*(198*x - 198) + 33*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)) + 16*(286*std::pow(y, 3) + std::pow(y, 2)*(198*x - 198) + 33*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (16*x - 1)*(858*std::pow(y, 2) + 2*y*(198*x - 198) + 33*std::pow(x - 1, 2))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (16*x - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(286*std::pow(y, 3) + std::pow(y, 2)*(198*x - 198) + 33*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (16*x - 1)*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))*(286*std::pow(y, 3) + std::pow(y, 2)*(198*x - 198) + 33*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 109
template<>
struct DGBasis<109> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(1365*std::pow(y, 4) + std::pow(y, 3)*(1456*x - 1456) + 468*std::pow(y, 2)*std::pow(x - 1, 2) + 48*y*std::pow(x - 1, 3) + std::pow(x - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(1365*std::pow(y, 4) + std::pow(y, 3)*(1456*x - 1456) + 468*std::pow(y, 2)*std::pow(x - 1, 2) + 48*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)) + (1456*std::pow(y, 3) + 468*std::pow(y, 2)*(2*x - 2) + 144*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(1365*std::pow(y, 4) + std::pow(y, 3)*(1456*x - 1456) + 468*std::pow(y, 2)*std::pow(x - 1, 2) + 48*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)) + (5460*std::pow(y, 3) + 3*std::pow(y, 2)*(1456*x - 1456) + 936*y*std::pow(x - 1, 2) + 48*std::pow(x - 1, 3))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))*(1365*std::pow(y, 4) + std::pow(y, 3)*(1456*x - 1456) + 468*std::pow(y, 2)*std::pow(x - 1, 2) + 48*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 110
template<>
struct DGBasis<110> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (7*x*(5*x*(16*x - 9) + 6) - 1)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)) + (35*x*(16*x - 9) + 7*x*(160*x - 45) + 42)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)),
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)),
            (7*x*(5*x*(16*x - 9) + 6) - 1)*(280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 111
template<>
struct DGBasis<111> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (30*x*(4*x - 1) + 1)*(x + 11*y - 1)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (240*x - 30)*(x + 11*y - 1)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (30*x*(4*x - 1) + 1)*(x + 11*y - 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)) + (30*x*(4*x - 1) + 1)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)),
            (30*x*(4*x - 1) + 1)*(x + 11*y - 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)) + 11*(30*x*(4*x - 1) + 1)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)),
            (30*x*(4*x - 1) + 1)*(x + 11*y - 1)*(280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 112
template<>
struct DGBasis<112> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (16*x - 1)*(78*std::pow(y, 2) + y*(24*x - 24) + std::pow(x - 1, 2))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (16*x - 1)*(2*x + 24*y - 2)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (16*x - 1)*(78*std::pow(y, 2) + y*(24*x - 24) + std::pow(x - 1, 2))*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)) + 16*(78*std::pow(y, 2) + y*(24*x - 24) + std::pow(x - 1, 2))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)),
            (16*x - 1)*(24*x + 156*y - 24)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (16*x - 1)*(78*std::pow(y, 2) + y*(24*x - 24) + std::pow(x - 1, 2))*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)),
            (16*x - 1)*(78*std::pow(y, 2) + y*(24*x - 24) + std::pow(x - 1, 2))*(280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 113
template<>
struct DGBasis<113> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (455*std::pow(y, 3) + std::pow(y, 2)*(273*x - 273) + 39*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (273*std::pow(y, 2) + 39*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (455*std::pow(y, 3) + std::pow(y, 2)*(273*x - 273) + 39*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)),
            (1365*std::pow(y, 2) + 2*y*(273*x - 273) + 39*std::pow(x - 1, 2))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (455*std::pow(y, 3) + std::pow(y, 2)*(273*x - 273) + 39*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)),
            (455*std::pow(y, 3) + std::pow(y, 2)*(273*x - 273) + 39*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 114
template<>
struct DGBasis<114> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (30*x*(4*x - 1) + 1)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (240*x - 30)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (30*x*(4*x - 1) + 1)*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)),
            (30*x*(4*x - 1) + 1)*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)),
            (30*x*(4*x - 1) + 1)*(1260*std::pow(z, 4) + 4*std::pow(z, 3)*(630*x + 630*y - 630) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 2) + 420*z*std::pow(x + y - 1, 3) + 30*std::pow(x + y - 1, 4))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 115
template<>
struct DGBasis<115> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (16*x - 1)*(x + 13*y - 1)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (16*x - 1)*(x + 13*y - 1)*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)) + (16*x - 1)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + 16*(x + 13*y - 1)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)),
            (16*x - 1)*(x + 13*y - 1)*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)) + 13*(16*x - 1)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)),
            (16*x - 1)*(x + 13*y - 1)*(1260*std::pow(z, 4) + 4*std::pow(z, 3)*(630*x + 630*y - 630) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 2) + 420*z*std::pow(x + y - 1, 3) + 30*std::pow(x + y - 1, 4))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 116
template<>
struct DGBasis<116> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (105*std::pow(y, 2) + y*(28*x - 28) + std::pow(x - 1, 2))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x + 28*y - 2)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (105*std::pow(y, 2) + y*(28*x - 28) + std::pow(x - 1, 2))*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)),
            (28*x + 210*y - 28)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (105*std::pow(y, 2) + y*(28*x - 28) + std::pow(x - 1, 2))*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)),
            (105*std::pow(y, 2) + y*(28*x - 28) + std::pow(x - 1, 2))*(1260*std::pow(z, 4) + 4*std::pow(z, 3)*(630*x + 630*y - 630) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 2) + 420*z*std::pow(x + y - 1, 3) + 30*std::pow(x + y - 1, 4))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 117
template<>
struct DGBasis<117> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (16*x - 1)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            14784*std::pow(z, 6) + 16*std::pow(z, 5)*(2772*x + 2772*y - 2772) + 50400*std::pow(z, 4)*std::pow(x + y - 1, 2) + 26880*std::pow(z, 3)*std::pow(x + y - 1, 3) + 6720*std::pow(z, 2)*std::pow(x + y - 1, 4) + 672*z*std::pow(x + y - 1, 5) + (16*x - 1)*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)) + 16*std::pow(x + y - 1, 6),
            (16*x - 1)*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)),
            (16*x - 1)*(5544*std::pow(z, 5) + 5*std::pow(z, 4)*(2772*x + 2772*y - 2772) + 12600*std::pow(z, 3)*std::pow(x + y - 1, 2) + 5040*std::pow(z, 2)*std::pow(x + y - 1, 3) + 840*z*std::pow(x + y - 1, 4) + 42*std::pow(x + y - 1, 5))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 118
template<>
struct DGBasis<118> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x + 15*y - 1)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6) + (x + 15*y - 1)*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)),
            13860*std::pow(z, 6) + 15*std::pow(z, 5)*(2772*x + 2772*y - 2772) + 47250*std::pow(z, 4)*std::pow(x + y - 1, 2) + 25200*std::pow(z, 3)*std::pow(x + y - 1, 3) + 6300*std::pow(z, 2)*std::pow(x + y - 1, 4) + 630*z*std::pow(x + y - 1, 5) + 15*std::pow(x + y - 1, 6) + (x + 15*y - 1)*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)),
            (x + 15*y - 1)*(5544*std::pow(z, 5) + 5*std::pow(z, 4)*(2772*x + 2772*y - 2772) + 12600*std::pow(z, 3)*std::pow(x + y - 1, 2) + 5040*std::pow(z, 2)*std::pow(x + y - 1, 3) + 840*z*std::pow(x + y - 1, 4) + 42*std::pow(x + y - 1, 5))
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 119
template<>
struct DGBasis<119> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            7*std::pow(x, 6) + 42*std::pow(x, 5)*y - 42*std::pow(x, 5) + 105*std::pow(x, 4)*std::pow(y, 2) - 210*std::pow(x, 4)*y + 105*std::pow(x, 4) + 140*std::pow(x, 3)*std::pow(y, 3) - 420*std::pow(x, 3)*std::pow(y, 2) + 420*std::pow(x, 3)*y - 140*std::pow(x, 3) + 105*std::pow(x, 2)*std::pow(y, 4) - 420*std::pow(x, 2)*std::pow(y, 3) + 630*std::pow(x, 2)*std::pow(y, 2) - 420*std::pow(x, 2)*y + 105*std::pow(x, 2) + 42*x*std::pow(y, 5) - 210*x*std::pow(y, 4) + 420*x*std::pow(y, 3) - 420*x*std::pow(y, 2) + 210*x*y - 42*x + 7*std::pow(y, 6) - 42*std::pow(y, 5) + 105*std::pow(y, 4) - 140*std::pow(y, 3) + 105*std::pow(y, 2) - 42*y + 12012*std::pow(z, 6) + 16632*std::pow(z, 5)*(2*x + 2*y - 2) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 2) + 16800*std::pow(z, 3)*std::pow(x + y - 1, 3) + 3780*std::pow(z, 2)*std::pow(x + y - 1, 4) + 336*z*std::pow(x + y - 1, 5) + 7,
            7*std::pow(x, 6) + 42*std::pow(x, 5)*y - 42*std::pow(x, 5) + 105*std::pow(x, 4)*std::pow(y, 2) - 210*std::pow(x, 4)*y + 105*std::pow(x, 4) + 140*std::pow(x, 3)*std::pow(y, 3) - 420*std::pow(x, 3)*std::pow(y, 2) + 420*std::pow(x, 3)*y - 140*std::pow(x, 3) + 105*std::pow(x, 2)*std::pow(y, 4) - 420*std::pow(x, 2)*std::pow(y, 3) + 630*std::pow(x, 2)*std::pow(y, 2) - 420*std::pow(x, 2)*y + 105*std::pow(x, 2) + 42*x*std::pow(y, 5) - 210*x*std::pow(y, 4) + 420*x*std::pow(y, 3) - 420*x*std::pow(y, 2) + 210*x*y - 42*x + 7*std::pow(y, 6) - 42*std::pow(y, 5) + 105*std::pow(y, 4) - 140*std::pow(y, 3) + 105*std::pow(y, 2) - 42*y + 12012*std::pow(z, 6) + 16632*std::pow(z, 5)*(2*x + 2*y - 2) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 2) + 16800*std::pow(z, 3)*std::pow(x + y - 1, 3) + 3780*std::pow(z, 2)*std::pow(x + y - 1, 4) + 336*z*std::pow(x + y - 1, 5) + 7,
            24024*std::pow(z, 6) + 6*std::pow(z, 5)*(12012*x + 12012*y - 12012) + 83160*std::pow(z, 4)*std::pow(x + y - 1, 2) + 46200*std::pow(z, 3)*std::pow(x + y - 1, 3) + 12600*std::pow(z, 2)*std::pow(x + y - 1, 4) + 1512*z*std::pow(x + y - 1, 5) + 56*std::pow(x + y - 1, 6)
        };
    }
    static constexpr uInt Order = 7;
};

// Basis 120
template<>
struct DGBasis<120> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return 22*x*(x*(13*x*(x*(x*(x*(17*x*(9*x - 32) + 784) - 588) + 245) - 56) + 84) - 4) + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x*(13*x*(x*(x*(8*x*(17*x*(18*x - 49) + 882) - 3675) + 980) - 126) + 84) - 1)*(x + 3*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (13*x*(7*x*(4*x*(3*x*(x*(17*x - 34) + 25) - 25) + 15) - 6) + 1)*(10*std::pow(y, 2) + y*(8*x - 8) + std::pow(x - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (13*x*(7*x*(4*x*(3*x*(x*(17*x - 34) + 25) - 25) + 15) - 6) + 1)*(2*x + 8*y - 2) + (10*std::pow(y, 2) + y*(8*x - 8) + std::pow(x - 1, 2))*(91*x*(4*x*(3*x*(x*(17*x - 34) + 25) - 25) + 15) + 13*x*(28*x*(3*x*(x*(17*x - 34) + 25) - 25) + 7*x*(12*x*(x*(17*x - 34) + 25) + 4*x*(3*x*(17*x - 34) + 3*x*(34*x - 34) + 75) - 100) + 105) - 78),
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(35*std::pow(y, 3) + std::pow(y, 2)*(45*x - 45) + 15*y*std::pow(x - 1, 2) + std::pow(x - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(45*std::pow(y, 2) + 15*y*(2*x - 2) + 3*std::pow(x - 1, 2)) + (14*x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 14*x*(2*x*(17*x*(18*x - 25) + 200) + x*(34*x*(18*x - 25) + 2*x*(612*x - 425) + 400) - 75) + 70)*(35*std::pow(y, 3) + std::pow(y, 2)*(45*x - 45) + 15*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(105*std::pow(y, 2) + 2*y*(45*x - 45) + 15*std::pow(x - 1, 2)),
            0
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 124
template<>
struct DGBasis<124> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(126*std::pow(y, 4) + std::pow(y, 3)*(224*x - 224) + 126*std::pow(y, 2)*std::pow(x - 1, 2) + 24*y*std::pow(x - 1, 3) + std::pow(x - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(224*std::pow(y, 3) + 126*std::pow(y, 2)*(2*x - 2) + 72*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3)) + (20*x*(17*x*(9*x - 8) + 36) + 20*x*(17*x*(9*x - 8) + x*(306*x - 136) + 36) - 60)*(126*std::pow(y, 4) + std::pow(y, 3)*(224*x - 224) + 126*std::pow(y, 2)*std::pow(x - 1, 2) + 24*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(504*std::pow(y, 3) + 3*std::pow(y, 2)*(224*x - 224) + 252*y*std::pow(x - 1, 2) + 24*std::pow(x - 1, 3)),
            0
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 125
template<>
struct DGBasis<125> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (24*x*(17*x*(2*x - 1) + 2) - 1)*(462*std::pow(y, 5) + std::pow(y, 4)*(1050*x - 1050) + 840*std::pow(y, 3)*std::pow(x - 1, 2) + 280*std::pow(y, 2)*std::pow(x - 1, 3) + 35*y*std::pow(x - 1, 4) + std::pow(x - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(1050*std::pow(y, 4) + 840*std::pow(y, 3)*(2*x - 2) + 840*std::pow(y, 2)*std::pow(x - 1, 2) + 140*y*std::pow(x - 1, 3) + 5*std::pow(x - 1, 4)) + (408*x*(2*x - 1) + 24*x*(68*x - 17) + 48)*(462*std::pow(y, 5) + std::pow(y, 4)*(1050*x - 1050) + 840*std::pow(y, 3)*std::pow(x - 1, 2) + 280*std::pow(y, 2)*std::pow(x - 1, 3) + 35*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)),
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(2310*std::pow(y, 4) + 4*std::pow(y, 3)*(1050*x - 1050) + 2520*std::pow(y, 2)*std::pow(x - 1, 2) + 560*y*std::pow(x - 1, 3) + 35*std::pow(x - 1, 4)),
            0
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 126
template<>
struct DGBasis<126> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (17*x*(9*x - 2) + 1)*(1716*std::pow(y, 6) + std::pow(y, 5)*(4752*x - 4752) + 4950*std::pow(y, 4)*std::pow(x - 1, 2) + 2400*std::pow(y, 3)*std::pow(x - 1, 3) + 540*std::pow(y, 2)*std::pow(x - 1, 4) + 48*y*std::pow(x - 1, 5) + std::pow(x - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (306*x - 34)*(1716*std::pow(y, 6) + std::pow(y, 5)*(4752*x - 4752) + 4950*std::pow(y, 4)*std::pow(x - 1, 2) + 2400*std::pow(y, 3)*std::pow(x - 1, 3) + 540*std::pow(y, 2)*std::pow(x - 1, 4) + 48*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)) + (17*x*(9*x - 2) + 1)*(4752*std::pow(y, 5) + 4950*std::pow(y, 4)*(2*x - 2) + 7200*std::pow(y, 3)*std::pow(x - 1, 2) + 2160*std::pow(y, 2)*std::pow(x - 1, 3) + 240*y*std::pow(x - 1, 4) + 6*std::pow(x - 1, 5)),
            (17*x*(9*x - 2) + 1)*(10296*std::pow(y, 5) + 5*std::pow(y, 4)*(4752*x - 4752) + 19800*std::pow(y, 3)*std::pow(x - 1, 2) + 7200*std::pow(y, 2)*std::pow(x - 1, 3) + 1080*y*std::pow(x - 1, 4) + 48*std::pow(x - 1, 5)),
            0
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 127
template<>
struct DGBasis<127> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (18*x - 1)*(6435*std::pow(y, 7) + std::pow(y, 6)*(21021*x - 21021) + 27027*std::pow(y, 5)*std::pow(x - 1, 2) + 17325*std::pow(y, 4)*std::pow(x - 1, 3) + 5775*std::pow(y, 3)*std::pow(x - 1, 4) + 945*std::pow(y, 2)*std::pow(x - 1, 5) + 63*y*std::pow(x - 1, 6) + std::pow(x - 1, 7));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            115830*std::pow(y, 7) + 18*std::pow(y, 6)*(21021*x - 21021) + 486486*std::pow(y, 5)*std::pow(x - 1, 2) + 311850*std::pow(y, 4)*std::pow(x - 1, 3) + 103950*std::pow(y, 3)*std::pow(x - 1, 4) + 17010*std::pow(y, 2)*std::pow(x - 1, 5) + 1134*y*std::pow(x - 1, 6) + 18*std::pow(x - 1, 7) + (18*x - 1)*(21021*std::pow(y, 6) + 27027*std::pow(y, 5)*(2*x - 2) + 51975*std::pow(y, 4)*std::pow(x - 1, 2) + 23100*std::pow(y, 3)*std::pow(x - 1, 3) + 4725*std::pow(y, 2)*std::pow(x - 1, 4) + 378*y*std::pow(x - 1, 5) + 7*std::pow(x - 1, 6)),
            (18*x - 1)*(45045*std::pow(y, 6) + 6*std::pow(y, 5)*(21021*x - 21021) + 135135*std::pow(y, 4)*std::pow(x - 1, 2) + 69300*std::pow(y, 3)*std::pow(x - 1, 3) + 17325*std::pow(y, 2)*std::pow(x - 1, 4) + 1890*y*std::pow(x - 1, 5) + 63*std::pow(x - 1, 6)),
            0
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 128
template<>
struct DGBasis<128> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return 24310*std::pow(y, 8) + std::pow(y, 7)*(91520*x - 91520) + 140140*std::pow(y, 6)*std::pow(x - 1, 2) + 112112*std::pow(y, 5)*std::pow(x - 1, 3) + 50050*std::pow(y, 4)*std::pow(x - 1, 4) + 12320*std::pow(y, 3)*std::pow(x - 1, 5) + 1540*std::pow(y, 2)*std::pow(x - 1, 6) + 80*y*std::pow(x - 1, 7) + std::pow(x - 1, 8);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            91520*std::pow(y, 7) + 140140*std::pow(y, 6)*(2*x - 2) + 336336*std::pow(y, 5)*std::pow(x - 1, 2) + 200200*std::pow(y, 4)*std::pow(x - 1, 3) + 61600*std::pow(y, 3)*std::pow(x - 1, 4) + 9240*std::pow(y, 2)*std::pow(x - 1, 5) + 560*y*std::pow(x - 1, 6) + 8*std::pow(x - 1, 7),
            194480*std::pow(y, 7) + 7*std::pow(y, 6)*(91520*x - 91520) + 840840*std::pow(y, 5)*std::pow(x - 1, 2) + 560560*std::pow(y, 4)*std::pow(x - 1, 3) + 200200*std::pow(y, 3)*std::pow(x - 1, 4) + 36960*std::pow(y, 2)*std::pow(x - 1, 5) + 3080*y*std::pow(x - 1, 6) + 80*std::pow(x - 1, 7),
            0
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 129
template<>
struct DGBasis<129> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x*(13*x*(x*(x*(8*x*(17*x*(18*x - 49) + 882) - 3675) + 980) - 126) + 84) - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (13*x*(7*x*(4*x*(3*x*(x*(17*x - 34) + 25) - 25) + 15) - 6) + 1)*(x + 5*y - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2))*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(2*x + 12*y - 2)*(x + y + 2*z - 1) + (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2)) + (21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2))*(14*x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 14*x*(2*x*(17*x*(18*x - 25) + 200) + x*(34*x*(18*x - 25) + 2*x*(612*x - 425) + 400) - 75) + 70)*(x + y + 2*z - 1),
            (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(12*x + 42*y - 12)*(x + y + 2*z - 1) + (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2)),
            2*(14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 132
template<>
struct DGBasis<132> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(x + y + 2*z - 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(84*std::pow(y, 2) + 21*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(x + y + 2*z - 1) + (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)) + (20*x*(17*x*(9*x - 8) + 36) + 20*x*(17*x*(9*x - 8) + x*(306*x - 136) + 36) - 60)*(x + y + 2*z - 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(252*std::pow(y, 2) + 2*y*(84*x - 84) + 21*std::pow(x - 1, 2))*(x + y + 2*z - 1) + (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            2*(20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 133
template<>
struct DGBasis<133> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (24*x*(17*x*(2*x - 1) + 2) - 1)*(x + y + 2*z - 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(x + y + 2*z - 1)*(480*std::pow(y, 3) + 216*std::pow(y, 2)*(2*x - 2) + 96*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3)) + (24*x*(17*x*(2*x - 1) + 2) - 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)) + (408*x*(2*x - 1) + 24*x*(68*x - 17) + 48)*(x + y + 2*z - 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(x + y + 2*z - 1)*(1320*std::pow(y, 3) + 3*std::pow(y, 2)*(480*x - 480) + 432*y*std::pow(x - 1, 2) + 32*std::pow(x - 1, 3)) + (24*x*(17*x*(2*x - 1) + 2) - 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            2*(24*x*(17*x*(2*x - 1) + 2) - 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 134
template<>
struct DGBasis<134> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (17*x*(9*x - 2) + 1)*(x + y + 2*z - 1)*(1287*std::pow(y, 5) + std::pow(y, 4)*(2475*x - 2475) + 1650*std::pow(y, 3)*std::pow(x - 1, 2) + 450*std::pow(y, 2)*std::pow(x - 1, 3) + 45*y*std::pow(x - 1, 4) + std::pow(x - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (306*x - 34)*(x + y + 2*z - 1)*(1287*std::pow(y, 5) + std::pow(y, 4)*(2475*x - 2475) + 1650*std::pow(y, 3)*std::pow(x - 1, 2) + 450*std::pow(y, 2)*std::pow(x - 1, 3) + 45*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)) + (17*x*(9*x - 2) + 1)*(x + y + 2*z - 1)*(2475*std::pow(y, 4) + 1650*std::pow(y, 3)*(2*x - 2) + 1350*std::pow(y, 2)*std::pow(x - 1, 2) + 180*y*std::pow(x - 1, 3) + 5*std::pow(x - 1, 4)) + (17*x*(9*x - 2) + 1)*(1287*std::pow(y, 5) + std::pow(y, 4)*(2475*x - 2475) + 1650*std::pow(y, 3)*std::pow(x - 1, 2) + 450*std::pow(y, 2)*std::pow(x - 1, 3) + 45*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)),
            (17*x*(9*x - 2) + 1)*(x + y + 2*z - 1)*(6435*std::pow(y, 4) + 4*std::pow(y, 3)*(2475*x - 2475) + 4950*std::pow(y, 2)*std::pow(x - 1, 2) + 900*y*std::pow(x - 1, 3) + 45*std::pow(x - 1, 4)) + (17*x*(9*x - 2) + 1)*(1287*std::pow(y, 5) + std::pow(y, 4)*(2475*x - 2475) + 1650*std::pow(y, 3)*std::pow(x - 1, 2) + 450*std::pow(y, 2)*std::pow(x - 1, 3) + 45*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)),
            2*(17*x*(9*x - 2) + 1)*(1287*std::pow(y, 5) + std::pow(y, 4)*(2475*x - 2475) + 1650*std::pow(y, 3)*std::pow(x - 1, 2) + 450*std::pow(y, 2)*std::pow(x - 1, 3) + 45*y*std::pow(x - 1, 4) + std::pow(x - 1, 5))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 135
template<>
struct DGBasis<135> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (18*x - 1)*(x + y + 2*z - 1)*(5005*std::pow(y, 6) + std::pow(y, 5)*(12012*x - 12012) + 10725*std::pow(y, 4)*std::pow(x - 1, 2) + 4400*std::pow(y, 3)*std::pow(x - 1, 3) + 825*std::pow(y, 2)*std::pow(x - 1, 4) + 60*y*std::pow(x - 1, 5) + std::pow(x - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (18*x - 1)*(x + y + 2*z - 1)*(12012*std::pow(y, 5) + 10725*std::pow(y, 4)*(2*x - 2) + 13200*std::pow(y, 3)*std::pow(x - 1, 2) + 3300*std::pow(y, 2)*std::pow(x - 1, 3) + 300*y*std::pow(x - 1, 4) + 6*std::pow(x - 1, 5)) + (18*x - 1)*(5005*std::pow(y, 6) + std::pow(y, 5)*(12012*x - 12012) + 10725*std::pow(y, 4)*std::pow(x - 1, 2) + 4400*std::pow(y, 3)*std::pow(x - 1, 3) + 825*std::pow(y, 2)*std::pow(x - 1, 4) + 60*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)) + 18*(x + y + 2*z - 1)*(5005*std::pow(y, 6) + std::pow(y, 5)*(12012*x - 12012) + 10725*std::pow(y, 4)*std::pow(x - 1, 2) + 4400*std::pow(y, 3)*std::pow(x - 1, 3) + 825*std::pow(y, 2)*std::pow(x - 1, 4) + 60*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)),
            (18*x - 1)*(x + y + 2*z - 1)*(30030*std::pow(y, 5) + 5*std::pow(y, 4)*(12012*x - 12012) + 42900*std::pow(y, 3)*std::pow(x - 1, 2) + 13200*std::pow(y, 2)*std::pow(x - 1, 3) + 1650*y*std::pow(x - 1, 4) + 60*std::pow(x - 1, 5)) + (18*x - 1)*(5005*std::pow(y, 6) + std::pow(y, 5)*(12012*x - 12012) + 10725*std::pow(y, 4)*std::pow(x - 1, 2) + 4400*std::pow(y, 3)*std::pow(x - 1, 3) + 825*std::pow(y, 2)*std::pow(x - 1, 4) + 60*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)),
            2*(18*x - 1)*(5005*std::pow(y, 6) + std::pow(y, 5)*(12012*x - 12012) + 10725*std::pow(y, 4)*std::pow(x - 1, 2) + 4400*std::pow(y, 3)*std::pow(x - 1, 3) + 825*std::pow(y, 2)*std::pow(x - 1, 4) + 60*y*std::pow(x - 1, 5) + std::pow(x - 1, 6))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 136
template<>
struct DGBasis<136> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x + y + 2*z - 1)*(19448*std::pow(y, 7) + std::pow(y, 6)*(56056*x - 56056) + 63063*std::pow(y, 5)*std::pow(x - 1, 2) + 35035*std::pow(y, 4)*std::pow(x - 1, 3) + 10010*std::pow(y, 3)*std::pow(x - 1, 4) + 1386*std::pow(y, 2)*std::pow(x - 1, 5) + 77*y*std::pow(x - 1, 6) + std::pow(x - 1, 7));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            19448*std::pow(y, 7) + std::pow(y, 6)*(56056*x - 56056) + 63063*std::pow(y, 5)*std::pow(x - 1, 2) + 35035*std::pow(y, 4)*std::pow(x - 1, 3) + 10010*std::pow(y, 3)*std::pow(x - 1, 4) + 1386*std::pow(y, 2)*std::pow(x - 1, 5) + 77*y*std::pow(x - 1, 6) + std::pow(x - 1, 7) + (x + y + 2*z - 1)*(56056*std::pow(y, 6) + 63063*std::pow(y, 5)*(2*x - 2) + 105105*std::pow(y, 4)*std::pow(x - 1, 2) + 40040*std::pow(y, 3)*std::pow(x - 1, 3) + 6930*std::pow(y, 2)*std::pow(x - 1, 4) + 462*y*std::pow(x - 1, 5) + 7*std::pow(x - 1, 6)),
            19448*std::pow(y, 7) + std::pow(y, 6)*(56056*x - 56056) + 63063*std::pow(y, 5)*std::pow(x - 1, 2) + 35035*std::pow(y, 4)*std::pow(x - 1, 3) + 10010*std::pow(y, 3)*std::pow(x - 1, 4) + 1386*std::pow(y, 2)*std::pow(x - 1, 5) + 77*y*std::pow(x - 1, 6) + std::pow(x - 1, 7) + (x + y + 2*z - 1)*(136136*std::pow(y, 6) + 6*std::pow(y, 5)*(56056*x - 56056) + 315315*std::pow(y, 4)*std::pow(x - 1, 2) + 140140*std::pow(y, 3)*std::pow(x - 1, 3) + 30030*std::pow(y, 2)*std::pow(x - 1, 4) + 2772*y*std::pow(x - 1, 5) + 77*std::pow(x - 1, 6)),
            38896*std::pow(y, 7) + 2*std::pow(y, 6)*(56056*x - 56056) + 126126*std::pow(y, 5)*std::pow(x - 1, 2) + 70070*std::pow(y, 4)*std::pow(x - 1, 3) + 20020*std::pow(y, 3)*std::pow(x - 1, 4) + 2772*std::pow(y, 2)*std::pow(x - 1, 5) + 154*y*std::pow(x - 1, 6) + 2*std::pow(x - 1, 7)
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 137
template<>
struct DGBasis<137> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (13*x*(7*x*(4*x*(3*x*(x*(17*x - 34) + 25) - 25) + 15) - 6) + 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (13*x*(7*x*(4*x*(3*x*(x*(17*x - 34) + 25) - 25) + 15) - 6) + 1)*(2*x + 2*y + 6*z - 2) + (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(91*x*(4*x*(3*x*(x*(17*x - 34) + 25) - 25) + 15) + 13*x*(28*x*(3*x*(x*(17*x - 34) + 25) - 25) + 7*x*(12*x*(x*(17*x - 34) + 25) + 4*x*(3*x*(17*x - 34) + 3*x*(34*x - 34) + 75) - 100) + 105) - 78),
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(x + 7*y - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (x + 7*y - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(14*x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 14*x*(2*x*(17*x*(18*x - 25) + 200) + x*(34*x*(18*x - 25) + 2*x*(612*x - 425) + 400) - 75) + 70),
            (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + 7*(14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)),
            (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(x + 7*y - 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 139
template<>
struct DGBasis<139> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(2*x + 16*y - 2)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(2*x + 2*y + 6*z - 2) + (36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(20*x*(17*x*(9*x - 8) + 36) + 20*x*(17*x*(9*x - 8) + x*(306*x - 136) + 36) - 60),
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(16*x + 72*y - 16)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(2*x + 2*y + 6*z - 2),
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 140
template<>
struct DGBasis<140> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (24*x*(17*x*(2*x - 1) + 2) - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(135*std::pow(y, 2) + 27*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (24*x*(17*x*(2*x - 1) + 2) - 1)*(2*x + 2*y + 6*z - 2)*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)) + (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(408*x*(2*x - 1) + 24*x*(68*x - 17) + 48)*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(495*std::pow(y, 2) + 2*y*(135*x - 135) + 27*std::pow(x - 1, 2))*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (24*x*(17*x*(2*x - 1) + 2) - 1)*(2*x + 2*y + 6*z - 2)*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(6*x + 6*y + 12*z - 6)*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 141
template<>
struct DGBasis<141> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (17*x*(9*x - 2) + 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(715*std::pow(y, 4) + std::pow(y, 3)*(880*x - 880) + 330*std::pow(y, 2)*std::pow(x - 1, 2) + 40*y*std::pow(x - 1, 3) + std::pow(x - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (306*x - 34)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(715*std::pow(y, 4) + std::pow(y, 3)*(880*x - 880) + 330*std::pow(y, 2)*std::pow(x - 1, 2) + 40*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)) + (17*x*(9*x - 2) + 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(880*std::pow(y, 3) + 330*std::pow(y, 2)*(2*x - 2) + 120*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3)) + (17*x*(9*x - 2) + 1)*(2*x + 2*y + 6*z - 2)*(715*std::pow(y, 4) + std::pow(y, 3)*(880*x - 880) + 330*std::pow(y, 2)*std::pow(x - 1, 2) + 40*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (17*x*(9*x - 2) + 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(2860*std::pow(y, 3) + 3*std::pow(y, 2)*(880*x - 880) + 660*y*std::pow(x - 1, 2) + 40*std::pow(x - 1, 3)) + (17*x*(9*x - 2) + 1)*(2*x + 2*y + 6*z - 2)*(715*std::pow(y, 4) + std::pow(y, 3)*(880*x - 880) + 330*std::pow(y, 2)*std::pow(x - 1, 2) + 40*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (17*x*(9*x - 2) + 1)*(6*x + 6*y + 12*z - 6)*(715*std::pow(y, 4) + std::pow(y, 3)*(880*x - 880) + 330*std::pow(y, 2)*std::pow(x - 1, 2) + 40*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 142
template<>
struct DGBasis<142> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (18*x - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(3003*std::pow(y, 5) + std::pow(y, 4)*(5005*x - 5005) + 2860*std::pow(y, 3)*std::pow(x - 1, 2) + 660*std::pow(y, 2)*std::pow(x - 1, 3) + 55*y*std::pow(x - 1, 4) + std::pow(x - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (18*x - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(5005*std::pow(y, 4) + 2860*std::pow(y, 3)*(2*x - 2) + 1980*std::pow(y, 2)*std::pow(x - 1, 2) + 220*y*std::pow(x - 1, 3) + 5*std::pow(x - 1, 4)) + (18*x - 1)*(2*x + 2*y + 6*z - 2)*(3003*std::pow(y, 5) + std::pow(y, 4)*(5005*x - 5005) + 2860*std::pow(y, 3)*std::pow(x - 1, 2) + 660*std::pow(y, 2)*std::pow(x - 1, 3) + 55*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)) + 18*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(3003*std::pow(y, 5) + std::pow(y, 4)*(5005*x - 5005) + 2860*std::pow(y, 3)*std::pow(x - 1, 2) + 660*std::pow(y, 2)*std::pow(x - 1, 3) + 55*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)),
            (18*x - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(15015*std::pow(y, 4) + 4*std::pow(y, 3)*(5005*x - 5005) + 8580*std::pow(y, 2)*std::pow(x - 1, 2) + 1320*y*std::pow(x - 1, 3) + 55*std::pow(x - 1, 4)) + (18*x - 1)*(2*x + 2*y + 6*z - 2)*(3003*std::pow(y, 5) + std::pow(y, 4)*(5005*x - 5005) + 2860*std::pow(y, 3)*std::pow(x - 1, 2) + 660*std::pow(y, 2)*std::pow(x - 1, 3) + 55*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)),
            (18*x - 1)*(6*x + 6*y + 12*z - 6)*(3003*std::pow(y, 5) + std::pow(y, 4)*(5005*x - 5005) + 2860*std::pow(y, 3)*std::pow(x - 1, 2) + 660*std::pow(y, 2)*std::pow(x - 1, 3) + 55*y*std::pow(x - 1, 4) + std::pow(x - 1, 5))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 143
template<>
struct DGBasis<143> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(12376*std::pow(y, 6) + std::pow(y, 5)*(26208*x - 26208) + 20475*std::pow(y, 4)*std::pow(x - 1, 2) + 7280*std::pow(y, 3)*std::pow(x - 1, 3) + 1170*std::pow(y, 2)*std::pow(x - 1, 4) + 72*y*std::pow(x - 1, 5) + std::pow(x - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(26208*std::pow(y, 5) + 20475*std::pow(y, 4)*(2*x - 2) + 21840*std::pow(y, 3)*std::pow(x - 1, 2) + 4680*std::pow(y, 2)*std::pow(x - 1, 3) + 360*y*std::pow(x - 1, 4) + 6*std::pow(x - 1, 5)) + (2*x + 2*y + 6*z - 2)*(12376*std::pow(y, 6) + std::pow(y, 5)*(26208*x - 26208) + 20475*std::pow(y, 4)*std::pow(x - 1, 2) + 7280*std::pow(y, 3)*std::pow(x - 1, 3) + 1170*std::pow(y, 2)*std::pow(x - 1, 4) + 72*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)),
            (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(74256*std::pow(y, 5) + 5*std::pow(y, 4)*(26208*x - 26208) + 81900*std::pow(y, 3)*std::pow(x - 1, 2) + 21840*std::pow(y, 2)*std::pow(x - 1, 3) + 2340*y*std::pow(x - 1, 4) + 72*std::pow(x - 1, 5)) + (2*x + 2*y + 6*z - 2)*(12376*std::pow(y, 6) + std::pow(y, 5)*(26208*x - 26208) + 20475*std::pow(y, 4)*std::pow(x - 1, 2) + 7280*std::pow(y, 3)*std::pow(x - 1, 3) + 1170*std::pow(y, 2)*std::pow(x - 1, 4) + 72*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)),
            (6*x + 6*y + 12*z - 6)*(12376*std::pow(y, 6) + std::pow(y, 5)*(26208*x - 26208) + 20475*std::pow(y, 4)*std::pow(x - 1, 2) + 7280*std::pow(y, 3)*std::pow(x - 1, 3) + 1170*std::pow(y, 2)*std::pow(x - 1, 4) + 72*y*std::pow(x - 1, 5) + std::pow(x - 1, 6))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 144
template<>
struct DGBasis<144> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)) + (14*x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 14*x*(2*x*(17*x*(18*x - 25) + 200) + x*(34*x*(18*x - 25) + 2*x*(612*x - 425) + 400) - 75) + 70)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)),
            (14*x*(x*(2*x*(17*x*(18*x - 25) + 200) - 75) + 5) - 1)*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 145
template<>
struct DGBasis<145> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(x + 9*y - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(x + 9*y - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)) + (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (x + 9*y - 1)*(20*x*(17*x*(9*x - 8) + 36) + 20*x*(17*x*(9*x - 8) + x*(306*x - 136) + 36) - 60)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(x + 9*y - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)) + 9*(20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(x + 9*y - 1)*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 146
template<>
struct DGBasis<146> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (24*x*(17*x*(2*x - 1) + 2) - 1)*(55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(2*x + 20*y - 2)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (24*x*(17*x*(2*x - 1) + 2) - 1)*(55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)) + (55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(408*x*(2*x - 1) + 24*x*(68*x - 17) + 48)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(20*x + 110*y - 20)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (24*x*(17*x*(2*x - 1) + 2) - 1)*(55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)),
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 147
template<>
struct DGBasis<147> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (17*x*(9*x - 2) + 1)*(286*std::pow(y, 3) + std::pow(y, 2)*(198*x - 198) + 33*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (306*x - 34)*(286*std::pow(y, 3) + std::pow(y, 2)*(198*x - 198) + 33*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (17*x*(9*x - 2) + 1)*(198*std::pow(y, 2) + 33*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (17*x*(9*x - 2) + 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(286*std::pow(y, 3) + std::pow(y, 2)*(198*x - 198) + 33*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (17*x*(9*x - 2) + 1)*(858*std::pow(y, 2) + 2*y*(198*x - 198) + 33*std::pow(x - 1, 2))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (17*x*(9*x - 2) + 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(286*std::pow(y, 3) + std::pow(y, 2)*(198*x - 198) + 33*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (17*x*(9*x - 2) + 1)*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))*(286*std::pow(y, 3) + std::pow(y, 2)*(198*x - 198) + 33*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 148
template<>
struct DGBasis<148> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (18*x - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(1365*std::pow(y, 4) + std::pow(y, 3)*(1456*x - 1456) + 468*std::pow(y, 2)*std::pow(x - 1, 2) + 48*y*std::pow(x - 1, 3) + std::pow(x - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (18*x - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(1365*std::pow(y, 4) + std::pow(y, 3)*(1456*x - 1456) + 468*std::pow(y, 2)*std::pow(x - 1, 2) + 48*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)) + (18*x - 1)*(1456*std::pow(y, 3) + 468*std::pow(y, 2)*(2*x - 2) + 144*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + 18*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(1365*std::pow(y, 4) + std::pow(y, 3)*(1456*x - 1456) + 468*std::pow(y, 2)*std::pow(x - 1, 2) + 48*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (18*x - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(1365*std::pow(y, 4) + std::pow(y, 3)*(1456*x - 1456) + 468*std::pow(y, 2)*std::pow(x - 1, 2) + 48*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)) + (18*x - 1)*(5460*std::pow(y, 3) + 3*std::pow(y, 2)*(1456*x - 1456) + 936*y*std::pow(x - 1, 2) + 48*std::pow(x - 1, 3))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (18*x - 1)*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))*(1365*std::pow(y, 4) + std::pow(y, 3)*(1456*x - 1456) + 468*std::pow(y, 2)*std::pow(x - 1, 2) + 48*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 149
template<>
struct DGBasis<149> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(6188*std::pow(y, 5) + std::pow(y, 4)*(9100*x - 9100) + 4550*std::pow(y, 3)*std::pow(x - 1, 2) + 910*std::pow(y, 2)*std::pow(x - 1, 3) + 65*y*std::pow(x - 1, 4) + std::pow(x - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(6188*std::pow(y, 5) + std::pow(y, 4)*(9100*x - 9100) + 4550*std::pow(y, 3)*std::pow(x - 1, 2) + 910*std::pow(y, 2)*std::pow(x - 1, 3) + 65*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)) + (20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(9100*std::pow(y, 4) + 4550*std::pow(y, 3)*(2*x - 2) + 2730*std::pow(y, 2)*std::pow(x - 1, 2) + 260*y*std::pow(x - 1, 3) + 5*std::pow(x - 1, 4)),
            (30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(6188*std::pow(y, 5) + std::pow(y, 4)*(9100*x - 9100) + 4550*std::pow(y, 3)*std::pow(x - 1, 2) + 910*std::pow(y, 2)*std::pow(x - 1, 3) + 65*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)) + (20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(30940*std::pow(y, 4) + 4*std::pow(y, 3)*(9100*x - 9100) + 13650*std::pow(y, 2)*std::pow(x - 1, 2) + 1820*y*std::pow(x - 1, 3) + 65*std::pow(x - 1, 4)),
            (60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))*(6188*std::pow(y, 5) + std::pow(y, 4)*(9100*x - 9100) + 4550*std::pow(y, 3)*std::pow(x - 1, 2) + 910*std::pow(y, 2)*std::pow(x - 1, 3) + 65*y*std::pow(x - 1, 4) + std::pow(x - 1, 5))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 150
template<>
struct DGBasis<150> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)) + (20*x*(17*x*(9*x - 8) + 36) + 20*x*(17*x*(9*x - 8) + x*(306*x - 136) + 36) - 60)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)),
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)),
            (20*x*(x*(17*x*(9*x - 8) + 36) - 3) + 1)*(280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 151
template<>
struct DGBasis<151> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (24*x*(17*x*(2*x - 1) + 2) - 1)*(x + 11*y - 1)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(x + 11*y - 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)) + (24*x*(17*x*(2*x - 1) + 2) - 1)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (x + 11*y - 1)*(408*x*(2*x - 1) + 24*x*(68*x - 17) + 48)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)),
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(x + 11*y - 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)) + 11*(24*x*(17*x*(2*x - 1) + 2) - 1)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)),
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(x + 11*y - 1)*(280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 152
template<>
struct DGBasis<152> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (17*x*(9*x - 2) + 1)*(78*std::pow(y, 2) + y*(24*x - 24) + std::pow(x - 1, 2))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (306*x - 34)*(78*std::pow(y, 2) + y*(24*x - 24) + std::pow(x - 1, 2))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (17*x*(9*x - 2) + 1)*(2*x + 24*y - 2)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (17*x*(9*x - 2) + 1)*(78*std::pow(y, 2) + y*(24*x - 24) + std::pow(x - 1, 2))*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)),
            (17*x*(9*x - 2) + 1)*(24*x + 156*y - 24)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (17*x*(9*x - 2) + 1)*(78*std::pow(y, 2) + y*(24*x - 24) + std::pow(x - 1, 2))*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)),
            (17*x*(9*x - 2) + 1)*(78*std::pow(y, 2) + y*(24*x - 24) + std::pow(x - 1, 2))*(280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 153
template<>
struct DGBasis<153> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (18*x - 1)*(455*std::pow(y, 3) + std::pow(y, 2)*(273*x - 273) + 39*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (18*x - 1)*(273*std::pow(y, 2) + 39*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (18*x - 1)*(455*std::pow(y, 3) + std::pow(y, 2)*(273*x - 273) + 39*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)) + 18*(455*std::pow(y, 3) + std::pow(y, 2)*(273*x - 273) + 39*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)),
            (18*x - 1)*(1365*std::pow(y, 2) + 2*y*(273*x - 273) + 39*std::pow(x - 1, 2))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (18*x - 1)*(455*std::pow(y, 3) + std::pow(y, 2)*(273*x - 273) + 39*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)),
            (18*x - 1)*(455*std::pow(y, 3) + std::pow(y, 2)*(273*x - 273) + 39*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 154
template<>
struct DGBasis<154> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (2380*std::pow(y, 4) + std::pow(y, 3)*(2240*x - 2240) + 630*std::pow(y, 2)*std::pow(x - 1, 2) + 56*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2240*std::pow(y, 3) + 630*std::pow(y, 2)*(2*x - 2) + 168*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3))*(2380*std::pow(y, 4) + std::pow(y, 3)*(2240*x - 2240) + 630*std::pow(y, 2)*std::pow(x - 1, 2) + 56*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (9520*std::pow(y, 3) + 3*std::pow(y, 2)*(2240*x - 2240) + 1260*y*std::pow(x - 1, 2) + 56*std::pow(x - 1, 3))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3))*(2380*std::pow(y, 4) + std::pow(y, 3)*(2240*x - 2240) + 630*std::pow(y, 2)*std::pow(x - 1, 2) + 56*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))*(2380*std::pow(y, 4) + std::pow(y, 3)*(2240*x - 2240) + 630*std::pow(y, 2)*std::pow(x - 1, 2) + 56*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 155
template<>
struct DGBasis<155> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (24*x*(17*x*(2*x - 1) + 2) - 1)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)) + (408*x*(2*x - 1) + 24*x*(68*x - 17) + 48)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)),
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)),
            (24*x*(17*x*(2*x - 1) + 2) - 1)*(1260*std::pow(z, 4) + 4*std::pow(z, 3)*(630*x + 630*y - 630) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 2) + 420*z*std::pow(x + y - 1, 3) + 30*std::pow(x + y - 1, 4))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 156
template<>
struct DGBasis<156> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (17*x*(9*x - 2) + 1)*(x + 13*y - 1)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (306*x - 34)*(x + 13*y - 1)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (17*x*(9*x - 2) + 1)*(x + 13*y - 1)*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)) + (17*x*(9*x - 2) + 1)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)),
            (17*x*(9*x - 2) + 1)*(x + 13*y - 1)*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)) + 13*(17*x*(9*x - 2) + 1)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)),
            (17*x*(9*x - 2) + 1)*(x + 13*y - 1)*(1260*std::pow(z, 4) + 4*std::pow(z, 3)*(630*x + 630*y - 630) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 2) + 420*z*std::pow(x + y - 1, 3) + 30*std::pow(x + y - 1, 4))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 157
template<>
struct DGBasis<157> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (18*x - 1)*(105*std::pow(y, 2) + y*(28*x - 28) + std::pow(x - 1, 2))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (18*x - 1)*(2*x + 28*y - 2)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (18*x - 1)*(105*std::pow(y, 2) + y*(28*x - 28) + std::pow(x - 1, 2))*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)) + 18*(105*std::pow(y, 2) + y*(28*x - 28) + std::pow(x - 1, 2))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)),
            (18*x - 1)*(28*x + 210*y - 28)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (18*x - 1)*(105*std::pow(y, 2) + y*(28*x - 28) + std::pow(x - 1, 2))*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)),
            (18*x - 1)*(105*std::pow(y, 2) + y*(28*x - 28) + std::pow(x - 1, 2))*(1260*std::pow(z, 4) + 4*std::pow(z, 3)*(630*x + 630*y - 630) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 2) + 420*z*std::pow(x + y - 1, 3) + 30*std::pow(x + y - 1, 4))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 158
template<>
struct DGBasis<158> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (680*std::pow(y, 3) + std::pow(y, 2)*(360*x - 360) + 45*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (360*std::pow(y, 2) + 45*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (680*std::pow(y, 3) + std::pow(y, 2)*(360*x - 360) + 45*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)),
            (2040*std::pow(y, 2) + 2*y*(360*x - 360) + 45*std::pow(x - 1, 2))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (680*std::pow(y, 3) + std::pow(y, 2)*(360*x - 360) + 45*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)),
            (680*std::pow(y, 3) + std::pow(y, 2)*(360*x - 360) + 45*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(1260*std::pow(z, 4) + 4*std::pow(z, 3)*(630*x + 630*y - 630) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 2) + 420*z*std::pow(x + y - 1, 3) + 30*std::pow(x + y - 1, 4))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 159
template<>
struct DGBasis<159> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (17*x*(9*x - 2) + 1)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (306*x - 34)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)) + (17*x*(9*x - 2) + 1)*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)),
            (17*x*(9*x - 2) + 1)*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)),
            (17*x*(9*x - 2) + 1)*(5544*std::pow(z, 5) + 5*std::pow(z, 4)*(2772*x + 2772*y - 2772) + 12600*std::pow(z, 3)*std::pow(x + y - 1, 2) + 5040*std::pow(z, 2)*std::pow(x + y - 1, 3) + 840*z*std::pow(x + y - 1, 4) + 42*std::pow(x + y - 1, 5))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 160
template<>
struct DGBasis<160> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (18*x - 1)*(x + 15*y - 1)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (18*x - 1)*(x + 15*y - 1)*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)) + (18*x - 1)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)) + 18*(x + 15*y - 1)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)),
            (18*x - 1)*(x + 15*y - 1)*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)) + 15*(18*x - 1)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)),
            (18*x - 1)*(x + 15*y - 1)*(5544*std::pow(z, 5) + 5*std::pow(z, 4)*(2772*x + 2772*y - 2772) + 12600*std::pow(z, 3)*std::pow(x + y - 1, 2) + 5040*std::pow(z, 2)*std::pow(x + y - 1, 3) + 840*z*std::pow(x + y - 1, 4) + 42*std::pow(x + y - 1, 5))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 161
template<>
struct DGBasis<161> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (136*std::pow(y, 2) + y*(32*x - 32) + std::pow(x - 1, 2))*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x + 32*y - 2)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)) + (136*std::pow(y, 2) + y*(32*x - 32) + std::pow(x - 1, 2))*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)),
            (32*x + 272*y - 32)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)) + (136*std::pow(y, 2) + y*(32*x - 32) + std::pow(x - 1, 2))*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)),
            (136*std::pow(y, 2) + y*(32*x - 32) + std::pow(x - 1, 2))*(5544*std::pow(z, 5) + 5*std::pow(z, 4)*(2772*x + 2772*y - 2772) + 12600*std::pow(z, 3)*std::pow(x + y - 1, 2) + 5040*std::pow(z, 2)*std::pow(x + y - 1, 3) + 840*z*std::pow(x + y - 1, 4) + 42*std::pow(x + y - 1, 5))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 162
template<>
struct DGBasis<162> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (18*x - 1)*(std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            18*std::pow(x, 7) + 126*std::pow(x, 6)*y - 126*std::pow(x, 6) + 378*std::pow(x, 5)*std::pow(y, 2) - 756*std::pow(x, 5)*y + 378*std::pow(x, 5) + 630*std::pow(x, 4)*std::pow(y, 3) - 1890*std::pow(x, 4)*std::pow(y, 2) + 1890*std::pow(x, 4)*y - 630*std::pow(x, 4) + 630*std::pow(x, 3)*std::pow(y, 4) - 2520*std::pow(x, 3)*std::pow(y, 3) + 3780*std::pow(x, 3)*std::pow(y, 2) - 2520*std::pow(x, 3)*y + 630*std::pow(x, 3) + 378*std::pow(x, 2)*std::pow(y, 5) - 1890*std::pow(x, 2)*std::pow(y, 4) + 3780*std::pow(x, 2)*std::pow(y, 3) - 3780*std::pow(x, 2)*std::pow(y, 2) + 1890*std::pow(x, 2)*y - 378*std::pow(x, 2) + 126*x*std::pow(y, 6) - 756*x*std::pow(y, 5) + 1890*x*std::pow(y, 4) - 2520*x*std::pow(y, 3) + 1890*x*std::pow(y, 2) - 756*x*y + 126*x + 18*std::pow(y, 7) - 126*std::pow(y, 6) + 378*std::pow(y, 5) - 630*std::pow(y, 4) + 630*std::pow(y, 3) - 378*std::pow(y, 2) + 126*y + 61776*std::pow(z, 7) + 18*std::pow(z, 6)*(12012*x + 12012*y - 12012) + 299376*std::pow(z, 5)*std::pow(x + y - 1, 2) + 207900*std::pow(z, 4)*std::pow(x + y - 1, 3) + 75600*std::pow(z, 3)*std::pow(x + y - 1, 4) + 13608*std::pow(z, 2)*std::pow(x + y - 1, 5) + 1008*z*std::pow(x + y - 1, 6) + (18*x - 1)*(7*std::pow(x, 6) + 42*std::pow(x, 5)*y - 42*std::pow(x, 5) + 105*std::pow(x, 4)*std::pow(y, 2) - 210*std::pow(x, 4)*y + 105*std::pow(x, 4) + 140*std::pow(x, 3)*std::pow(y, 3) - 420*std::pow(x, 3)*std::pow(y, 2) + 420*std::pow(x, 3)*y - 140*std::pow(x, 3) + 105*std::pow(x, 2)*std::pow(y, 4) - 420*std::pow(x, 2)*std::pow(y, 3) + 630*std::pow(x, 2)*std::pow(y, 2) - 420*std::pow(x, 2)*y + 105*std::pow(x, 2) + 42*x*std::pow(y, 5) - 210*x*std::pow(y, 4) + 420*x*std::pow(y, 3) - 420*x*std::pow(y, 2) + 210*x*y - 42*x + 7*std::pow(y, 6) - 42*std::pow(y, 5) + 105*std::pow(y, 4) - 140*std::pow(y, 3) + 105*std::pow(y, 2) - 42*y + 12012*std::pow(z, 6) + 16632*std::pow(z, 5)*(2*x + 2*y - 2) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 2) + 16800*std::pow(z, 3)*std::pow(x + y - 1, 3) + 3780*std::pow(z, 2)*std::pow(x + y - 1, 4) + 336*z*std::pow(x + y - 1, 5) + 7) - 18,
            (18*x - 1)*(7*std::pow(x, 6) + 42*std::pow(x, 5)*y - 42*std::pow(x, 5) + 105*std::pow(x, 4)*std::pow(y, 2) - 210*std::pow(x, 4)*y + 105*std::pow(x, 4) + 140*std::pow(x, 3)*std::pow(y, 3) - 420*std::pow(x, 3)*std::pow(y, 2) + 420*std::pow(x, 3)*y - 140*std::pow(x, 3) + 105*std::pow(x, 2)*std::pow(y, 4) - 420*std::pow(x, 2)*std::pow(y, 3) + 630*std::pow(x, 2)*std::pow(y, 2) - 420*std::pow(x, 2)*y + 105*std::pow(x, 2) + 42*x*std::pow(y, 5) - 210*x*std::pow(y, 4) + 420*x*std::pow(y, 3) - 420*x*std::pow(y, 2) + 210*x*y - 42*x + 7*std::pow(y, 6) - 42*std::pow(y, 5) + 105*std::pow(y, 4) - 140*std::pow(y, 3) + 105*std::pow(y, 2) - 42*y + 12012*std::pow(z, 6) + 16632*std::pow(z, 5)*(2*x + 2*y - 2) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 2) + 16800*std::pow(z, 3)*std::pow(x + y - 1, 3) + 3780*std::pow(z, 2)*std::pow(x + y - 1, 4) + 336*z*std::pow(x + y - 1, 5) + 7),
            (18*x - 1)*(24024*std::pow(z, 6) + 6*std::pow(z, 5)*(12012*x + 12012*y - 12012) + 83160*std::pow(z, 4)*std::pow(x + y - 1, 2) + 46200*std::pow(z, 3)*std::pow(x + y - 1, 3) + 12600*std::pow(z, 2)*std::pow(x + y - 1, 4) + 1512*z*std::pow(x + y - 1, 5) + 56*std::pow(x + y - 1, 6))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 163
template<>
struct DGBasis<163> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x + 17*y - 1)*(std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) + (x + 17*y - 1)*(7*std::pow(x, 6) + 42*std::pow(x, 5)*y - 42*std::pow(x, 5) + 105*std::pow(x, 4)*std::pow(y, 2) - 210*std::pow(x, 4)*y + 105*std::pow(x, 4) + 140*std::pow(x, 3)*std::pow(y, 3) - 420*std::pow(x, 3)*std::pow(y, 2) + 420*std::pow(x, 3)*y - 140*std::pow(x, 3) + 105*std::pow(x, 2)*std::pow(y, 4) - 420*std::pow(x, 2)*std::pow(y, 3) + 630*std::pow(x, 2)*std::pow(y, 2) - 420*std::pow(x, 2)*y + 105*std::pow(x, 2) + 42*x*std::pow(y, 5) - 210*x*std::pow(y, 4) + 420*x*std::pow(y, 3) - 420*x*std::pow(y, 2) + 210*x*y - 42*x + 7*std::pow(y, 6) - 42*std::pow(y, 5) + 105*std::pow(y, 4) - 140*std::pow(y, 3) + 105*std::pow(y, 2) - 42*y + 12012*std::pow(z, 6) + 16632*std::pow(z, 5)*(2*x + 2*y - 2) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 2) + 16800*std::pow(z, 3)*std::pow(x + y - 1, 3) + 3780*std::pow(z, 2)*std::pow(x + y - 1, 4) + 336*z*std::pow(x + y - 1, 5) + 7) - 1,
            17*std::pow(x, 7) + 119*std::pow(x, 6)*y - 119*std::pow(x, 6) + 357*std::pow(x, 5)*std::pow(y, 2) - 714*std::pow(x, 5)*y + 357*std::pow(x, 5) + 595*std::pow(x, 4)*std::pow(y, 3) - 1785*std::pow(x, 4)*std::pow(y, 2) + 1785*std::pow(x, 4)*y - 595*std::pow(x, 4) + 595*std::pow(x, 3)*std::pow(y, 4) - 2380*std::pow(x, 3)*std::pow(y, 3) + 3570*std::pow(x, 3)*std::pow(y, 2) - 2380*std::pow(x, 3)*y + 595*std::pow(x, 3) + 357*std::pow(x, 2)*std::pow(y, 5) - 1785*std::pow(x, 2)*std::pow(y, 4) + 3570*std::pow(x, 2)*std::pow(y, 3) - 3570*std::pow(x, 2)*std::pow(y, 2) + 1785*std::pow(x, 2)*y - 357*std::pow(x, 2) + 119*x*std::pow(y, 6) - 714*x*std::pow(y, 5) + 1785*x*std::pow(y, 4) - 2380*x*std::pow(y, 3) + 1785*x*std::pow(y, 2) - 714*x*y + 119*x + 17*std::pow(y, 7) - 119*std::pow(y, 6) + 357*std::pow(y, 5) - 595*std::pow(y, 4) + 595*std::pow(y, 3) - 357*std::pow(y, 2) + 119*y + 58344*std::pow(z, 7) + 17*std::pow(z, 6)*(12012*x + 12012*y - 12012) + 282744*std::pow(z, 5)*std::pow(x + y - 1, 2) + 196350*std::pow(z, 4)*std::pow(x + y - 1, 3) + 71400*std::pow(z, 3)*std::pow(x + y - 1, 4) + 12852*std::pow(z, 2)*std::pow(x + y - 1, 5) + 952*z*std::pow(x + y - 1, 6) + (x + 17*y - 1)*(7*std::pow(x, 6) + 42*std::pow(x, 5)*y - 42*std::pow(x, 5) + 105*std::pow(x, 4)*std::pow(y, 2) - 210*std::pow(x, 4)*y + 105*std::pow(x, 4) + 140*std::pow(x, 3)*std::pow(y, 3) - 420*std::pow(x, 3)*std::pow(y, 2) + 420*std::pow(x, 3)*y - 140*std::pow(x, 3) + 105*std::pow(x, 2)*std::pow(y, 4) - 420*std::pow(x, 2)*std::pow(y, 3) + 630*std::pow(x, 2)*std::pow(y, 2) - 420*std::pow(x, 2)*y + 105*std::pow(x, 2) + 42*x*std::pow(y, 5) - 210*x*std::pow(y, 4) + 420*x*std::pow(y, 3) - 420*x*std::pow(y, 2) + 210*x*y - 42*x + 7*std::pow(y, 6) - 42*std::pow(y, 5) + 105*std::pow(y, 4) - 140*std::pow(y, 3) + 105*std::pow(y, 2) - 42*y + 12012*std::pow(z, 6) + 16632*std::pow(z, 5)*(2*x + 2*y - 2) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 2) + 16800*std::pow(z, 3)*std::pow(x + y - 1, 3) + 3780*std::pow(z, 2)*std::pow(x + y - 1, 4) + 336*z*std::pow(x + y - 1, 5) + 7) - 17,
            (x + 17*y - 1)*(24024*std::pow(z, 6) + 6*std::pow(z, 5)*(12012*x + 12012*y - 12012) + 83160*std::pow(z, 4)*std::pow(x + y - 1, 2) + 46200*std::pow(z, 3)*std::pow(x + y - 1, 3) + 12600*std::pow(z, 2)*std::pow(x + y - 1, 4) + 1512*z*std::pow(x + y - 1, 5) + 56*std::pow(x + y - 1, 6))
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 164
template<>
struct DGBasis<164> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return std::pow(x, 8) + 8*std::pow(x, 7)*y - 8*std::pow(x, 7) + 28*std::pow(x, 6)*std::pow(y, 2) - 56*std::pow(x, 6)*y + 28*std::pow(x, 6) + 56*std::pow(x, 5)*std::pow(y, 3) - 168*std::pow(x, 5)*std::pow(y, 2) + 168*std::pow(x, 5)*y - 56*std::pow(x, 5) + 70*std::pow(x, 4)*std::pow(y, 4) - 280*std::pow(x, 4)*std::pow(y, 3) + 420*std::pow(x, 4)*std::pow(y, 2) - 280*std::pow(x, 4)*y + 70*std::pow(x, 4) + 56*std::pow(x, 3)*std::pow(y, 5) - 280*std::pow(x, 3)*std::pow(y, 4) + 560*std::pow(x, 3)*std::pow(y, 3) - 560*std::pow(x, 3)*std::pow(y, 2) + 280*std::pow(x, 3)*y - 56*std::pow(x, 3) + 28*std::pow(x, 2)*std::pow(y, 6) - 168*std::pow(x, 2)*std::pow(y, 5) + 420*std::pow(x, 2)*std::pow(y, 4) - 560*std::pow(x, 2)*std::pow(y, 3) + 420*std::pow(x, 2)*std::pow(y, 2) - 168*std::pow(x, 2)*y + 28*std::pow(x, 2) + 8*x*std::pow(y, 7) - 56*x*std::pow(y, 6) + 168*x*std::pow(y, 5) - 280*x*std::pow(y, 4) + 280*x*std::pow(y, 3) - 168*x*std::pow(y, 2) + 56*x*y - 8*x + std::pow(y, 8) - 8*std::pow(y, 7) + 28*std::pow(y, 6) - 56*std::pow(y, 5) + 70*std::pow(y, 4) - 56*std::pow(y, 3) + 28*std::pow(y, 2) - 8*y + 12870*std::pow(z, 8) + std::pow(z, 7)*(51480*x + 51480*y - 51480) + 84084*std::pow(z, 6)*std::pow(x + y - 1, 2) + 72072*std::pow(z, 5)*std::pow(x + y - 1, 3) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 4) + 9240*std::pow(z, 3)*std::pow(x + y - 1, 5) + 1260*std::pow(z, 2)*std::pow(x + y - 1, 6) + 72*z*std::pow(x + y - 1, 7) + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            8*std::pow(x, 7) + 56*std::pow(x, 6)*y - 56*std::pow(x, 6) + 168*std::pow(x, 5)*std::pow(y, 2) - 336*std::pow(x, 5)*y + 168*std::pow(x, 5) + 280*std::pow(x, 4)*std::pow(y, 3) - 840*std::pow(x, 4)*std::pow(y, 2) + 840*std::pow(x, 4)*y - 280*std::pow(x, 4) + 280*std::pow(x, 3)*std::pow(y, 4) - 1120*std::pow(x, 3)*std::pow(y, 3) + 1680*std::pow(x, 3)*std::pow(y, 2) - 1120*std::pow(x, 3)*y + 280*std::pow(x, 3) + 168*std::pow(x, 2)*std::pow(y, 5) - 840*std::pow(x, 2)*std::pow(y, 4) + 1680*std::pow(x, 2)*std::pow(y, 3) - 1680*std::pow(x, 2)*std::pow(y, 2) + 840*std::pow(x, 2)*y - 168*std::pow(x, 2) + 56*x*std::pow(y, 6) - 336*x*std::pow(y, 5) + 840*x*std::pow(y, 4) - 1120*x*std::pow(y, 3) + 840*x*std::pow(y, 2) - 336*x*y + 56*x + 8*std::pow(y, 7) - 56*std::pow(y, 6) + 168*std::pow(y, 5) - 280*std::pow(y, 4) + 280*std::pow(y, 3) - 168*std::pow(y, 2) + 56*y + 51480*std::pow(z, 7) + 84084*std::pow(z, 6)*(2*x + 2*y - 2) + 216216*std::pow(z, 5)*std::pow(x + y - 1, 2) + 138600*std::pow(z, 4)*std::pow(x + y - 1, 3) + 46200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 7560*std::pow(z, 2)*std::pow(x + y - 1, 5) + 504*z*std::pow(x + y - 1, 6) - 8,
            8*std::pow(x, 7) + 56*std::pow(x, 6)*y - 56*std::pow(x, 6) + 168*std::pow(x, 5)*std::pow(y, 2) - 336*std::pow(x, 5)*y + 168*std::pow(x, 5) + 280*std::pow(x, 4)*std::pow(y, 3) - 840*std::pow(x, 4)*std::pow(y, 2) + 840*std::pow(x, 4)*y - 280*std::pow(x, 4) + 280*std::pow(x, 3)*std::pow(y, 4) - 1120*std::pow(x, 3)*std::pow(y, 3) + 1680*std::pow(x, 3)*std::pow(y, 2) - 1120*std::pow(x, 3)*y + 280*std::pow(x, 3) + 168*std::pow(x, 2)*std::pow(y, 5) - 840*std::pow(x, 2)*std::pow(y, 4) + 1680*std::pow(x, 2)*std::pow(y, 3) - 1680*std::pow(x, 2)*std::pow(y, 2) + 840*std::pow(x, 2)*y - 168*std::pow(x, 2) + 56*x*std::pow(y, 6) - 336*x*std::pow(y, 5) + 840*x*std::pow(y, 4) - 1120*x*std::pow(y, 3) + 840*x*std::pow(y, 2) - 336*x*y + 56*x + 8*std::pow(y, 7) - 56*std::pow(y, 6) + 168*std::pow(y, 5) - 280*std::pow(y, 4) + 280*std::pow(y, 3) - 168*std::pow(y, 2) + 56*y + 51480*std::pow(z, 7) + 84084*std::pow(z, 6)*(2*x + 2*y - 2) + 216216*std::pow(z, 5)*std::pow(x + y - 1, 2) + 138600*std::pow(z, 4)*std::pow(x + y - 1, 3) + 46200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 7560*std::pow(z, 2)*std::pow(x + y - 1, 5) + 504*z*std::pow(x + y - 1, 6) - 8,
            102960*std::pow(z, 7) + 7*std::pow(z, 6)*(51480*x + 51480*y - 51480) + 504504*std::pow(z, 5)*std::pow(x + y - 1, 2) + 360360*std::pow(z, 4)*std::pow(x + y - 1, 3) + 138600*std::pow(z, 3)*std::pow(x + y - 1, 4) + 27720*std::pow(z, 2)*std::pow(x + y - 1, 5) + 2520*z*std::pow(x + y - 1, 6) + 72*std::pow(x + y - 1, 7)
        };
    }
    static constexpr uInt Order = 8;
};

// Basis 165
template<>
struct DGBasis<165> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return 2*x*(13*x*(x*(x*(x*(17*x*(x*(19*x*(20*x - 81) + 2592) - 2352) + 21168) - 6615) + 1176) - 108) + 54) - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (26*x*(x*(x*(x*(17*x*(3*x*(19*x*(5*x - 16) + 392) - 784) + 4900) - 980) + 98) - 4) + 1)*(x + 3*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x*(x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) - 2205) + 98) - 1)*(10*std::pow(y, 2) + y*(8*x - 8) + std::pow(x - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (x*(x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) - 2205) + 98) - 1)*(2*x + 8*y - 2) + (10*std::pow(y, 2) + y*(8*x - 8) + std::pow(x - 1, 2))*(x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) - 2205) + x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) + x*(68*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4*x*(51*x*(19*x*(20*x - 49) + 882) + 17*x*(57*x*(20*x - 49) + 3*x*(760*x - 931) + 2646) - 20825) + 19600) - 2205) + 98),
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(35*std::pow(y, 3) + std::pow(y, 2)*(45*x - 45) + 15*y*std::pow(x - 1, 2) + std::pow(x - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(45*std::pow(y, 2) + 15*y*(2*x - 2) + 3*std::pow(x - 1, 2)) + (4*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) + 2*x*(34*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 2*x*(51*x*(38*x*(5*x - 9) + 225) + 17*x*(114*x*(5*x - 9) + 3*x*(380*x - 342) + 675) - 3400) + 900) - 90)*(35*std::pow(y, 3) + std::pow(y, 2)*(45*x - 45) + 15*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(105*std::pow(y, 2) + 2*y*(45*x - 45) + 15*std::pow(x - 1, 2)),
            0
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 169
template<>
struct DGBasis<169> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(126*std::pow(y, 4) + std::pow(y, 3)*(224*x - 224) + 126*std::pow(y, 2)*std::pow(x - 1, 2) + 24*y*std::pow(x - 1, 3) + std::pow(x - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(224*std::pow(y, 3) + 126*std::pow(y, 2)*(2*x - 2) + 72*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3)) + (68*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 4*x*(51*x*(19*x*(4*x - 5) + 40) + 17*x*(57*x*(4*x - 5) + 3*x*(152*x - 95) + 120) - 340) + 80)*(126*std::pow(y, 4) + std::pow(y, 3)*(224*x - 224) + 126*std::pow(y, 2)*std::pow(x - 1, 2) + 24*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(504*std::pow(y, 3) + 3*std::pow(y, 2)*(224*x - 224) + 252*y*std::pow(x - 1, 2) + 24*std::pow(x - 1, 3)),
            0
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 170
template<>
struct DGBasis<170> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(462*std::pow(y, 5) + std::pow(y, 4)*(1050*x - 1050) + 840*std::pow(y, 3)*std::pow(x - 1, 2) + 280*std::pow(y, 2)*std::pow(x - 1, 3) + 35*y*std::pow(x - 1, 4) + std::pow(x - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(1050*std::pow(y, 4) + 840*std::pow(y, 3)*(2*x - 2) + 840*std::pow(y, 2)*std::pow(x - 1, 2) + 140*y*std::pow(x - 1, 3) + 5*std::pow(x - 1, 4)) + (51*x*(19*x*(5*x - 4) + 18) + 17*x*(57*x*(5*x - 4) + 3*x*(190*x - 76) + 54) - 68)*(462*std::pow(y, 5) + std::pow(y, 4)*(1050*x - 1050) + 840*std::pow(y, 3)*std::pow(x - 1, 2) + 280*std::pow(y, 2)*std::pow(x - 1, 3) + 35*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)),
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(2310*std::pow(y, 4) + 4*std::pow(y, 3)*(1050*x - 1050) + 2520*std::pow(y, 2)*std::pow(x - 1, 2) + 560*y*std::pow(x - 1, 3) + 35*std::pow(x - 1, 4)),
            0
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 171
template<>
struct DGBasis<171> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(20*x - 9) + 18) - 1)*(1716*std::pow(y, 6) + std::pow(y, 5)*(4752*x - 4752) + 4950*std::pow(y, 4)*std::pow(x - 1, 2) + 2400*std::pow(y, 3)*std::pow(x - 1, 3) + 540*std::pow(y, 2)*std::pow(x - 1, 4) + 48*y*std::pow(x - 1, 5) + std::pow(x - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(4752*std::pow(y, 5) + 4950*std::pow(y, 4)*(2*x - 2) + 7200*std::pow(y, 3)*std::pow(x - 1, 2) + 2160*std::pow(y, 2)*std::pow(x - 1, 3) + 240*y*std::pow(x - 1, 4) + 6*std::pow(x - 1, 5)) + (57*x*(20*x - 9) + 3*x*(760*x - 171) + 54)*(1716*std::pow(y, 6) + std::pow(y, 5)*(4752*x - 4752) + 4950*std::pow(y, 4)*std::pow(x - 1, 2) + 2400*std::pow(y, 3)*std::pow(x - 1, 3) + 540*std::pow(y, 2)*std::pow(x - 1, 4) + 48*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)),
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(10296*std::pow(y, 5) + 5*std::pow(y, 4)*(4752*x - 4752) + 19800*std::pow(y, 3)*std::pow(x - 1, 2) + 7200*std::pow(y, 2)*std::pow(x - 1, 3) + 1080*y*std::pow(x - 1, 4) + 48*std::pow(x - 1, 5)),
            0
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 172
template<>
struct DGBasis<172> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (38*x*(5*x - 1) + 1)*(6435*std::pow(y, 7) + std::pow(y, 6)*(21021*x - 21021) + 27027*std::pow(y, 5)*std::pow(x - 1, 2) + 17325*std::pow(y, 4)*std::pow(x - 1, 3) + 5775*std::pow(y, 3)*std::pow(x - 1, 4) + 945*std::pow(y, 2)*std::pow(x - 1, 5) + 63*y*std::pow(x - 1, 6) + std::pow(x - 1, 7));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (380*x - 38)*(6435*std::pow(y, 7) + std::pow(y, 6)*(21021*x - 21021) + 27027*std::pow(y, 5)*std::pow(x - 1, 2) + 17325*std::pow(y, 4)*std::pow(x - 1, 3) + 5775*std::pow(y, 3)*std::pow(x - 1, 4) + 945*std::pow(y, 2)*std::pow(x - 1, 5) + 63*y*std::pow(x - 1, 6) + std::pow(x - 1, 7)) + (38*x*(5*x - 1) + 1)*(21021*std::pow(y, 6) + 27027*std::pow(y, 5)*(2*x - 2) + 51975*std::pow(y, 4)*std::pow(x - 1, 2) + 23100*std::pow(y, 3)*std::pow(x - 1, 3) + 4725*std::pow(y, 2)*std::pow(x - 1, 4) + 378*y*std::pow(x - 1, 5) + 7*std::pow(x - 1, 6)),
            (38*x*(5*x - 1) + 1)*(45045*std::pow(y, 6) + 6*std::pow(y, 5)*(21021*x - 21021) + 135135*std::pow(y, 4)*std::pow(x - 1, 2) + 69300*std::pow(y, 3)*std::pow(x - 1, 3) + 17325*std::pow(y, 2)*std::pow(x - 1, 4) + 1890*y*std::pow(x - 1, 5) + 63*std::pow(x - 1, 6)),
            0
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 173
template<>
struct DGBasis<173> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (20*x - 1)*(24310*std::pow(y, 8) + std::pow(y, 7)*(91520*x - 91520) + 140140*std::pow(y, 6)*std::pow(x - 1, 2) + 112112*std::pow(y, 5)*std::pow(x - 1, 3) + 50050*std::pow(y, 4)*std::pow(x - 1, 4) + 12320*std::pow(y, 3)*std::pow(x - 1, 5) + 1540*std::pow(y, 2)*std::pow(x - 1, 6) + 80*y*std::pow(x - 1, 7) + std::pow(x - 1, 8));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            486200*std::pow(y, 8) + 20*std::pow(y, 7)*(91520*x - 91520) + 2802800*std::pow(y, 6)*std::pow(x - 1, 2) + 2242240*std::pow(y, 5)*std::pow(x - 1, 3) + 1001000*std::pow(y, 4)*std::pow(x - 1, 4) + 246400*std::pow(y, 3)*std::pow(x - 1, 5) + 30800*std::pow(y, 2)*std::pow(x - 1, 6) + 1600*y*std::pow(x - 1, 7) + 20*std::pow(x - 1, 8) + (20*x - 1)*(91520*std::pow(y, 7) + 140140*std::pow(y, 6)*(2*x - 2) + 336336*std::pow(y, 5)*std::pow(x - 1, 2) + 200200*std::pow(y, 4)*std::pow(x - 1, 3) + 61600*std::pow(y, 3)*std::pow(x - 1, 4) + 9240*std::pow(y, 2)*std::pow(x - 1, 5) + 560*y*std::pow(x - 1, 6) + 8*std::pow(x - 1, 7)),
            (20*x - 1)*(194480*std::pow(y, 7) + 7*std::pow(y, 6)*(91520*x - 91520) + 840840*std::pow(y, 5)*std::pow(x - 1, 2) + 560560*std::pow(y, 4)*std::pow(x - 1, 3) + 200200*std::pow(y, 3)*std::pow(x - 1, 4) + 36960*std::pow(y, 2)*std::pow(x - 1, 5) + 3080*y*std::pow(x - 1, 6) + 80*std::pow(x - 1, 7)),
            0
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 174
template<>
struct DGBasis<174> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return 92378*std::pow(y, 9) + std::pow(y, 8)*(393822*x - 393822) + 700128*std::pow(y, 7)*std::pow(x - 1, 2) + 672672*std::pow(y, 6)*std::pow(x - 1, 3) + 378378*std::pow(y, 5)*std::pow(x - 1, 4) + 126126*std::pow(y, 4)*std::pow(x - 1, 5) + 24024*std::pow(y, 3)*std::pow(x - 1, 6) + 2376*std::pow(y, 2)*std::pow(x - 1, 7) + 99*y*std::pow(x - 1, 8) + std::pow(x - 1, 9);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            393822*std::pow(y, 8) + 700128*std::pow(y, 7)*(2*x - 2) + 2018016*std::pow(y, 6)*std::pow(x - 1, 2) + 1513512*std::pow(y, 5)*std::pow(x - 1, 3) + 630630*std::pow(y, 4)*std::pow(x - 1, 4) + 144144*std::pow(y, 3)*std::pow(x - 1, 5) + 16632*std::pow(y, 2)*std::pow(x - 1, 6) + 792*y*std::pow(x - 1, 7) + 9*std::pow(x - 1, 8),
            831402*std::pow(y, 8) + 8*std::pow(y, 7)*(393822*x - 393822) + 4900896*std::pow(y, 6)*std::pow(x - 1, 2) + 4036032*std::pow(y, 5)*std::pow(x - 1, 3) + 1891890*std::pow(y, 4)*std::pow(x - 1, 4) + 504504*std::pow(y, 3)*std::pow(x - 1, 5) + 72072*std::pow(y, 2)*std::pow(x - 1, 6) + 4752*y*std::pow(x - 1, 7) + 99*std::pow(x - 1, 8),
            0
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 175
template<>
struct DGBasis<175> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (26*x*(x*(x*(x*(17*x*(3*x*(19*x*(5*x - 16) + 392) - 784) + 4900) - 980) + 98) - 4) + 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x*(x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) - 2205) + 98) - 1)*(x + 5*y - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2))*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(2*x + 12*y - 2)*(x + y + 2*z - 1) + (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2)) + (21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2))*(4*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) + 2*x*(34*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 2*x*(51*x*(38*x*(5*x - 9) + 225) + 17*x*(114*x*(5*x - 9) + 3*x*(380*x - 342) + 675) - 3400) + 900) - 90)*(x + y + 2*z - 1),
            (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(12*x + 42*y - 12)*(x + y + 2*z - 1) + (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2)),
            2*(2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 178
template<>
struct DGBasis<178> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(x + y + 2*z - 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(84*std::pow(y, 2) + 21*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(x + y + 2*z - 1) + (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)) + (68*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 4*x*(51*x*(19*x*(4*x - 5) + 40) + 17*x*(57*x*(4*x - 5) + 3*x*(152*x - 95) + 120) - 340) + 80)*(x + y + 2*z - 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(252*std::pow(y, 2) + 2*y*(84*x - 84) + 21*std::pow(x - 1, 2))*(x + y + 2*z - 1) + (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            2*(4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 179
template<>
struct DGBasis<179> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(x + y + 2*z - 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(x + y + 2*z - 1)*(480*std::pow(y, 3) + 216*std::pow(y, 2)*(2*x - 2) + 96*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3)) + (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)) + (51*x*(19*x*(5*x - 4) + 18) + 17*x*(57*x*(5*x - 4) + 3*x*(190*x - 76) + 54) - 68)*(x + y + 2*z - 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(x + y + 2*z - 1)*(1320*std::pow(y, 3) + 3*std::pow(y, 2)*(480*x - 480) + 432*y*std::pow(x - 1, 2) + 32*std::pow(x - 1, 3)) + (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            2*(17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 180
template<>
struct DGBasis<180> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(20*x - 9) + 18) - 1)*(x + y + 2*z - 1)*(1287*std::pow(y, 5) + std::pow(y, 4)*(2475*x - 2475) + 1650*std::pow(y, 3)*std::pow(x - 1, 2) + 450*std::pow(y, 2)*std::pow(x - 1, 3) + 45*y*std::pow(x - 1, 4) + std::pow(x - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(x + y + 2*z - 1)*(2475*std::pow(y, 4) + 1650*std::pow(y, 3)*(2*x - 2) + 1350*std::pow(y, 2)*std::pow(x - 1, 2) + 180*y*std::pow(x - 1, 3) + 5*std::pow(x - 1, 4)) + (3*x*(19*x*(20*x - 9) + 18) - 1)*(1287*std::pow(y, 5) + std::pow(y, 4)*(2475*x - 2475) + 1650*std::pow(y, 3)*std::pow(x - 1, 2) + 450*std::pow(y, 2)*std::pow(x - 1, 3) + 45*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)) + (57*x*(20*x - 9) + 3*x*(760*x - 171) + 54)*(x + y + 2*z - 1)*(1287*std::pow(y, 5) + std::pow(y, 4)*(2475*x - 2475) + 1650*std::pow(y, 3)*std::pow(x - 1, 2) + 450*std::pow(y, 2)*std::pow(x - 1, 3) + 45*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)),
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(x + y + 2*z - 1)*(6435*std::pow(y, 4) + 4*std::pow(y, 3)*(2475*x - 2475) + 4950*std::pow(y, 2)*std::pow(x - 1, 2) + 900*y*std::pow(x - 1, 3) + 45*std::pow(x - 1, 4)) + (3*x*(19*x*(20*x - 9) + 18) - 1)*(1287*std::pow(y, 5) + std::pow(y, 4)*(2475*x - 2475) + 1650*std::pow(y, 3)*std::pow(x - 1, 2) + 450*std::pow(y, 2)*std::pow(x - 1, 3) + 45*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)),
            2*(3*x*(19*x*(20*x - 9) + 18) - 1)*(1287*std::pow(y, 5) + std::pow(y, 4)*(2475*x - 2475) + 1650*std::pow(y, 3)*std::pow(x - 1, 2) + 450*std::pow(y, 2)*std::pow(x - 1, 3) + 45*y*std::pow(x - 1, 4) + std::pow(x - 1, 5))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 181
template<>
struct DGBasis<181> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (38*x*(5*x - 1) + 1)*(x + y + 2*z - 1)*(5005*std::pow(y, 6) + std::pow(y, 5)*(12012*x - 12012) + 10725*std::pow(y, 4)*std::pow(x - 1, 2) + 4400*std::pow(y, 3)*std::pow(x - 1, 3) + 825*std::pow(y, 2)*std::pow(x - 1, 4) + 60*y*std::pow(x - 1, 5) + std::pow(x - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (380*x - 38)*(x + y + 2*z - 1)*(5005*std::pow(y, 6) + std::pow(y, 5)*(12012*x - 12012) + 10725*std::pow(y, 4)*std::pow(x - 1, 2) + 4400*std::pow(y, 3)*std::pow(x - 1, 3) + 825*std::pow(y, 2)*std::pow(x - 1, 4) + 60*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)) + (38*x*(5*x - 1) + 1)*(x + y + 2*z - 1)*(12012*std::pow(y, 5) + 10725*std::pow(y, 4)*(2*x - 2) + 13200*std::pow(y, 3)*std::pow(x - 1, 2) + 3300*std::pow(y, 2)*std::pow(x - 1, 3) + 300*y*std::pow(x - 1, 4) + 6*std::pow(x - 1, 5)) + (38*x*(5*x - 1) + 1)*(5005*std::pow(y, 6) + std::pow(y, 5)*(12012*x - 12012) + 10725*std::pow(y, 4)*std::pow(x - 1, 2) + 4400*std::pow(y, 3)*std::pow(x - 1, 3) + 825*std::pow(y, 2)*std::pow(x - 1, 4) + 60*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)),
            (38*x*(5*x - 1) + 1)*(x + y + 2*z - 1)*(30030*std::pow(y, 5) + 5*std::pow(y, 4)*(12012*x - 12012) + 42900*std::pow(y, 3)*std::pow(x - 1, 2) + 13200*std::pow(y, 2)*std::pow(x - 1, 3) + 1650*y*std::pow(x - 1, 4) + 60*std::pow(x - 1, 5)) + (38*x*(5*x - 1) + 1)*(5005*std::pow(y, 6) + std::pow(y, 5)*(12012*x - 12012) + 10725*std::pow(y, 4)*std::pow(x - 1, 2) + 4400*std::pow(y, 3)*std::pow(x - 1, 3) + 825*std::pow(y, 2)*std::pow(x - 1, 4) + 60*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)),
            2*(38*x*(5*x - 1) + 1)*(5005*std::pow(y, 6) + std::pow(y, 5)*(12012*x - 12012) + 10725*std::pow(y, 4)*std::pow(x - 1, 2) + 4400*std::pow(y, 3)*std::pow(x - 1, 3) + 825*std::pow(y, 2)*std::pow(x - 1, 4) + 60*y*std::pow(x - 1, 5) + std::pow(x - 1, 6))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 182
template<>
struct DGBasis<182> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (20*x - 1)*(x + y + 2*z - 1)*(19448*std::pow(y, 7) + std::pow(y, 6)*(56056*x - 56056) + 63063*std::pow(y, 5)*std::pow(x - 1, 2) + 35035*std::pow(y, 4)*std::pow(x - 1, 3) + 10010*std::pow(y, 3)*std::pow(x - 1, 4) + 1386*std::pow(y, 2)*std::pow(x - 1, 5) + 77*y*std::pow(x - 1, 6) + std::pow(x - 1, 7));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (20*x - 1)*(x + y + 2*z - 1)*(56056*std::pow(y, 6) + 63063*std::pow(y, 5)*(2*x - 2) + 105105*std::pow(y, 4)*std::pow(x - 1, 2) + 40040*std::pow(y, 3)*std::pow(x - 1, 3) + 6930*std::pow(y, 2)*std::pow(x - 1, 4) + 462*y*std::pow(x - 1, 5) + 7*std::pow(x - 1, 6)) + (20*x - 1)*(19448*std::pow(y, 7) + std::pow(y, 6)*(56056*x - 56056) + 63063*std::pow(y, 5)*std::pow(x - 1, 2) + 35035*std::pow(y, 4)*std::pow(x - 1, 3) + 10010*std::pow(y, 3)*std::pow(x - 1, 4) + 1386*std::pow(y, 2)*std::pow(x - 1, 5) + 77*y*std::pow(x - 1, 6) + std::pow(x - 1, 7)) + 20*(x + y + 2*z - 1)*(19448*std::pow(y, 7) + std::pow(y, 6)*(56056*x - 56056) + 63063*std::pow(y, 5)*std::pow(x - 1, 2) + 35035*std::pow(y, 4)*std::pow(x - 1, 3) + 10010*std::pow(y, 3)*std::pow(x - 1, 4) + 1386*std::pow(y, 2)*std::pow(x - 1, 5) + 77*y*std::pow(x - 1, 6) + std::pow(x - 1, 7)),
            (20*x - 1)*(x + y + 2*z - 1)*(136136*std::pow(y, 6) + 6*std::pow(y, 5)*(56056*x - 56056) + 315315*std::pow(y, 4)*std::pow(x - 1, 2) + 140140*std::pow(y, 3)*std::pow(x - 1, 3) + 30030*std::pow(y, 2)*std::pow(x - 1, 4) + 2772*y*std::pow(x - 1, 5) + 77*std::pow(x - 1, 6)) + (20*x - 1)*(19448*std::pow(y, 7) + std::pow(y, 6)*(56056*x - 56056) + 63063*std::pow(y, 5)*std::pow(x - 1, 2) + 35035*std::pow(y, 4)*std::pow(x - 1, 3) + 10010*std::pow(y, 3)*std::pow(x - 1, 4) + 1386*std::pow(y, 2)*std::pow(x - 1, 5) + 77*y*std::pow(x - 1, 6) + std::pow(x - 1, 7)),
            2*(20*x - 1)*(19448*std::pow(y, 7) + std::pow(y, 6)*(56056*x - 56056) + 63063*std::pow(y, 5)*std::pow(x - 1, 2) + 35035*std::pow(y, 4)*std::pow(x - 1, 3) + 10010*std::pow(y, 3)*std::pow(x - 1, 4) + 1386*std::pow(y, 2)*std::pow(x - 1, 5) + 77*y*std::pow(x - 1, 6) + std::pow(x - 1, 7))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 183
template<>
struct DGBasis<183> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x + y + 2*z - 1)*(75582*std::pow(y, 8) + std::pow(y, 7)*(254592*x - 254592) + 346528*std::pow(y, 6)*std::pow(x - 1, 2) + 244608*std::pow(y, 5)*std::pow(x - 1, 3) + 95550*std::pow(y, 4)*std::pow(x - 1, 4) + 20384*std::pow(y, 3)*std::pow(x - 1, 5) + 2184*std::pow(y, 2)*std::pow(x - 1, 6) + 96*y*std::pow(x - 1, 7) + std::pow(x - 1, 8));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            75582*std::pow(y, 8) + std::pow(y, 7)*(254592*x - 254592) + 346528*std::pow(y, 6)*std::pow(x - 1, 2) + 244608*std::pow(y, 5)*std::pow(x - 1, 3) + 95550*std::pow(y, 4)*std::pow(x - 1, 4) + 20384*std::pow(y, 3)*std::pow(x - 1, 5) + 2184*std::pow(y, 2)*std::pow(x - 1, 6) + 96*y*std::pow(x - 1, 7) + std::pow(x - 1, 8) + (x + y + 2*z - 1)*(254592*std::pow(y, 7) + 346528*std::pow(y, 6)*(2*x - 2) + 733824*std::pow(y, 5)*std::pow(x - 1, 2) + 382200*std::pow(y, 4)*std::pow(x - 1, 3) + 101920*std::pow(y, 3)*std::pow(x - 1, 4) + 13104*std::pow(y, 2)*std::pow(x - 1, 5) + 672*y*std::pow(x - 1, 6) + 8*std::pow(x - 1, 7)),
            75582*std::pow(y, 8) + std::pow(y, 7)*(254592*x - 254592) + 346528*std::pow(y, 6)*std::pow(x - 1, 2) + 244608*std::pow(y, 5)*std::pow(x - 1, 3) + 95550*std::pow(y, 4)*std::pow(x - 1, 4) + 20384*std::pow(y, 3)*std::pow(x - 1, 5) + 2184*std::pow(y, 2)*std::pow(x - 1, 6) + 96*y*std::pow(x - 1, 7) + std::pow(x - 1, 8) + (x + y + 2*z - 1)*(604656*std::pow(y, 7) + 7*std::pow(y, 6)*(254592*x - 254592) + 2079168*std::pow(y, 5)*std::pow(x - 1, 2) + 1223040*std::pow(y, 4)*std::pow(x - 1, 3) + 382200*std::pow(y, 3)*std::pow(x - 1, 4) + 61152*std::pow(y, 2)*std::pow(x - 1, 5) + 4368*y*std::pow(x - 1, 6) + 96*std::pow(x - 1, 7)),
            151164*std::pow(y, 8) + 2*std::pow(y, 7)*(254592*x - 254592) + 693056*std::pow(y, 6)*std::pow(x - 1, 2) + 489216*std::pow(y, 5)*std::pow(x - 1, 3) + 191100*std::pow(y, 4)*std::pow(x - 1, 4) + 40768*std::pow(y, 3)*std::pow(x - 1, 5) + 4368*std::pow(y, 2)*std::pow(x - 1, 6) + 192*y*std::pow(x - 1, 7) + 2*std::pow(x - 1, 8)
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 184
template<>
struct DGBasis<184> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x*(x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) - 2205) + 98) - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (x*(x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) - 2205) + 98) - 1)*(2*x + 2*y + 6*z - 2) + (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) - 2205) + x*(4*x*(17*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4900) + x*(68*x*(3*x*(19*x*(20*x - 49) + 882) - 1225) + 4*x*(51*x*(19*x*(20*x - 49) + 882) + 17*x*(57*x*(20*x - 49) + 3*x*(760*x - 931) + 2646) - 20825) + 19600) - 2205) + 98),
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(x + 7*y - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (x + 7*y - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(4*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) + 2*x*(34*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 2*x*(51*x*(38*x*(5*x - 9) + 225) + 17*x*(114*x*(5*x - 9) + 3*x*(380*x - 342) + 675) - 3400) + 900) - 90),
            (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + 7*(2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)),
            (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(x + 7*y - 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 186
template<>
struct DGBasis<186> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(2*x + 16*y - 2)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(2*x + 2*y + 6*z - 2) + (36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(68*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 4*x*(51*x*(19*x*(4*x - 5) + 40) + 17*x*(57*x*(4*x - 5) + 3*x*(152*x - 95) + 120) - 340) + 80),
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(16*x + 72*y - 16)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(2*x + 2*y + 6*z - 2),
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 187
template<>
struct DGBasis<187> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(135*std::pow(y, 2) + 27*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(2*x + 2*y + 6*z - 2)*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)) + (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(51*x*(19*x*(5*x - 4) + 18) + 17*x*(57*x*(5*x - 4) + 3*x*(190*x - 76) + 54) - 68)*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(495*std::pow(y, 2) + 2*y*(135*x - 135) + 27*std::pow(x - 1, 2))*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(2*x + 2*y + 6*z - 2)*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(6*x + 6*y + 12*z - 6)*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 188
template<>
struct DGBasis<188> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(20*x - 9) + 18) - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(715*std::pow(y, 4) + std::pow(y, 3)*(880*x - 880) + 330*std::pow(y, 2)*std::pow(x - 1, 2) + 40*y*std::pow(x - 1, 3) + std::pow(x - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(880*std::pow(y, 3) + 330*std::pow(y, 2)*(2*x - 2) + 120*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3)) + (3*x*(19*x*(20*x - 9) + 18) - 1)*(2*x + 2*y + 6*z - 2)*(715*std::pow(y, 4) + std::pow(y, 3)*(880*x - 880) + 330*std::pow(y, 2)*std::pow(x - 1, 2) + 40*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)) + (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(57*x*(20*x - 9) + 3*x*(760*x - 171) + 54)*(715*std::pow(y, 4) + std::pow(y, 3)*(880*x - 880) + 330*std::pow(y, 2)*std::pow(x - 1, 2) + 40*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(2860*std::pow(y, 3) + 3*std::pow(y, 2)*(880*x - 880) + 660*y*std::pow(x - 1, 2) + 40*std::pow(x - 1, 3)) + (3*x*(19*x*(20*x - 9) + 18) - 1)*(2*x + 2*y + 6*z - 2)*(715*std::pow(y, 4) + std::pow(y, 3)*(880*x - 880) + 330*std::pow(y, 2)*std::pow(x - 1, 2) + 40*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(6*x + 6*y + 12*z - 6)*(715*std::pow(y, 4) + std::pow(y, 3)*(880*x - 880) + 330*std::pow(y, 2)*std::pow(x - 1, 2) + 40*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 189
template<>
struct DGBasis<189> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (38*x*(5*x - 1) + 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(3003*std::pow(y, 5) + std::pow(y, 4)*(5005*x - 5005) + 2860*std::pow(y, 3)*std::pow(x - 1, 2) + 660*std::pow(y, 2)*std::pow(x - 1, 3) + 55*y*std::pow(x - 1, 4) + std::pow(x - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (380*x - 38)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(3003*std::pow(y, 5) + std::pow(y, 4)*(5005*x - 5005) + 2860*std::pow(y, 3)*std::pow(x - 1, 2) + 660*std::pow(y, 2)*std::pow(x - 1, 3) + 55*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)) + (38*x*(5*x - 1) + 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(5005*std::pow(y, 4) + 2860*std::pow(y, 3)*(2*x - 2) + 1980*std::pow(y, 2)*std::pow(x - 1, 2) + 220*y*std::pow(x - 1, 3) + 5*std::pow(x - 1, 4)) + (38*x*(5*x - 1) + 1)*(2*x + 2*y + 6*z - 2)*(3003*std::pow(y, 5) + std::pow(y, 4)*(5005*x - 5005) + 2860*std::pow(y, 3)*std::pow(x - 1, 2) + 660*std::pow(y, 2)*std::pow(x - 1, 3) + 55*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)),
            (38*x*(5*x - 1) + 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(15015*std::pow(y, 4) + 4*std::pow(y, 3)*(5005*x - 5005) + 8580*std::pow(y, 2)*std::pow(x - 1, 2) + 1320*y*std::pow(x - 1, 3) + 55*std::pow(x - 1, 4)) + (38*x*(5*x - 1) + 1)*(2*x + 2*y + 6*z - 2)*(3003*std::pow(y, 5) + std::pow(y, 4)*(5005*x - 5005) + 2860*std::pow(y, 3)*std::pow(x - 1, 2) + 660*std::pow(y, 2)*std::pow(x - 1, 3) + 55*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)),
            (38*x*(5*x - 1) + 1)*(6*x + 6*y + 12*z - 6)*(3003*std::pow(y, 5) + std::pow(y, 4)*(5005*x - 5005) + 2860*std::pow(y, 3)*std::pow(x - 1, 2) + 660*std::pow(y, 2)*std::pow(x - 1, 3) + 55*y*std::pow(x - 1, 4) + std::pow(x - 1, 5))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 190
template<>
struct DGBasis<190> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (20*x - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(12376*std::pow(y, 6) + std::pow(y, 5)*(26208*x - 26208) + 20475*std::pow(y, 4)*std::pow(x - 1, 2) + 7280*std::pow(y, 3)*std::pow(x - 1, 3) + 1170*std::pow(y, 2)*std::pow(x - 1, 4) + 72*y*std::pow(x - 1, 5) + std::pow(x - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (20*x - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(26208*std::pow(y, 5) + 20475*std::pow(y, 4)*(2*x - 2) + 21840*std::pow(y, 3)*std::pow(x - 1, 2) + 4680*std::pow(y, 2)*std::pow(x - 1, 3) + 360*y*std::pow(x - 1, 4) + 6*std::pow(x - 1, 5)) + (20*x - 1)*(2*x + 2*y + 6*z - 2)*(12376*std::pow(y, 6) + std::pow(y, 5)*(26208*x - 26208) + 20475*std::pow(y, 4)*std::pow(x - 1, 2) + 7280*std::pow(y, 3)*std::pow(x - 1, 3) + 1170*std::pow(y, 2)*std::pow(x - 1, 4) + 72*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)) + 20*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(12376*std::pow(y, 6) + std::pow(y, 5)*(26208*x - 26208) + 20475*std::pow(y, 4)*std::pow(x - 1, 2) + 7280*std::pow(y, 3)*std::pow(x - 1, 3) + 1170*std::pow(y, 2)*std::pow(x - 1, 4) + 72*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)),
            (20*x - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(74256*std::pow(y, 5) + 5*std::pow(y, 4)*(26208*x - 26208) + 81900*std::pow(y, 3)*std::pow(x - 1, 2) + 21840*std::pow(y, 2)*std::pow(x - 1, 3) + 2340*y*std::pow(x - 1, 4) + 72*std::pow(x - 1, 5)) + (20*x - 1)*(2*x + 2*y + 6*z - 2)*(12376*std::pow(y, 6) + std::pow(y, 5)*(26208*x - 26208) + 20475*std::pow(y, 4)*std::pow(x - 1, 2) + 7280*std::pow(y, 3)*std::pow(x - 1, 3) + 1170*std::pow(y, 2)*std::pow(x - 1, 4) + 72*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)),
            (20*x - 1)*(6*x + 6*y + 12*z - 6)*(12376*std::pow(y, 6) + std::pow(y, 5)*(26208*x - 26208) + 20475*std::pow(y, 4)*std::pow(x - 1, 2) + 7280*std::pow(y, 3)*std::pow(x - 1, 3) + 1170*std::pow(y, 2)*std::pow(x - 1, 4) + 72*y*std::pow(x - 1, 5) + std::pow(x - 1, 6))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 191
template<>
struct DGBasis<191> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(50388*std::pow(y, 7) + std::pow(y, 6)*(129948*x - 129948) + 129948*std::pow(y, 5)*std::pow(x - 1, 2) + 63700*std::pow(y, 4)*std::pow(x - 1, 3) + 15925*std::pow(y, 3)*std::pow(x - 1, 4) + 1911*std::pow(y, 2)*std::pow(x - 1, 5) + 91*y*std::pow(x - 1, 6) + std::pow(x - 1, 7));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(129948*std::pow(y, 6) + 129948*std::pow(y, 5)*(2*x - 2) + 191100*std::pow(y, 4)*std::pow(x - 1, 2) + 63700*std::pow(y, 3)*std::pow(x - 1, 3) + 9555*std::pow(y, 2)*std::pow(x - 1, 4) + 546*y*std::pow(x - 1, 5) + 7*std::pow(x - 1, 6)) + (2*x + 2*y + 6*z - 2)*(50388*std::pow(y, 7) + std::pow(y, 6)*(129948*x - 129948) + 129948*std::pow(y, 5)*std::pow(x - 1, 2) + 63700*std::pow(y, 4)*std::pow(x - 1, 3) + 15925*std::pow(y, 3)*std::pow(x - 1, 4) + 1911*std::pow(y, 2)*std::pow(x - 1, 5) + 91*y*std::pow(x - 1, 6) + std::pow(x - 1, 7)),
            (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(352716*std::pow(y, 6) + 6*std::pow(y, 5)*(129948*x - 129948) + 649740*std::pow(y, 4)*std::pow(x - 1, 2) + 254800*std::pow(y, 3)*std::pow(x - 1, 3) + 47775*std::pow(y, 2)*std::pow(x - 1, 4) + 3822*y*std::pow(x - 1, 5) + 91*std::pow(x - 1, 6)) + (2*x + 2*y + 6*z - 2)*(50388*std::pow(y, 7) + std::pow(y, 6)*(129948*x - 129948) + 129948*std::pow(y, 5)*std::pow(x - 1, 2) + 63700*std::pow(y, 4)*std::pow(x - 1, 3) + 15925*std::pow(y, 3)*std::pow(x - 1, 4) + 1911*std::pow(y, 2)*std::pow(x - 1, 5) + 91*y*std::pow(x - 1, 6) + std::pow(x - 1, 7)),
            (6*x + 6*y + 12*z - 6)*(50388*std::pow(y, 7) + std::pow(y, 6)*(129948*x - 129948) + 129948*std::pow(y, 5)*std::pow(x - 1, 2) + 63700*std::pow(y, 4)*std::pow(x - 1, 3) + 15925*std::pow(y, 3)*std::pow(x - 1, 4) + 1911*std::pow(y, 2)*std::pow(x - 1, 5) + 91*y*std::pow(x - 1, 6) + std::pow(x - 1, 7))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 192
template<>
struct DGBasis<192> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)) + (4*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) + 2*x*(34*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 2*x*(51*x*(38*x*(5*x - 9) + 225) + 17*x*(114*x*(5*x - 9) + 3*x*(380*x - 342) + 675) - 3400) + 900) - 90)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)),
            (2*x*(2*x*(17*x*(3*x*(38*x*(5*x - 9) + 225) - 200) + 450) - 45) + 1)*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 193
template<>
struct DGBasis<193> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(x + 9*y - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(x + 9*y - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)) + (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (x + 9*y - 1)*(68*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 4*x*(51*x*(19*x*(4*x - 5) + 40) + 17*x*(57*x*(4*x - 5) + 3*x*(152*x - 95) + 120) - 340) + 80)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(x + 9*y - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)) + 9*(4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(x + 9*y - 1)*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 194
template<>
struct DGBasis<194> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(2*x + 20*y - 2)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)) + (55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(51*x*(19*x*(5*x - 4) + 18) + 17*x*(57*x*(5*x - 4) + 3*x*(190*x - 76) + 54) - 68)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(20*x + 110*y - 20)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)),
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 195
template<>
struct DGBasis<195> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(20*x - 9) + 18) - 1)*(286*std::pow(y, 3) + std::pow(y, 2)*(198*x - 198) + 33*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(198*std::pow(y, 2) + 33*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (3*x*(19*x*(20*x - 9) + 18) - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(286*std::pow(y, 3) + std::pow(y, 2)*(198*x - 198) + 33*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)) + (57*x*(20*x - 9) + 3*x*(760*x - 171) + 54)*(286*std::pow(y, 3) + std::pow(y, 2)*(198*x - 198) + 33*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(858*std::pow(y, 2) + 2*y*(198*x - 198) + 33*std::pow(x - 1, 2))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (3*x*(19*x*(20*x - 9) + 18) - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(286*std::pow(y, 3) + std::pow(y, 2)*(198*x - 198) + 33*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))*(286*std::pow(y, 3) + std::pow(y, 2)*(198*x - 198) + 33*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 196
template<>
struct DGBasis<196> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (38*x*(5*x - 1) + 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(1365*std::pow(y, 4) + std::pow(y, 3)*(1456*x - 1456) + 468*std::pow(y, 2)*std::pow(x - 1, 2) + 48*y*std::pow(x - 1, 3) + std::pow(x - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (380*x - 38)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(1365*std::pow(y, 4) + std::pow(y, 3)*(1456*x - 1456) + 468*std::pow(y, 2)*std::pow(x - 1, 2) + 48*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)) + (38*x*(5*x - 1) + 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(1365*std::pow(y, 4) + std::pow(y, 3)*(1456*x - 1456) + 468*std::pow(y, 2)*std::pow(x - 1, 2) + 48*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)) + (38*x*(5*x - 1) + 1)*(1456*std::pow(y, 3) + 468*std::pow(y, 2)*(2*x - 2) + 144*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (38*x*(5*x - 1) + 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(1365*std::pow(y, 4) + std::pow(y, 3)*(1456*x - 1456) + 468*std::pow(y, 2)*std::pow(x - 1, 2) + 48*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)) + (38*x*(5*x - 1) + 1)*(5460*std::pow(y, 3) + 3*std::pow(y, 2)*(1456*x - 1456) + 936*y*std::pow(x - 1, 2) + 48*std::pow(x - 1, 3))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (38*x*(5*x - 1) + 1)*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))*(1365*std::pow(y, 4) + std::pow(y, 3)*(1456*x - 1456) + 468*std::pow(y, 2)*std::pow(x - 1, 2) + 48*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 197
template<>
struct DGBasis<197> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (20*x - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(6188*std::pow(y, 5) + std::pow(y, 4)*(9100*x - 9100) + 4550*std::pow(y, 3)*std::pow(x - 1, 2) + 910*std::pow(y, 2)*std::pow(x - 1, 3) + 65*y*std::pow(x - 1, 4) + std::pow(x - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (20*x - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(6188*std::pow(y, 5) + std::pow(y, 4)*(9100*x - 9100) + 4550*std::pow(y, 3)*std::pow(x - 1, 2) + 910*std::pow(y, 2)*std::pow(x - 1, 3) + 65*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)) + (20*x - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(9100*std::pow(y, 4) + 4550*std::pow(y, 3)*(2*x - 2) + 2730*std::pow(y, 2)*std::pow(x - 1, 2) + 260*y*std::pow(x - 1, 3) + 5*std::pow(x - 1, 4)) + 20*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(6188*std::pow(y, 5) + std::pow(y, 4)*(9100*x - 9100) + 4550*std::pow(y, 3)*std::pow(x - 1, 2) + 910*std::pow(y, 2)*std::pow(x - 1, 3) + 65*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)),
            (20*x - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(6188*std::pow(y, 5) + std::pow(y, 4)*(9100*x - 9100) + 4550*std::pow(y, 3)*std::pow(x - 1, 2) + 910*std::pow(y, 2)*std::pow(x - 1, 3) + 65*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)) + (20*x - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(30940*std::pow(y, 4) + 4*std::pow(y, 3)*(9100*x - 9100) + 13650*std::pow(y, 2)*std::pow(x - 1, 2) + 1820*y*std::pow(x - 1, 3) + 65*std::pow(x - 1, 4)),
            (20*x - 1)*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))*(6188*std::pow(y, 5) + std::pow(y, 4)*(9100*x - 9100) + 4550*std::pow(y, 3)*std::pow(x - 1, 2) + 910*std::pow(y, 2)*std::pow(x - 1, 3) + 65*y*std::pow(x - 1, 4) + std::pow(x - 1, 5))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 198
template<>
struct DGBasis<198> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(27132*std::pow(y, 6) + std::pow(y, 5)*(51408*x - 51408) + 35700*std::pow(y, 4)*std::pow(x - 1, 2) + 11200*std::pow(y, 3)*std::pow(x - 1, 3) + 1575*std::pow(y, 2)*std::pow(x - 1, 4) + 84*y*std::pow(x - 1, 5) + std::pow(x - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(27132*std::pow(y, 6) + std::pow(y, 5)*(51408*x - 51408) + 35700*std::pow(y, 4)*std::pow(x - 1, 2) + 11200*std::pow(y, 3)*std::pow(x - 1, 3) + 1575*std::pow(y, 2)*std::pow(x - 1, 4) + 84*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)) + (20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(51408*std::pow(y, 5) + 35700*std::pow(y, 4)*(2*x - 2) + 33600*std::pow(y, 3)*std::pow(x - 1, 2) + 6300*std::pow(y, 2)*std::pow(x - 1, 3) + 420*y*std::pow(x - 1, 4) + 6*std::pow(x - 1, 5)),
            (30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(27132*std::pow(y, 6) + std::pow(y, 5)*(51408*x - 51408) + 35700*std::pow(y, 4)*std::pow(x - 1, 2) + 11200*std::pow(y, 3)*std::pow(x - 1, 3) + 1575*std::pow(y, 2)*std::pow(x - 1, 4) + 84*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)) + (20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(162792*std::pow(y, 5) + 5*std::pow(y, 4)*(51408*x - 51408) + 142800*std::pow(y, 3)*std::pow(x - 1, 2) + 33600*std::pow(y, 2)*std::pow(x - 1, 3) + 3150*y*std::pow(x - 1, 4) + 84*std::pow(x - 1, 5)),
            (60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))*(27132*std::pow(y, 6) + std::pow(y, 5)*(51408*x - 51408) + 35700*std::pow(y, 4)*std::pow(x - 1, 2) + 11200*std::pow(y, 3)*std::pow(x - 1, 3) + 1575*std::pow(y, 2)*std::pow(x - 1, 4) + 84*y*std::pow(x - 1, 5) + std::pow(x - 1, 6))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 199
template<>
struct DGBasis<199> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)) + (68*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 4*x*(51*x*(19*x*(4*x - 5) + 40) + 17*x*(57*x*(4*x - 5) + 3*x*(152*x - 95) + 120) - 340) + 80)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)),
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)),
            (4*x*(17*x*(3*x*(19*x*(4*x - 5) + 40) - 20) + 20) - 1)*(280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 200
template<>
struct DGBasis<200> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(x + 11*y - 1)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(x + 11*y - 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)) + (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (x + 11*y - 1)*(51*x*(19*x*(5*x - 4) + 18) + 17*x*(57*x*(5*x - 4) + 3*x*(190*x - 76) + 54) - 68)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)),
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(x + 11*y - 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)) + 11*(17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)),
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(x + 11*y - 1)*(280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 201
template<>
struct DGBasis<201> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(20*x - 9) + 18) - 1)*(78*std::pow(y, 2) + y*(24*x - 24) + std::pow(x - 1, 2))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(2*x + 24*y - 2)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (3*x*(19*x*(20*x - 9) + 18) - 1)*(78*std::pow(y, 2) + y*(24*x - 24) + std::pow(x - 1, 2))*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)) + (78*std::pow(y, 2) + y*(24*x - 24) + std::pow(x - 1, 2))*(57*x*(20*x - 9) + 3*x*(760*x - 171) + 54)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)),
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(24*x + 156*y - 24)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (3*x*(19*x*(20*x - 9) + 18) - 1)*(78*std::pow(y, 2) + y*(24*x - 24) + std::pow(x - 1, 2))*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)),
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(78*std::pow(y, 2) + y*(24*x - 24) + std::pow(x - 1, 2))*(280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 202
template<>
struct DGBasis<202> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (38*x*(5*x - 1) + 1)*(455*std::pow(y, 3) + std::pow(y, 2)*(273*x - 273) + 39*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (380*x - 38)*(455*std::pow(y, 3) + std::pow(y, 2)*(273*x - 273) + 39*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (38*x*(5*x - 1) + 1)*(273*std::pow(y, 2) + 39*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (38*x*(5*x - 1) + 1)*(455*std::pow(y, 3) + std::pow(y, 2)*(273*x - 273) + 39*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)),
            (38*x*(5*x - 1) + 1)*(1365*std::pow(y, 2) + 2*y*(273*x - 273) + 39*std::pow(x - 1, 2))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (38*x*(5*x - 1) + 1)*(455*std::pow(y, 3) + std::pow(y, 2)*(273*x - 273) + 39*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)),
            (38*x*(5*x - 1) + 1)*(455*std::pow(y, 3) + std::pow(y, 2)*(273*x - 273) + 39*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 203
template<>
struct DGBasis<203> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (20*x - 1)*(2380*std::pow(y, 4) + std::pow(y, 3)*(2240*x - 2240) + 630*std::pow(y, 2)*std::pow(x - 1, 2) + 56*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (20*x - 1)*(2240*std::pow(y, 3) + 630*std::pow(y, 2)*(2*x - 2) + 168*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (20*x - 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3))*(2380*std::pow(y, 4) + std::pow(y, 3)*(2240*x - 2240) + 630*std::pow(y, 2)*std::pow(x - 1, 2) + 56*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)) + 20*(2380*std::pow(y, 4) + std::pow(y, 3)*(2240*x - 2240) + 630*std::pow(y, 2)*std::pow(x - 1, 2) + 56*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)),
            (20*x - 1)*(9520*std::pow(y, 3) + 3*std::pow(y, 2)*(2240*x - 2240) + 1260*y*std::pow(x - 1, 2) + 56*std::pow(x - 1, 3))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (20*x - 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3))*(2380*std::pow(y, 4) + std::pow(y, 3)*(2240*x - 2240) + 630*std::pow(y, 2)*std::pow(x - 1, 2) + 56*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (20*x - 1)*(280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))*(2380*std::pow(y, 4) + std::pow(y, 3)*(2240*x - 2240) + 630*std::pow(y, 2)*std::pow(x - 1, 2) + 56*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 204
template<>
struct DGBasis<204> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4))*(11628*std::pow(y, 5) + std::pow(y, 4)*(15300*x - 15300) + 6800*std::pow(y, 3)*std::pow(x - 1, 2) + 1200*std::pow(y, 2)*std::pow(x - 1, 3) + 75*y*std::pow(x - 1, 4) + std::pow(x - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3))*(11628*std::pow(y, 5) + std::pow(y, 4)*(15300*x - 15300) + 6800*std::pow(y, 3)*std::pow(x - 1, 2) + 1200*std::pow(y, 2)*std::pow(x - 1, 3) + 75*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)) + (15300*std::pow(y, 4) + 6800*std::pow(y, 3)*(2*x - 2) + 3600*std::pow(y, 2)*std::pow(x - 1, 2) + 300*y*std::pow(x - 1, 3) + 5*std::pow(x - 1, 4))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)),
            (140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3))*(11628*std::pow(y, 5) + std::pow(y, 4)*(15300*x - 15300) + 6800*std::pow(y, 3)*std::pow(x - 1, 2) + 1200*std::pow(y, 2)*std::pow(x - 1, 3) + 75*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)) + (58140*std::pow(y, 4) + 4*std::pow(y, 3)*(15300*x - 15300) + 20400*std::pow(y, 2)*std::pow(x - 1, 2) + 2400*y*std::pow(x - 1, 3) + 75*std::pow(x - 1, 4))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)),
            (280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))*(11628*std::pow(y, 5) + std::pow(y, 4)*(15300*x - 15300) + 6800*std::pow(y, 3)*std::pow(x - 1, 2) + 1200*std::pow(y, 2)*std::pow(x - 1, 3) + 75*y*std::pow(x - 1, 4) + std::pow(x - 1, 5))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 205
template<>
struct DGBasis<205> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)) + (51*x*(19*x*(5*x - 4) + 18) + 17*x*(57*x*(5*x - 4) + 3*x*(190*x - 76) + 54) - 68)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)),
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)),
            (17*x*(3*x*(19*x*(5*x - 4) + 18) - 4) + 1)*(1260*std::pow(z, 4) + 4*std::pow(z, 3)*(630*x + 630*y - 630) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 2) + 420*z*std::pow(x + y - 1, 3) + 30*std::pow(x + y - 1, 4))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 206
template<>
struct DGBasis<206> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(20*x - 9) + 18) - 1)*(x + 13*y - 1)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(x + 13*y - 1)*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)) + (3*x*(19*x*(20*x - 9) + 18) - 1)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (x + 13*y - 1)*(57*x*(20*x - 9) + 3*x*(760*x - 171) + 54)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)),
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(x + 13*y - 1)*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)) + 13*(3*x*(19*x*(20*x - 9) + 18) - 1)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)),
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(x + 13*y - 1)*(1260*std::pow(z, 4) + 4*std::pow(z, 3)*(630*x + 630*y - 630) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 2) + 420*z*std::pow(x + y - 1, 3) + 30*std::pow(x + y - 1, 4))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 207
template<>
struct DGBasis<207> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (38*x*(5*x - 1) + 1)*(105*std::pow(y, 2) + y*(28*x - 28) + std::pow(x - 1, 2))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (380*x - 38)*(105*std::pow(y, 2) + y*(28*x - 28) + std::pow(x - 1, 2))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (38*x*(5*x - 1) + 1)*(2*x + 28*y - 2)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (38*x*(5*x - 1) + 1)*(105*std::pow(y, 2) + y*(28*x - 28) + std::pow(x - 1, 2))*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)),
            (38*x*(5*x - 1) + 1)*(28*x + 210*y - 28)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (38*x*(5*x - 1) + 1)*(105*std::pow(y, 2) + y*(28*x - 28) + std::pow(x - 1, 2))*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)),
            (38*x*(5*x - 1) + 1)*(105*std::pow(y, 2) + y*(28*x - 28) + std::pow(x - 1, 2))*(1260*std::pow(z, 4) + 4*std::pow(z, 3)*(630*x + 630*y - 630) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 2) + 420*z*std::pow(x + y - 1, 3) + 30*std::pow(x + y - 1, 4))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 208
template<>
struct DGBasis<208> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (20*x - 1)*(680*std::pow(y, 3) + std::pow(y, 2)*(360*x - 360) + 45*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (20*x - 1)*(360*std::pow(y, 2) + 45*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (20*x - 1)*(680*std::pow(y, 3) + std::pow(y, 2)*(360*x - 360) + 45*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)) + 20*(680*std::pow(y, 3) + std::pow(y, 2)*(360*x - 360) + 45*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)),
            (20*x - 1)*(2040*std::pow(y, 2) + 2*y*(360*x - 360) + 45*std::pow(x - 1, 2))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (20*x - 1)*(680*std::pow(y, 3) + std::pow(y, 2)*(360*x - 360) + 45*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)),
            (20*x - 1)*(680*std::pow(y, 3) + std::pow(y, 2)*(360*x - 360) + 45*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(1260*std::pow(z, 4) + 4*std::pow(z, 3)*(630*x + 630*y - 630) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 2) + 420*z*std::pow(x + y - 1, 3) + 30*std::pow(x + y - 1, 4))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 209
template<>
struct DGBasis<209> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (3876*std::pow(y, 4) + std::pow(y, 3)*(3264*x - 3264) + 816*std::pow(y, 2)*std::pow(x - 1, 2) + 64*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3264*std::pow(y, 3) + 816*std::pow(y, 2)*(2*x - 2) + 192*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (3876*std::pow(y, 4) + std::pow(y, 3)*(3264*x - 3264) + 816*std::pow(y, 2)*std::pow(x - 1, 2) + 64*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)),
            (15504*std::pow(y, 3) + 3*std::pow(y, 2)*(3264*x - 3264) + 1632*y*std::pow(x - 1, 2) + 64*std::pow(x - 1, 3))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (3876*std::pow(y, 4) + std::pow(y, 3)*(3264*x - 3264) + 816*std::pow(y, 2)*std::pow(x - 1, 2) + 64*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)),
            (3876*std::pow(y, 4) + std::pow(y, 3)*(3264*x - 3264) + 816*std::pow(y, 2)*std::pow(x - 1, 2) + 64*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))*(1260*std::pow(z, 4) + 4*std::pow(z, 3)*(630*x + 630*y - 630) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 2) + 420*z*std::pow(x + y - 1, 3) + 30*std::pow(x + y - 1, 4))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 210
template<>
struct DGBasis<210> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(20*x - 9) + 18) - 1)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)) + (57*x*(20*x - 9) + 3*x*(760*x - 171) + 54)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)),
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)),
            (3*x*(19*x*(20*x - 9) + 18) - 1)*(5544*std::pow(z, 5) + 5*std::pow(z, 4)*(2772*x + 2772*y - 2772) + 12600*std::pow(z, 3)*std::pow(x + y - 1, 2) + 5040*std::pow(z, 2)*std::pow(x + y - 1, 3) + 840*z*std::pow(x + y - 1, 4) + 42*std::pow(x + y - 1, 5))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 211
template<>
struct DGBasis<211> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (38*x*(5*x - 1) + 1)*(x + 15*y - 1)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (380*x - 38)*(x + 15*y - 1)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)) + (38*x*(5*x - 1) + 1)*(x + 15*y - 1)*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)) + (38*x*(5*x - 1) + 1)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)),
            (38*x*(5*x - 1) + 1)*(x + 15*y - 1)*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)) + 15*(38*x*(5*x - 1) + 1)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)),
            (38*x*(5*x - 1) + 1)*(x + 15*y - 1)*(5544*std::pow(z, 5) + 5*std::pow(z, 4)*(2772*x + 2772*y - 2772) + 12600*std::pow(z, 3)*std::pow(x + y - 1, 2) + 5040*std::pow(z, 2)*std::pow(x + y - 1, 3) + 840*z*std::pow(x + y - 1, 4) + 42*std::pow(x + y - 1, 5))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 212
template<>
struct DGBasis<212> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (20*x - 1)*(136*std::pow(y, 2) + y*(32*x - 32) + std::pow(x - 1, 2))*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (20*x - 1)*(2*x + 32*y - 2)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)) + (20*x - 1)*(136*std::pow(y, 2) + y*(32*x - 32) + std::pow(x - 1, 2))*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)) + 20*(136*std::pow(y, 2) + y*(32*x - 32) + std::pow(x - 1, 2))*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)),
            (20*x - 1)*(32*x + 272*y - 32)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)) + (20*x - 1)*(136*std::pow(y, 2) + y*(32*x - 32) + std::pow(x - 1, 2))*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)),
            (20*x - 1)*(136*std::pow(y, 2) + y*(32*x - 32) + std::pow(x - 1, 2))*(5544*std::pow(z, 5) + 5*std::pow(z, 4)*(2772*x + 2772*y - 2772) + 12600*std::pow(z, 3)*std::pow(x + y - 1, 2) + 5040*std::pow(z, 2)*std::pow(x + y - 1, 3) + 840*z*std::pow(x + y - 1, 4) + 42*std::pow(x + y - 1, 5))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 213
template<>
struct DGBasis<213> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (969*std::pow(y, 3) + std::pow(y, 2)*(459*x - 459) + 51*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (459*std::pow(y, 2) + 51*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)) + (969*std::pow(y, 3) + std::pow(y, 2)*(459*x - 459) + 51*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)),
            (2907*std::pow(y, 2) + 2*y*(459*x - 459) + 51*std::pow(x - 1, 2))*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)) + (969*std::pow(y, 3) + std::pow(y, 2)*(459*x - 459) + 51*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)),
            (969*std::pow(y, 3) + std::pow(y, 2)*(459*x - 459) + 51*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(5544*std::pow(z, 5) + 5*std::pow(z, 4)*(2772*x + 2772*y - 2772) + 12600*std::pow(z, 3)*std::pow(x + y - 1, 2) + 5040*std::pow(z, 2)*std::pow(x + y - 1, 3) + 840*z*std::pow(x + y - 1, 4) + 42*std::pow(x + y - 1, 5))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 214
template<>
struct DGBasis<214> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (38*x*(5*x - 1) + 1)*(std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (380*x - 38)*(std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) - 1) + (38*x*(5*x - 1) + 1)*(7*std::pow(x, 6) + 42*std::pow(x, 5)*y - 42*std::pow(x, 5) + 105*std::pow(x, 4)*std::pow(y, 2) - 210*std::pow(x, 4)*y + 105*std::pow(x, 4) + 140*std::pow(x, 3)*std::pow(y, 3) - 420*std::pow(x, 3)*std::pow(y, 2) + 420*std::pow(x, 3)*y - 140*std::pow(x, 3) + 105*std::pow(x, 2)*std::pow(y, 4) - 420*std::pow(x, 2)*std::pow(y, 3) + 630*std::pow(x, 2)*std::pow(y, 2) - 420*std::pow(x, 2)*y + 105*std::pow(x, 2) + 42*x*std::pow(y, 5) - 210*x*std::pow(y, 4) + 420*x*std::pow(y, 3) - 420*x*std::pow(y, 2) + 210*x*y - 42*x + 7*std::pow(y, 6) - 42*std::pow(y, 5) + 105*std::pow(y, 4) - 140*std::pow(y, 3) + 105*std::pow(y, 2) - 42*y + 12012*std::pow(z, 6) + 16632*std::pow(z, 5)*(2*x + 2*y - 2) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 2) + 16800*std::pow(z, 3)*std::pow(x + y - 1, 3) + 3780*std::pow(z, 2)*std::pow(x + y - 1, 4) + 336*z*std::pow(x + y - 1, 5) + 7),
            (38*x*(5*x - 1) + 1)*(7*std::pow(x, 6) + 42*std::pow(x, 5)*y - 42*std::pow(x, 5) + 105*std::pow(x, 4)*std::pow(y, 2) - 210*std::pow(x, 4)*y + 105*std::pow(x, 4) + 140*std::pow(x, 3)*std::pow(y, 3) - 420*std::pow(x, 3)*std::pow(y, 2) + 420*std::pow(x, 3)*y - 140*std::pow(x, 3) + 105*std::pow(x, 2)*std::pow(y, 4) - 420*std::pow(x, 2)*std::pow(y, 3) + 630*std::pow(x, 2)*std::pow(y, 2) - 420*std::pow(x, 2)*y + 105*std::pow(x, 2) + 42*x*std::pow(y, 5) - 210*x*std::pow(y, 4) + 420*x*std::pow(y, 3) - 420*x*std::pow(y, 2) + 210*x*y - 42*x + 7*std::pow(y, 6) - 42*std::pow(y, 5) + 105*std::pow(y, 4) - 140*std::pow(y, 3) + 105*std::pow(y, 2) - 42*y + 12012*std::pow(z, 6) + 16632*std::pow(z, 5)*(2*x + 2*y - 2) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 2) + 16800*std::pow(z, 3)*std::pow(x + y - 1, 3) + 3780*std::pow(z, 2)*std::pow(x + y - 1, 4) + 336*z*std::pow(x + y - 1, 5) + 7),
            (38*x*(5*x - 1) + 1)*(24024*std::pow(z, 6) + 6*std::pow(z, 5)*(12012*x + 12012*y - 12012) + 83160*std::pow(z, 4)*std::pow(x + y - 1, 2) + 46200*std::pow(z, 3)*std::pow(x + y - 1, 3) + 12600*std::pow(z, 2)*std::pow(x + y - 1, 4) + 1512*z*std::pow(x + y - 1, 5) + 56*std::pow(x + y - 1, 6))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 215
template<>
struct DGBasis<215> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (20*x - 1)*(x + 17*y - 1)*(std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (20*x - 1)*(x + 17*y - 1)*(7*std::pow(x, 6) + 42*std::pow(x, 5)*y - 42*std::pow(x, 5) + 105*std::pow(x, 4)*std::pow(y, 2) - 210*std::pow(x, 4)*y + 105*std::pow(x, 4) + 140*std::pow(x, 3)*std::pow(y, 3) - 420*std::pow(x, 3)*std::pow(y, 2) + 420*std::pow(x, 3)*y - 140*std::pow(x, 3) + 105*std::pow(x, 2)*std::pow(y, 4) - 420*std::pow(x, 2)*std::pow(y, 3) + 630*std::pow(x, 2)*std::pow(y, 2) - 420*std::pow(x, 2)*y + 105*std::pow(x, 2) + 42*x*std::pow(y, 5) - 210*x*std::pow(y, 4) + 420*x*std::pow(y, 3) - 420*x*std::pow(y, 2) + 210*x*y - 42*x + 7*std::pow(y, 6) - 42*std::pow(y, 5) + 105*std::pow(y, 4) - 140*std::pow(y, 3) + 105*std::pow(y, 2) - 42*y + 12012*std::pow(z, 6) + 16632*std::pow(z, 5)*(2*x + 2*y - 2) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 2) + 16800*std::pow(z, 3)*std::pow(x + y - 1, 3) + 3780*std::pow(z, 2)*std::pow(x + y - 1, 4) + 336*z*std::pow(x + y - 1, 5) + 7) + (20*x - 1)*(std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) - 1) + 20*(x + 17*y - 1)*(std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) - 1),
            (20*x - 1)*(x + 17*y - 1)*(7*std::pow(x, 6) + 42*std::pow(x, 5)*y - 42*std::pow(x, 5) + 105*std::pow(x, 4)*std::pow(y, 2) - 210*std::pow(x, 4)*y + 105*std::pow(x, 4) + 140*std::pow(x, 3)*std::pow(y, 3) - 420*std::pow(x, 3)*std::pow(y, 2) + 420*std::pow(x, 3)*y - 140*std::pow(x, 3) + 105*std::pow(x, 2)*std::pow(y, 4) - 420*std::pow(x, 2)*std::pow(y, 3) + 630*std::pow(x, 2)*std::pow(y, 2) - 420*std::pow(x, 2)*y + 105*std::pow(x, 2) + 42*x*std::pow(y, 5) - 210*x*std::pow(y, 4) + 420*x*std::pow(y, 3) - 420*x*std::pow(y, 2) + 210*x*y - 42*x + 7*std::pow(y, 6) - 42*std::pow(y, 5) + 105*std::pow(y, 4) - 140*std::pow(y, 3) + 105*std::pow(y, 2) - 42*y + 12012*std::pow(z, 6) + 16632*std::pow(z, 5)*(2*x + 2*y - 2) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 2) + 16800*std::pow(z, 3)*std::pow(x + y - 1, 3) + 3780*std::pow(z, 2)*std::pow(x + y - 1, 4) + 336*z*std::pow(x + y - 1, 5) + 7) + 17*(20*x - 1)*(std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) - 1),
            (20*x - 1)*(x + 17*y - 1)*(24024*std::pow(z, 6) + 6*std::pow(z, 5)*(12012*x + 12012*y - 12012) + 83160*std::pow(z, 4)*std::pow(x + y - 1, 2) + 46200*std::pow(z, 3)*std::pow(x + y - 1, 3) + 12600*std::pow(z, 2)*std::pow(x + y - 1, 4) + 1512*z*std::pow(x + y - 1, 5) + 56*std::pow(x + y - 1, 6))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 216
template<>
struct DGBasis<216> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (171*std::pow(y, 2) + y*(36*x - 36) + std::pow(x - 1, 2))*(std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x + 36*y - 2)*(std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) - 1) + (171*std::pow(y, 2) + y*(36*x - 36) + std::pow(x - 1, 2))*(7*std::pow(x, 6) + 42*std::pow(x, 5)*y - 42*std::pow(x, 5) + 105*std::pow(x, 4)*std::pow(y, 2) - 210*std::pow(x, 4)*y + 105*std::pow(x, 4) + 140*std::pow(x, 3)*std::pow(y, 3) - 420*std::pow(x, 3)*std::pow(y, 2) + 420*std::pow(x, 3)*y - 140*std::pow(x, 3) + 105*std::pow(x, 2)*std::pow(y, 4) - 420*std::pow(x, 2)*std::pow(y, 3) + 630*std::pow(x, 2)*std::pow(y, 2) - 420*std::pow(x, 2)*y + 105*std::pow(x, 2) + 42*x*std::pow(y, 5) - 210*x*std::pow(y, 4) + 420*x*std::pow(y, 3) - 420*x*std::pow(y, 2) + 210*x*y - 42*x + 7*std::pow(y, 6) - 42*std::pow(y, 5) + 105*std::pow(y, 4) - 140*std::pow(y, 3) + 105*std::pow(y, 2) - 42*y + 12012*std::pow(z, 6) + 16632*std::pow(z, 5)*(2*x + 2*y - 2) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 2) + 16800*std::pow(z, 3)*std::pow(x + y - 1, 3) + 3780*std::pow(z, 2)*std::pow(x + y - 1, 4) + 336*z*std::pow(x + y - 1, 5) + 7),
            (36*x + 342*y - 36)*(std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) - 1) + (171*std::pow(y, 2) + y*(36*x - 36) + std::pow(x - 1, 2))*(7*std::pow(x, 6) + 42*std::pow(x, 5)*y - 42*std::pow(x, 5) + 105*std::pow(x, 4)*std::pow(y, 2) - 210*std::pow(x, 4)*y + 105*std::pow(x, 4) + 140*std::pow(x, 3)*std::pow(y, 3) - 420*std::pow(x, 3)*std::pow(y, 2) + 420*std::pow(x, 3)*y - 140*std::pow(x, 3) + 105*std::pow(x, 2)*std::pow(y, 4) - 420*std::pow(x, 2)*std::pow(y, 3) + 630*std::pow(x, 2)*std::pow(y, 2) - 420*std::pow(x, 2)*y + 105*std::pow(x, 2) + 42*x*std::pow(y, 5) - 210*x*std::pow(y, 4) + 420*x*std::pow(y, 3) - 420*x*std::pow(y, 2) + 210*x*y - 42*x + 7*std::pow(y, 6) - 42*std::pow(y, 5) + 105*std::pow(y, 4) - 140*std::pow(y, 3) + 105*std::pow(y, 2) - 42*y + 12012*std::pow(z, 6) + 16632*std::pow(z, 5)*(2*x + 2*y - 2) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 2) + 16800*std::pow(z, 3)*std::pow(x + y - 1, 3) + 3780*std::pow(z, 2)*std::pow(x + y - 1, 4) + 336*z*std::pow(x + y - 1, 5) + 7),
            (171*std::pow(y, 2) + y*(36*x - 36) + std::pow(x - 1, 2))*(24024*std::pow(z, 6) + 6*std::pow(z, 5)*(12012*x + 12012*y - 12012) + 83160*std::pow(z, 4)*std::pow(x + y - 1, 2) + 46200*std::pow(z, 3)*std::pow(x + y - 1, 3) + 12600*std::pow(z, 2)*std::pow(x + y - 1, 4) + 1512*z*std::pow(x + y - 1, 5) + 56*std::pow(x + y - 1, 6))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 217
template<>
struct DGBasis<217> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (20*x - 1)*(std::pow(x, 8) + 8*std::pow(x, 7)*y - 8*std::pow(x, 7) + 28*std::pow(x, 6)*std::pow(y, 2) - 56*std::pow(x, 6)*y + 28*std::pow(x, 6) + 56*std::pow(x, 5)*std::pow(y, 3) - 168*std::pow(x, 5)*std::pow(y, 2) + 168*std::pow(x, 5)*y - 56*std::pow(x, 5) + 70*std::pow(x, 4)*std::pow(y, 4) - 280*std::pow(x, 4)*std::pow(y, 3) + 420*std::pow(x, 4)*std::pow(y, 2) - 280*std::pow(x, 4)*y + 70*std::pow(x, 4) + 56*std::pow(x, 3)*std::pow(y, 5) - 280*std::pow(x, 3)*std::pow(y, 4) + 560*std::pow(x, 3)*std::pow(y, 3) - 560*std::pow(x, 3)*std::pow(y, 2) + 280*std::pow(x, 3)*y - 56*std::pow(x, 3) + 28*std::pow(x, 2)*std::pow(y, 6) - 168*std::pow(x, 2)*std::pow(y, 5) + 420*std::pow(x, 2)*std::pow(y, 4) - 560*std::pow(x, 2)*std::pow(y, 3) + 420*std::pow(x, 2)*std::pow(y, 2) - 168*std::pow(x, 2)*y + 28*std::pow(x, 2) + 8*x*std::pow(y, 7) - 56*x*std::pow(y, 6) + 168*x*std::pow(y, 5) - 280*x*std::pow(y, 4) + 280*x*std::pow(y, 3) - 168*x*std::pow(y, 2) + 56*x*y - 8*x + std::pow(y, 8) - 8*std::pow(y, 7) + 28*std::pow(y, 6) - 56*std::pow(y, 5) + 70*std::pow(y, 4) - 56*std::pow(y, 3) + 28*std::pow(y, 2) - 8*y + 12870*std::pow(z, 8) + std::pow(z, 7)*(51480*x + 51480*y - 51480) + 84084*std::pow(z, 6)*std::pow(x + y - 1, 2) + 72072*std::pow(z, 5)*std::pow(x + y - 1, 3) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 4) + 9240*std::pow(z, 3)*std::pow(x + y - 1, 5) + 1260*std::pow(z, 2)*std::pow(x + y - 1, 6) + 72*z*std::pow(x + y - 1, 7) + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            20*std::pow(x, 8) + 160*std::pow(x, 7)*y - 160*std::pow(x, 7) + 560*std::pow(x, 6)*std::pow(y, 2) - 1120*std::pow(x, 6)*y + 560*std::pow(x, 6) + 1120*std::pow(x, 5)*std::pow(y, 3) - 3360*std::pow(x, 5)*std::pow(y, 2) + 3360*std::pow(x, 5)*y - 1120*std::pow(x, 5) + 1400*std::pow(x, 4)*std::pow(y, 4) - 5600*std::pow(x, 4)*std::pow(y, 3) + 8400*std::pow(x, 4)*std::pow(y, 2) - 5600*std::pow(x, 4)*y + 1400*std::pow(x, 4) + 1120*std::pow(x, 3)*std::pow(y, 5) - 5600*std::pow(x, 3)*std::pow(y, 4) + 11200*std::pow(x, 3)*std::pow(y, 3) - 11200*std::pow(x, 3)*std::pow(y, 2) + 5600*std::pow(x, 3)*y - 1120*std::pow(x, 3) + 560*std::pow(x, 2)*std::pow(y, 6) - 3360*std::pow(x, 2)*std::pow(y, 5) + 8400*std::pow(x, 2)*std::pow(y, 4) - 11200*std::pow(x, 2)*std::pow(y, 3) + 8400*std::pow(x, 2)*std::pow(y, 2) - 3360*std::pow(x, 2)*y + 560*std::pow(x, 2) + 160*x*std::pow(y, 7) - 1120*x*std::pow(y, 6) + 3360*x*std::pow(y, 5) - 5600*x*std::pow(y, 4) + 5600*x*std::pow(y, 3) - 3360*x*std::pow(y, 2) + 1120*x*y - 160*x + 20*std::pow(y, 8) - 160*std::pow(y, 7) + 560*std::pow(y, 6) - 1120*std::pow(y, 5) + 1400*std::pow(y, 4) - 1120*std::pow(y, 3) + 560*std::pow(y, 2) - 160*y + 257400*std::pow(z, 8) + 20*std::pow(z, 7)*(51480*x + 51480*y - 51480) + 1681680*std::pow(z, 6)*std::pow(x + y - 1, 2) + 1441440*std::pow(z, 5)*std::pow(x + y - 1, 3) + 693000*std::pow(z, 4)*std::pow(x + y - 1, 4) + 184800*std::pow(z, 3)*std::pow(x + y - 1, 5) + 25200*std::pow(z, 2)*std::pow(x + y - 1, 6) + 1440*z*std::pow(x + y - 1, 7) + (20*x - 1)*(8*std::pow(x, 7) + 56*std::pow(x, 6)*y - 56*std::pow(x, 6) + 168*std::pow(x, 5)*std::pow(y, 2) - 336*std::pow(x, 5)*y + 168*std::pow(x, 5) + 280*std::pow(x, 4)*std::pow(y, 3) - 840*std::pow(x, 4)*std::pow(y, 2) + 840*std::pow(x, 4)*y - 280*std::pow(x, 4) + 280*std::pow(x, 3)*std::pow(y, 4) - 1120*std::pow(x, 3)*std::pow(y, 3) + 1680*std::pow(x, 3)*std::pow(y, 2) - 1120*std::pow(x, 3)*y + 280*std::pow(x, 3) + 168*std::pow(x, 2)*std::pow(y, 5) - 840*std::pow(x, 2)*std::pow(y, 4) + 1680*std::pow(x, 2)*std::pow(y, 3) - 1680*std::pow(x, 2)*std::pow(y, 2) + 840*std::pow(x, 2)*y - 168*std::pow(x, 2) + 56*x*std::pow(y, 6) - 336*x*std::pow(y, 5) + 840*x*std::pow(y, 4) - 1120*x*std::pow(y, 3) + 840*x*std::pow(y, 2) - 336*x*y + 56*x + 8*std::pow(y, 7) - 56*std::pow(y, 6) + 168*std::pow(y, 5) - 280*std::pow(y, 4) + 280*std::pow(y, 3) - 168*std::pow(y, 2) + 56*y + 51480*std::pow(z, 7) + 84084*std::pow(z, 6)*(2*x + 2*y - 2) + 216216*std::pow(z, 5)*std::pow(x + y - 1, 2) + 138600*std::pow(z, 4)*std::pow(x + y - 1, 3) + 46200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 7560*std::pow(z, 2)*std::pow(x + y - 1, 5) + 504*z*std::pow(x + y - 1, 6) - 8) + 20,
            (20*x - 1)*(8*std::pow(x, 7) + 56*std::pow(x, 6)*y - 56*std::pow(x, 6) + 168*std::pow(x, 5)*std::pow(y, 2) - 336*std::pow(x, 5)*y + 168*std::pow(x, 5) + 280*std::pow(x, 4)*std::pow(y, 3) - 840*std::pow(x, 4)*std::pow(y, 2) + 840*std::pow(x, 4)*y - 280*std::pow(x, 4) + 280*std::pow(x, 3)*std::pow(y, 4) - 1120*std::pow(x, 3)*std::pow(y, 3) + 1680*std::pow(x, 3)*std::pow(y, 2) - 1120*std::pow(x, 3)*y + 280*std::pow(x, 3) + 168*std::pow(x, 2)*std::pow(y, 5) - 840*std::pow(x, 2)*std::pow(y, 4) + 1680*std::pow(x, 2)*std::pow(y, 3) - 1680*std::pow(x, 2)*std::pow(y, 2) + 840*std::pow(x, 2)*y - 168*std::pow(x, 2) + 56*x*std::pow(y, 6) - 336*x*std::pow(y, 5) + 840*x*std::pow(y, 4) - 1120*x*std::pow(y, 3) + 840*x*std::pow(y, 2) - 336*x*y + 56*x + 8*std::pow(y, 7) - 56*std::pow(y, 6) + 168*std::pow(y, 5) - 280*std::pow(y, 4) + 280*std::pow(y, 3) - 168*std::pow(y, 2) + 56*y + 51480*std::pow(z, 7) + 84084*std::pow(z, 6)*(2*x + 2*y - 2) + 216216*std::pow(z, 5)*std::pow(x + y - 1, 2) + 138600*std::pow(z, 4)*std::pow(x + y - 1, 3) + 46200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 7560*std::pow(z, 2)*std::pow(x + y - 1, 5) + 504*z*std::pow(x + y - 1, 6) - 8),
            (20*x - 1)*(102960*std::pow(z, 7) + 7*std::pow(z, 6)*(51480*x + 51480*y - 51480) + 504504*std::pow(z, 5)*std::pow(x + y - 1, 2) + 360360*std::pow(z, 4)*std::pow(x + y - 1, 3) + 138600*std::pow(z, 3)*std::pow(x + y - 1, 4) + 27720*std::pow(z, 2)*std::pow(x + y - 1, 5) + 2520*z*std::pow(x + y - 1, 6) + 72*std::pow(x + y - 1, 7))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 218
template<>
struct DGBasis<218> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x + 19*y - 1)*(std::pow(x, 8) + 8*std::pow(x, 7)*y - 8*std::pow(x, 7) + 28*std::pow(x, 6)*std::pow(y, 2) - 56*std::pow(x, 6)*y + 28*std::pow(x, 6) + 56*std::pow(x, 5)*std::pow(y, 3) - 168*std::pow(x, 5)*std::pow(y, 2) + 168*std::pow(x, 5)*y - 56*std::pow(x, 5) + 70*std::pow(x, 4)*std::pow(y, 4) - 280*std::pow(x, 4)*std::pow(y, 3) + 420*std::pow(x, 4)*std::pow(y, 2) - 280*std::pow(x, 4)*y + 70*std::pow(x, 4) + 56*std::pow(x, 3)*std::pow(y, 5) - 280*std::pow(x, 3)*std::pow(y, 4) + 560*std::pow(x, 3)*std::pow(y, 3) - 560*std::pow(x, 3)*std::pow(y, 2) + 280*std::pow(x, 3)*y - 56*std::pow(x, 3) + 28*std::pow(x, 2)*std::pow(y, 6) - 168*std::pow(x, 2)*std::pow(y, 5) + 420*std::pow(x, 2)*std::pow(y, 4) - 560*std::pow(x, 2)*std::pow(y, 3) + 420*std::pow(x, 2)*std::pow(y, 2) - 168*std::pow(x, 2)*y + 28*std::pow(x, 2) + 8*x*std::pow(y, 7) - 56*x*std::pow(y, 6) + 168*x*std::pow(y, 5) - 280*x*std::pow(y, 4) + 280*x*std::pow(y, 3) - 168*x*std::pow(y, 2) + 56*x*y - 8*x + std::pow(y, 8) - 8*std::pow(y, 7) + 28*std::pow(y, 6) - 56*std::pow(y, 5) + 70*std::pow(y, 4) - 56*std::pow(y, 3) + 28*std::pow(y, 2) - 8*y + 12870*std::pow(z, 8) + std::pow(z, 7)*(51480*x + 51480*y - 51480) + 84084*std::pow(z, 6)*std::pow(x + y - 1, 2) + 72072*std::pow(z, 5)*std::pow(x + y - 1, 3) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 4) + 9240*std::pow(z, 3)*std::pow(x + y - 1, 5) + 1260*std::pow(z, 2)*std::pow(x + y - 1, 6) + 72*z*std::pow(x + y - 1, 7) + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            std::pow(x, 8) + 8*std::pow(x, 7)*y - 8*std::pow(x, 7) + 28*std::pow(x, 6)*std::pow(y, 2) - 56*std::pow(x, 6)*y + 28*std::pow(x, 6) + 56*std::pow(x, 5)*std::pow(y, 3) - 168*std::pow(x, 5)*std::pow(y, 2) + 168*std::pow(x, 5)*y - 56*std::pow(x, 5) + 70*std::pow(x, 4)*std::pow(y, 4) - 280*std::pow(x, 4)*std::pow(y, 3) + 420*std::pow(x, 4)*std::pow(y, 2) - 280*std::pow(x, 4)*y + 70*std::pow(x, 4) + 56*std::pow(x, 3)*std::pow(y, 5) - 280*std::pow(x, 3)*std::pow(y, 4) + 560*std::pow(x, 3)*std::pow(y, 3) - 560*std::pow(x, 3)*std::pow(y, 2) + 280*std::pow(x, 3)*y - 56*std::pow(x, 3) + 28*std::pow(x, 2)*std::pow(y, 6) - 168*std::pow(x, 2)*std::pow(y, 5) + 420*std::pow(x, 2)*std::pow(y, 4) - 560*std::pow(x, 2)*std::pow(y, 3) + 420*std::pow(x, 2)*std::pow(y, 2) - 168*std::pow(x, 2)*y + 28*std::pow(x, 2) + 8*x*std::pow(y, 7) - 56*x*std::pow(y, 6) + 168*x*std::pow(y, 5) - 280*x*std::pow(y, 4) + 280*x*std::pow(y, 3) - 168*x*std::pow(y, 2) + 56*x*y - 8*x + std::pow(y, 8) - 8*std::pow(y, 7) + 28*std::pow(y, 6) - 56*std::pow(y, 5) + 70*std::pow(y, 4) - 56*std::pow(y, 3) + 28*std::pow(y, 2) - 8*y + 12870*std::pow(z, 8) + std::pow(z, 7)*(51480*x + 51480*y - 51480) + 84084*std::pow(z, 6)*std::pow(x + y - 1, 2) + 72072*std::pow(z, 5)*std::pow(x + y - 1, 3) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 4) + 9240*std::pow(z, 3)*std::pow(x + y - 1, 5) + 1260*std::pow(z, 2)*std::pow(x + y - 1, 6) + 72*z*std::pow(x + y - 1, 7) + (x + 19*y - 1)*(8*std::pow(x, 7) + 56*std::pow(x, 6)*y - 56*std::pow(x, 6) + 168*std::pow(x, 5)*std::pow(y, 2) - 336*std::pow(x, 5)*y + 168*std::pow(x, 5) + 280*std::pow(x, 4)*std::pow(y, 3) - 840*std::pow(x, 4)*std::pow(y, 2) + 840*std::pow(x, 4)*y - 280*std::pow(x, 4) + 280*std::pow(x, 3)*std::pow(y, 4) - 1120*std::pow(x, 3)*std::pow(y, 3) + 1680*std::pow(x, 3)*std::pow(y, 2) - 1120*std::pow(x, 3)*y + 280*std::pow(x, 3) + 168*std::pow(x, 2)*std::pow(y, 5) - 840*std::pow(x, 2)*std::pow(y, 4) + 1680*std::pow(x, 2)*std::pow(y, 3) - 1680*std::pow(x, 2)*std::pow(y, 2) + 840*std::pow(x, 2)*y - 168*std::pow(x, 2) + 56*x*std::pow(y, 6) - 336*x*std::pow(y, 5) + 840*x*std::pow(y, 4) - 1120*x*std::pow(y, 3) + 840*x*std::pow(y, 2) - 336*x*y + 56*x + 8*std::pow(y, 7) - 56*std::pow(y, 6) + 168*std::pow(y, 5) - 280*std::pow(y, 4) + 280*std::pow(y, 3) - 168*std::pow(y, 2) + 56*y + 51480*std::pow(z, 7) + 84084*std::pow(z, 6)*(2*x + 2*y - 2) + 216216*std::pow(z, 5)*std::pow(x + y - 1, 2) + 138600*std::pow(z, 4)*std::pow(x + y - 1, 3) + 46200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 7560*std::pow(z, 2)*std::pow(x + y - 1, 5) + 504*z*std::pow(x + y - 1, 6) - 8) + 1,
            19*std::pow(x, 8) + 152*std::pow(x, 7)*y - 152*std::pow(x, 7) + 532*std::pow(x, 6)*std::pow(y, 2) - 1064*std::pow(x, 6)*y + 532*std::pow(x, 6) + 1064*std::pow(x, 5)*std::pow(y, 3) - 3192*std::pow(x, 5)*std::pow(y, 2) + 3192*std::pow(x, 5)*y - 1064*std::pow(x, 5) + 1330*std::pow(x, 4)*std::pow(y, 4) - 5320*std::pow(x, 4)*std::pow(y, 3) + 7980*std::pow(x, 4)*std::pow(y, 2) - 5320*std::pow(x, 4)*y + 1330*std::pow(x, 4) + 1064*std::pow(x, 3)*std::pow(y, 5) - 5320*std::pow(x, 3)*std::pow(y, 4) + 10640*std::pow(x, 3)*std::pow(y, 3) - 10640*std::pow(x, 3)*std::pow(y, 2) + 5320*std::pow(x, 3)*y - 1064*std::pow(x, 3) + 532*std::pow(x, 2)*std::pow(y, 6) - 3192*std::pow(x, 2)*std::pow(y, 5) + 7980*std::pow(x, 2)*std::pow(y, 4) - 10640*std::pow(x, 2)*std::pow(y, 3) + 7980*std::pow(x, 2)*std::pow(y, 2) - 3192*std::pow(x, 2)*y + 532*std::pow(x, 2) + 152*x*std::pow(y, 7) - 1064*x*std::pow(y, 6) + 3192*x*std::pow(y, 5) - 5320*x*std::pow(y, 4) + 5320*x*std::pow(y, 3) - 3192*x*std::pow(y, 2) + 1064*x*y - 152*x + 19*std::pow(y, 8) - 152*std::pow(y, 7) + 532*std::pow(y, 6) - 1064*std::pow(y, 5) + 1330*std::pow(y, 4) - 1064*std::pow(y, 3) + 532*std::pow(y, 2) - 152*y + 244530*std::pow(z, 8) + 19*std::pow(z, 7)*(51480*x + 51480*y - 51480) + 1597596*std::pow(z, 6)*std::pow(x + y - 1, 2) + 1369368*std::pow(z, 5)*std::pow(x + y - 1, 3) + 658350*std::pow(z, 4)*std::pow(x + y - 1, 4) + 175560*std::pow(z, 3)*std::pow(x + y - 1, 5) + 23940*std::pow(z, 2)*std::pow(x + y - 1, 6) + 1368*z*std::pow(x + y - 1, 7) + (x + 19*y - 1)*(8*std::pow(x, 7) + 56*std::pow(x, 6)*y - 56*std::pow(x, 6) + 168*std::pow(x, 5)*std::pow(y, 2) - 336*std::pow(x, 5)*y + 168*std::pow(x, 5) + 280*std::pow(x, 4)*std::pow(y, 3) - 840*std::pow(x, 4)*std::pow(y, 2) + 840*std::pow(x, 4)*y - 280*std::pow(x, 4) + 280*std::pow(x, 3)*std::pow(y, 4) - 1120*std::pow(x, 3)*std::pow(y, 3) + 1680*std::pow(x, 3)*std::pow(y, 2) - 1120*std::pow(x, 3)*y + 280*std::pow(x, 3) + 168*std::pow(x, 2)*std::pow(y, 5) - 840*std::pow(x, 2)*std::pow(y, 4) + 1680*std::pow(x, 2)*std::pow(y, 3) - 1680*std::pow(x, 2)*std::pow(y, 2) + 840*std::pow(x, 2)*y - 168*std::pow(x, 2) + 56*x*std::pow(y, 6) - 336*x*std::pow(y, 5) + 840*x*std::pow(y, 4) - 1120*x*std::pow(y, 3) + 840*x*std::pow(y, 2) - 336*x*y + 56*x + 8*std::pow(y, 7) - 56*std::pow(y, 6) + 168*std::pow(y, 5) - 280*std::pow(y, 4) + 280*std::pow(y, 3) - 168*std::pow(y, 2) + 56*y + 51480*std::pow(z, 7) + 84084*std::pow(z, 6)*(2*x + 2*y - 2) + 216216*std::pow(z, 5)*std::pow(x + y - 1, 2) + 138600*std::pow(z, 4)*std::pow(x + y - 1, 3) + 46200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 7560*std::pow(z, 2)*std::pow(x + y - 1, 5) + 504*z*std::pow(x + y - 1, 6) - 8) + 19,
            (x + 19*y - 1)*(102960*std::pow(z, 7) + 7*std::pow(z, 6)*(51480*x + 51480*y - 51480) + 504504*std::pow(z, 5)*std::pow(x + y - 1, 2) + 360360*std::pow(z, 4)*std::pow(x + y - 1, 3) + 138600*std::pow(z, 3)*std::pow(x + y - 1, 4) + 27720*std::pow(z, 2)*std::pow(x + y - 1, 5) + 2520*z*std::pow(x + y - 1, 6) + 72*std::pow(x + y - 1, 7))
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 219
template<>
struct DGBasis<219> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return std::pow(x, 9) + 9*std::pow(x, 8)*y - 9*std::pow(x, 8) + 36*std::pow(x, 7)*std::pow(y, 2) - 72*std::pow(x, 7)*y + 36*std::pow(x, 7) + 84*std::pow(x, 6)*std::pow(y, 3) - 252*std::pow(x, 6)*std::pow(y, 2) + 252*std::pow(x, 6)*y - 84*std::pow(x, 6) + 126*std::pow(x, 5)*std::pow(y, 4) - 504*std::pow(x, 5)*std::pow(y, 3) + 756*std::pow(x, 5)*std::pow(y, 2) - 504*std::pow(x, 5)*y + 126*std::pow(x, 5) + 126*std::pow(x, 4)*std::pow(y, 5) - 630*std::pow(x, 4)*std::pow(y, 4) + 1260*std::pow(x, 4)*std::pow(y, 3) - 1260*std::pow(x, 4)*std::pow(y, 2) + 630*std::pow(x, 4)*y - 126*std::pow(x, 4) + 84*std::pow(x, 3)*std::pow(y, 6) - 504*std::pow(x, 3)*std::pow(y, 5) + 1260*std::pow(x, 3)*std::pow(y, 4) - 1680*std::pow(x, 3)*std::pow(y, 3) + 1260*std::pow(x, 3)*std::pow(y, 2) - 504*std::pow(x, 3)*y + 84*std::pow(x, 3) + 36*std::pow(x, 2)*std::pow(y, 7) - 252*std::pow(x, 2)*std::pow(y, 6) + 756*std::pow(x, 2)*std::pow(y, 5) - 1260*std::pow(x, 2)*std::pow(y, 4) + 1260*std::pow(x, 2)*std::pow(y, 3) - 756*std::pow(x, 2)*std::pow(y, 2) + 252*std::pow(x, 2)*y - 36*std::pow(x, 2) + 9*x*std::pow(y, 8) - 72*x*std::pow(y, 7) + 252*x*std::pow(y, 6) - 504*x*std::pow(y, 5) + 630*x*std::pow(y, 4) - 504*x*std::pow(y, 3) + 252*x*std::pow(y, 2) - 72*x*y + 9*x + std::pow(y, 9) - 9*std::pow(y, 8) + 36*std::pow(y, 7) - 84*std::pow(y, 6) + 126*std::pow(y, 5) - 126*std::pow(y, 4) + 84*std::pow(y, 3) - 36*std::pow(y, 2) + 9*y + 48620*std::pow(z, 9) + std::pow(z, 8)*(218790*x + 218790*y - 218790) + 411840*std::pow(z, 7)*std::pow(x + y - 1, 2) + 420420*std::pow(z, 6)*std::pow(x + y - 1, 3) + 252252*std::pow(z, 5)*std::pow(x + y - 1, 4) + 90090*std::pow(z, 4)*std::pow(x + y - 1, 5) + 18480*std::pow(z, 3)*std::pow(x + y - 1, 6) + 1980*std::pow(z, 2)*std::pow(x + y - 1, 7) + 90*z*std::pow(x + y - 1, 8) - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            9*std::pow(x, 8) + 72*std::pow(x, 7)*y - 72*std::pow(x, 7) + 252*std::pow(x, 6)*std::pow(y, 2) - 504*std::pow(x, 6)*y + 252*std::pow(x, 6) + 504*std::pow(x, 5)*std::pow(y, 3) - 1512*std::pow(x, 5)*std::pow(y, 2) + 1512*std::pow(x, 5)*y - 504*std::pow(x, 5) + 630*std::pow(x, 4)*std::pow(y, 4) - 2520*std::pow(x, 4)*std::pow(y, 3) + 3780*std::pow(x, 4)*std::pow(y, 2) - 2520*std::pow(x, 4)*y + 630*std::pow(x, 4) + 504*std::pow(x, 3)*std::pow(y, 5) - 2520*std::pow(x, 3)*std::pow(y, 4) + 5040*std::pow(x, 3)*std::pow(y, 3) - 5040*std::pow(x, 3)*std::pow(y, 2) + 2520*std::pow(x, 3)*y - 504*std::pow(x, 3) + 252*std::pow(x, 2)*std::pow(y, 6) - 1512*std::pow(x, 2)*std::pow(y, 5) + 3780*std::pow(x, 2)*std::pow(y, 4) - 5040*std::pow(x, 2)*std::pow(y, 3) + 3780*std::pow(x, 2)*std::pow(y, 2) - 1512*std::pow(x, 2)*y + 252*std::pow(x, 2) + 72*x*std::pow(y, 7) - 504*x*std::pow(y, 6) + 1512*x*std::pow(y, 5) - 2520*x*std::pow(y, 4) + 2520*x*std::pow(y, 3) - 1512*x*std::pow(y, 2) + 504*x*y - 72*x + 9*std::pow(y, 8) - 72*std::pow(y, 7) + 252*std::pow(y, 6) - 504*std::pow(y, 5) + 630*std::pow(y, 4) - 504*std::pow(y, 3) + 252*std::pow(y, 2) - 72*y + 218790*std::pow(z, 8) + 411840*std::pow(z, 7)*(2*x + 2*y - 2) + 1261260*std::pow(z, 6)*std::pow(x + y - 1, 2) + 1009008*std::pow(z, 5)*std::pow(x + y - 1, 3) + 450450*std::pow(z, 4)*std::pow(x + y - 1, 4) + 110880*std::pow(z, 3)*std::pow(x + y - 1, 5) + 13860*std::pow(z, 2)*std::pow(x + y - 1, 6) + 720*z*std::pow(x + y - 1, 7) + 9,
            9*std::pow(x, 8) + 72*std::pow(x, 7)*y - 72*std::pow(x, 7) + 252*std::pow(x, 6)*std::pow(y, 2) - 504*std::pow(x, 6)*y + 252*std::pow(x, 6) + 504*std::pow(x, 5)*std::pow(y, 3) - 1512*std::pow(x, 5)*std::pow(y, 2) + 1512*std::pow(x, 5)*y - 504*std::pow(x, 5) + 630*std::pow(x, 4)*std::pow(y, 4) - 2520*std::pow(x, 4)*std::pow(y, 3) + 3780*std::pow(x, 4)*std::pow(y, 2) - 2520*std::pow(x, 4)*y + 630*std::pow(x, 4) + 504*std::pow(x, 3)*std::pow(y, 5) - 2520*std::pow(x, 3)*std::pow(y, 4) + 5040*std::pow(x, 3)*std::pow(y, 3) - 5040*std::pow(x, 3)*std::pow(y, 2) + 2520*std::pow(x, 3)*y - 504*std::pow(x, 3) + 252*std::pow(x, 2)*std::pow(y, 6) - 1512*std::pow(x, 2)*std::pow(y, 5) + 3780*std::pow(x, 2)*std::pow(y, 4) - 5040*std::pow(x, 2)*std::pow(y, 3) + 3780*std::pow(x, 2)*std::pow(y, 2) - 1512*std::pow(x, 2)*y + 252*std::pow(x, 2) + 72*x*std::pow(y, 7) - 504*x*std::pow(y, 6) + 1512*x*std::pow(y, 5) - 2520*x*std::pow(y, 4) + 2520*x*std::pow(y, 3) - 1512*x*std::pow(y, 2) + 504*x*y - 72*x + 9*std::pow(y, 8) - 72*std::pow(y, 7) + 252*std::pow(y, 6) - 504*std::pow(y, 5) + 630*std::pow(y, 4) - 504*std::pow(y, 3) + 252*std::pow(y, 2) - 72*y + 218790*std::pow(z, 8) + 411840*std::pow(z, 7)*(2*x + 2*y - 2) + 1261260*std::pow(z, 6)*std::pow(x + y - 1, 2) + 1009008*std::pow(z, 5)*std::pow(x + y - 1, 3) + 450450*std::pow(z, 4)*std::pow(x + y - 1, 4) + 110880*std::pow(z, 3)*std::pow(x + y - 1, 5) + 13860*std::pow(z, 2)*std::pow(x + y - 1, 6) + 720*z*std::pow(x + y - 1, 7) + 9,
            437580*std::pow(z, 8) + 8*std::pow(z, 7)*(218790*x + 218790*y - 218790) + 2882880*std::pow(z, 6)*std::pow(x + y - 1, 2) + 2522520*std::pow(z, 5)*std::pow(x + y - 1, 3) + 1261260*std::pow(z, 4)*std::pow(x + y - 1, 4) + 360360*std::pow(z, 3)*std::pow(x + y - 1, 5) + 55440*std::pow(z, 2)*std::pow(x + y - 1, 6) + 3960*z*std::pow(x + y - 1, 7) + 90*std::pow(x + y - 1, 8)
        };
    }
    static constexpr uInt Order = 9;
};

// Basis 220
template<>
struct DGBasis<220> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return 13*x*(x*(2*x*(x*(17*x*(x*(19*x*(x*(7*x*(11*x - 50) + 675) - 720) + 8820) - 3528) + 14700) - 2100) + 315) - 10) + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (2*x*(x*(x*(17*x*(x*(19*x*(5*x*(7*x*(22*x - 81) + 864) - 3528) + 31752) - 8820) + 23520) - 1890) + 63) - 1)*(x + 3*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*std::pow(x, 2) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(10*std::pow(y, 2) + y*(8*x - 8) + std::pow(x - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*std::pow(x, 2) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(2*x + 8*y - 2) + (10*std::pow(y, 2) + y*(8*x - 8) + std::pow(x - 1, 2))*(2*x*(17*x*(3*x*(19*x*(5*x*(33*std::pow(x, 2) - 96*x + 112) - 336) + 2100) - 1120) + 1680) + 2*x*(17*x*(3*x*(19*x*(5*x*(33*std::pow(x, 2) - 96*x + 112) - 336) + 2100) - 1120) + x*(51*x*(19*x*(5*x*(33*std::pow(x, 2) - 96*x + 112) - 336) + 2100) + 17*x*(57*x*(5*x*(33*std::pow(x, 2) - 96*x + 112) - 336) + 3*x*(95*x*(33*std::pow(x, 2) - 96*x + 112) + 19*x*(165*std::pow(x, 2) + 5*x*(66*x - 96) - 480*x + 560) - 6384) + 6300) - 19040) + 1680) - 120),
            (2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*std::pow(x, 2) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(8*x + 20*y - 8),
            0
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 223
template<>
struct DGBasis<223> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(35*std::pow(y, 3) + std::pow(y, 2)*(45*x - 45) + 15*y*std::pow(x - 1, 2) + std::pow(x - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(45*std::pow(y, 2) + 15*y*(2*x - 2) + 3*std::pow(x - 1, 2)) + (204*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 4*x*(51*x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) + 51*x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + x*(38*x*(x*(22*x - 49) + 42) + 19*x*(2*x*(22*x - 49) + 2*x*(44*x - 49) + 84) - 665) + 140) - 714) + 112)*(35*std::pow(y, 3) + std::pow(y, 2)*(45*x - 45) + 15*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(105*std::pow(y, 2) + 2*y*(45*x - 45) + 15*std::pow(x - 1, 2)),
            0
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 224
template<>
struct DGBasis<224> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(126*std::pow(y, 4) + std::pow(y, 3)*(224*x - 224) + 126*std::pow(y, 2)*std::pow(x - 1, 2) + 24*y*std::pow(x - 1, 3) + std::pow(x - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(224*std::pow(y, 3) + 126*std::pow(y, 2)*(2*x - 2) + 72*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3)) + (51*x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) + 51*x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + x*(19*x*(7*x*(11*x - 18) + 75) + 19*x*(7*x*(11*x - 18) + x*(154*x - 126) + 75) - 380) + 45) - 102)*(126*std::pow(y, 4) + std::pow(y, 3)*(224*x - 224) + 126*std::pow(y, 2)*std::pow(x - 1, 2) + 24*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(504*std::pow(y, 3) + 3*std::pow(y, 2)*(224*x - 224) + 252*y*std::pow(x - 1, 2) + 24*std::pow(x - 1, 3)),
            0
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 225
template<>
struct DGBasis<225> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(462*std::pow(y, 5) + std::pow(y, 4)*(1050*x - 1050) + 840*std::pow(y, 3)*std::pow(x - 1, 2) + 280*std::pow(y, 2)*std::pow(x - 1, 3) + 35*y*std::pow(x - 1, 4) + std::pow(x - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(1050*std::pow(y, 4) + 840*std::pow(y, 3)*(2*x - 2) + 840*std::pow(y, 2)*std::pow(x - 1, 2) + 140*y*std::pow(x - 1, 3) + 5*std::pow(x - 1, 4)) + (57*x*(x*(21*x*(22*x - 25) + 200) - 30) + 3*x*(19*x*(21*x*(22*x - 25) + 200) + 19*x*(21*x*(22*x - 25) + x*(924*x - 525) + 200) - 570) + 90)*(462*std::pow(y, 5) + std::pow(y, 4)*(1050*x - 1050) + 840*std::pow(y, 3)*std::pow(x - 1, 2) + 280*std::pow(y, 2)*std::pow(x - 1, 3) + 35*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)),
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(2310*std::pow(y, 4) + 4*std::pow(y, 3)*(1050*x - 1050) + 2520*std::pow(y, 2)*std::pow(x - 1, 2) + 560*y*std::pow(x - 1, 3) + 35*std::pow(x - 1, 4)),
            0
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 226
template<>
struct DGBasis<226> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(1716*std::pow(y, 6) + std::pow(y, 5)*(4752*x - 4752) + 4950*std::pow(y, 4)*std::pow(x - 1, 2) + 2400*std::pow(y, 3)*std::pow(x - 1, 3) + 540*std::pow(y, 2)*std::pow(x - 1, 4) + 48*y*std::pow(x - 1, 5) + std::pow(x - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(4752*std::pow(y, 5) + 4950*std::pow(y, 4)*(2*x - 2) + 7200*std::pow(y, 3)*std::pow(x - 1, 2) + 2160*std::pow(y, 2)*std::pow(x - 1, 3) + 240*y*std::pow(x - 1, 4) + 6*std::pow(x - 1, 5)) + (95*x*(7*x*(11*x - 8) + 12) + 19*x*(35*x*(11*x - 8) + 5*x*(154*x - 56) + 60) - 76)*(1716*std::pow(y, 6) + std::pow(y, 5)*(4752*x - 4752) + 4950*std::pow(y, 4)*std::pow(x - 1, 2) + 2400*std::pow(y, 3)*std::pow(x - 1, 3) + 540*std::pow(y, 2)*std::pow(x - 1, 4) + 48*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)),
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(10296*std::pow(y, 5) + 5*std::pow(y, 4)*(4752*x - 4752) + 19800*std::pow(y, 3)*std::pow(x - 1, 2) + 7200*std::pow(y, 2)*std::pow(x - 1, 3) + 1080*y*std::pow(x - 1, 4) + 48*std::pow(x - 1, 5)),
            0
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 227
template<>
struct DGBasis<227> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (10*x*(7*x*(22*x - 9) + 6) - 1)*(6435*std::pow(y, 7) + std::pow(y, 6)*(21021*x - 21021) + 27027*std::pow(y, 5)*std::pow(x - 1, 2) + 17325*std::pow(y, 4)*std::pow(x - 1, 3) + 5775*std::pow(y, 3)*std::pow(x - 1, 4) + 945*std::pow(y, 2)*std::pow(x - 1, 5) + 63*y*std::pow(x - 1, 6) + std::pow(x - 1, 7));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(21021*std::pow(y, 6) + 27027*std::pow(y, 5)*(2*x - 2) + 51975*std::pow(y, 4)*std::pow(x - 1, 2) + 23100*std::pow(y, 3)*std::pow(x - 1, 3) + 4725*std::pow(y, 2)*std::pow(x - 1, 4) + 378*y*std::pow(x - 1, 5) + 7*std::pow(x - 1, 6)) + (70*x*(22*x - 9) + 10*x*(308*x - 63) + 60)*(6435*std::pow(y, 7) + std::pow(y, 6)*(21021*x - 21021) + 27027*std::pow(y, 5)*std::pow(x - 1, 2) + 17325*std::pow(y, 4)*std::pow(x - 1, 3) + 5775*std::pow(y, 3)*std::pow(x - 1, 4) + 945*std::pow(y, 2)*std::pow(x - 1, 5) + 63*y*std::pow(x - 1, 6) + std::pow(x - 1, 7)),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(45045*std::pow(y, 6) + 6*std::pow(y, 5)*(21021*x - 21021) + 135135*std::pow(y, 4)*std::pow(x - 1, 2) + 69300*std::pow(y, 3)*std::pow(x - 1, 3) + 17325*std::pow(y, 2)*std::pow(x - 1, 4) + 1890*y*std::pow(x - 1, 5) + 63*std::pow(x - 1, 6)),
            0
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 228
template<>
struct DGBasis<228> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (21*x*(11*x - 2) + 1)*(24310*std::pow(y, 8) + std::pow(y, 7)*(91520*x - 91520) + 140140*std::pow(y, 6)*std::pow(x - 1, 2) + 112112*std::pow(y, 5)*std::pow(x - 1, 3) + 50050*std::pow(y, 4)*std::pow(x - 1, 4) + 12320*std::pow(y, 3)*std::pow(x - 1, 5) + 1540*std::pow(y, 2)*std::pow(x - 1, 6) + 80*y*std::pow(x - 1, 7) + std::pow(x - 1, 8));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (462*x - 42)*(24310*std::pow(y, 8) + std::pow(y, 7)*(91520*x - 91520) + 140140*std::pow(y, 6)*std::pow(x - 1, 2) + 112112*std::pow(y, 5)*std::pow(x - 1, 3) + 50050*std::pow(y, 4)*std::pow(x - 1, 4) + 12320*std::pow(y, 3)*std::pow(x - 1, 5) + 1540*std::pow(y, 2)*std::pow(x - 1, 6) + 80*y*std::pow(x - 1, 7) + std::pow(x - 1, 8)) + (21*x*(11*x - 2) + 1)*(91520*std::pow(y, 7) + 140140*std::pow(y, 6)*(2*x - 2) + 336336*std::pow(y, 5)*std::pow(x - 1, 2) + 200200*std::pow(y, 4)*std::pow(x - 1, 3) + 61600*std::pow(y, 3)*std::pow(x - 1, 4) + 9240*std::pow(y, 2)*std::pow(x - 1, 5) + 560*y*std::pow(x - 1, 6) + 8*std::pow(x - 1, 7)),
            (21*x*(11*x - 2) + 1)*(194480*std::pow(y, 7) + 7*std::pow(y, 6)*(91520*x - 91520) + 840840*std::pow(y, 5)*std::pow(x - 1, 2) + 560560*std::pow(y, 4)*std::pow(x - 1, 3) + 200200*std::pow(y, 3)*std::pow(x - 1, 4) + 36960*std::pow(y, 2)*std::pow(x - 1, 5) + 3080*y*std::pow(x - 1, 6) + 80*std::pow(x - 1, 7)),
            0
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 229
template<>
struct DGBasis<229> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (22*x - 1)*(92378*std::pow(y, 9) + std::pow(y, 8)*(393822*x - 393822) + 700128*std::pow(y, 7)*std::pow(x - 1, 2) + 672672*std::pow(y, 6)*std::pow(x - 1, 3) + 378378*std::pow(y, 5)*std::pow(x - 1, 4) + 126126*std::pow(y, 4)*std::pow(x - 1, 5) + 24024*std::pow(y, 3)*std::pow(x - 1, 6) + 2376*std::pow(y, 2)*std::pow(x - 1, 7) + 99*y*std::pow(x - 1, 8) + std::pow(x - 1, 9));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            2032316*std::pow(y, 9) + 22*std::pow(y, 8)*(393822*x - 393822) + 15402816*std::pow(y, 7)*std::pow(x - 1, 2) + 14798784*std::pow(y, 6)*std::pow(x - 1, 3) + 8324316*std::pow(y, 5)*std::pow(x - 1, 4) + 2774772*std::pow(y, 4)*std::pow(x - 1, 5) + 528528*std::pow(y, 3)*std::pow(x - 1, 6) + 52272*std::pow(y, 2)*std::pow(x - 1, 7) + 2178*y*std::pow(x - 1, 8) + 22*std::pow(x - 1, 9) + (22*x - 1)*(393822*std::pow(y, 8) + 700128*std::pow(y, 7)*(2*x - 2) + 2018016*std::pow(y, 6)*std::pow(x - 1, 2) + 1513512*std::pow(y, 5)*std::pow(x - 1, 3) + 630630*std::pow(y, 4)*std::pow(x - 1, 4) + 144144*std::pow(y, 3)*std::pow(x - 1, 5) + 16632*std::pow(y, 2)*std::pow(x - 1, 6) + 792*y*std::pow(x - 1, 7) + 9*std::pow(x - 1, 8)),
            (22*x - 1)*(831402*std::pow(y, 8) + 8*std::pow(y, 7)*(393822*x - 393822) + 4900896*std::pow(y, 6)*std::pow(x - 1, 2) + 4036032*std::pow(y, 5)*std::pow(x - 1, 3) + 1891890*std::pow(y, 4)*std::pow(x - 1, 4) + 504504*std::pow(y, 3)*std::pow(x - 1, 5) + 72072*std::pow(y, 2)*std::pow(x - 1, 6) + 4752*y*std::pow(x - 1, 7) + 99*std::pow(x - 1, 8)),
            0
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 230
template<>
struct DGBasis<230> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return 352716*std::pow(y, 10) + std::pow(y, 9)*(1679600*x - 1679600) + 3401190*std::pow(y, 8)*std::pow(x - 1, 2) + 3818880*std::pow(y, 7)*std::pow(x - 1, 3) + 2598960*std::pow(y, 6)*std::pow(x - 1, 4) + 1100736*std::pow(y, 5)*std::pow(x - 1, 5) + 286650*std::pow(y, 4)*std::pow(x - 1, 6) + 43680*std::pow(y, 3)*std::pow(x - 1, 7) + 3510*std::pow(y, 2)*std::pow(x - 1, 8) + 120*y*std::pow(x - 1, 9) + std::pow(x - 1, 10);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            1679600*std::pow(y, 9) + 3401190*std::pow(y, 8)*(2*x - 2) + 11456640*std::pow(y, 7)*std::pow(x - 1, 2) + 10395840*std::pow(y, 6)*std::pow(x - 1, 3) + 5503680*std::pow(y, 5)*std::pow(x - 1, 4) + 1719900*std::pow(y, 4)*std::pow(x - 1, 5) + 305760*std::pow(y, 3)*std::pow(x - 1, 6) + 28080*std::pow(y, 2)*std::pow(x - 1, 7) + 1080*y*std::pow(x - 1, 8) + 10*std::pow(x - 1, 9),
            3527160*std::pow(y, 9) + 9*std::pow(y, 8)*(1679600*x - 1679600) + 27209520*std::pow(y, 7)*std::pow(x - 1, 2) + 26732160*std::pow(y, 6)*std::pow(x - 1, 3) + 15593760*std::pow(y, 5)*std::pow(x - 1, 4) + 5503680*std::pow(y, 4)*std::pow(x - 1, 5) + 1146600*std::pow(y, 3)*std::pow(x - 1, 6) + 131040*std::pow(y, 2)*std::pow(x - 1, 7) + 7020*y*std::pow(x - 1, 8) + 120*std::pow(x - 1, 9),
            0
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 231
template<>
struct DGBasis<231> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (2*x*(x*(x*(17*x*(x*(19*x*(5*x*(7*x*(22*x - 81) + 864) - 3528) + 31752) - 8820) + 23520) - 1890) + 63) - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
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
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*std::pow(x, 2) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(x + 5*y - 1)*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*std::pow(x, 2) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(x + 5*y - 1) + (2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*std::pow(x, 2) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(x + y + 2*z - 1) + (x + 5*y - 1)*(2*x*(17*x*(3*x*(19*x*(5*x*(33*std::pow(x, 2) - 96*x + 112) - 336) + 2100) - 1120) + 1680) + 2*x*(17*x*(3*x*(19*x*(5*x*(33*std::pow(x, 2) - 96*x + 112) - 336) + 2100) - 1120) + x*(51*x*(19*x*(5*x*(33*std::pow(x, 2) - 96*x + 112) - 336) + 2100) + 17*x*(57*x*(5*x*(33*std::pow(x, 2) - 96*x + 112) - 336) + 3*x*(95*x*(33*std::pow(x, 2) - 96*x + 112) + 19*x*(165*std::pow(x, 2) + 5*x*(66*x - 96) - 480*x + 560) - 6384) + 6300) - 19040) + 1680) - 120)*(x + y + 2*z - 1),
            (2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*std::pow(x, 2) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(x + 5*y - 1) + 5*(2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*std::pow(x, 2) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(x + y + 2*z - 1),
            2*(2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*std::pow(x, 2) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(x + 5*y - 1)
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 233
template<>
struct DGBasis<233> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2))*(x + y + 2*z - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(2*x + 12*y - 2)*(x + y + 2*z - 1) + (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2)) + (21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2))*(204*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 4*x*(51*x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) + 51*x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + x*(38*x*(x*(22*x - 49) + 42) + 19*x*(2*x*(22*x - 49) + 2*x*(44*x - 49) + 84) - 665) + 140) - 714) + 112)*(x + y + 2*z - 1),
            (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(12*x + 42*y - 12)*(x + y + 2*z - 1) + (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2)),
            2*(4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(21*std::pow(y, 2) + y*(12*x - 12) + std::pow(x - 1, 2))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 234
template<>
struct DGBasis<234> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(x + y + 2*z - 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(84*std::pow(y, 2) + 21*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(x + y + 2*z - 1) + (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)) + (51*x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) + 51*x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + x*(19*x*(7*x*(11*x - 18) + 75) + 19*x*(7*x*(11*x - 18) + x*(154*x - 126) + 75) - 380) + 45) - 102)*(x + y + 2*z - 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(252*std::pow(y, 2) + 2*y*(84*x - 84) + 21*std::pow(x - 1, 2))*(x + y + 2*z - 1) + (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            2*(51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(84*std::pow(y, 3) + std::pow(y, 2)*(84*x - 84) + 21*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 235
template<>
struct DGBasis<235> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(x + y + 2*z - 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(x + y + 2*z - 1)*(480*std::pow(y, 3) + 216*std::pow(y, 2)*(2*x - 2) + 96*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3)) + (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)) + (57*x*(x*(21*x*(22*x - 25) + 200) - 30) + 3*x*(19*x*(21*x*(22*x - 25) + 200) + 19*x*(21*x*(22*x - 25) + x*(924*x - 525) + 200) - 570) + 90)*(x + y + 2*z - 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(x + y + 2*z - 1)*(1320*std::pow(y, 3) + 3*std::pow(y, 2)*(480*x - 480) + 432*y*std::pow(x - 1, 2) + 32*std::pow(x - 1, 3)) + (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            2*(3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(330*std::pow(y, 4) + std::pow(y, 3)*(480*x - 480) + 216*std::pow(y, 2)*std::pow(x - 1, 2) + 32*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 236
template<>
struct DGBasis<236> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(x + y + 2*z - 1)*(1287*std::pow(y, 5) + std::pow(y, 4)*(2475*x - 2475) + 1650*std::pow(y, 3)*std::pow(x - 1, 2) + 450*std::pow(y, 2)*std::pow(x - 1, 3) + 45*y*std::pow(x - 1, 4) + std::pow(x - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(x + y + 2*z - 1)*(2475*std::pow(y, 4) + 1650*std::pow(y, 3)*(2*x - 2) + 1350*std::pow(y, 2)*std::pow(x - 1, 2) + 180*y*std::pow(x - 1, 3) + 5*std::pow(x - 1, 4)) + (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(1287*std::pow(y, 5) + std::pow(y, 4)*(2475*x - 2475) + 1650*std::pow(y, 3)*std::pow(x - 1, 2) + 450*std::pow(y, 2)*std::pow(x - 1, 3) + 45*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)) + (95*x*(7*x*(11*x - 8) + 12) + 19*x*(35*x*(11*x - 8) + 5*x*(154*x - 56) + 60) - 76)*(x + y + 2*z - 1)*(1287*std::pow(y, 5) + std::pow(y, 4)*(2475*x - 2475) + 1650*std::pow(y, 3)*std::pow(x - 1, 2) + 450*std::pow(y, 2)*std::pow(x - 1, 3) + 45*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)),
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(x + y + 2*z - 1)*(6435*std::pow(y, 4) + 4*std::pow(y, 3)*(2475*x - 2475) + 4950*std::pow(y, 2)*std::pow(x - 1, 2) + 900*y*std::pow(x - 1, 3) + 45*std::pow(x - 1, 4)) + (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(1287*std::pow(y, 5) + std::pow(y, 4)*(2475*x - 2475) + 1650*std::pow(y, 3)*std::pow(x - 1, 2) + 450*std::pow(y, 2)*std::pow(x - 1, 3) + 45*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)),
            2*(19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(1287*std::pow(y, 5) + std::pow(y, 4)*(2475*x - 2475) + 1650*std::pow(y, 3)*std::pow(x - 1, 2) + 450*std::pow(y, 2)*std::pow(x - 1, 3) + 45*y*std::pow(x - 1, 4) + std::pow(x - 1, 5))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 237
template<>
struct DGBasis<237> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (10*x*(7*x*(22*x - 9) + 6) - 1)*(x + y + 2*z - 1)*(5005*std::pow(y, 6) + std::pow(y, 5)*(12012*x - 12012) + 10725*std::pow(y, 4)*std::pow(x - 1, 2) + 4400*std::pow(y, 3)*std::pow(x - 1, 3) + 825*std::pow(y, 2)*std::pow(x - 1, 4) + 60*y*std::pow(x - 1, 5) + std::pow(x - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(x + y + 2*z - 1)*(12012*std::pow(y, 5) + 10725*std::pow(y, 4)*(2*x - 2) + 13200*std::pow(y, 3)*std::pow(x - 1, 2) + 3300*std::pow(y, 2)*std::pow(x - 1, 3) + 300*y*std::pow(x - 1, 4) + 6*std::pow(x - 1, 5)) + (10*x*(7*x*(22*x - 9) + 6) - 1)*(5005*std::pow(y, 6) + std::pow(y, 5)*(12012*x - 12012) + 10725*std::pow(y, 4)*std::pow(x - 1, 2) + 4400*std::pow(y, 3)*std::pow(x - 1, 3) + 825*std::pow(y, 2)*std::pow(x - 1, 4) + 60*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)) + (70*x*(22*x - 9) + 10*x*(308*x - 63) + 60)*(x + y + 2*z - 1)*(5005*std::pow(y, 6) + std::pow(y, 5)*(12012*x - 12012) + 10725*std::pow(y, 4)*std::pow(x - 1, 2) + 4400*std::pow(y, 3)*std::pow(x - 1, 3) + 825*std::pow(y, 2)*std::pow(x - 1, 4) + 60*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(x + y + 2*z - 1)*(30030*std::pow(y, 5) + 5*std::pow(y, 4)*(12012*x - 12012) + 42900*std::pow(y, 3)*std::pow(x - 1, 2) + 13200*std::pow(y, 2)*std::pow(x - 1, 3) + 1650*y*std::pow(x - 1, 4) + 60*std::pow(x - 1, 5)) + (10*x*(7*x*(22*x - 9) + 6) - 1)*(5005*std::pow(y, 6) + std::pow(y, 5)*(12012*x - 12012) + 10725*std::pow(y, 4)*std::pow(x - 1, 2) + 4400*std::pow(y, 3)*std::pow(x - 1, 3) + 825*std::pow(y, 2)*std::pow(x - 1, 4) + 60*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)),
            2*(10*x*(7*x*(22*x - 9) + 6) - 1)*(5005*std::pow(y, 6) + std::pow(y, 5)*(12012*x - 12012) + 10725*std::pow(y, 4)*std::pow(x - 1, 2) + 4400*std::pow(y, 3)*std::pow(x - 1, 3) + 825*std::pow(y, 2)*std::pow(x - 1, 4) + 60*y*std::pow(x - 1, 5) + std::pow(x - 1, 6))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 238
template<>
struct DGBasis<238> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (21*x*(11*x - 2) + 1)*(x + y + 2*z - 1)*(19448*std::pow(y, 7) + std::pow(y, 6)*(56056*x - 56056) + 63063*std::pow(y, 5)*std::pow(x - 1, 2) + 35035*std::pow(y, 4)*std::pow(x - 1, 3) + 10010*std::pow(y, 3)*std::pow(x - 1, 4) + 1386*std::pow(y, 2)*std::pow(x - 1, 5) + 77*y*std::pow(x - 1, 6) + std::pow(x - 1, 7));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (462*x - 42)*(x + y + 2*z - 1)*(19448*std::pow(y, 7) + std::pow(y, 6)*(56056*x - 56056) + 63063*std::pow(y, 5)*std::pow(x - 1, 2) + 35035*std::pow(y, 4)*std::pow(x - 1, 3) + 10010*std::pow(y, 3)*std::pow(x - 1, 4) + 1386*std::pow(y, 2)*std::pow(x - 1, 5) + 77*y*std::pow(x - 1, 6) + std::pow(x - 1, 7)) + (21*x*(11*x - 2) + 1)*(x + y + 2*z - 1)*(56056*std::pow(y, 6) + 63063*std::pow(y, 5)*(2*x - 2) + 105105*std::pow(y, 4)*std::pow(x - 1, 2) + 40040*std::pow(y, 3)*std::pow(x - 1, 3) + 6930*std::pow(y, 2)*std::pow(x - 1, 4) + 462*y*std::pow(x - 1, 5) + 7*std::pow(x - 1, 6)) + (21*x*(11*x - 2) + 1)*(19448*std::pow(y, 7) + std::pow(y, 6)*(56056*x - 56056) + 63063*std::pow(y, 5)*std::pow(x - 1, 2) + 35035*std::pow(y, 4)*std::pow(x - 1, 3) + 10010*std::pow(y, 3)*std::pow(x - 1, 4) + 1386*std::pow(y, 2)*std::pow(x - 1, 5) + 77*y*std::pow(x - 1, 6) + std::pow(x - 1, 7)),
            (21*x*(11*x - 2) + 1)*(x + y + 2*z - 1)*(136136*std::pow(y, 6) + 6*std::pow(y, 5)*(56056*x - 56056) + 315315*std::pow(y, 4)*std::pow(x - 1, 2) + 140140*std::pow(y, 3)*std::pow(x - 1, 3) + 30030*std::pow(y, 2)*std::pow(x - 1, 4) + 2772*y*std::pow(x - 1, 5) + 77*std::pow(x - 1, 6)) + (21*x*(11*x - 2) + 1)*(19448*std::pow(y, 7) + std::pow(y, 6)*(56056*x - 56056) + 63063*std::pow(y, 5)*std::pow(x - 1, 2) + 35035*std::pow(y, 4)*std::pow(x - 1, 3) + 10010*std::pow(y, 3)*std::pow(x - 1, 4) + 1386*std::pow(y, 2)*std::pow(x - 1, 5) + 77*y*std::pow(x - 1, 6) + std::pow(x - 1, 7)),
            2*(21*x*(11*x - 2) + 1)*(19448*std::pow(y, 7) + std::pow(y, 6)*(56056*x - 56056) + 63063*std::pow(y, 5)*std::pow(x - 1, 2) + 35035*std::pow(y, 4)*std::pow(x - 1, 3) + 10010*std::pow(y, 3)*std::pow(x - 1, 4) + 1386*std::pow(y, 2)*std::pow(x - 1, 5) + 77*y*std::pow(x - 1, 6) + std::pow(x - 1, 7))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 239
template<>
struct DGBasis<239> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (22*x - 1)*(x + y + 2*z - 1)*(75582*std::pow(y, 8) + std::pow(y, 7)*(254592*x - 254592) + 346528*std::pow(y, 6)*std::pow(x - 1, 2) + 244608*std::pow(y, 5)*std::pow(x - 1, 3) + 95550*std::pow(y, 4)*std::pow(x - 1, 4) + 20384*std::pow(y, 3)*std::pow(x - 1, 5) + 2184*std::pow(y, 2)*std::pow(x - 1, 6) + 96*y*std::pow(x - 1, 7) + std::pow(x - 1, 8));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (22*x - 1)*(x + y + 2*z - 1)*(254592*std::pow(y, 7) + 346528*std::pow(y, 6)*(2*x - 2) + 733824*std::pow(y, 5)*std::pow(x - 1, 2) + 382200*std::pow(y, 4)*std::pow(x - 1, 3) + 101920*std::pow(y, 3)*std::pow(x - 1, 4) + 13104*std::pow(y, 2)*std::pow(x - 1, 5) + 672*y*std::pow(x - 1, 6) + 8*std::pow(x - 1, 7)) + (22*x - 1)*(75582*std::pow(y, 8) + std::pow(y, 7)*(254592*x - 254592) + 346528*std::pow(y, 6)*std::pow(x - 1, 2) + 244608*std::pow(y, 5)*std::pow(x - 1, 3) + 95550*std::pow(y, 4)*std::pow(x - 1, 4) + 20384*std::pow(y, 3)*std::pow(x - 1, 5) + 2184*std::pow(y, 2)*std::pow(x - 1, 6) + 96*y*std::pow(x - 1, 7) + std::pow(x - 1, 8)) + 22*(x + y + 2*z - 1)*(75582*std::pow(y, 8) + std::pow(y, 7)*(254592*x - 254592) + 346528*std::pow(y, 6)*std::pow(x - 1, 2) + 244608*std::pow(y, 5)*std::pow(x - 1, 3) + 95550*std::pow(y, 4)*std::pow(x - 1, 4) + 20384*std::pow(y, 3)*std::pow(x - 1, 5) + 2184*std::pow(y, 2)*std::pow(x - 1, 6) + 96*y*std::pow(x - 1, 7) + std::pow(x - 1, 8)),
            (22*x - 1)*(x + y + 2*z - 1)*(604656*std::pow(y, 7) + 7*std::pow(y, 6)*(254592*x - 254592) + 2079168*std::pow(y, 5)*std::pow(x - 1, 2) + 1223040*std::pow(y, 4)*std::pow(x - 1, 3) + 382200*std::pow(y, 3)*std::pow(x - 1, 4) + 61152*std::pow(y, 2)*std::pow(x - 1, 5) + 4368*y*std::pow(x - 1, 6) + 96*std::pow(x - 1, 7)) + (22*x - 1)*(75582*std::pow(y, 8) + std::pow(y, 7)*(254592*x - 254592) + 346528*std::pow(y, 6)*std::pow(x - 1, 2) + 244608*std::pow(y, 5)*std::pow(x - 1, 3) + 95550*std::pow(y, 4)*std::pow(x - 1, 4) + 20384*std::pow(y, 3)*std::pow(x - 1, 5) + 2184*std::pow(y, 2)*std::pow(x - 1, 6) + 96*y*std::pow(x - 1, 7) + std::pow(x - 1, 8)),
            2*(22*x - 1)*(75582*std::pow(y, 8) + std::pow(y, 7)*(254592*x - 254592) + 346528*std::pow(y, 6)*std::pow(x - 1, 2) + 244608*std::pow(y, 5)*std::pow(x - 1, 3) + 95550*std::pow(y, 4)*std::pow(x - 1, 4) + 20384*std::pow(y, 3)*std::pow(x - 1, 5) + 2184*std::pow(y, 2)*std::pow(x - 1, 6) + 96*y*std::pow(x - 1, 7) + std::pow(x - 1, 8))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 240
template<>
struct DGBasis<240> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x + y + 2*z - 1)*(293930*std::pow(y, 9) + std::pow(y, 8)*(1133730*x - 1133730) + 1813968*std::pow(y, 7)*std::pow(x - 1, 2) + 1559376*std::pow(y, 6)*std::pow(x - 1, 3) + 779688*std::pow(y, 5)*std::pow(x - 1, 4) + 229320*std::pow(y, 4)*std::pow(x - 1, 5) + 38220*std::pow(y, 3)*std::pow(x - 1, 6) + 3276*std::pow(y, 2)*std::pow(x - 1, 7) + 117*y*std::pow(x - 1, 8) + std::pow(x - 1, 9));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            293930*std::pow(y, 9) + std::pow(y, 8)*(1133730*x - 1133730) + 1813968*std::pow(y, 7)*std::pow(x - 1, 2) + 1559376*std::pow(y, 6)*std::pow(x - 1, 3) + 779688*std::pow(y, 5)*std::pow(x - 1, 4) + 229320*std::pow(y, 4)*std::pow(x - 1, 5) + 38220*std::pow(y, 3)*std::pow(x - 1, 6) + 3276*std::pow(y, 2)*std::pow(x - 1, 7) + 117*y*std::pow(x - 1, 8) + std::pow(x - 1, 9) + (x + y + 2*z - 1)*(1133730*std::pow(y, 8) + 1813968*std::pow(y, 7)*(2*x - 2) + 4678128*std::pow(y, 6)*std::pow(x - 1, 2) + 3118752*std::pow(y, 5)*std::pow(x - 1, 3) + 1146600*std::pow(y, 4)*std::pow(x - 1, 4) + 229320*std::pow(y, 3)*std::pow(x - 1, 5) + 22932*std::pow(y, 2)*std::pow(x - 1, 6) + 936*y*std::pow(x - 1, 7) + 9*std::pow(x - 1, 8)),
            293930*std::pow(y, 9) + std::pow(y, 8)*(1133730*x - 1133730) + 1813968*std::pow(y, 7)*std::pow(x - 1, 2) + 1559376*std::pow(y, 6)*std::pow(x - 1, 3) + 779688*std::pow(y, 5)*std::pow(x - 1, 4) + 229320*std::pow(y, 4)*std::pow(x - 1, 5) + 38220*std::pow(y, 3)*std::pow(x - 1, 6) + 3276*std::pow(y, 2)*std::pow(x - 1, 7) + 117*y*std::pow(x - 1, 8) + std::pow(x - 1, 9) + (x + y + 2*z - 1)*(2645370*std::pow(y, 8) + 8*std::pow(y, 7)*(1133730*x - 1133730) + 12697776*std::pow(y, 6)*std::pow(x - 1, 2) + 9356256*std::pow(y, 5)*std::pow(x - 1, 3) + 3898440*std::pow(y, 4)*std::pow(x - 1, 4) + 917280*std::pow(y, 3)*std::pow(x - 1, 5) + 114660*std::pow(y, 2)*std::pow(x - 1, 6) + 6552*y*std::pow(x - 1, 7) + 117*std::pow(x - 1, 8)),
            587860*std::pow(y, 9) + 2*std::pow(y, 8)*(1133730*x - 1133730) + 3627936*std::pow(y, 7)*std::pow(x - 1, 2) + 3118752*std::pow(y, 6)*std::pow(x - 1, 3) + 1559376*std::pow(y, 5)*std::pow(x - 1, 4) + 458640*std::pow(y, 4)*std::pow(x - 1, 5) + 76440*std::pow(y, 3)*std::pow(x - 1, 6) + 6552*std::pow(y, 2)*std::pow(x - 1, 7) + 234*y*std::pow(x - 1, 8) + 2*std::pow(x - 1, 9)
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 241
template<>
struct DGBasis<241> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*std::pow(x, 2) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*std::pow(x, 2) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(2*x + 2*y + 6*z - 2) + (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(2*x*(17*x*(3*x*(19*x*(5*x*(33*std::pow(x, 2) - 96*x + 112) - 336) + 2100) - 1120) + 1680) + 2*x*(17*x*(3*x*(19*x*(5*x*(33*std::pow(x, 2) - 96*x + 112) - 336) + 2100) - 1120) + x*(51*x*(19*x*(5*x*(33*std::pow(x, 2) - 96*x + 112) - 336) + 2100) + 17*x*(57*x*(5*x*(33*std::pow(x, 2) - 96*x + 112) - 336) + 3*x*(95*x*(33*std::pow(x, 2) - 96*x + 112) + 19*x*(165*std::pow(x, 2) + 5*x*(66*x - 96) - 480*x + 560) - 6384) + 6300) - 19040) + 1680) - 120),
            (2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*std::pow(x, 2) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(2*x + 2*y + 6*z - 2),
            (2*x*(x*(17*x*(3*x*(19*x*(5*x*(33*std::pow(x, 2) - 96*x + 112) - 336) + 2100) - 1120) + 1680) - 60) + 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 242
template<>
struct DGBasis<242> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(x + 7*y - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (x + 7*y - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(204*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 4*x*(51*x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) + 51*x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + x*(38*x*(x*(22*x - 49) + 42) + 19*x*(2*x*(22*x - 49) + 2*x*(44*x - 49) + 84) - 665) + 140) - 714) + 112),
            (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(x + 7*y - 1)*(2*x + 2*y + 6*z - 2) + 7*(4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)),
            (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(x + 7*y - 1)*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 243
template<>
struct DGBasis<243> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(2*x + 16*y - 2)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(2*x + 2*y + 6*z - 2) + (36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(51*x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) + 51*x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + x*(19*x*(7*x*(11*x - 18) + 75) + 19*x*(7*x*(11*x - 18) + x*(154*x - 126) + 75) - 380) + 45) - 102),
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(16*x + 72*y - 16)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(2*x + 2*y + 6*z - 2),
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(36*std::pow(y, 2) + y*(16*x - 16) + std::pow(x - 1, 2))*(6*x + 6*y + 12*z - 6)
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 244
template<>
struct DGBasis<244> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(135*std::pow(y, 2) + 27*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(2*x + 2*y + 6*z - 2)*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)) + (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(57*x*(x*(21*x*(22*x - 25) + 200) - 30) + 3*x*(19*x*(21*x*(22*x - 25) + 200) + 19*x*(21*x*(22*x - 25) + x*(924*x - 525) + 200) - 570) + 90)*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(495*std::pow(y, 2) + 2*y*(135*x - 135) + 27*std::pow(x - 1, 2))*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2)) + (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(2*x + 2*y + 6*z - 2)*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(6*x + 6*y + 12*z - 6)*(165*std::pow(y, 3) + std::pow(y, 2)*(135*x - 135) + 27*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 245
template<>
struct DGBasis<245> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(715*std::pow(y, 4) + std::pow(y, 3)*(880*x - 880) + 330*std::pow(y, 2)*std::pow(x - 1, 2) + 40*y*std::pow(x - 1, 3) + std::pow(x - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(880*std::pow(y, 3) + 330*std::pow(y, 2)*(2*x - 2) + 120*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3)) + (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(2*x + 2*y + 6*z - 2)*(715*std::pow(y, 4) + std::pow(y, 3)*(880*x - 880) + 330*std::pow(y, 2)*std::pow(x - 1, 2) + 40*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)) + (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(95*x*(7*x*(11*x - 8) + 12) + 19*x*(35*x*(11*x - 8) + 5*x*(154*x - 56) + 60) - 76)*(715*std::pow(y, 4) + std::pow(y, 3)*(880*x - 880) + 330*std::pow(y, 2)*std::pow(x - 1, 2) + 40*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(2860*std::pow(y, 3) + 3*std::pow(y, 2)*(880*x - 880) + 660*y*std::pow(x - 1, 2) + 40*std::pow(x - 1, 3)) + (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(2*x + 2*y + 6*z - 2)*(715*std::pow(y, 4) + std::pow(y, 3)*(880*x - 880) + 330*std::pow(y, 2)*std::pow(x - 1, 2) + 40*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(6*x + 6*y + 12*z - 6)*(715*std::pow(y, 4) + std::pow(y, 3)*(880*x - 880) + 330*std::pow(y, 2)*std::pow(x - 1, 2) + 40*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 246
template<>
struct DGBasis<246> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (10*x*(7*x*(22*x - 9) + 6) - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(3003*std::pow(y, 5) + std::pow(y, 4)*(5005*x - 5005) + 2860*std::pow(y, 3)*std::pow(x - 1, 2) + 660*std::pow(y, 2)*std::pow(x - 1, 3) + 55*y*std::pow(x - 1, 4) + std::pow(x - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(5005*std::pow(y, 4) + 2860*std::pow(y, 3)*(2*x - 2) + 1980*std::pow(y, 2)*std::pow(x - 1, 2) + 220*y*std::pow(x - 1, 3) + 5*std::pow(x - 1, 4)) + (10*x*(7*x*(22*x - 9) + 6) - 1)*(2*x + 2*y + 6*z - 2)*(3003*std::pow(y, 5) + std::pow(y, 4)*(5005*x - 5005) + 2860*std::pow(y, 3)*std::pow(x - 1, 2) + 660*std::pow(y, 2)*std::pow(x - 1, 3) + 55*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)) + (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(70*x*(22*x - 9) + 10*x*(308*x - 63) + 60)*(3003*std::pow(y, 5) + std::pow(y, 4)*(5005*x - 5005) + 2860*std::pow(y, 3)*std::pow(x - 1, 2) + 660*std::pow(y, 2)*std::pow(x - 1, 3) + 55*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(15015*std::pow(y, 4) + 4*std::pow(y, 3)*(5005*x - 5005) + 8580*std::pow(y, 2)*std::pow(x - 1, 2) + 1320*y*std::pow(x - 1, 3) + 55*std::pow(x - 1, 4)) + (10*x*(7*x*(22*x - 9) + 6) - 1)*(2*x + 2*y + 6*z - 2)*(3003*std::pow(y, 5) + std::pow(y, 4)*(5005*x - 5005) + 2860*std::pow(y, 3)*std::pow(x - 1, 2) + 660*std::pow(y, 2)*std::pow(x - 1, 3) + 55*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(6*x + 6*y + 12*z - 6)*(3003*std::pow(y, 5) + std::pow(y, 4)*(5005*x - 5005) + 2860*std::pow(y, 3)*std::pow(x - 1, 2) + 660*std::pow(y, 2)*std::pow(x - 1, 3) + 55*y*std::pow(x - 1, 4) + std::pow(x - 1, 5))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 247
template<>
struct DGBasis<247> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (21*x*(11*x - 2) + 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(12376*std::pow(y, 6) + std::pow(y, 5)*(26208*x - 26208) + 20475*std::pow(y, 4)*std::pow(x - 1, 2) + 7280*std::pow(y, 3)*std::pow(x - 1, 3) + 1170*std::pow(y, 2)*std::pow(x - 1, 4) + 72*y*std::pow(x - 1, 5) + std::pow(x - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (462*x - 42)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(12376*std::pow(y, 6) + std::pow(y, 5)*(26208*x - 26208) + 20475*std::pow(y, 4)*std::pow(x - 1, 2) + 7280*std::pow(y, 3)*std::pow(x - 1, 3) + 1170*std::pow(y, 2)*std::pow(x - 1, 4) + 72*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)) + (21*x*(11*x - 2) + 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(26208*std::pow(y, 5) + 20475*std::pow(y, 4)*(2*x - 2) + 21840*std::pow(y, 3)*std::pow(x - 1, 2) + 4680*std::pow(y, 2)*std::pow(x - 1, 3) + 360*y*std::pow(x - 1, 4) + 6*std::pow(x - 1, 5)) + (21*x*(11*x - 2) + 1)*(2*x + 2*y + 6*z - 2)*(12376*std::pow(y, 6) + std::pow(y, 5)*(26208*x - 26208) + 20475*std::pow(y, 4)*std::pow(x - 1, 2) + 7280*std::pow(y, 3)*std::pow(x - 1, 3) + 1170*std::pow(y, 2)*std::pow(x - 1, 4) + 72*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)),
            (21*x*(11*x - 2) + 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(74256*std::pow(y, 5) + 5*std::pow(y, 4)*(26208*x - 26208) + 81900*std::pow(y, 3)*std::pow(x - 1, 2) + 21840*std::pow(y, 2)*std::pow(x - 1, 3) + 2340*y*std::pow(x - 1, 4) + 72*std::pow(x - 1, 5)) + (21*x*(11*x - 2) + 1)*(2*x + 2*y + 6*z - 2)*(12376*std::pow(y, 6) + std::pow(y, 5)*(26208*x - 26208) + 20475*std::pow(y, 4)*std::pow(x - 1, 2) + 7280*std::pow(y, 3)*std::pow(x - 1, 3) + 1170*std::pow(y, 2)*std::pow(x - 1, 4) + 72*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)),
            (21*x*(11*x - 2) + 1)*(6*x + 6*y + 12*z - 6)*(12376*std::pow(y, 6) + std::pow(y, 5)*(26208*x - 26208) + 20475*std::pow(y, 4)*std::pow(x - 1, 2) + 7280*std::pow(y, 3)*std::pow(x - 1, 3) + 1170*std::pow(y, 2)*std::pow(x - 1, 4) + 72*y*std::pow(x - 1, 5) + std::pow(x - 1, 6))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 248
template<>
struct DGBasis<248> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (22*x - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(50388*std::pow(y, 7) + std::pow(y, 6)*(129948*x - 129948) + 129948*std::pow(y, 5)*std::pow(x - 1, 2) + 63700*std::pow(y, 4)*std::pow(x - 1, 3) + 15925*std::pow(y, 3)*std::pow(x - 1, 4) + 1911*std::pow(y, 2)*std::pow(x - 1, 5) + 91*y*std::pow(x - 1, 6) + std::pow(x - 1, 7));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (22*x - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(129948*std::pow(y, 6) + 129948*std::pow(y, 5)*(2*x - 2) + 191100*std::pow(y, 4)*std::pow(x - 1, 2) + 63700*std::pow(y, 3)*std::pow(x - 1, 3) + 9555*std::pow(y, 2)*std::pow(x - 1, 4) + 546*y*std::pow(x - 1, 5) + 7*std::pow(x - 1, 6)) + (22*x - 1)*(2*x + 2*y + 6*z - 2)*(50388*std::pow(y, 7) + std::pow(y, 6)*(129948*x - 129948) + 129948*std::pow(y, 5)*std::pow(x - 1, 2) + 63700*std::pow(y, 4)*std::pow(x - 1, 3) + 15925*std::pow(y, 3)*std::pow(x - 1, 4) + 1911*std::pow(y, 2)*std::pow(x - 1, 5) + 91*y*std::pow(x - 1, 6) + std::pow(x - 1, 7)) + 22*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(50388*std::pow(y, 7) + std::pow(y, 6)*(129948*x - 129948) + 129948*std::pow(y, 5)*std::pow(x - 1, 2) + 63700*std::pow(y, 4)*std::pow(x - 1, 3) + 15925*std::pow(y, 3)*std::pow(x - 1, 4) + 1911*std::pow(y, 2)*std::pow(x - 1, 5) + 91*y*std::pow(x - 1, 6) + std::pow(x - 1, 7)),
            (22*x - 1)*(6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(352716*std::pow(y, 6) + 6*std::pow(y, 5)*(129948*x - 129948) + 649740*std::pow(y, 4)*std::pow(x - 1, 2) + 254800*std::pow(y, 3)*std::pow(x - 1, 3) + 47775*std::pow(y, 2)*std::pow(x - 1, 4) + 3822*y*std::pow(x - 1, 5) + 91*std::pow(x - 1, 6)) + (22*x - 1)*(2*x + 2*y + 6*z - 2)*(50388*std::pow(y, 7) + std::pow(y, 6)*(129948*x - 129948) + 129948*std::pow(y, 5)*std::pow(x - 1, 2) + 63700*std::pow(y, 4)*std::pow(x - 1, 3) + 15925*std::pow(y, 3)*std::pow(x - 1, 4) + 1911*std::pow(y, 2)*std::pow(x - 1, 5) + 91*y*std::pow(x - 1, 6) + std::pow(x - 1, 7)),
            (22*x - 1)*(6*x + 6*y + 12*z - 6)*(50388*std::pow(y, 7) + std::pow(y, 6)*(129948*x - 129948) + 129948*std::pow(y, 5)*std::pow(x - 1, 2) + 63700*std::pow(y, 4)*std::pow(x - 1, 3) + 15925*std::pow(y, 3)*std::pow(x - 1, 4) + 1911*std::pow(y, 2)*std::pow(x - 1, 5) + 91*y*std::pow(x - 1, 6) + std::pow(x - 1, 7))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 249
template<>
struct DGBasis<249> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(203490*std::pow(y, 8) + std::pow(y, 7)*(620160*x - 620160) + 759696*std::pow(y, 6)*std::pow(x - 1, 2) + 479808*std::pow(y, 5)*std::pow(x - 1, 3) + 166600*std::pow(y, 4)*std::pow(x - 1, 4) + 31360*std::pow(y, 3)*std::pow(x - 1, 5) + 2940*std::pow(y, 2)*std::pow(x - 1, 6) + 112*y*std::pow(x - 1, 7) + std::pow(x - 1, 8));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(620160*std::pow(y, 7) + 759696*std::pow(y, 6)*(2*x - 2) + 1439424*std::pow(y, 5)*std::pow(x - 1, 2) + 666400*std::pow(y, 4)*std::pow(x - 1, 3) + 156800*std::pow(y, 3)*std::pow(x - 1, 4) + 17640*std::pow(y, 2)*std::pow(x - 1, 5) + 784*y*std::pow(x - 1, 6) + 8*std::pow(x - 1, 7)) + (2*x + 2*y + 6*z - 2)*(203490*std::pow(y, 8) + std::pow(y, 7)*(620160*x - 620160) + 759696*std::pow(y, 6)*std::pow(x - 1, 2) + 479808*std::pow(y, 5)*std::pow(x - 1, 3) + 166600*std::pow(y, 4)*std::pow(x - 1, 4) + 31360*std::pow(y, 3)*std::pow(x - 1, 5) + 2940*std::pow(y, 2)*std::pow(x - 1, 6) + 112*y*std::pow(x - 1, 7) + std::pow(x - 1, 8)),
            (6*std::pow(z, 2) + z*(6*x + 6*y - 6) + std::pow(x + y - 1, 2))*(1627920*std::pow(y, 7) + 7*std::pow(y, 6)*(620160*x - 620160) + 4558176*std::pow(y, 5)*std::pow(x - 1, 2) + 2399040*std::pow(y, 4)*std::pow(x - 1, 3) + 666400*std::pow(y, 3)*std::pow(x - 1, 4) + 94080*std::pow(y, 2)*std::pow(x - 1, 5) + 5880*y*std::pow(x - 1, 6) + 112*std::pow(x - 1, 7)) + (2*x + 2*y + 6*z - 2)*(203490*std::pow(y, 8) + std::pow(y, 7)*(620160*x - 620160) + 759696*std::pow(y, 6)*std::pow(x - 1, 2) + 479808*std::pow(y, 5)*std::pow(x - 1, 3) + 166600*std::pow(y, 4)*std::pow(x - 1, 4) + 31360*std::pow(y, 3)*std::pow(x - 1, 5) + 2940*std::pow(y, 2)*std::pow(x - 1, 6) + 112*y*std::pow(x - 1, 7) + std::pow(x - 1, 8)),
            (6*x + 6*y + 12*z - 6)*(203490*std::pow(y, 8) + std::pow(y, 7)*(620160*x - 620160) + 759696*std::pow(y, 6)*std::pow(x - 1, 2) + 479808*std::pow(y, 5)*std::pow(x - 1, 3) + 166600*std::pow(y, 4)*std::pow(x - 1, 4) + 31360*std::pow(y, 3)*std::pow(x - 1, 5) + 2940*std::pow(y, 2)*std::pow(x - 1, 6) + 112*y*std::pow(x - 1, 7) + std::pow(x - 1, 8))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 250
template<>
struct DGBasis<250> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)) + (204*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 4*x*(51*x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) + 51*x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + x*(38*x*(x*(22*x - 49) + 42) + 19*x*(2*x*(22*x - 49) + 2*x*(44*x - 49) + 84) - 665) + 140) - 714) + 112)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)),
            (4*x*(51*x*(x*(19*x*(2*x*(x*(22*x - 49) + 42) - 35) + 140) - 14) + 28) - 1)*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 251
template<>
struct DGBasis<251> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(x + 9*y - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(x + 9*y - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)) + (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (x + 9*y - 1)*(51*x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) + 51*x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + x*(19*x*(7*x*(11*x - 18) + 75) + 19*x*(7*x*(11*x - 18) + x*(154*x - 126) + 75) - 380) + 45) - 102)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(x + 9*y - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)) + 9*(51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(x + 9*y - 1)*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 252
template<>
struct DGBasis<252> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(2*x + 20*y - 2)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)) + (55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(57*x*(x*(21*x*(22*x - 25) + 200) - 30) + 3*x*(19*x*(21*x*(22*x - 25) + 200) + 19*x*(21*x*(22*x - 25) + x*(924*x - 525) + 200) - 570) + 90)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(20*x + 110*y - 20)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2)),
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(55*std::pow(y, 2) + y*(20*x - 20) + std::pow(x - 1, 2))*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 253
template<>
struct DGBasis<253> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(286*std::pow(y, 3) + std::pow(y, 2)*(198*x - 198) + 33*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(198*std::pow(y, 2) + 33*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(286*std::pow(y, 3) + std::pow(y, 2)*(198*x - 198) + 33*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)) + (95*x*(7*x*(11*x - 8) + 12) + 19*x*(35*x*(11*x - 8) + 5*x*(154*x - 56) + 60) - 76)*(286*std::pow(y, 3) + std::pow(y, 2)*(198*x - 198) + 33*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(858*std::pow(y, 2) + 2*y*(198*x - 198) + 33*std::pow(x - 1, 2))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(286*std::pow(y, 3) + std::pow(y, 2)*(198*x - 198) + 33*y*std::pow(x - 1, 2) + std::pow(x - 1, 3)),
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))*(286*std::pow(y, 3) + std::pow(y, 2)*(198*x - 198) + 33*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 254
template<>
struct DGBasis<254> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (10*x*(7*x*(22*x - 9) + 6) - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(1365*std::pow(y, 4) + std::pow(y, 3)*(1456*x - 1456) + 468*std::pow(y, 2)*std::pow(x - 1, 2) + 48*y*std::pow(x - 1, 3) + std::pow(x - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(1365*std::pow(y, 4) + std::pow(y, 3)*(1456*x - 1456) + 468*std::pow(y, 2)*std::pow(x - 1, 2) + 48*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)) + (10*x*(7*x*(22*x - 9) + 6) - 1)*(1456*std::pow(y, 3) + 468*std::pow(y, 2)*(2*x - 2) + 144*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)) + (70*x*(22*x - 9) + 10*x*(308*x - 63) + 60)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(1365*std::pow(y, 4) + std::pow(y, 3)*(1456*x - 1456) + 468*std::pow(y, 2)*std::pow(x - 1, 2) + 48*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(1365*std::pow(y, 4) + std::pow(y, 3)*(1456*x - 1456) + 468*std::pow(y, 2)*std::pow(x - 1, 2) + 48*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)) + (10*x*(7*x*(22*x - 9) + 6) - 1)*(5460*std::pow(y, 3) + 3*std::pow(y, 2)*(1456*x - 1456) + 936*y*std::pow(x - 1, 2) + 48*std::pow(x - 1, 3))*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3)),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))*(1365*std::pow(y, 4) + std::pow(y, 3)*(1456*x - 1456) + 468*std::pow(y, 2)*std::pow(x - 1, 2) + 48*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 255
template<>
struct DGBasis<255> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (21*x*(11*x - 2) + 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(6188*std::pow(y, 5) + std::pow(y, 4)*(9100*x - 9100) + 4550*std::pow(y, 3)*std::pow(x - 1, 2) + 910*std::pow(y, 2)*std::pow(x - 1, 3) + 65*y*std::pow(x - 1, 4) + std::pow(x - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (462*x - 42)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(6188*std::pow(y, 5) + std::pow(y, 4)*(9100*x - 9100) + 4550*std::pow(y, 3)*std::pow(x - 1, 2) + 910*std::pow(y, 2)*std::pow(x - 1, 3) + 65*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)) + (21*x*(11*x - 2) + 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(6188*std::pow(y, 5) + std::pow(y, 4)*(9100*x - 9100) + 4550*std::pow(y, 3)*std::pow(x - 1, 2) + 910*std::pow(y, 2)*std::pow(x - 1, 3) + 65*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)) + (21*x*(11*x - 2) + 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(9100*std::pow(y, 4) + 4550*std::pow(y, 3)*(2*x - 2) + 2730*std::pow(y, 2)*std::pow(x - 1, 2) + 260*y*std::pow(x - 1, 3) + 5*std::pow(x - 1, 4)),
            (21*x*(11*x - 2) + 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(6188*std::pow(y, 5) + std::pow(y, 4)*(9100*x - 9100) + 4550*std::pow(y, 3)*std::pow(x - 1, 2) + 910*std::pow(y, 2)*std::pow(x - 1, 3) + 65*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)) + (21*x*(11*x - 2) + 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(30940*std::pow(y, 4) + 4*std::pow(y, 3)*(9100*x - 9100) + 13650*std::pow(y, 2)*std::pow(x - 1, 2) + 1820*y*std::pow(x - 1, 3) + 65*std::pow(x - 1, 4)),
            (21*x*(11*x - 2) + 1)*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))*(6188*std::pow(y, 5) + std::pow(y, 4)*(9100*x - 9100) + 4550*std::pow(y, 3)*std::pow(x - 1, 2) + 910*std::pow(y, 2)*std::pow(x - 1, 3) + 65*y*std::pow(x - 1, 4) + std::pow(x - 1, 5))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 256
template<>
struct DGBasis<256> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (22*x - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(27132*std::pow(y, 6) + std::pow(y, 5)*(51408*x - 51408) + 35700*std::pow(y, 4)*std::pow(x - 1, 2) + 11200*std::pow(y, 3)*std::pow(x - 1, 3) + 1575*std::pow(y, 2)*std::pow(x - 1, 4) + 84*y*std::pow(x - 1, 5) + std::pow(x - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (22*x - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(27132*std::pow(y, 6) + std::pow(y, 5)*(51408*x - 51408) + 35700*std::pow(y, 4)*std::pow(x - 1, 2) + 11200*std::pow(y, 3)*std::pow(x - 1, 3) + 1575*std::pow(y, 2)*std::pow(x - 1, 4) + 84*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)) + (22*x - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(51408*std::pow(y, 5) + 35700*std::pow(y, 4)*(2*x - 2) + 33600*std::pow(y, 3)*std::pow(x - 1, 2) + 6300*std::pow(y, 2)*std::pow(x - 1, 3) + 420*y*std::pow(x - 1, 4) + 6*std::pow(x - 1, 5)) + 22*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(27132*std::pow(y, 6) + std::pow(y, 5)*(51408*x - 51408) + 35700*std::pow(y, 4)*std::pow(x - 1, 2) + 11200*std::pow(y, 3)*std::pow(x - 1, 3) + 1575*std::pow(y, 2)*std::pow(x - 1, 4) + 84*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)),
            (22*x - 1)*(30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(27132*std::pow(y, 6) + std::pow(y, 5)*(51408*x - 51408) + 35700*std::pow(y, 4)*std::pow(x - 1, 2) + 11200*std::pow(y, 3)*std::pow(x - 1, 3) + 1575*std::pow(y, 2)*std::pow(x - 1, 4) + 84*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)) + (22*x - 1)*(20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(162792*std::pow(y, 5) + 5*std::pow(y, 4)*(51408*x - 51408) + 142800*std::pow(y, 3)*std::pow(x - 1, 2) + 33600*std::pow(y, 2)*std::pow(x - 1, 3) + 3150*y*std::pow(x - 1, 4) + 84*std::pow(x - 1, 5)),
            (22*x - 1)*(60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))*(27132*std::pow(y, 6) + std::pow(y, 5)*(51408*x - 51408) + 35700*std::pow(y, 4)*std::pow(x - 1, 2) + 11200*std::pow(y, 3)*std::pow(x - 1, 3) + 1575*std::pow(y, 2)*std::pow(x - 1, 4) + 84*y*std::pow(x - 1, 5) + std::pow(x - 1, 6))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 257
template<>
struct DGBasis<257> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(116280*std::pow(y, 7) + std::pow(y, 6)*(271320*x - 271320) + 244188*std::pow(y, 5)*std::pow(x - 1, 2) + 107100*std::pow(y, 4)*std::pow(x - 1, 3) + 23800*std::pow(y, 3)*std::pow(x - 1, 4) + 2520*std::pow(y, 2)*std::pow(x - 1, 5) + 105*y*std::pow(x - 1, 6) + std::pow(x - 1, 7));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(116280*std::pow(y, 7) + std::pow(y, 6)*(271320*x - 271320) + 244188*std::pow(y, 5)*std::pow(x - 1, 2) + 107100*std::pow(y, 4)*std::pow(x - 1, 3) + 23800*std::pow(y, 3)*std::pow(x - 1, 4) + 2520*std::pow(y, 2)*std::pow(x - 1, 5) + 105*y*std::pow(x - 1, 6) + std::pow(x - 1, 7)) + (20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(271320*std::pow(y, 6) + 244188*std::pow(y, 5)*(2*x - 2) + 321300*std::pow(y, 4)*std::pow(x - 1, 2) + 95200*std::pow(y, 3)*std::pow(x - 1, 3) + 12600*std::pow(y, 2)*std::pow(x - 1, 4) + 630*y*std::pow(x - 1, 5) + 7*std::pow(x - 1, 6)),
            (30*std::pow(z, 2) + 12*z*(2*x + 2*y - 2) + 3*std::pow(x + y - 1, 2))*(116280*std::pow(y, 7) + std::pow(y, 6)*(271320*x - 271320) + 244188*std::pow(y, 5)*std::pow(x - 1, 2) + 107100*std::pow(y, 4)*std::pow(x - 1, 3) + 23800*std::pow(y, 3)*std::pow(x - 1, 4) + 2520*std::pow(y, 2)*std::pow(x - 1, 5) + 105*y*std::pow(x - 1, 6) + std::pow(x - 1, 7)) + (20*std::pow(z, 3) + std::pow(z, 2)*(30*x + 30*y - 30) + 12*z*std::pow(x + y - 1, 2) + std::pow(x + y - 1, 3))*(813960*std::pow(y, 6) + 6*std::pow(y, 5)*(271320*x - 271320) + 1220940*std::pow(y, 4)*std::pow(x - 1, 2) + 428400*std::pow(y, 3)*std::pow(x - 1, 3) + 71400*std::pow(y, 2)*std::pow(x - 1, 4) + 5040*y*std::pow(x - 1, 5) + 105*std::pow(x - 1, 6)),
            (60*std::pow(z, 2) + 2*z*(30*x + 30*y - 30) + 12*std::pow(x + y - 1, 2))*(116280*std::pow(y, 7) + std::pow(y, 6)*(271320*x - 271320) + 244188*std::pow(y, 5)*std::pow(x - 1, 2) + 107100*std::pow(y, 4)*std::pow(x - 1, 3) + 23800*std::pow(y, 3)*std::pow(x - 1, 4) + 2520*std::pow(y, 2)*std::pow(x - 1, 5) + 105*y*std::pow(x - 1, 6) + std::pow(x - 1, 7))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 258
template<>
struct DGBasis<258> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)) + (51*x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) + 51*x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + x*(19*x*(7*x*(11*x - 18) + 75) + 19*x*(7*x*(11*x - 18) + x*(154*x - 126) + 75) - 380) + 45) - 102)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)),
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)),
            (51*x*(x*(19*x*(x*(7*x*(11*x - 18) + 75) - 20) + 45) - 2) + 1)*(280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 259
template<>
struct DGBasis<259> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(x + 11*y - 1)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(x + 11*y - 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)) + (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (x + 11*y - 1)*(57*x*(x*(21*x*(22*x - 25) + 200) - 30) + 3*x*(19*x*(21*x*(22*x - 25) + 200) + 19*x*(21*x*(22*x - 25) + x*(924*x - 525) + 200) - 570) + 90)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)),
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(x + 11*y - 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)) + 11*(3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)),
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(x + 11*y - 1)*(280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 260
template<>
struct DGBasis<260> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(78*std::pow(y, 2) + y*(24*x - 24) + std::pow(x - 1, 2))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(2*x + 24*y - 2)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(78*std::pow(y, 2) + y*(24*x - 24) + std::pow(x - 1, 2))*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)) + (78*std::pow(y, 2) + y*(24*x - 24) + std::pow(x - 1, 2))*(95*x*(7*x*(11*x - 8) + 12) + 19*x*(35*x*(11*x - 8) + 5*x*(154*x - 56) + 60) - 76)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)),
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(24*x + 156*y - 24)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(78*std::pow(y, 2) + y*(24*x - 24) + std::pow(x - 1, 2))*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)),
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(78*std::pow(y, 2) + y*(24*x - 24) + std::pow(x - 1, 2))*(280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 261
template<>
struct DGBasis<261> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (10*x*(7*x*(22*x - 9) + 6) - 1)*(455*std::pow(y, 3) + std::pow(y, 2)*(273*x - 273) + 39*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(273*std::pow(y, 2) + 39*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (10*x*(7*x*(22*x - 9) + 6) - 1)*(455*std::pow(y, 3) + std::pow(y, 2)*(273*x - 273) + 39*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)) + (70*x*(22*x - 9) + 10*x*(308*x - 63) + 60)*(455*std::pow(y, 3) + std::pow(y, 2)*(273*x - 273) + 39*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(1365*std::pow(y, 2) + 2*y*(273*x - 273) + 39*std::pow(x - 1, 2))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (10*x*(7*x*(22*x - 9) + 6) - 1)*(455*std::pow(y, 3) + std::pow(y, 2)*(273*x - 273) + 39*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3)),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(455*std::pow(y, 3) + std::pow(y, 2)*(273*x - 273) + 39*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 262
template<>
struct DGBasis<262> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (21*x*(11*x - 2) + 1)*(2380*std::pow(y, 4) + std::pow(y, 3)*(2240*x - 2240) + 630*std::pow(y, 2)*std::pow(x - 1, 2) + 56*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (462*x - 42)*(2380*std::pow(y, 4) + std::pow(y, 3)*(2240*x - 2240) + 630*std::pow(y, 2)*std::pow(x - 1, 2) + 56*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (21*x*(11*x - 2) + 1)*(2240*std::pow(y, 3) + 630*std::pow(y, 2)*(2*x - 2) + 168*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (21*x*(11*x - 2) + 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3))*(2380*std::pow(y, 4) + std::pow(y, 3)*(2240*x - 2240) + 630*std::pow(y, 2)*std::pow(x - 1, 2) + 56*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (21*x*(11*x - 2) + 1)*(9520*std::pow(y, 3) + 3*std::pow(y, 2)*(2240*x - 2240) + 1260*y*std::pow(x - 1, 2) + 56*std::pow(x - 1, 3))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + (21*x*(11*x - 2) + 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3))*(2380*std::pow(y, 4) + std::pow(y, 3)*(2240*x - 2240) + 630*std::pow(y, 2)*std::pow(x - 1, 2) + 56*y*std::pow(x - 1, 3) + std::pow(x - 1, 4)),
            (21*x*(11*x - 2) + 1)*(280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))*(2380*std::pow(y, 4) + std::pow(y, 3)*(2240*x - 2240) + 630*std::pow(y, 2)*std::pow(x - 1, 2) + 56*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 263
template<>
struct DGBasis<263> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (22*x - 1)*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4))*(11628*std::pow(y, 5) + std::pow(y, 4)*(15300*x - 15300) + 6800*std::pow(y, 3)*std::pow(x - 1, 2) + 1200*std::pow(y, 2)*std::pow(x - 1, 3) + 75*y*std::pow(x - 1, 4) + std::pow(x - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (22*x - 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3))*(11628*std::pow(y, 5) + std::pow(y, 4)*(15300*x - 15300) + 6800*std::pow(y, 3)*std::pow(x - 1, 2) + 1200*std::pow(y, 2)*std::pow(x - 1, 3) + 75*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)) + (22*x - 1)*(15300*std::pow(y, 4) + 6800*std::pow(y, 3)*(2*x - 2) + 3600*std::pow(y, 2)*std::pow(x - 1, 2) + 300*y*std::pow(x - 1, 3) + 5*std::pow(x - 1, 4))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)) + 22*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4))*(11628*std::pow(y, 5) + std::pow(y, 4)*(15300*x - 15300) + 6800*std::pow(y, 3)*std::pow(x - 1, 2) + 1200*std::pow(y, 2)*std::pow(x - 1, 3) + 75*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)),
            (22*x - 1)*(140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3))*(11628*std::pow(y, 5) + std::pow(y, 4)*(15300*x - 15300) + 6800*std::pow(y, 3)*std::pow(x - 1, 2) + 1200*std::pow(y, 2)*std::pow(x - 1, 3) + 75*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)) + (22*x - 1)*(58140*std::pow(y, 4) + 4*std::pow(y, 3)*(15300*x - 15300) + 20400*std::pow(y, 2)*std::pow(x - 1, 2) + 2400*y*std::pow(x - 1, 3) + 75*std::pow(x - 1, 4))*(70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4)),
            (22*x - 1)*(280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))*(11628*std::pow(y, 5) + std::pow(y, 4)*(15300*x - 15300) + 6800*std::pow(y, 3)*std::pow(x - 1, 2) + 1200*std::pow(y, 2)*std::pow(x - 1, 3) + 75*y*std::pow(x - 1, 4) + std::pow(x - 1, 5))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 264
template<>
struct DGBasis<264> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4))*(54264*std::pow(y, 6) + std::pow(y, 5)*(93024*x - 93024) + 58140*std::pow(y, 4)*std::pow(x - 1, 2) + 16320*std::pow(y, 3)*std::pow(x - 1, 3) + 2040*std::pow(y, 2)*std::pow(x - 1, 4) + 96*y*std::pow(x - 1, 5) + std::pow(x - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3))*(54264*std::pow(y, 6) + std::pow(y, 5)*(93024*x - 93024) + 58140*std::pow(y, 4)*std::pow(x - 1, 2) + 16320*std::pow(y, 3)*std::pow(x - 1, 3) + 2040*std::pow(y, 2)*std::pow(x - 1, 4) + 96*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)) + (70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4))*(93024*std::pow(y, 5) + 58140*std::pow(y, 4)*(2*x - 2) + 48960*std::pow(y, 3)*std::pow(x - 1, 2) + 8160*std::pow(y, 2)*std::pow(x - 1, 3) + 480*y*std::pow(x - 1, 4) + 6*std::pow(x - 1, 5)),
            (140*std::pow(z, 3) + 90*std::pow(z, 2)*(2*x + 2*y - 2) + 60*z*std::pow(x + y - 1, 2) + 4*std::pow(x + y - 1, 3))*(54264*std::pow(y, 6) + std::pow(y, 5)*(93024*x - 93024) + 58140*std::pow(y, 4)*std::pow(x - 1, 2) + 16320*std::pow(y, 3)*std::pow(x - 1, 3) + 2040*std::pow(y, 2)*std::pow(x - 1, 4) + 96*y*std::pow(x - 1, 5) + std::pow(x - 1, 6)) + (70*std::pow(z, 4) + std::pow(z, 3)*(140*x + 140*y - 140) + 90*std::pow(z, 2)*std::pow(x + y - 1, 2) + 20*z*std::pow(x + y - 1, 3) + std::pow(x + y - 1, 4))*(325584*std::pow(y, 5) + 5*std::pow(y, 4)*(93024*x - 93024) + 232560*std::pow(y, 3)*std::pow(x - 1, 2) + 48960*std::pow(y, 2)*std::pow(x - 1, 3) + 4080*y*std::pow(x - 1, 4) + 96*std::pow(x - 1, 5)),
            (280*std::pow(z, 3) + 3*std::pow(z, 2)*(140*x + 140*y - 140) + 180*z*std::pow(x + y - 1, 2) + 20*std::pow(x + y - 1, 3))*(54264*std::pow(y, 6) + std::pow(y, 5)*(93024*x - 93024) + 58140*std::pow(y, 4)*std::pow(x - 1, 2) + 16320*std::pow(y, 3)*std::pow(x - 1, 3) + 2040*std::pow(y, 2)*std::pow(x - 1, 4) + 96*y*std::pow(x - 1, 5) + std::pow(x - 1, 6))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 265
template<>
struct DGBasis<265> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)) + (57*x*(x*(21*x*(22*x - 25) + 200) - 30) + 3*x*(19*x*(21*x*(22*x - 25) + 200) + 19*x*(21*x*(22*x - 25) + x*(924*x - 525) + 200) - 570) + 90)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)),
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)),
            (3*x*(19*x*(x*(21*x*(22*x - 25) + 200) - 30) + 30) - 1)*(1260*std::pow(z, 4) + 4*std::pow(z, 3)*(630*x + 630*y - 630) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 2) + 420*z*std::pow(x + y - 1, 3) + 30*std::pow(x + y - 1, 4))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 266
template<>
struct DGBasis<266> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(x + 13*y - 1)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(x + 13*y - 1)*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)) + (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (x + 13*y - 1)*(95*x*(7*x*(11*x - 8) + 12) + 19*x*(35*x*(11*x - 8) + 5*x*(154*x - 56) + 60) - 76)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)),
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(x + 13*y - 1)*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)) + 13*(19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)),
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(x + 13*y - 1)*(1260*std::pow(z, 4) + 4*std::pow(z, 3)*(630*x + 630*y - 630) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 2) + 420*z*std::pow(x + y - 1, 3) + 30*std::pow(x + y - 1, 4))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 267
template<>
struct DGBasis<267> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (10*x*(7*x*(22*x - 9) + 6) - 1)*(105*std::pow(y, 2) + y*(28*x - 28) + std::pow(x - 1, 2))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(2*x + 28*y - 2)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (10*x*(7*x*(22*x - 9) + 6) - 1)*(105*std::pow(y, 2) + y*(28*x - 28) + std::pow(x - 1, 2))*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)) + (105*std::pow(y, 2) + y*(28*x - 28) + std::pow(x - 1, 2))*(70*x*(22*x - 9) + 10*x*(308*x - 63) + 60)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(28*x + 210*y - 28)*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (10*x*(7*x*(22*x - 9) + 6) - 1)*(105*std::pow(y, 2) + y*(28*x - 28) + std::pow(x - 1, 2))*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(105*std::pow(y, 2) + y*(28*x - 28) + std::pow(x - 1, 2))*(1260*std::pow(z, 4) + 4*std::pow(z, 3)*(630*x + 630*y - 630) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 2) + 420*z*std::pow(x + y - 1, 3) + 30*std::pow(x + y - 1, 4))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 268
template<>
struct DGBasis<268> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (21*x*(11*x - 2) + 1)*(680*std::pow(y, 3) + std::pow(y, 2)*(360*x - 360) + 45*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (462*x - 42)*(680*std::pow(y, 3) + std::pow(y, 2)*(360*x - 360) + 45*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (21*x*(11*x - 2) + 1)*(360*std::pow(y, 2) + 45*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (21*x*(11*x - 2) + 1)*(680*std::pow(y, 3) + std::pow(y, 2)*(360*x - 360) + 45*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)),
            (21*x*(11*x - 2) + 1)*(2040*std::pow(y, 2) + 2*y*(360*x - 360) + 45*std::pow(x - 1, 2))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (21*x*(11*x - 2) + 1)*(680*std::pow(y, 3) + std::pow(y, 2)*(360*x - 360) + 45*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)),
            (21*x*(11*x - 2) + 1)*(680*std::pow(y, 3) + std::pow(y, 2)*(360*x - 360) + 45*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(1260*std::pow(z, 4) + 4*std::pow(z, 3)*(630*x + 630*y - 630) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 2) + 420*z*std::pow(x + y - 1, 3) + 30*std::pow(x + y - 1, 4))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 269
template<>
struct DGBasis<269> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (22*x - 1)*(3876*std::pow(y, 4) + std::pow(y, 3)*(3264*x - 3264) + 816*std::pow(y, 2)*std::pow(x - 1, 2) + 64*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (22*x - 1)*(3264*std::pow(y, 3) + 816*std::pow(y, 2)*(2*x - 2) + 192*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (22*x - 1)*(3876*std::pow(y, 4) + std::pow(y, 3)*(3264*x - 3264) + 816*std::pow(y, 2)*std::pow(x - 1, 2) + 64*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)) + 22*(3876*std::pow(y, 4) + std::pow(y, 3)*(3264*x - 3264) + 816*std::pow(y, 2)*std::pow(x - 1, 2) + 64*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)),
            (22*x - 1)*(15504*std::pow(y, 3) + 3*std::pow(y, 2)*(3264*x - 3264) + 1632*y*std::pow(x - 1, 2) + 64*std::pow(x - 1, 3))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (22*x - 1)*(3876*std::pow(y, 4) + std::pow(y, 3)*(3264*x - 3264) + 816*std::pow(y, 2)*std::pow(x - 1, 2) + 64*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))*(630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4)),
            (22*x - 1)*(3876*std::pow(y, 4) + std::pow(y, 3)*(3264*x - 3264) + 816*std::pow(y, 2)*std::pow(x - 1, 2) + 64*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))*(1260*std::pow(z, 4) + 4*std::pow(z, 3)*(630*x + 630*y - 630) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 2) + 420*z*std::pow(x + y - 1, 3) + 30*std::pow(x + y - 1, 4))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 270
template<>
struct DGBasis<270> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (20349*std::pow(y, 5) + std::pow(y, 4)*(24225*x - 24225) + 9690*std::pow(y, 3)*std::pow(x - 1, 2) + 1530*std::pow(y, 2)*std::pow(x - 1, 3) + 85*y*std::pow(x - 1, 4) + std::pow(x - 1, 5))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (24225*std::pow(y, 4) + 9690*std::pow(y, 3)*(2*x - 2) + 4590*std::pow(y, 2)*std::pow(x - 1, 2) + 340*y*std::pow(x - 1, 3) + 5*std::pow(x - 1, 4))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4))*(20349*std::pow(y, 5) + std::pow(y, 4)*(24225*x - 24225) + 9690*std::pow(y, 3)*std::pow(x - 1, 2) + 1530*std::pow(y, 2)*std::pow(x - 1, 3) + 85*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)),
            (101745*std::pow(y, 4) + 4*std::pow(y, 3)*(24225*x - 24225) + 29070*std::pow(y, 2)*std::pow(x - 1, 2) + 3060*y*std::pow(x - 1, 3) + 85*std::pow(x - 1, 4))*(252*std::pow(z, 5) + std::pow(z, 4)*(630*x + 630*y - 630) + 560*std::pow(z, 3)*std::pow(x + y - 1, 2) + 210*std::pow(z, 2)*std::pow(x + y - 1, 3) + 30*z*std::pow(x + y - 1, 4) + std::pow(x + y - 1, 5)) + (630*std::pow(z, 4) + 560*std::pow(z, 3)*(2*x + 2*y - 2) + 630*std::pow(z, 2)*std::pow(x + y - 1, 2) + 120*z*std::pow(x + y - 1, 3) + 5*std::pow(x + y - 1, 4))*(20349*std::pow(y, 5) + std::pow(y, 4)*(24225*x - 24225) + 9690*std::pow(y, 3)*std::pow(x - 1, 2) + 1530*std::pow(y, 2)*std::pow(x - 1, 3) + 85*y*std::pow(x - 1, 4) + std::pow(x - 1, 5)),
            (1260*std::pow(z, 4) + 4*std::pow(z, 3)*(630*x + 630*y - 630) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 2) + 420*z*std::pow(x + y - 1, 3) + 30*std::pow(x + y - 1, 4))*(20349*std::pow(y, 5) + std::pow(y, 4)*(24225*x - 24225) + 9690*std::pow(y, 3)*std::pow(x - 1, 2) + 1530*std::pow(y, 2)*std::pow(x - 1, 3) + 85*y*std::pow(x - 1, 4) + std::pow(x - 1, 5))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 271
template<>
struct DGBasis<271> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)) + (95*x*(7*x*(11*x - 8) + 12) + 19*x*(35*x*(11*x - 8) + 5*x*(154*x - 56) + 60) - 76)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)),
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)),
            (19*x*(5*x*(7*x*(11*x - 8) + 12) - 4) + 1)*(5544*std::pow(z, 5) + 5*std::pow(z, 4)*(2772*x + 2772*y - 2772) + 12600*std::pow(z, 3)*std::pow(x + y - 1, 2) + 5040*std::pow(z, 2)*std::pow(x + y - 1, 3) + 840*z*std::pow(x + y - 1, 4) + 42*std::pow(x + y - 1, 5))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 272
template<>
struct DGBasis<272> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (10*x*(7*x*(22*x - 9) + 6) - 1)*(x + 15*y - 1)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(x + 15*y - 1)*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)) + (10*x*(7*x*(22*x - 9) + 6) - 1)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)) + (x + 15*y - 1)*(70*x*(22*x - 9) + 10*x*(308*x - 63) + 60)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(x + 15*y - 1)*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)) + 15*(10*x*(7*x*(22*x - 9) + 6) - 1)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(x + 15*y - 1)*(5544*std::pow(z, 5) + 5*std::pow(z, 4)*(2772*x + 2772*y - 2772) + 12600*std::pow(z, 3)*std::pow(x + y - 1, 2) + 5040*std::pow(z, 2)*std::pow(x + y - 1, 3) + 840*z*std::pow(x + y - 1, 4) + 42*std::pow(x + y - 1, 5))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 273
template<>
struct DGBasis<273> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (21*x*(11*x - 2) + 1)*(136*std::pow(y, 2) + y*(32*x - 32) + std::pow(x - 1, 2))*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (462*x - 42)*(136*std::pow(y, 2) + y*(32*x - 32) + std::pow(x - 1, 2))*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)) + (21*x*(11*x - 2) + 1)*(2*x + 32*y - 2)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)) + (21*x*(11*x - 2) + 1)*(136*std::pow(y, 2) + y*(32*x - 32) + std::pow(x - 1, 2))*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)),
            (21*x*(11*x - 2) + 1)*(32*x + 272*y - 32)*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)) + (21*x*(11*x - 2) + 1)*(136*std::pow(y, 2) + y*(32*x - 32) + std::pow(x - 1, 2))*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)),
            (21*x*(11*x - 2) + 1)*(136*std::pow(y, 2) + y*(32*x - 32) + std::pow(x - 1, 2))*(5544*std::pow(z, 5) + 5*std::pow(z, 4)*(2772*x + 2772*y - 2772) + 12600*std::pow(z, 3)*std::pow(x + y - 1, 2) + 5040*std::pow(z, 2)*std::pow(x + y - 1, 3) + 840*z*std::pow(x + y - 1, 4) + 42*std::pow(x + y - 1, 5))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 274
template<>
struct DGBasis<274> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (22*x - 1)*(969*std::pow(y, 3) + std::pow(y, 2)*(459*x - 459) + 51*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (22*x - 1)*(459*std::pow(y, 2) + 51*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)) + (22*x - 1)*(969*std::pow(y, 3) + std::pow(y, 2)*(459*x - 459) + 51*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)) + 22*(969*std::pow(y, 3) + std::pow(y, 2)*(459*x - 459) + 51*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)),
            (22*x - 1)*(2907*std::pow(y, 2) + 2*y*(459*x - 459) + 51*std::pow(x - 1, 2))*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)) + (22*x - 1)*(969*std::pow(y, 3) + std::pow(y, 2)*(459*x - 459) + 51*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)),
            (22*x - 1)*(969*std::pow(y, 3) + std::pow(y, 2)*(459*x - 459) + 51*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(5544*std::pow(z, 5) + 5*std::pow(z, 4)*(2772*x + 2772*y - 2772) + 12600*std::pow(z, 3)*std::pow(x + y - 1, 2) + 5040*std::pow(z, 2)*std::pow(x + y - 1, 3) + 840*z*std::pow(x + y - 1, 4) + 42*std::pow(x + y - 1, 5))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 275
template<>
struct DGBasis<275> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (5985*std::pow(y, 4) + std::pow(y, 3)*(4560*x - 4560) + 1026*std::pow(y, 2)*std::pow(x - 1, 2) + 72*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6));
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (4560*std::pow(y, 3) + 1026*std::pow(y, 2)*(2*x - 2) + 216*y*std::pow(x - 1, 2) + 4*std::pow(x - 1, 3))*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)) + (5985*std::pow(y, 4) + std::pow(y, 3)*(4560*x - 4560) + 1026*std::pow(y, 2)*std::pow(x - 1, 2) + 72*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)),
            (23940*std::pow(y, 3) + 3*std::pow(y, 2)*(4560*x - 4560) + 2052*y*std::pow(x - 1, 2) + 72*std::pow(x - 1, 3))*(924*std::pow(z, 6) + std::pow(z, 5)*(2772*x + 2772*y - 2772) + 3150*std::pow(z, 4)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 3)*std::pow(x + y - 1, 3) + 420*std::pow(z, 2)*std::pow(x + y - 1, 4) + 42*z*std::pow(x + y - 1, 5) + std::pow(x + y - 1, 6)) + (5985*std::pow(y, 4) + std::pow(y, 3)*(4560*x - 4560) + 1026*std::pow(y, 2)*std::pow(x - 1, 2) + 72*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))*(2772*std::pow(z, 5) + 3150*std::pow(z, 4)*(2*x + 2*y - 2) + 5040*std::pow(z, 3)*std::pow(x + y - 1, 2) + 1680*std::pow(z, 2)*std::pow(x + y - 1, 3) + 210*z*std::pow(x + y - 1, 4) + 6*std::pow(x + y - 1, 5)),
            (5985*std::pow(y, 4) + std::pow(y, 3)*(4560*x - 4560) + 1026*std::pow(y, 2)*std::pow(x - 1, 2) + 72*y*std::pow(x - 1, 3) + std::pow(x - 1, 4))*(5544*std::pow(z, 5) + 5*std::pow(z, 4)*(2772*x + 2772*y - 2772) + 12600*std::pow(z, 3)*std::pow(x + y - 1, 2) + 5040*std::pow(z, 2)*std::pow(x + y - 1, 3) + 840*z*std::pow(x + y - 1, 4) + 42*std::pow(x + y - 1, 5))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 276
template<>
struct DGBasis<276> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (10*x*(7*x*(22*x - 9) + 6) - 1)*(std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(7*std::pow(x, 6) + 42*std::pow(x, 5)*y - 42*std::pow(x, 5) + 105*std::pow(x, 4)*std::pow(y, 2) - 210*std::pow(x, 4)*y + 105*std::pow(x, 4) + 140*std::pow(x, 3)*std::pow(y, 3) - 420*std::pow(x, 3)*std::pow(y, 2) + 420*std::pow(x, 3)*y - 140*std::pow(x, 3) + 105*std::pow(x, 2)*std::pow(y, 4) - 420*std::pow(x, 2)*std::pow(y, 3) + 630*std::pow(x, 2)*std::pow(y, 2) - 420*std::pow(x, 2)*y + 105*std::pow(x, 2) + 42*x*std::pow(y, 5) - 210*x*std::pow(y, 4) + 420*x*std::pow(y, 3) - 420*x*std::pow(y, 2) + 210*x*y - 42*x + 7*std::pow(y, 6) - 42*std::pow(y, 5) + 105*std::pow(y, 4) - 140*std::pow(y, 3) + 105*std::pow(y, 2) - 42*y + 12012*std::pow(z, 6) + 16632*std::pow(z, 5)*(2*x + 2*y - 2) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 2) + 16800*std::pow(z, 3)*std::pow(x + y - 1, 3) + 3780*std::pow(z, 2)*std::pow(x + y - 1, 4) + 336*z*std::pow(x + y - 1, 5) + 7) + (70*x*(22*x - 9) + 10*x*(308*x - 63) + 60)*(std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) - 1),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(7*std::pow(x, 6) + 42*std::pow(x, 5)*y - 42*std::pow(x, 5) + 105*std::pow(x, 4)*std::pow(y, 2) - 210*std::pow(x, 4)*y + 105*std::pow(x, 4) + 140*std::pow(x, 3)*std::pow(y, 3) - 420*std::pow(x, 3)*std::pow(y, 2) + 420*std::pow(x, 3)*y - 140*std::pow(x, 3) + 105*std::pow(x, 2)*std::pow(y, 4) - 420*std::pow(x, 2)*std::pow(y, 3) + 630*std::pow(x, 2)*std::pow(y, 2) - 420*std::pow(x, 2)*y + 105*std::pow(x, 2) + 42*x*std::pow(y, 5) - 210*x*std::pow(y, 4) + 420*x*std::pow(y, 3) - 420*x*std::pow(y, 2) + 210*x*y - 42*x + 7*std::pow(y, 6) - 42*std::pow(y, 5) + 105*std::pow(y, 4) - 140*std::pow(y, 3) + 105*std::pow(y, 2) - 42*y + 12012*std::pow(z, 6) + 16632*std::pow(z, 5)*(2*x + 2*y - 2) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 2) + 16800*std::pow(z, 3)*std::pow(x + y - 1, 3) + 3780*std::pow(z, 2)*std::pow(x + y - 1, 4) + 336*z*std::pow(x + y - 1, 5) + 7),
            (10*x*(7*x*(22*x - 9) + 6) - 1)*(24024*std::pow(z, 6) + 6*std::pow(z, 5)*(12012*x + 12012*y - 12012) + 83160*std::pow(z, 4)*std::pow(x + y - 1, 2) + 46200*std::pow(z, 3)*std::pow(x + y - 1, 3) + 12600*std::pow(z, 2)*std::pow(x + y - 1, 4) + 1512*z*std::pow(x + y - 1, 5) + 56*std::pow(x + y - 1, 6))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 277
template<>
struct DGBasis<277> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (21*x*(11*x - 2) + 1)*(x + 17*y - 1)*(std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (462*x - 42)*(x + 17*y - 1)*(std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) - 1) + (21*x*(11*x - 2) + 1)*(x + 17*y - 1)*(7*std::pow(x, 6) + 42*std::pow(x, 5)*y - 42*std::pow(x, 5) + 105*std::pow(x, 4)*std::pow(y, 2) - 210*std::pow(x, 4)*y + 105*std::pow(x, 4) + 140*std::pow(x, 3)*std::pow(y, 3) - 420*std::pow(x, 3)*std::pow(y, 2) + 420*std::pow(x, 3)*y - 140*std::pow(x, 3) + 105*std::pow(x, 2)*std::pow(y, 4) - 420*std::pow(x, 2)*std::pow(y, 3) + 630*std::pow(x, 2)*std::pow(y, 2) - 420*std::pow(x, 2)*y + 105*std::pow(x, 2) + 42*x*std::pow(y, 5) - 210*x*std::pow(y, 4) + 420*x*std::pow(y, 3) - 420*x*std::pow(y, 2) + 210*x*y - 42*x + 7*std::pow(y, 6) - 42*std::pow(y, 5) + 105*std::pow(y, 4) - 140*std::pow(y, 3) + 105*std::pow(y, 2) - 42*y + 12012*std::pow(z, 6) + 16632*std::pow(z, 5)*(2*x + 2*y - 2) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 2) + 16800*std::pow(z, 3)*std::pow(x + y - 1, 3) + 3780*std::pow(z, 2)*std::pow(x + y - 1, 4) + 336*z*std::pow(x + y - 1, 5) + 7) + (21*x*(11*x - 2) + 1)*(std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) - 1),
            (21*x*(11*x - 2) + 1)*(x + 17*y - 1)*(7*std::pow(x, 6) + 42*std::pow(x, 5)*y - 42*std::pow(x, 5) + 105*std::pow(x, 4)*std::pow(y, 2) - 210*std::pow(x, 4)*y + 105*std::pow(x, 4) + 140*std::pow(x, 3)*std::pow(y, 3) - 420*std::pow(x, 3)*std::pow(y, 2) + 420*std::pow(x, 3)*y - 140*std::pow(x, 3) + 105*std::pow(x, 2)*std::pow(y, 4) - 420*std::pow(x, 2)*std::pow(y, 3) + 630*std::pow(x, 2)*std::pow(y, 2) - 420*std::pow(x, 2)*y + 105*std::pow(x, 2) + 42*x*std::pow(y, 5) - 210*x*std::pow(y, 4) + 420*x*std::pow(y, 3) - 420*x*std::pow(y, 2) + 210*x*y - 42*x + 7*std::pow(y, 6) - 42*std::pow(y, 5) + 105*std::pow(y, 4) - 140*std::pow(y, 3) + 105*std::pow(y, 2) - 42*y + 12012*std::pow(z, 6) + 16632*std::pow(z, 5)*(2*x + 2*y - 2) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 2) + 16800*std::pow(z, 3)*std::pow(x + y - 1, 3) + 3780*std::pow(z, 2)*std::pow(x + y - 1, 4) + 336*z*std::pow(x + y - 1, 5) + 7) + 17*(21*x*(11*x - 2) + 1)*(std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) - 1),
            (21*x*(11*x - 2) + 1)*(x + 17*y - 1)*(24024*std::pow(z, 6) + 6*std::pow(z, 5)*(12012*x + 12012*y - 12012) + 83160*std::pow(z, 4)*std::pow(x + y - 1, 2) + 46200*std::pow(z, 3)*std::pow(x + y - 1, 3) + 12600*std::pow(z, 2)*std::pow(x + y - 1, 4) + 1512*z*std::pow(x + y - 1, 5) + 56*std::pow(x + y - 1, 6))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 278
template<>
struct DGBasis<278> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (22*x - 1)*(171*std::pow(y, 2) + y*(36*x - 36) + std::pow(x - 1, 2))*(std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (22*x - 1)*(2*x + 36*y - 2)*(std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) - 1) + (22*x - 1)*(171*std::pow(y, 2) + y*(36*x - 36) + std::pow(x - 1, 2))*(7*std::pow(x, 6) + 42*std::pow(x, 5)*y - 42*std::pow(x, 5) + 105*std::pow(x, 4)*std::pow(y, 2) - 210*std::pow(x, 4)*y + 105*std::pow(x, 4) + 140*std::pow(x, 3)*std::pow(y, 3) - 420*std::pow(x, 3)*std::pow(y, 2) + 420*std::pow(x, 3)*y - 140*std::pow(x, 3) + 105*std::pow(x, 2)*std::pow(y, 4) - 420*std::pow(x, 2)*std::pow(y, 3) + 630*std::pow(x, 2)*std::pow(y, 2) - 420*std::pow(x, 2)*y + 105*std::pow(x, 2) + 42*x*std::pow(y, 5) - 210*x*std::pow(y, 4) + 420*x*std::pow(y, 3) - 420*x*std::pow(y, 2) + 210*x*y - 42*x + 7*std::pow(y, 6) - 42*std::pow(y, 5) + 105*std::pow(y, 4) - 140*std::pow(y, 3) + 105*std::pow(y, 2) - 42*y + 12012*std::pow(z, 6) + 16632*std::pow(z, 5)*(2*x + 2*y - 2) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 2) + 16800*std::pow(z, 3)*std::pow(x + y - 1, 3) + 3780*std::pow(z, 2)*std::pow(x + y - 1, 4) + 336*z*std::pow(x + y - 1, 5) + 7) + 22*(171*std::pow(y, 2) + y*(36*x - 36) + std::pow(x - 1, 2))*(std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) - 1),
            (22*x - 1)*(36*x + 342*y - 36)*(std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) - 1) + (22*x - 1)*(171*std::pow(y, 2) + y*(36*x - 36) + std::pow(x - 1, 2))*(7*std::pow(x, 6) + 42*std::pow(x, 5)*y - 42*std::pow(x, 5) + 105*std::pow(x, 4)*std::pow(y, 2) - 210*std::pow(x, 4)*y + 105*std::pow(x, 4) + 140*std::pow(x, 3)*std::pow(y, 3) - 420*std::pow(x, 3)*std::pow(y, 2) + 420*std::pow(x, 3)*y - 140*std::pow(x, 3) + 105*std::pow(x, 2)*std::pow(y, 4) - 420*std::pow(x, 2)*std::pow(y, 3) + 630*std::pow(x, 2)*std::pow(y, 2) - 420*std::pow(x, 2)*y + 105*std::pow(x, 2) + 42*x*std::pow(y, 5) - 210*x*std::pow(y, 4) + 420*x*std::pow(y, 3) - 420*x*std::pow(y, 2) + 210*x*y - 42*x + 7*std::pow(y, 6) - 42*std::pow(y, 5) + 105*std::pow(y, 4) - 140*std::pow(y, 3) + 105*std::pow(y, 2) - 42*y + 12012*std::pow(z, 6) + 16632*std::pow(z, 5)*(2*x + 2*y - 2) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 2) + 16800*std::pow(z, 3)*std::pow(x + y - 1, 3) + 3780*std::pow(z, 2)*std::pow(x + y - 1, 4) + 336*z*std::pow(x + y - 1, 5) + 7),
            (22*x - 1)*(171*std::pow(y, 2) + y*(36*x - 36) + std::pow(x - 1, 2))*(24024*std::pow(z, 6) + 6*std::pow(z, 5)*(12012*x + 12012*y - 12012) + 83160*std::pow(z, 4)*std::pow(x + y - 1, 2) + 46200*std::pow(z, 3)*std::pow(x + y - 1, 3) + 12600*std::pow(z, 2)*std::pow(x + y - 1, 4) + 1512*z*std::pow(x + y - 1, 5) + 56*std::pow(x + y - 1, 6))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 279
template<>
struct DGBasis<279> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (1330*std::pow(y, 3) + std::pow(y, 2)*(570*x - 570) + 57*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (570*std::pow(y, 2) + 57*y*(2*x - 2) + 3*std::pow(x - 1, 2))*(std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) - 1) + (1330*std::pow(y, 3) + std::pow(y, 2)*(570*x - 570) + 57*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(7*std::pow(x, 6) + 42*std::pow(x, 5)*y - 42*std::pow(x, 5) + 105*std::pow(x, 4)*std::pow(y, 2) - 210*std::pow(x, 4)*y + 105*std::pow(x, 4) + 140*std::pow(x, 3)*std::pow(y, 3) - 420*std::pow(x, 3)*std::pow(y, 2) + 420*std::pow(x, 3)*y - 140*std::pow(x, 3) + 105*std::pow(x, 2)*std::pow(y, 4) - 420*std::pow(x, 2)*std::pow(y, 3) + 630*std::pow(x, 2)*std::pow(y, 2) - 420*std::pow(x, 2)*y + 105*std::pow(x, 2) + 42*x*std::pow(y, 5) - 210*x*std::pow(y, 4) + 420*x*std::pow(y, 3) - 420*x*std::pow(y, 2) + 210*x*y - 42*x + 7*std::pow(y, 6) - 42*std::pow(y, 5) + 105*std::pow(y, 4) - 140*std::pow(y, 3) + 105*std::pow(y, 2) - 42*y + 12012*std::pow(z, 6) + 16632*std::pow(z, 5)*(2*x + 2*y - 2) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 2) + 16800*std::pow(z, 3)*std::pow(x + y - 1, 3) + 3780*std::pow(z, 2)*std::pow(x + y - 1, 4) + 336*z*std::pow(x + y - 1, 5) + 7),
            (3990*std::pow(y, 2) + 2*y*(570*x - 570) + 57*std::pow(x - 1, 2))*(std::pow(x, 7) + 7*std::pow(x, 6)*y - 7*std::pow(x, 6) + 21*std::pow(x, 5)*std::pow(y, 2) - 42*std::pow(x, 5)*y + 21*std::pow(x, 5) + 35*std::pow(x, 4)*std::pow(y, 3) - 105*std::pow(x, 4)*std::pow(y, 2) + 105*std::pow(x, 4)*y - 35*std::pow(x, 4) + 35*std::pow(x, 3)*std::pow(y, 4) - 140*std::pow(x, 3)*std::pow(y, 3) + 210*std::pow(x, 3)*std::pow(y, 2) - 140*std::pow(x, 3)*y + 35*std::pow(x, 3) + 21*std::pow(x, 2)*std::pow(y, 5) - 105*std::pow(x, 2)*std::pow(y, 4) + 210*std::pow(x, 2)*std::pow(y, 3) - 210*std::pow(x, 2)*std::pow(y, 2) + 105*std::pow(x, 2)*y - 21*std::pow(x, 2) + 7*x*std::pow(y, 6) - 42*x*std::pow(y, 5) + 105*x*std::pow(y, 4) - 140*x*std::pow(y, 3) + 105*x*std::pow(y, 2) - 42*x*y + 7*x + std::pow(y, 7) - 7*std::pow(y, 6) + 21*std::pow(y, 5) - 35*std::pow(y, 4) + 35*std::pow(y, 3) - 21*std::pow(y, 2) + 7*y + 3432*std::pow(z, 7) + std::pow(z, 6)*(12012*x + 12012*y - 12012) + 16632*std::pow(z, 5)*std::pow(x + y - 1, 2) + 11550*std::pow(z, 4)*std::pow(x + y - 1, 3) + 4200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 756*std::pow(z, 2)*std::pow(x + y - 1, 5) + 56*z*std::pow(x + y - 1, 6) - 1) + (1330*std::pow(y, 3) + std::pow(y, 2)*(570*x - 570) + 57*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(7*std::pow(x, 6) + 42*std::pow(x, 5)*y - 42*std::pow(x, 5) + 105*std::pow(x, 4)*std::pow(y, 2) - 210*std::pow(x, 4)*y + 105*std::pow(x, 4) + 140*std::pow(x, 3)*std::pow(y, 3) - 420*std::pow(x, 3)*std::pow(y, 2) + 420*std::pow(x, 3)*y - 140*std::pow(x, 3) + 105*std::pow(x, 2)*std::pow(y, 4) - 420*std::pow(x, 2)*std::pow(y, 3) + 630*std::pow(x, 2)*std::pow(y, 2) - 420*std::pow(x, 2)*y + 105*std::pow(x, 2) + 42*x*std::pow(y, 5) - 210*x*std::pow(y, 4) + 420*x*std::pow(y, 3) - 420*x*std::pow(y, 2) + 210*x*y - 42*x + 7*std::pow(y, 6) - 42*std::pow(y, 5) + 105*std::pow(y, 4) - 140*std::pow(y, 3) + 105*std::pow(y, 2) - 42*y + 12012*std::pow(z, 6) + 16632*std::pow(z, 5)*(2*x + 2*y - 2) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 2) + 16800*std::pow(z, 3)*std::pow(x + y - 1, 3) + 3780*std::pow(z, 2)*std::pow(x + y - 1, 4) + 336*z*std::pow(x + y - 1, 5) + 7),
            (1330*std::pow(y, 3) + std::pow(y, 2)*(570*x - 570) + 57*y*std::pow(x - 1, 2) + std::pow(x - 1, 3))*(24024*std::pow(z, 6) + 6*std::pow(z, 5)*(12012*x + 12012*y - 12012) + 83160*std::pow(z, 4)*std::pow(x + y - 1, 2) + 46200*std::pow(z, 3)*std::pow(x + y - 1, 3) + 12600*std::pow(z, 2)*std::pow(x + y - 1, 4) + 1512*z*std::pow(x + y - 1, 5) + 56*std::pow(x + y - 1, 6))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 280
template<>
struct DGBasis<280> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (21*x*(11*x - 2) + 1)*(std::pow(x, 8) + 8*std::pow(x, 7)*y - 8*std::pow(x, 7) + 28*std::pow(x, 6)*std::pow(y, 2) - 56*std::pow(x, 6)*y + 28*std::pow(x, 6) + 56*std::pow(x, 5)*std::pow(y, 3) - 168*std::pow(x, 5)*std::pow(y, 2) + 168*std::pow(x, 5)*y - 56*std::pow(x, 5) + 70*std::pow(x, 4)*std::pow(y, 4) - 280*std::pow(x, 4)*std::pow(y, 3) + 420*std::pow(x, 4)*std::pow(y, 2) - 280*std::pow(x, 4)*y + 70*std::pow(x, 4) + 56*std::pow(x, 3)*std::pow(y, 5) - 280*std::pow(x, 3)*std::pow(y, 4) + 560*std::pow(x, 3)*std::pow(y, 3) - 560*std::pow(x, 3)*std::pow(y, 2) + 280*std::pow(x, 3)*y - 56*std::pow(x, 3) + 28*std::pow(x, 2)*std::pow(y, 6) - 168*std::pow(x, 2)*std::pow(y, 5) + 420*std::pow(x, 2)*std::pow(y, 4) - 560*std::pow(x, 2)*std::pow(y, 3) + 420*std::pow(x, 2)*std::pow(y, 2) - 168*std::pow(x, 2)*y + 28*std::pow(x, 2) + 8*x*std::pow(y, 7) - 56*x*std::pow(y, 6) + 168*x*std::pow(y, 5) - 280*x*std::pow(y, 4) + 280*x*std::pow(y, 3) - 168*x*std::pow(y, 2) + 56*x*y - 8*x + std::pow(y, 8) - 8*std::pow(y, 7) + 28*std::pow(y, 6) - 56*std::pow(y, 5) + 70*std::pow(y, 4) - 56*std::pow(y, 3) + 28*std::pow(y, 2) - 8*y + 12870*std::pow(z, 8) + std::pow(z, 7)*(51480*x + 51480*y - 51480) + 84084*std::pow(z, 6)*std::pow(x + y - 1, 2) + 72072*std::pow(z, 5)*std::pow(x + y - 1, 3) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 4) + 9240*std::pow(z, 3)*std::pow(x + y - 1, 5) + 1260*std::pow(z, 2)*std::pow(x + y - 1, 6) + 72*z*std::pow(x + y - 1, 7) + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (462*x - 42)*(std::pow(x, 8) + 8*std::pow(x, 7)*y - 8*std::pow(x, 7) + 28*std::pow(x, 6)*std::pow(y, 2) - 56*std::pow(x, 6)*y + 28*std::pow(x, 6) + 56*std::pow(x, 5)*std::pow(y, 3) - 168*std::pow(x, 5)*std::pow(y, 2) + 168*std::pow(x, 5)*y - 56*std::pow(x, 5) + 70*std::pow(x, 4)*std::pow(y, 4) - 280*std::pow(x, 4)*std::pow(y, 3) + 420*std::pow(x, 4)*std::pow(y, 2) - 280*std::pow(x, 4)*y + 70*std::pow(x, 4) + 56*std::pow(x, 3)*std::pow(y, 5) - 280*std::pow(x, 3)*std::pow(y, 4) + 560*std::pow(x, 3)*std::pow(y, 3) - 560*std::pow(x, 3)*std::pow(y, 2) + 280*std::pow(x, 3)*y - 56*std::pow(x, 3) + 28*std::pow(x, 2)*std::pow(y, 6) - 168*std::pow(x, 2)*std::pow(y, 5) + 420*std::pow(x, 2)*std::pow(y, 4) - 560*std::pow(x, 2)*std::pow(y, 3) + 420*std::pow(x, 2)*std::pow(y, 2) - 168*std::pow(x, 2)*y + 28*std::pow(x, 2) + 8*x*std::pow(y, 7) - 56*x*std::pow(y, 6) + 168*x*std::pow(y, 5) - 280*x*std::pow(y, 4) + 280*x*std::pow(y, 3) - 168*x*std::pow(y, 2) + 56*x*y - 8*x + std::pow(y, 8) - 8*std::pow(y, 7) + 28*std::pow(y, 6) - 56*std::pow(y, 5) + 70*std::pow(y, 4) - 56*std::pow(y, 3) + 28*std::pow(y, 2) - 8*y + 12870*std::pow(z, 8) + std::pow(z, 7)*(51480*x + 51480*y - 51480) + 84084*std::pow(z, 6)*std::pow(x + y - 1, 2) + 72072*std::pow(z, 5)*std::pow(x + y - 1, 3) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 4) + 9240*std::pow(z, 3)*std::pow(x + y - 1, 5) + 1260*std::pow(z, 2)*std::pow(x + y - 1, 6) + 72*z*std::pow(x + y - 1, 7) + 1) + (21*x*(11*x - 2) + 1)*(8*std::pow(x, 7) + 56*std::pow(x, 6)*y - 56*std::pow(x, 6) + 168*std::pow(x, 5)*std::pow(y, 2) - 336*std::pow(x, 5)*y + 168*std::pow(x, 5) + 280*std::pow(x, 4)*std::pow(y, 3) - 840*std::pow(x, 4)*std::pow(y, 2) + 840*std::pow(x, 4)*y - 280*std::pow(x, 4) + 280*std::pow(x, 3)*std::pow(y, 4) - 1120*std::pow(x, 3)*std::pow(y, 3) + 1680*std::pow(x, 3)*std::pow(y, 2) - 1120*std::pow(x, 3)*y + 280*std::pow(x, 3) + 168*std::pow(x, 2)*std::pow(y, 5) - 840*std::pow(x, 2)*std::pow(y, 4) + 1680*std::pow(x, 2)*std::pow(y, 3) - 1680*std::pow(x, 2)*std::pow(y, 2) + 840*std::pow(x, 2)*y - 168*std::pow(x, 2) + 56*x*std::pow(y, 6) - 336*x*std::pow(y, 5) + 840*x*std::pow(y, 4) - 1120*x*std::pow(y, 3) + 840*x*std::pow(y, 2) - 336*x*y + 56*x + 8*std::pow(y, 7) - 56*std::pow(y, 6) + 168*std::pow(y, 5) - 280*std::pow(y, 4) + 280*std::pow(y, 3) - 168*std::pow(y, 2) + 56*y + 51480*std::pow(z, 7) + 84084*std::pow(z, 6)*(2*x + 2*y - 2) + 216216*std::pow(z, 5)*std::pow(x + y - 1, 2) + 138600*std::pow(z, 4)*std::pow(x + y - 1, 3) + 46200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 7560*std::pow(z, 2)*std::pow(x + y - 1, 5) + 504*z*std::pow(x + y - 1, 6) - 8),
            (21*x*(11*x - 2) + 1)*(8*std::pow(x, 7) + 56*std::pow(x, 6)*y - 56*std::pow(x, 6) + 168*std::pow(x, 5)*std::pow(y, 2) - 336*std::pow(x, 5)*y + 168*std::pow(x, 5) + 280*std::pow(x, 4)*std::pow(y, 3) - 840*std::pow(x, 4)*std::pow(y, 2) + 840*std::pow(x, 4)*y - 280*std::pow(x, 4) + 280*std::pow(x, 3)*std::pow(y, 4) - 1120*std::pow(x, 3)*std::pow(y, 3) + 1680*std::pow(x, 3)*std::pow(y, 2) - 1120*std::pow(x, 3)*y + 280*std::pow(x, 3) + 168*std::pow(x, 2)*std::pow(y, 5) - 840*std::pow(x, 2)*std::pow(y, 4) + 1680*std::pow(x, 2)*std::pow(y, 3) - 1680*std::pow(x, 2)*std::pow(y, 2) + 840*std::pow(x, 2)*y - 168*std::pow(x, 2) + 56*x*std::pow(y, 6) - 336*x*std::pow(y, 5) + 840*x*std::pow(y, 4) - 1120*x*std::pow(y, 3) + 840*x*std::pow(y, 2) - 336*x*y + 56*x + 8*std::pow(y, 7) - 56*std::pow(y, 6) + 168*std::pow(y, 5) - 280*std::pow(y, 4) + 280*std::pow(y, 3) - 168*std::pow(y, 2) + 56*y + 51480*std::pow(z, 7) + 84084*std::pow(z, 6)*(2*x + 2*y - 2) + 216216*std::pow(z, 5)*std::pow(x + y - 1, 2) + 138600*std::pow(z, 4)*std::pow(x + y - 1, 3) + 46200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 7560*std::pow(z, 2)*std::pow(x + y - 1, 5) + 504*z*std::pow(x + y - 1, 6) - 8),
            (21*x*(11*x - 2) + 1)*(102960*std::pow(z, 7) + 7*std::pow(z, 6)*(51480*x + 51480*y - 51480) + 504504*std::pow(z, 5)*std::pow(x + y - 1, 2) + 360360*std::pow(z, 4)*std::pow(x + y - 1, 3) + 138600*std::pow(z, 3)*std::pow(x + y - 1, 4) + 27720*std::pow(z, 2)*std::pow(x + y - 1, 5) + 2520*z*std::pow(x + y - 1, 6) + 72*std::pow(x + y - 1, 7))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 281
template<>
struct DGBasis<281> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (22*x - 1)*(x + 19*y - 1)*(std::pow(x, 8) + 8*std::pow(x, 7)*y - 8*std::pow(x, 7) + 28*std::pow(x, 6)*std::pow(y, 2) - 56*std::pow(x, 6)*y + 28*std::pow(x, 6) + 56*std::pow(x, 5)*std::pow(y, 3) - 168*std::pow(x, 5)*std::pow(y, 2) + 168*std::pow(x, 5)*y - 56*std::pow(x, 5) + 70*std::pow(x, 4)*std::pow(y, 4) - 280*std::pow(x, 4)*std::pow(y, 3) + 420*std::pow(x, 4)*std::pow(y, 2) - 280*std::pow(x, 4)*y + 70*std::pow(x, 4) + 56*std::pow(x, 3)*std::pow(y, 5) - 280*std::pow(x, 3)*std::pow(y, 4) + 560*std::pow(x, 3)*std::pow(y, 3) - 560*std::pow(x, 3)*std::pow(y, 2) + 280*std::pow(x, 3)*y - 56*std::pow(x, 3) + 28*std::pow(x, 2)*std::pow(y, 6) - 168*std::pow(x, 2)*std::pow(y, 5) + 420*std::pow(x, 2)*std::pow(y, 4) - 560*std::pow(x, 2)*std::pow(y, 3) + 420*std::pow(x, 2)*std::pow(y, 2) - 168*std::pow(x, 2)*y + 28*std::pow(x, 2) + 8*x*std::pow(y, 7) - 56*x*std::pow(y, 6) + 168*x*std::pow(y, 5) - 280*x*std::pow(y, 4) + 280*x*std::pow(y, 3) - 168*x*std::pow(y, 2) + 56*x*y - 8*x + std::pow(y, 8) - 8*std::pow(y, 7) + 28*std::pow(y, 6) - 56*std::pow(y, 5) + 70*std::pow(y, 4) - 56*std::pow(y, 3) + 28*std::pow(y, 2) - 8*y + 12870*std::pow(z, 8) + std::pow(z, 7)*(51480*x + 51480*y - 51480) + 84084*std::pow(z, 6)*std::pow(x + y - 1, 2) + 72072*std::pow(z, 5)*std::pow(x + y - 1, 3) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 4) + 9240*std::pow(z, 3)*std::pow(x + y - 1, 5) + 1260*std::pow(z, 2)*std::pow(x + y - 1, 6) + 72*z*std::pow(x + y - 1, 7) + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (22*x - 1)*(x + 19*y - 1)*(8*std::pow(x, 7) + 56*std::pow(x, 6)*y - 56*std::pow(x, 6) + 168*std::pow(x, 5)*std::pow(y, 2) - 336*std::pow(x, 5)*y + 168*std::pow(x, 5) + 280*std::pow(x, 4)*std::pow(y, 3) - 840*std::pow(x, 4)*std::pow(y, 2) + 840*std::pow(x, 4)*y - 280*std::pow(x, 4) + 280*std::pow(x, 3)*std::pow(y, 4) - 1120*std::pow(x, 3)*std::pow(y, 3) + 1680*std::pow(x, 3)*std::pow(y, 2) - 1120*std::pow(x, 3)*y + 280*std::pow(x, 3) + 168*std::pow(x, 2)*std::pow(y, 5) - 840*std::pow(x, 2)*std::pow(y, 4) + 1680*std::pow(x, 2)*std::pow(y, 3) - 1680*std::pow(x, 2)*std::pow(y, 2) + 840*std::pow(x, 2)*y - 168*std::pow(x, 2) + 56*x*std::pow(y, 6) - 336*x*std::pow(y, 5) + 840*x*std::pow(y, 4) - 1120*x*std::pow(y, 3) + 840*x*std::pow(y, 2) - 336*x*y + 56*x + 8*std::pow(y, 7) - 56*std::pow(y, 6) + 168*std::pow(y, 5) - 280*std::pow(y, 4) + 280*std::pow(y, 3) - 168*std::pow(y, 2) + 56*y + 51480*std::pow(z, 7) + 84084*std::pow(z, 6)*(2*x + 2*y - 2) + 216216*std::pow(z, 5)*std::pow(x + y - 1, 2) + 138600*std::pow(z, 4)*std::pow(x + y - 1, 3) + 46200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 7560*std::pow(z, 2)*std::pow(x + y - 1, 5) + 504*z*std::pow(x + y - 1, 6) - 8) + (22*x - 1)*(std::pow(x, 8) + 8*std::pow(x, 7)*y - 8*std::pow(x, 7) + 28*std::pow(x, 6)*std::pow(y, 2) - 56*std::pow(x, 6)*y + 28*std::pow(x, 6) + 56*std::pow(x, 5)*std::pow(y, 3) - 168*std::pow(x, 5)*std::pow(y, 2) + 168*std::pow(x, 5)*y - 56*std::pow(x, 5) + 70*std::pow(x, 4)*std::pow(y, 4) - 280*std::pow(x, 4)*std::pow(y, 3) + 420*std::pow(x, 4)*std::pow(y, 2) - 280*std::pow(x, 4)*y + 70*std::pow(x, 4) + 56*std::pow(x, 3)*std::pow(y, 5) - 280*std::pow(x, 3)*std::pow(y, 4) + 560*std::pow(x, 3)*std::pow(y, 3) - 560*std::pow(x, 3)*std::pow(y, 2) + 280*std::pow(x, 3)*y - 56*std::pow(x, 3) + 28*std::pow(x, 2)*std::pow(y, 6) - 168*std::pow(x, 2)*std::pow(y, 5) + 420*std::pow(x, 2)*std::pow(y, 4) - 560*std::pow(x, 2)*std::pow(y, 3) + 420*std::pow(x, 2)*std::pow(y, 2) - 168*std::pow(x, 2)*y + 28*std::pow(x, 2) + 8*x*std::pow(y, 7) - 56*x*std::pow(y, 6) + 168*x*std::pow(y, 5) - 280*x*std::pow(y, 4) + 280*x*std::pow(y, 3) - 168*x*std::pow(y, 2) + 56*x*y - 8*x + std::pow(y, 8) - 8*std::pow(y, 7) + 28*std::pow(y, 6) - 56*std::pow(y, 5) + 70*std::pow(y, 4) - 56*std::pow(y, 3) + 28*std::pow(y, 2) - 8*y + 12870*std::pow(z, 8) + std::pow(z, 7)*(51480*x + 51480*y - 51480) + 84084*std::pow(z, 6)*std::pow(x + y - 1, 2) + 72072*std::pow(z, 5)*std::pow(x + y - 1, 3) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 4) + 9240*std::pow(z, 3)*std::pow(x + y - 1, 5) + 1260*std::pow(z, 2)*std::pow(x + y - 1, 6) + 72*z*std::pow(x + y - 1, 7) + 1) + 22*(x + 19*y - 1)*(std::pow(x, 8) + 8*std::pow(x, 7)*y - 8*std::pow(x, 7) + 28*std::pow(x, 6)*std::pow(y, 2) - 56*std::pow(x, 6)*y + 28*std::pow(x, 6) + 56*std::pow(x, 5)*std::pow(y, 3) - 168*std::pow(x, 5)*std::pow(y, 2) + 168*std::pow(x, 5)*y - 56*std::pow(x, 5) + 70*std::pow(x, 4)*std::pow(y, 4) - 280*std::pow(x, 4)*std::pow(y, 3) + 420*std::pow(x, 4)*std::pow(y, 2) - 280*std::pow(x, 4)*y + 70*std::pow(x, 4) + 56*std::pow(x, 3)*std::pow(y, 5) - 280*std::pow(x, 3)*std::pow(y, 4) + 560*std::pow(x, 3)*std::pow(y, 3) - 560*std::pow(x, 3)*std::pow(y, 2) + 280*std::pow(x, 3)*y - 56*std::pow(x, 3) + 28*std::pow(x, 2)*std::pow(y, 6) - 168*std::pow(x, 2)*std::pow(y, 5) + 420*std::pow(x, 2)*std::pow(y, 4) - 560*std::pow(x, 2)*std::pow(y, 3) + 420*std::pow(x, 2)*std::pow(y, 2) - 168*std::pow(x, 2)*y + 28*std::pow(x, 2) + 8*x*std::pow(y, 7) - 56*x*std::pow(y, 6) + 168*x*std::pow(y, 5) - 280*x*std::pow(y, 4) + 280*x*std::pow(y, 3) - 168*x*std::pow(y, 2) + 56*x*y - 8*x + std::pow(y, 8) - 8*std::pow(y, 7) + 28*std::pow(y, 6) - 56*std::pow(y, 5) + 70*std::pow(y, 4) - 56*std::pow(y, 3) + 28*std::pow(y, 2) - 8*y + 12870*std::pow(z, 8) + std::pow(z, 7)*(51480*x + 51480*y - 51480) + 84084*std::pow(z, 6)*std::pow(x + y - 1, 2) + 72072*std::pow(z, 5)*std::pow(x + y - 1, 3) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 4) + 9240*std::pow(z, 3)*std::pow(x + y - 1, 5) + 1260*std::pow(z, 2)*std::pow(x + y - 1, 6) + 72*z*std::pow(x + y - 1, 7) + 1),
            (22*x - 1)*(x + 19*y - 1)*(8*std::pow(x, 7) + 56*std::pow(x, 6)*y - 56*std::pow(x, 6) + 168*std::pow(x, 5)*std::pow(y, 2) - 336*std::pow(x, 5)*y + 168*std::pow(x, 5) + 280*std::pow(x, 4)*std::pow(y, 3) - 840*std::pow(x, 4)*std::pow(y, 2) + 840*std::pow(x, 4)*y - 280*std::pow(x, 4) + 280*std::pow(x, 3)*std::pow(y, 4) - 1120*std::pow(x, 3)*std::pow(y, 3) + 1680*std::pow(x, 3)*std::pow(y, 2) - 1120*std::pow(x, 3)*y + 280*std::pow(x, 3) + 168*std::pow(x, 2)*std::pow(y, 5) - 840*std::pow(x, 2)*std::pow(y, 4) + 1680*std::pow(x, 2)*std::pow(y, 3) - 1680*std::pow(x, 2)*std::pow(y, 2) + 840*std::pow(x, 2)*y - 168*std::pow(x, 2) + 56*x*std::pow(y, 6) - 336*x*std::pow(y, 5) + 840*x*std::pow(y, 4) - 1120*x*std::pow(y, 3) + 840*x*std::pow(y, 2) - 336*x*y + 56*x + 8*std::pow(y, 7) - 56*std::pow(y, 6) + 168*std::pow(y, 5) - 280*std::pow(y, 4) + 280*std::pow(y, 3) - 168*std::pow(y, 2) + 56*y + 51480*std::pow(z, 7) + 84084*std::pow(z, 6)*(2*x + 2*y - 2) + 216216*std::pow(z, 5)*std::pow(x + y - 1, 2) + 138600*std::pow(z, 4)*std::pow(x + y - 1, 3) + 46200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 7560*std::pow(z, 2)*std::pow(x + y - 1, 5) + 504*z*std::pow(x + y - 1, 6) - 8) + 19*(22*x - 1)*(std::pow(x, 8) + 8*std::pow(x, 7)*y - 8*std::pow(x, 7) + 28*std::pow(x, 6)*std::pow(y, 2) - 56*std::pow(x, 6)*y + 28*std::pow(x, 6) + 56*std::pow(x, 5)*std::pow(y, 3) - 168*std::pow(x, 5)*std::pow(y, 2) + 168*std::pow(x, 5)*y - 56*std::pow(x, 5) + 70*std::pow(x, 4)*std::pow(y, 4) - 280*std::pow(x, 4)*std::pow(y, 3) + 420*std::pow(x, 4)*std::pow(y, 2) - 280*std::pow(x, 4)*y + 70*std::pow(x, 4) + 56*std::pow(x, 3)*std::pow(y, 5) - 280*std::pow(x, 3)*std::pow(y, 4) + 560*std::pow(x, 3)*std::pow(y, 3) - 560*std::pow(x, 3)*std::pow(y, 2) + 280*std::pow(x, 3)*y - 56*std::pow(x, 3) + 28*std::pow(x, 2)*std::pow(y, 6) - 168*std::pow(x, 2)*std::pow(y, 5) + 420*std::pow(x, 2)*std::pow(y, 4) - 560*std::pow(x, 2)*std::pow(y, 3) + 420*std::pow(x, 2)*std::pow(y, 2) - 168*std::pow(x, 2)*y + 28*std::pow(x, 2) + 8*x*std::pow(y, 7) - 56*x*std::pow(y, 6) + 168*x*std::pow(y, 5) - 280*x*std::pow(y, 4) + 280*x*std::pow(y, 3) - 168*x*std::pow(y, 2) + 56*x*y - 8*x + std::pow(y, 8) - 8*std::pow(y, 7) + 28*std::pow(y, 6) - 56*std::pow(y, 5) + 70*std::pow(y, 4) - 56*std::pow(y, 3) + 28*std::pow(y, 2) - 8*y + 12870*std::pow(z, 8) + std::pow(z, 7)*(51480*x + 51480*y - 51480) + 84084*std::pow(z, 6)*std::pow(x + y - 1, 2) + 72072*std::pow(z, 5)*std::pow(x + y - 1, 3) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 4) + 9240*std::pow(z, 3)*std::pow(x + y - 1, 5) + 1260*std::pow(z, 2)*std::pow(x + y - 1, 6) + 72*z*std::pow(x + y - 1, 7) + 1),
            (22*x - 1)*(x + 19*y - 1)*(102960*std::pow(z, 7) + 7*std::pow(z, 6)*(51480*x + 51480*y - 51480) + 504504*std::pow(z, 5)*std::pow(x + y - 1, 2) + 360360*std::pow(z, 4)*std::pow(x + y - 1, 3) + 138600*std::pow(z, 3)*std::pow(x + y - 1, 4) + 27720*std::pow(z, 2)*std::pow(x + y - 1, 5) + 2520*z*std::pow(x + y - 1, 6) + 72*std::pow(x + y - 1, 7))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 282
template<>
struct DGBasis<282> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (210*std::pow(y, 2) + y*(40*x - 40) + std::pow(x - 1, 2))*(std::pow(x, 8) + 8*std::pow(x, 7)*y - 8*std::pow(x, 7) + 28*std::pow(x, 6)*std::pow(y, 2) - 56*std::pow(x, 6)*y + 28*std::pow(x, 6) + 56*std::pow(x, 5)*std::pow(y, 3) - 168*std::pow(x, 5)*std::pow(y, 2) + 168*std::pow(x, 5)*y - 56*std::pow(x, 5) + 70*std::pow(x, 4)*std::pow(y, 4) - 280*std::pow(x, 4)*std::pow(y, 3) + 420*std::pow(x, 4)*std::pow(y, 2) - 280*std::pow(x, 4)*y + 70*std::pow(x, 4) + 56*std::pow(x, 3)*std::pow(y, 5) - 280*std::pow(x, 3)*std::pow(y, 4) + 560*std::pow(x, 3)*std::pow(y, 3) - 560*std::pow(x, 3)*std::pow(y, 2) + 280*std::pow(x, 3)*y - 56*std::pow(x, 3) + 28*std::pow(x, 2)*std::pow(y, 6) - 168*std::pow(x, 2)*std::pow(y, 5) + 420*std::pow(x, 2)*std::pow(y, 4) - 560*std::pow(x, 2)*std::pow(y, 3) + 420*std::pow(x, 2)*std::pow(y, 2) - 168*std::pow(x, 2)*y + 28*std::pow(x, 2) + 8*x*std::pow(y, 7) - 56*x*std::pow(y, 6) + 168*x*std::pow(y, 5) - 280*x*std::pow(y, 4) + 280*x*std::pow(y, 3) - 168*x*std::pow(y, 2) + 56*x*y - 8*x + std::pow(y, 8) - 8*std::pow(y, 7) + 28*std::pow(y, 6) - 56*std::pow(y, 5) + 70*std::pow(y, 4) - 56*std::pow(y, 3) + 28*std::pow(y, 2) - 8*y + 12870*std::pow(z, 8) + std::pow(z, 7)*(51480*x + 51480*y - 51480) + 84084*std::pow(z, 6)*std::pow(x + y - 1, 2) + 72072*std::pow(z, 5)*std::pow(x + y - 1, 3) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 4) + 9240*std::pow(z, 3)*std::pow(x + y - 1, 5) + 1260*std::pow(z, 2)*std::pow(x + y - 1, 6) + 72*z*std::pow(x + y - 1, 7) + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            (2*x + 40*y - 2)*(std::pow(x, 8) + 8*std::pow(x, 7)*y - 8*std::pow(x, 7) + 28*std::pow(x, 6)*std::pow(y, 2) - 56*std::pow(x, 6)*y + 28*std::pow(x, 6) + 56*std::pow(x, 5)*std::pow(y, 3) - 168*std::pow(x, 5)*std::pow(y, 2) + 168*std::pow(x, 5)*y - 56*std::pow(x, 5) + 70*std::pow(x, 4)*std::pow(y, 4) - 280*std::pow(x, 4)*std::pow(y, 3) + 420*std::pow(x, 4)*std::pow(y, 2) - 280*std::pow(x, 4)*y + 70*std::pow(x, 4) + 56*std::pow(x, 3)*std::pow(y, 5) - 280*std::pow(x, 3)*std::pow(y, 4) + 560*std::pow(x, 3)*std::pow(y, 3) - 560*std::pow(x, 3)*std::pow(y, 2) + 280*std::pow(x, 3)*y - 56*std::pow(x, 3) + 28*std::pow(x, 2)*std::pow(y, 6) - 168*std::pow(x, 2)*std::pow(y, 5) + 420*std::pow(x, 2)*std::pow(y, 4) - 560*std::pow(x, 2)*std::pow(y, 3) + 420*std::pow(x, 2)*std::pow(y, 2) - 168*std::pow(x, 2)*y + 28*std::pow(x, 2) + 8*x*std::pow(y, 7) - 56*x*std::pow(y, 6) + 168*x*std::pow(y, 5) - 280*x*std::pow(y, 4) + 280*x*std::pow(y, 3) - 168*x*std::pow(y, 2) + 56*x*y - 8*x + std::pow(y, 8) - 8*std::pow(y, 7) + 28*std::pow(y, 6) - 56*std::pow(y, 5) + 70*std::pow(y, 4) - 56*std::pow(y, 3) + 28*std::pow(y, 2) - 8*y + 12870*std::pow(z, 8) + std::pow(z, 7)*(51480*x + 51480*y - 51480) + 84084*std::pow(z, 6)*std::pow(x + y - 1, 2) + 72072*std::pow(z, 5)*std::pow(x + y - 1, 3) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 4) + 9240*std::pow(z, 3)*std::pow(x + y - 1, 5) + 1260*std::pow(z, 2)*std::pow(x + y - 1, 6) + 72*z*std::pow(x + y - 1, 7) + 1) + (210*std::pow(y, 2) + y*(40*x - 40) + std::pow(x - 1, 2))*(8*std::pow(x, 7) + 56*std::pow(x, 6)*y - 56*std::pow(x, 6) + 168*std::pow(x, 5)*std::pow(y, 2) - 336*std::pow(x, 5)*y + 168*std::pow(x, 5) + 280*std::pow(x, 4)*std::pow(y, 3) - 840*std::pow(x, 4)*std::pow(y, 2) + 840*std::pow(x, 4)*y - 280*std::pow(x, 4) + 280*std::pow(x, 3)*std::pow(y, 4) - 1120*std::pow(x, 3)*std::pow(y, 3) + 1680*std::pow(x, 3)*std::pow(y, 2) - 1120*std::pow(x, 3)*y + 280*std::pow(x, 3) + 168*std::pow(x, 2)*std::pow(y, 5) - 840*std::pow(x, 2)*std::pow(y, 4) + 1680*std::pow(x, 2)*std::pow(y, 3) - 1680*std::pow(x, 2)*std::pow(y, 2) + 840*std::pow(x, 2)*y - 168*std::pow(x, 2) + 56*x*std::pow(y, 6) - 336*x*std::pow(y, 5) + 840*x*std::pow(y, 4) - 1120*x*std::pow(y, 3) + 840*x*std::pow(y, 2) - 336*x*y + 56*x + 8*std::pow(y, 7) - 56*std::pow(y, 6) + 168*std::pow(y, 5) - 280*std::pow(y, 4) + 280*std::pow(y, 3) - 168*std::pow(y, 2) + 56*y + 51480*std::pow(z, 7) + 84084*std::pow(z, 6)*(2*x + 2*y - 2) + 216216*std::pow(z, 5)*std::pow(x + y - 1, 2) + 138600*std::pow(z, 4)*std::pow(x + y - 1, 3) + 46200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 7560*std::pow(z, 2)*std::pow(x + y - 1, 5) + 504*z*std::pow(x + y - 1, 6) - 8),
            (40*x + 420*y - 40)*(std::pow(x, 8) + 8*std::pow(x, 7)*y - 8*std::pow(x, 7) + 28*std::pow(x, 6)*std::pow(y, 2) - 56*std::pow(x, 6)*y + 28*std::pow(x, 6) + 56*std::pow(x, 5)*std::pow(y, 3) - 168*std::pow(x, 5)*std::pow(y, 2) + 168*std::pow(x, 5)*y - 56*std::pow(x, 5) + 70*std::pow(x, 4)*std::pow(y, 4) - 280*std::pow(x, 4)*std::pow(y, 3) + 420*std::pow(x, 4)*std::pow(y, 2) - 280*std::pow(x, 4)*y + 70*std::pow(x, 4) + 56*std::pow(x, 3)*std::pow(y, 5) - 280*std::pow(x, 3)*std::pow(y, 4) + 560*std::pow(x, 3)*std::pow(y, 3) - 560*std::pow(x, 3)*std::pow(y, 2) + 280*std::pow(x, 3)*y - 56*std::pow(x, 3) + 28*std::pow(x, 2)*std::pow(y, 6) - 168*std::pow(x, 2)*std::pow(y, 5) + 420*std::pow(x, 2)*std::pow(y, 4) - 560*std::pow(x, 2)*std::pow(y, 3) + 420*std::pow(x, 2)*std::pow(y, 2) - 168*std::pow(x, 2)*y + 28*std::pow(x, 2) + 8*x*std::pow(y, 7) - 56*x*std::pow(y, 6) + 168*x*std::pow(y, 5) - 280*x*std::pow(y, 4) + 280*x*std::pow(y, 3) - 168*x*std::pow(y, 2) + 56*x*y - 8*x + std::pow(y, 8) - 8*std::pow(y, 7) + 28*std::pow(y, 6) - 56*std::pow(y, 5) + 70*std::pow(y, 4) - 56*std::pow(y, 3) + 28*std::pow(y, 2) - 8*y + 12870*std::pow(z, 8) + std::pow(z, 7)*(51480*x + 51480*y - 51480) + 84084*std::pow(z, 6)*std::pow(x + y - 1, 2) + 72072*std::pow(z, 5)*std::pow(x + y - 1, 3) + 34650*std::pow(z, 4)*std::pow(x + y - 1, 4) + 9240*std::pow(z, 3)*std::pow(x + y - 1, 5) + 1260*std::pow(z, 2)*std::pow(x + y - 1, 6) + 72*z*std::pow(x + y - 1, 7) + 1) + (210*std::pow(y, 2) + y*(40*x - 40) + std::pow(x - 1, 2))*(8*std::pow(x, 7) + 56*std::pow(x, 6)*y - 56*std::pow(x, 6) + 168*std::pow(x, 5)*std::pow(y, 2) - 336*std::pow(x, 5)*y + 168*std::pow(x, 5) + 280*std::pow(x, 4)*std::pow(y, 3) - 840*std::pow(x, 4)*std::pow(y, 2) + 840*std::pow(x, 4)*y - 280*std::pow(x, 4) + 280*std::pow(x, 3)*std::pow(y, 4) - 1120*std::pow(x, 3)*std::pow(y, 3) + 1680*std::pow(x, 3)*std::pow(y, 2) - 1120*std::pow(x, 3)*y + 280*std::pow(x, 3) + 168*std::pow(x, 2)*std::pow(y, 5) - 840*std::pow(x, 2)*std::pow(y, 4) + 1680*std::pow(x, 2)*std::pow(y, 3) - 1680*std::pow(x, 2)*std::pow(y, 2) + 840*std::pow(x, 2)*y - 168*std::pow(x, 2) + 56*x*std::pow(y, 6) - 336*x*std::pow(y, 5) + 840*x*std::pow(y, 4) - 1120*x*std::pow(y, 3) + 840*x*std::pow(y, 2) - 336*x*y + 56*x + 8*std::pow(y, 7) - 56*std::pow(y, 6) + 168*std::pow(y, 5) - 280*std::pow(y, 4) + 280*std::pow(y, 3) - 168*std::pow(y, 2) + 56*y + 51480*std::pow(z, 7) + 84084*std::pow(z, 6)*(2*x + 2*y - 2) + 216216*std::pow(z, 5)*std::pow(x + y - 1, 2) + 138600*std::pow(z, 4)*std::pow(x + y - 1, 3) + 46200*std::pow(z, 3)*std::pow(x + y - 1, 4) + 7560*std::pow(z, 2)*std::pow(x + y - 1, 5) + 504*z*std::pow(x + y - 1, 6) - 8),
            (210*std::pow(y, 2) + y*(40*x - 40) + std::pow(x - 1, 2))*(102960*std::pow(z, 7) + 7*std::pow(z, 6)*(51480*x + 51480*y - 51480) + 504504*std::pow(z, 5)*std::pow(x + y - 1, 2) + 360360*std::pow(z, 4)*std::pow(x + y - 1, 3) + 138600*std::pow(z, 3)*std::pow(x + y - 1, 4) + 27720*std::pow(z, 2)*std::pow(x + y - 1, 5) + 2520*z*std::pow(x + y - 1, 6) + 72*std::pow(x + y - 1, 7))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 283
template<>
struct DGBasis<283> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (22*x - 1)*(std::pow(x, 9) + 9*std::pow(x, 8)*y - 9*std::pow(x, 8) + 36*std::pow(x, 7)*std::pow(y, 2) - 72*std::pow(x, 7)*y + 36*std::pow(x, 7) + 84*std::pow(x, 6)*std::pow(y, 3) - 252*std::pow(x, 6)*std::pow(y, 2) + 252*std::pow(x, 6)*y - 84*std::pow(x, 6) + 126*std::pow(x, 5)*std::pow(y, 4) - 504*std::pow(x, 5)*std::pow(y, 3) + 756*std::pow(x, 5)*std::pow(y, 2) - 504*std::pow(x, 5)*y + 126*std::pow(x, 5) + 126*std::pow(x, 4)*std::pow(y, 5) - 630*std::pow(x, 4)*std::pow(y, 4) + 1260*std::pow(x, 4)*std::pow(y, 3) - 1260*std::pow(x, 4)*std::pow(y, 2) + 630*std::pow(x, 4)*y - 126*std::pow(x, 4) + 84*std::pow(x, 3)*std::pow(y, 6) - 504*std::pow(x, 3)*std::pow(y, 5) + 1260*std::pow(x, 3)*std::pow(y, 4) - 1680*std::pow(x, 3)*std::pow(y, 3) + 1260*std::pow(x, 3)*std::pow(y, 2) - 504*std::pow(x, 3)*y + 84*std::pow(x, 3) + 36*std::pow(x, 2)*std::pow(y, 7) - 252*std::pow(x, 2)*std::pow(y, 6) + 756*std::pow(x, 2)*std::pow(y, 5) - 1260*std::pow(x, 2)*std::pow(y, 4) + 1260*std::pow(x, 2)*std::pow(y, 3) - 756*std::pow(x, 2)*std::pow(y, 2) + 252*std::pow(x, 2)*y - 36*std::pow(x, 2) + 9*x*std::pow(y, 8) - 72*x*std::pow(y, 7) + 252*x*std::pow(y, 6) - 504*x*std::pow(y, 5) + 630*x*std::pow(y, 4) - 504*x*std::pow(y, 3) + 252*x*std::pow(y, 2) - 72*x*y + 9*x + std::pow(y, 9) - 9*std::pow(y, 8) + 36*std::pow(y, 7) - 84*std::pow(y, 6) + 126*std::pow(y, 5) - 126*std::pow(y, 4) + 84*std::pow(y, 3) - 36*std::pow(y, 2) + 9*y + 48620*std::pow(z, 9) + std::pow(z, 8)*(218790*x + 218790*y - 218790) + 411840*std::pow(z, 7)*std::pow(x + y - 1, 2) + 420420*std::pow(z, 6)*std::pow(x + y - 1, 3) + 252252*std::pow(z, 5)*std::pow(x + y - 1, 4) + 90090*std::pow(z, 4)*std::pow(x + y - 1, 5) + 18480*std::pow(z, 3)*std::pow(x + y - 1, 6) + 1980*std::pow(z, 2)*std::pow(x + y - 1, 7) + 90*z*std::pow(x + y - 1, 8) - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            22*std::pow(x, 9) + 198*std::pow(x, 8)*y - 198*std::pow(x, 8) + 792*std::pow(x, 7)*std::pow(y, 2) - 1584*std::pow(x, 7)*y + 792*std::pow(x, 7) + 1848*std::pow(x, 6)*std::pow(y, 3) - 5544*std::pow(x, 6)*std::pow(y, 2) + 5544*std::pow(x, 6)*y - 1848*std::pow(x, 6) + 2772*std::pow(x, 5)*std::pow(y, 4) - 11088*std::pow(x, 5)*std::pow(y, 3) + 16632*std::pow(x, 5)*std::pow(y, 2) - 11088*std::pow(x, 5)*y + 2772*std::pow(x, 5) + 2772*std::pow(x, 4)*std::pow(y, 5) - 13860*std::pow(x, 4)*std::pow(y, 4) + 27720*std::pow(x, 4)*std::pow(y, 3) - 27720*std::pow(x, 4)*std::pow(y, 2) + 13860*std::pow(x, 4)*y - 2772*std::pow(x, 4) + 1848*std::pow(x, 3)*std::pow(y, 6) - 11088*std::pow(x, 3)*std::pow(y, 5) + 27720*std::pow(x, 3)*std::pow(y, 4) - 36960*std::pow(x, 3)*std::pow(y, 3) + 27720*std::pow(x, 3)*std::pow(y, 2) - 11088*std::pow(x, 3)*y + 1848*std::pow(x, 3) + 792*std::pow(x, 2)*std::pow(y, 7) - 5544*std::pow(x, 2)*std::pow(y, 6) + 16632*std::pow(x, 2)*std::pow(y, 5) - 27720*std::pow(x, 2)*std::pow(y, 4) + 27720*std::pow(x, 2)*std::pow(y, 3) - 16632*std::pow(x, 2)*std::pow(y, 2) + 5544*std::pow(x, 2)*y - 792*std::pow(x, 2) + 198*x*std::pow(y, 8) - 1584*x*std::pow(y, 7) + 5544*x*std::pow(y, 6) - 11088*x*std::pow(y, 5) + 13860*x*std::pow(y, 4) - 11088*x*std::pow(y, 3) + 5544*x*std::pow(y, 2) - 1584*x*y + 198*x + 22*std::pow(y, 9) - 198*std::pow(y, 8) + 792*std::pow(y, 7) - 1848*std::pow(y, 6) + 2772*std::pow(y, 5) - 2772*std::pow(y, 4) + 1848*std::pow(y, 3) - 792*std::pow(y, 2) + 198*y + 1069640*std::pow(z, 9) + 22*std::pow(z, 8)*(218790*x + 218790*y - 218790) + 9060480*std::pow(z, 7)*std::pow(x + y - 1, 2) + 9249240*std::pow(z, 6)*std::pow(x + y - 1, 3) + 5549544*std::pow(z, 5)*std::pow(x + y - 1, 4) + 1981980*std::pow(z, 4)*std::pow(x + y - 1, 5) + 406560*std::pow(z, 3)*std::pow(x + y - 1, 6) + 43560*std::pow(z, 2)*std::pow(x + y - 1, 7) + 1980*z*std::pow(x + y - 1, 8) + (22*x - 1)*(9*std::pow(x, 8) + 72*std::pow(x, 7)*y - 72*std::pow(x, 7) + 252*std::pow(x, 6)*std::pow(y, 2) - 504*std::pow(x, 6)*y + 252*std::pow(x, 6) + 504*std::pow(x, 5)*std::pow(y, 3) - 1512*std::pow(x, 5)*std::pow(y, 2) + 1512*std::pow(x, 5)*y - 504*std::pow(x, 5) + 630*std::pow(x, 4)*std::pow(y, 4) - 2520*std::pow(x, 4)*std::pow(y, 3) + 3780*std::pow(x, 4)*std::pow(y, 2) - 2520*std::pow(x, 4)*y + 630*std::pow(x, 4) + 504*std::pow(x, 3)*std::pow(y, 5) - 2520*std::pow(x, 3)*std::pow(y, 4) + 5040*std::pow(x, 3)*std::pow(y, 3) - 5040*std::pow(x, 3)*std::pow(y, 2) + 2520*std::pow(x, 3)*y - 504*std::pow(x, 3) + 252*std::pow(x, 2)*std::pow(y, 6) - 1512*std::pow(x, 2)*std::pow(y, 5) + 3780*std::pow(x, 2)*std::pow(y, 4) - 5040*std::pow(x, 2)*std::pow(y, 3) + 3780*std::pow(x, 2)*std::pow(y, 2) - 1512*std::pow(x, 2)*y + 252*std::pow(x, 2) + 72*x*std::pow(y, 7) - 504*x*std::pow(y, 6) + 1512*x*std::pow(y, 5) - 2520*x*std::pow(y, 4) + 2520*x*std::pow(y, 3) - 1512*x*std::pow(y, 2) + 504*x*y - 72*x + 9*std::pow(y, 8) - 72*std::pow(y, 7) + 252*std::pow(y, 6) - 504*std::pow(y, 5) + 630*std::pow(y, 4) - 504*std::pow(y, 3) + 252*std::pow(y, 2) - 72*y + 218790*std::pow(z, 8) + 411840*std::pow(z, 7)*(2*x + 2*y - 2) + 1261260*std::pow(z, 6)*std::pow(x + y - 1, 2) + 1009008*std::pow(z, 5)*std::pow(x + y - 1, 3) + 450450*std::pow(z, 4)*std::pow(x + y - 1, 4) + 110880*std::pow(z, 3)*std::pow(x + y - 1, 5) + 13860*std::pow(z, 2)*std::pow(x + y - 1, 6) + 720*z*std::pow(x + y - 1, 7) + 9) - 22,
            (22*x - 1)*(9*std::pow(x, 8) + 72*std::pow(x, 7)*y - 72*std::pow(x, 7) + 252*std::pow(x, 6)*std::pow(y, 2) - 504*std::pow(x, 6)*y + 252*std::pow(x, 6) + 504*std::pow(x, 5)*std::pow(y, 3) - 1512*std::pow(x, 5)*std::pow(y, 2) + 1512*std::pow(x, 5)*y - 504*std::pow(x, 5) + 630*std::pow(x, 4)*std::pow(y, 4) - 2520*std::pow(x, 4)*std::pow(y, 3) + 3780*std::pow(x, 4)*std::pow(y, 2) - 2520*std::pow(x, 4)*y + 630*std::pow(x, 4) + 504*std::pow(x, 3)*std::pow(y, 5) - 2520*std::pow(x, 3)*std::pow(y, 4) + 5040*std::pow(x, 3)*std::pow(y, 3) - 5040*std::pow(x, 3)*std::pow(y, 2) + 2520*std::pow(x, 3)*y - 504*std::pow(x, 3) + 252*std::pow(x, 2)*std::pow(y, 6) - 1512*std::pow(x, 2)*std::pow(y, 5) + 3780*std::pow(x, 2)*std::pow(y, 4) - 5040*std::pow(x, 2)*std::pow(y, 3) + 3780*std::pow(x, 2)*std::pow(y, 2) - 1512*std::pow(x, 2)*y + 252*std::pow(x, 2) + 72*x*std::pow(y, 7) - 504*x*std::pow(y, 6) + 1512*x*std::pow(y, 5) - 2520*x*std::pow(y, 4) + 2520*x*std::pow(y, 3) - 1512*x*std::pow(y, 2) + 504*x*y - 72*x + 9*std::pow(y, 8) - 72*std::pow(y, 7) + 252*std::pow(y, 6) - 504*std::pow(y, 5) + 630*std::pow(y, 4) - 504*std::pow(y, 3) + 252*std::pow(y, 2) - 72*y + 218790*std::pow(z, 8) + 411840*std::pow(z, 7)*(2*x + 2*y - 2) + 1261260*std::pow(z, 6)*std::pow(x + y - 1, 2) + 1009008*std::pow(z, 5)*std::pow(x + y - 1, 3) + 450450*std::pow(z, 4)*std::pow(x + y - 1, 4) + 110880*std::pow(z, 3)*std::pow(x + y - 1, 5) + 13860*std::pow(z, 2)*std::pow(x + y - 1, 6) + 720*z*std::pow(x + y - 1, 7) + 9),
            (22*x - 1)*(437580*std::pow(z, 8) + 8*std::pow(z, 7)*(218790*x + 218790*y - 218790) + 2882880*std::pow(z, 6)*std::pow(x + y - 1, 2) + 2522520*std::pow(z, 5)*std::pow(x + y - 1, 3) + 1261260*std::pow(z, 4)*std::pow(x + y - 1, 4) + 360360*std::pow(z, 3)*std::pow(x + y - 1, 5) + 55440*std::pow(z, 2)*std::pow(x + y - 1, 6) + 3960*z*std::pow(x + y - 1, 7) + 90*std::pow(x + y - 1, 8))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 284
template<>
struct DGBasis<284> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return (x + 21*y - 1)*(std::pow(x, 9) + 9*std::pow(x, 8)*y - 9*std::pow(x, 8) + 36*std::pow(x, 7)*std::pow(y, 2) - 72*std::pow(x, 7)*y + 36*std::pow(x, 7) + 84*std::pow(x, 6)*std::pow(y, 3) - 252*std::pow(x, 6)*std::pow(y, 2) + 252*std::pow(x, 6)*y - 84*std::pow(x, 6) + 126*std::pow(x, 5)*std::pow(y, 4) - 504*std::pow(x, 5)*std::pow(y, 3) + 756*std::pow(x, 5)*std::pow(y, 2) - 504*std::pow(x, 5)*y + 126*std::pow(x, 5) + 126*std::pow(x, 4)*std::pow(y, 5) - 630*std::pow(x, 4)*std::pow(y, 4) + 1260*std::pow(x, 4)*std::pow(y, 3) - 1260*std::pow(x, 4)*std::pow(y, 2) + 630*std::pow(x, 4)*y - 126*std::pow(x, 4) + 84*std::pow(x, 3)*std::pow(y, 6) - 504*std::pow(x, 3)*std::pow(y, 5) + 1260*std::pow(x, 3)*std::pow(y, 4) - 1680*std::pow(x, 3)*std::pow(y, 3) + 1260*std::pow(x, 3)*std::pow(y, 2) - 504*std::pow(x, 3)*y + 84*std::pow(x, 3) + 36*std::pow(x, 2)*std::pow(y, 7) - 252*std::pow(x, 2)*std::pow(y, 6) + 756*std::pow(x, 2)*std::pow(y, 5) - 1260*std::pow(x, 2)*std::pow(y, 4) + 1260*std::pow(x, 2)*std::pow(y, 3) - 756*std::pow(x, 2)*std::pow(y, 2) + 252*std::pow(x, 2)*y - 36*std::pow(x, 2) + 9*x*std::pow(y, 8) - 72*x*std::pow(y, 7) + 252*x*std::pow(y, 6) - 504*x*std::pow(y, 5) + 630*x*std::pow(y, 4) - 504*x*std::pow(y, 3) + 252*x*std::pow(y, 2) - 72*x*y + 9*x + std::pow(y, 9) - 9*std::pow(y, 8) + 36*std::pow(y, 7) - 84*std::pow(y, 6) + 126*std::pow(y, 5) - 126*std::pow(y, 4) + 84*std::pow(y, 3) - 36*std::pow(y, 2) + 9*y + 48620*std::pow(z, 9) + std::pow(z, 8)*(218790*x + 218790*y - 218790) + 411840*std::pow(z, 7)*std::pow(x + y - 1, 2) + 420420*std::pow(z, 6)*std::pow(x + y - 1, 3) + 252252*std::pow(z, 5)*std::pow(x + y - 1, 4) + 90090*std::pow(z, 4)*std::pow(x + y - 1, 5) + 18480*std::pow(z, 3)*std::pow(x + y - 1, 6) + 1980*std::pow(z, 2)*std::pow(x + y - 1, 7) + 90*z*std::pow(x + y - 1, 8) - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            std::pow(x, 9) + 9*std::pow(x, 8)*y - 9*std::pow(x, 8) + 36*std::pow(x, 7)*std::pow(y, 2) - 72*std::pow(x, 7)*y + 36*std::pow(x, 7) + 84*std::pow(x, 6)*std::pow(y, 3) - 252*std::pow(x, 6)*std::pow(y, 2) + 252*std::pow(x, 6)*y - 84*std::pow(x, 6) + 126*std::pow(x, 5)*std::pow(y, 4) - 504*std::pow(x, 5)*std::pow(y, 3) + 756*std::pow(x, 5)*std::pow(y, 2) - 504*std::pow(x, 5)*y + 126*std::pow(x, 5) + 126*std::pow(x, 4)*std::pow(y, 5) - 630*std::pow(x, 4)*std::pow(y, 4) + 1260*std::pow(x, 4)*std::pow(y, 3) - 1260*std::pow(x, 4)*std::pow(y, 2) + 630*std::pow(x, 4)*y - 126*std::pow(x, 4) + 84*std::pow(x, 3)*std::pow(y, 6) - 504*std::pow(x, 3)*std::pow(y, 5) + 1260*std::pow(x, 3)*std::pow(y, 4) - 1680*std::pow(x, 3)*std::pow(y, 3) + 1260*std::pow(x, 3)*std::pow(y, 2) - 504*std::pow(x, 3)*y + 84*std::pow(x, 3) + 36*std::pow(x, 2)*std::pow(y, 7) - 252*std::pow(x, 2)*std::pow(y, 6) + 756*std::pow(x, 2)*std::pow(y, 5) - 1260*std::pow(x, 2)*std::pow(y, 4) + 1260*std::pow(x, 2)*std::pow(y, 3) - 756*std::pow(x, 2)*std::pow(y, 2) + 252*std::pow(x, 2)*y - 36*std::pow(x, 2) + 9*x*std::pow(y, 8) - 72*x*std::pow(y, 7) + 252*x*std::pow(y, 6) - 504*x*std::pow(y, 5) + 630*x*std::pow(y, 4) - 504*x*std::pow(y, 3) + 252*x*std::pow(y, 2) - 72*x*y + 9*x + std::pow(y, 9) - 9*std::pow(y, 8) + 36*std::pow(y, 7) - 84*std::pow(y, 6) + 126*std::pow(y, 5) - 126*std::pow(y, 4) + 84*std::pow(y, 3) - 36*std::pow(y, 2) + 9*y + 48620*std::pow(z, 9) + std::pow(z, 8)*(218790*x + 218790*y - 218790) + 411840*std::pow(z, 7)*std::pow(x + y - 1, 2) + 420420*std::pow(z, 6)*std::pow(x + y - 1, 3) + 252252*std::pow(z, 5)*std::pow(x + y - 1, 4) + 90090*std::pow(z, 4)*std::pow(x + y - 1, 5) + 18480*std::pow(z, 3)*std::pow(x + y - 1, 6) + 1980*std::pow(z, 2)*std::pow(x + y - 1, 7) + 90*z*std::pow(x + y - 1, 8) + (x + 21*y - 1)*(9*std::pow(x, 8) + 72*std::pow(x, 7)*y - 72*std::pow(x, 7) + 252*std::pow(x, 6)*std::pow(y, 2) - 504*std::pow(x, 6)*y + 252*std::pow(x, 6) + 504*std::pow(x, 5)*std::pow(y, 3) - 1512*std::pow(x, 5)*std::pow(y, 2) + 1512*std::pow(x, 5)*y - 504*std::pow(x, 5) + 630*std::pow(x, 4)*std::pow(y, 4) - 2520*std::pow(x, 4)*std::pow(y, 3) + 3780*std::pow(x, 4)*std::pow(y, 2) - 2520*std::pow(x, 4)*y + 630*std::pow(x, 4) + 504*std::pow(x, 3)*std::pow(y, 5) - 2520*std::pow(x, 3)*std::pow(y, 4) + 5040*std::pow(x, 3)*std::pow(y, 3) - 5040*std::pow(x, 3)*std::pow(y, 2) + 2520*std::pow(x, 3)*y - 504*std::pow(x, 3) + 252*std::pow(x, 2)*std::pow(y, 6) - 1512*std::pow(x, 2)*std::pow(y, 5) + 3780*std::pow(x, 2)*std::pow(y, 4) - 5040*std::pow(x, 2)*std::pow(y, 3) + 3780*std::pow(x, 2)*std::pow(y, 2) - 1512*std::pow(x, 2)*y + 252*std::pow(x, 2) + 72*x*std::pow(y, 7) - 504*x*std::pow(y, 6) + 1512*x*std::pow(y, 5) - 2520*x*std::pow(y, 4) + 2520*x*std::pow(y, 3) - 1512*x*std::pow(y, 2) + 504*x*y - 72*x + 9*std::pow(y, 8) - 72*std::pow(y, 7) + 252*std::pow(y, 6) - 504*std::pow(y, 5) + 630*std::pow(y, 4) - 504*std::pow(y, 3) + 252*std::pow(y, 2) - 72*y + 218790*std::pow(z, 8) + 411840*std::pow(z, 7)*(2*x + 2*y - 2) + 1261260*std::pow(z, 6)*std::pow(x + y - 1, 2) + 1009008*std::pow(z, 5)*std::pow(x + y - 1, 3) + 450450*std::pow(z, 4)*std::pow(x + y - 1, 4) + 110880*std::pow(z, 3)*std::pow(x + y - 1, 5) + 13860*std::pow(z, 2)*std::pow(x + y - 1, 6) + 720*z*std::pow(x + y - 1, 7) + 9) - 1,
            21*std::pow(x, 9) + 189*std::pow(x, 8)*y - 189*std::pow(x, 8) + 756*std::pow(x, 7)*std::pow(y, 2) - 1512*std::pow(x, 7)*y + 756*std::pow(x, 7) + 1764*std::pow(x, 6)*std::pow(y, 3) - 5292*std::pow(x, 6)*std::pow(y, 2) + 5292*std::pow(x, 6)*y - 1764*std::pow(x, 6) + 2646*std::pow(x, 5)*std::pow(y, 4) - 10584*std::pow(x, 5)*std::pow(y, 3) + 15876*std::pow(x, 5)*std::pow(y, 2) - 10584*std::pow(x, 5)*y + 2646*std::pow(x, 5) + 2646*std::pow(x, 4)*std::pow(y, 5) - 13230*std::pow(x, 4)*std::pow(y, 4) + 26460*std::pow(x, 4)*std::pow(y, 3) - 26460*std::pow(x, 4)*std::pow(y, 2) + 13230*std::pow(x, 4)*y - 2646*std::pow(x, 4) + 1764*std::pow(x, 3)*std::pow(y, 6) - 10584*std::pow(x, 3)*std::pow(y, 5) + 26460*std::pow(x, 3)*std::pow(y, 4) - 35280*std::pow(x, 3)*std::pow(y, 3) + 26460*std::pow(x, 3)*std::pow(y, 2) - 10584*std::pow(x, 3)*y + 1764*std::pow(x, 3) + 756*std::pow(x, 2)*std::pow(y, 7) - 5292*std::pow(x, 2)*std::pow(y, 6) + 15876*std::pow(x, 2)*std::pow(y, 5) - 26460*std::pow(x, 2)*std::pow(y, 4) + 26460*std::pow(x, 2)*std::pow(y, 3) - 15876*std::pow(x, 2)*std::pow(y, 2) + 5292*std::pow(x, 2)*y - 756*std::pow(x, 2) + 189*x*std::pow(y, 8) - 1512*x*std::pow(y, 7) + 5292*x*std::pow(y, 6) - 10584*x*std::pow(y, 5) + 13230*x*std::pow(y, 4) - 10584*x*std::pow(y, 3) + 5292*x*std::pow(y, 2) - 1512*x*y + 189*x + 21*std::pow(y, 9) - 189*std::pow(y, 8) + 756*std::pow(y, 7) - 1764*std::pow(y, 6) + 2646*std::pow(y, 5) - 2646*std::pow(y, 4) + 1764*std::pow(y, 3) - 756*std::pow(y, 2) + 189*y + 1021020*std::pow(z, 9) + 21*std::pow(z, 8)*(218790*x + 218790*y - 218790) + 8648640*std::pow(z, 7)*std::pow(x + y - 1, 2) + 8828820*std::pow(z, 6)*std::pow(x + y - 1, 3) + 5297292*std::pow(z, 5)*std::pow(x + y - 1, 4) + 1891890*std::pow(z, 4)*std::pow(x + y - 1, 5) + 388080*std::pow(z, 3)*std::pow(x + y - 1, 6) + 41580*std::pow(z, 2)*std::pow(x + y - 1, 7) + 1890*z*std::pow(x + y - 1, 8) + (x + 21*y - 1)*(9*std::pow(x, 8) + 72*std::pow(x, 7)*y - 72*std::pow(x, 7) + 252*std::pow(x, 6)*std::pow(y, 2) - 504*std::pow(x, 6)*y + 252*std::pow(x, 6) + 504*std::pow(x, 5)*std::pow(y, 3) - 1512*std::pow(x, 5)*std::pow(y, 2) + 1512*std::pow(x, 5)*y - 504*std::pow(x, 5) + 630*std::pow(x, 4)*std::pow(y, 4) - 2520*std::pow(x, 4)*std::pow(y, 3) + 3780*std::pow(x, 4)*std::pow(y, 2) - 2520*std::pow(x, 4)*y + 630*std::pow(x, 4) + 504*std::pow(x, 3)*std::pow(y, 5) - 2520*std::pow(x, 3)*std::pow(y, 4) + 5040*std::pow(x, 3)*std::pow(y, 3) - 5040*std::pow(x, 3)*std::pow(y, 2) + 2520*std::pow(x, 3)*y - 504*std::pow(x, 3) + 252*std::pow(x, 2)*std::pow(y, 6) - 1512*std::pow(x, 2)*std::pow(y, 5) + 3780*std::pow(x, 2)*std::pow(y, 4) - 5040*std::pow(x, 2)*std::pow(y, 3) + 3780*std::pow(x, 2)*std::pow(y, 2) - 1512*std::pow(x, 2)*y + 252*std::pow(x, 2) + 72*x*std::pow(y, 7) - 504*x*std::pow(y, 6) + 1512*x*std::pow(y, 5) - 2520*x*std::pow(y, 4) + 2520*x*std::pow(y, 3) - 1512*x*std::pow(y, 2) + 504*x*y - 72*x + 9*std::pow(y, 8) - 72*std::pow(y, 7) + 252*std::pow(y, 6) - 504*std::pow(y, 5) + 630*std::pow(y, 4) - 504*std::pow(y, 3) + 252*std::pow(y, 2) - 72*y + 218790*std::pow(z, 8) + 411840*std::pow(z, 7)*(2*x + 2*y - 2) + 1261260*std::pow(z, 6)*std::pow(x + y - 1, 2) + 1009008*std::pow(z, 5)*std::pow(x + y - 1, 3) + 450450*std::pow(z, 4)*std::pow(x + y - 1, 4) + 110880*std::pow(z, 3)*std::pow(x + y - 1, 5) + 13860*std::pow(z, 2)*std::pow(x + y - 1, 6) + 720*z*std::pow(x + y - 1, 7) + 9) - 21,
            (x + 21*y - 1)*(437580*std::pow(z, 8) + 8*std::pow(z, 7)*(218790*x + 218790*y - 218790) + 2882880*std::pow(z, 6)*std::pow(x + y - 1, 2) + 2522520*std::pow(z, 5)*std::pow(x + y - 1, 3) + 1261260*std::pow(z, 4)*std::pow(x + y - 1, 4) + 360360*std::pow(z, 3)*std::pow(x + y - 1, 5) + 55440*std::pow(z, 2)*std::pow(x + y - 1, 6) + 3960*z*std::pow(x + y - 1, 7) + 90*std::pow(x + y - 1, 8))
        };
    }
    static constexpr uInt Order = 10;
};

// Basis 285
template<>
struct DGBasis<285> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z) {
        return 184756*std::pow(z, 10) + std::pow(z, 9)*(923780*x + 923780*y - 923780) + 1969110*std::pow(z, 8)*std::pow(x + y - 1, 2) + 2333760*std::pow(z, 7)*std::pow(x + y - 1, 3) + 1681680*std::pow(z, 6)*std::pow(x + y - 1, 4) + 756756*std::pow(z, 5)*std::pow(x + y - 1, 5) + 210210*std::pow(z, 4)*std::pow(x + y - 1, 6) + 34320*std::pow(z, 3)*std::pow(x + y - 1, 7) + 2970*std::pow(z, 2)*std::pow(x + y - 1, 8) + 110*z*std::pow(x + y - 1, 9) + std::pow(x + y - 1, 10);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z) {
        return {
            923780*std::pow(z, 9) + 1969110*std::pow(z, 8)*(2*x + 2*y - 2) + 7001280*std::pow(z, 7)*std::pow(x + y - 1, 2) + 6726720*std::pow(z, 6)*std::pow(x + y - 1, 3) + 3783780*std::pow(z, 5)*std::pow(x + y - 1, 4) + 1261260*std::pow(z, 4)*std::pow(x + y - 1, 5) + 240240*std::pow(z, 3)*std::pow(x + y - 1, 6) + 23760*std::pow(z, 2)*std::pow(x + y - 1, 7) + 990*z*std::pow(x + y - 1, 8) + 10*std::pow(x + y - 1, 9),
            923780*std::pow(z, 9) + 1969110*std::pow(z, 8)*(2*x + 2*y - 2) + 7001280*std::pow(z, 7)*std::pow(x + y - 1, 2) + 6726720*std::pow(z, 6)*std::pow(x + y - 1, 3) + 3783780*std::pow(z, 5)*std::pow(x + y - 1, 4) + 1261260*std::pow(z, 4)*std::pow(x + y - 1, 5) + 240240*std::pow(z, 3)*std::pow(x + y - 1, 6) + 23760*std::pow(z, 2)*std::pow(x + y - 1, 7) + 990*z*std::pow(x + y - 1, 8) + 10*std::pow(x + y - 1, 9),
            1847560*std::pow(z, 9) + 9*std::pow(z, 8)*(923780*x + 923780*y - 923780) + 15752880*std::pow(z, 7)*std::pow(x + y - 1, 2) + 16336320*std::pow(z, 6)*std::pow(x + y - 1, 3) + 10090080*std::pow(z, 5)*std::pow(x + y - 1, 4) + 3783780*std::pow(z, 4)*std::pow(x + y - 1, 5) + 840840*std::pow(z, 3)*std::pow(x + y - 1, 6) + 102960*std::pow(z, 2)*std::pow(x + y - 1, 7) + 5940*z*std::pow(x + y - 1, 8) + 110*std::pow(x + y - 1, 9)
        };
    }
    static constexpr uInt Order = 10;
};