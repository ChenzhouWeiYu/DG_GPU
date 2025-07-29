
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
        return 10*std::pow(x, 2) - 8*x + 1;
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
        return std::pow(x, 2) + x*(6*y - 2) + 6*std::pow(y, 2) - 6*y + 1;
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
        return 35*std::pow(x, 3) - 45*std::pow(x, 2) + 15*x - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            105*std::pow(x, 2) - 90*x + 15,
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
        return (x + 2*y - 1)*(21*std::pow(x, 2) - 12*x + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            21*std::pow(x, 2) - 12*x + (42*x - 12)*(x + 2*y - 1) + 1,
            42*std::pow(x, 2) - 24*x + 2
        };
    }
    static constexpr uInt Order = 2;
};

// Basis 8
template<>
struct DGBasis2D<8> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (7*x - 1)*(std::pow(x, 2) + x*(6*y - 2) + 6*std::pow(y, 2) - 6*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            7*std::pow(x, 2) + 7*x*(6*y - 2) + 42*std::pow(y, 2) - 42*y + (7*x - 1)*(2*x + 6*y - 2) + 7,
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
        return std::pow(x, 3) + 3*std::pow(x, 2)*(4*y - 1) + 3*x*(10*std::pow(y, 2) - 8*y + 1) + 20*std::pow(y, 3) - 30*std::pow(y, 2) + 12*y - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            3*std::pow(x, 2) + 6*x*(4*y - 1) + 30*std::pow(y, 2) - 24*y + 3,
            12*std::pow(x, 2) + 3*x*(20*y - 8) + 60*std::pow(y, 2) - 60*y + 12
        };
    }
    static constexpr uInt Order = 2;
};

// Basis 10
template<>
struct DGBasis2D<10> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return 126*std::pow(x, 4) - 224*std::pow(x, 3) + 126*std::pow(x, 2) - 24*x + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            504*std::pow(x, 3) - 672*std::pow(x, 2) + 252*x - 24,
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
        return (x + 2*y - 1)*(84*std::pow(x, 3) - 84*std::pow(x, 2) + 21*x - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            84*std::pow(x, 3) - 84*std::pow(x, 2) + 21*x + (x + 2*y - 1)*(252*std::pow(x, 2) - 168*x + 21) - 1,
            168*std::pow(x, 3) - 168*std::pow(x, 2) + 42*x - 2
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 12
template<>
struct DGBasis2D<12> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (36*std::pow(x, 2) - 16*x + 1)*(std::pow(x, 2) + x*(6*y - 2) + 6*std::pow(y, 2) - 6*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (72*x - 16)*(std::pow(x, 2) + x*(6*y - 2) + 6*std::pow(y, 2) - 6*y + 1) + (2*x + 6*y - 2)*(36*std::pow(x, 2) - 16*x + 1),
            (6*x + 12*y - 6)*(36*std::pow(x, 2) - 16*x + 1)
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 13
template<>
struct DGBasis2D<13> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (9*x - 1)*(std::pow(x, 3) + 3*std::pow(x, 2)*(4*y - 1) + 3*x*(10*std::pow(y, 2) - 8*y + 1) + 20*std::pow(y, 3) - 30*std::pow(y, 2) + 12*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            9*std::pow(x, 3) + 27*std::pow(x, 2)*(4*y - 1) + 27*x*(10*std::pow(y, 2) - 8*y + 1) + 180*std::pow(y, 3) - 270*std::pow(y, 2) + 108*y + (9*x - 1)*(3*std::pow(x, 2) + 6*x*(4*y - 1) + 30*std::pow(y, 2) - 24*y + 3) - 9,
            (9*x - 1)*(12*std::pow(x, 2) + 3*x*(20*y - 8) + 60*std::pow(y, 2) - 60*y + 12)
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 14
template<>
struct DGBasis2D<14> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return std::pow(x, 4) + 4*std::pow(x, 3)*(5*y - 1) + std::pow(x, 2)*(90*std::pow(y, 2) - 60*y + 6) + 4*x*(35*std::pow(y, 3) - 45*std::pow(y, 2) + 15*y - 1) + 70*std::pow(y, 4) - 140*std::pow(y, 3) + 90*std::pow(y, 2) - 20*y + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            4*std::pow(x, 3) + 12*std::pow(x, 2)*(5*y - 1) + 2*x*(90*std::pow(y, 2) - 60*y + 6) + 140*std::pow(y, 3) - 180*std::pow(y, 2) + 60*y - 4,
            20*std::pow(x, 3) + std::pow(x, 2)*(180*y - 60) + 4*x*(105*std::pow(y, 2) - 90*y + 15) + 280*std::pow(y, 3) - 420*std::pow(y, 2) + 180*y - 20
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 15
template<>
struct DGBasis2D<15> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return 462*std::pow(x, 5) - 1050*std::pow(x, 4) + 840*std::pow(x, 3) - 280*std::pow(x, 2) + 35*x - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            2310*std::pow(x, 4) - 4200*std::pow(x, 3) + 2520*std::pow(x, 2) - 560*x + 35,
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
        return (x + 2*y - 1)*(330*std::pow(x, 4) - 480*std::pow(x, 3) + 216*std::pow(x, 2) - 32*x + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            330*std::pow(x, 4) - 480*std::pow(x, 3) + 216*std::pow(x, 2) - 32*x + (x + 2*y - 1)*(1320*std::pow(x, 3) - 1440*std::pow(x, 2) + 432*x - 32) + 1,
            660*std::pow(x, 4) - 960*std::pow(x, 3) + 432*std::pow(x, 2) - 64*x + 2
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 17
template<>
struct DGBasis2D<17> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (165*std::pow(x, 3) - 135*std::pow(x, 2) + 27*x - 1)*(std::pow(x, 2) + x*(6*y - 2) + 6*std::pow(y, 2) - 6*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (2*x + 6*y - 2)*(165*std::pow(x, 3) - 135*std::pow(x, 2) + 27*x - 1) + (495*std::pow(x, 2) - 270*x + 27)*(std::pow(x, 2) + x*(6*y - 2) + 6*std::pow(y, 2) - 6*y + 1),
            (6*x + 12*y - 6)*(165*std::pow(x, 3) - 135*std::pow(x, 2) + 27*x - 1)
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 18
template<>
struct DGBasis2D<18> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (55*std::pow(x, 2) - 20*x + 1)*(std::pow(x, 3) + 3*std::pow(x, 2)*(4*y - 1) + 3*x*(10*std::pow(y, 2) - 8*y + 1) + 20*std::pow(y, 3) - 30*std::pow(y, 2) + 12*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (110*x - 20)*(std::pow(x, 3) + 3*std::pow(x, 2)*(4*y - 1) + 3*x*(10*std::pow(y, 2) - 8*y + 1) + 20*std::pow(y, 3) - 30*std::pow(y, 2) + 12*y - 1) + (55*std::pow(x, 2) - 20*x + 1)*(3*std::pow(x, 2) + 6*x*(4*y - 1) + 30*std::pow(y, 2) - 24*y + 3),
            (55*std::pow(x, 2) - 20*x + 1)*(12*std::pow(x, 2) + 3*x*(20*y - 8) + 60*std::pow(y, 2) - 60*y + 12)
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 19
template<>
struct DGBasis2D<19> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (11*x - 1)*(std::pow(x, 4) + 4*std::pow(x, 3)*(5*y - 1) + std::pow(x, 2)*(90*std::pow(y, 2) - 60*y + 6) + 4*x*(35*std::pow(y, 3) - 45*std::pow(y, 2) + 15*y - 1) + 70*std::pow(y, 4) - 140*std::pow(y, 3) + 90*std::pow(y, 2) - 20*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            11*std::pow(x, 4) + 44*std::pow(x, 3)*(5*y - 1) + 11*std::pow(x, 2)*(90*std::pow(y, 2) - 60*y + 6) + 44*x*(35*std::pow(y, 3) - 45*std::pow(y, 2) + 15*y - 1) + 770*std::pow(y, 4) - 1540*std::pow(y, 3) + 990*std::pow(y, 2) - 220*y + (11*x - 1)*(4*std::pow(x, 3) + 12*std::pow(x, 2)*(5*y - 1) + 2*x*(90*std::pow(y, 2) - 60*y + 6) + 140*std::pow(y, 3) - 180*std::pow(y, 2) + 60*y - 4) + 11,
            (11*x - 1)*(20*std::pow(x, 3) + std::pow(x, 2)*(180*y - 60) + 4*x*(105*std::pow(y, 2) - 90*y + 15) + 280*std::pow(y, 3) - 420*std::pow(y, 2) + 180*y - 20)
        };
    }
    static constexpr uInt Order = 3;
};

// Basis 20
template<>
struct DGBasis2D<20> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return std::pow(x, 5) + 5*std::pow(x, 4)*(6*y - 1) + 10*std::pow(x, 3)*(21*std::pow(y, 2) - 12*y + 1) + 10*std::pow(x, 2)*(56*std::pow(y, 3) - 63*std::pow(y, 2) + 18*y - 1) + 5*x*(126*std::pow(y, 4) - 224*std::pow(y, 3) + 126*std::pow(y, 2) - 24*y + 1) + 252*std::pow(y, 5) - 630*std::pow(y, 4) + 560*std::pow(y, 3) - 210*std::pow(y, 2) + 30*y - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            5*std::pow(x, 4) + 20*std::pow(x, 3)*(6*y - 1) + 30*std::pow(x, 2)*(21*std::pow(y, 2) - 12*y + 1) + 20*x*(56*std::pow(y, 3) - 63*std::pow(y, 2) + 18*y - 1) + 630*std::pow(y, 4) - 1120*std::pow(y, 3) + 630*std::pow(y, 2) - 120*y + 5,
            30*std::pow(x, 4) + 10*std::pow(x, 3)*(42*y - 12) + 10*std::pow(x, 2)*(168*std::pow(y, 2) - 126*y + 18) + 5*x*(504*std::pow(y, 3) - 672*std::pow(y, 2) + 252*y - 24) + 1260*std::pow(y, 4) - 2520*std::pow(y, 3) + 1680*std::pow(y, 2) - 420*y + 30
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 21
template<>
struct DGBasis2D<21> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return 1716*std::pow(x, 6) - 4752*std::pow(x, 5) + 4950*std::pow(x, 4) - 2400*std::pow(x, 3) + 540*std::pow(x, 2) - 48*x + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            10296*std::pow(x, 5) - 23760*std::pow(x, 4) + 19800*std::pow(x, 3) - 7200*std::pow(x, 2) + 1080*x - 48,
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
        return (x + 2*y - 1)*(1287*std::pow(x, 5) - 2475*std::pow(x, 4) + 1650*std::pow(x, 3) - 450*std::pow(x, 2) + 45*x - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            1287*std::pow(x, 5) - 2475*std::pow(x, 4) + 1650*std::pow(x, 3) - 450*std::pow(x, 2) + 45*x + (x + 2*y - 1)*(6435*std::pow(x, 4) - 9900*std::pow(x, 3) + 4950*std::pow(x, 2) - 900*x + 45) - 1,
            2574*std::pow(x, 5) - 4950*std::pow(x, 4) + 3300*std::pow(x, 3) - 900*std::pow(x, 2) + 90*x - 2
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 23
template<>
struct DGBasis2D<23> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (std::pow(x, 2) + x*(6*y - 2) + 6*std::pow(y, 2) - 6*y + 1)*(715*std::pow(x, 4) - 880*std::pow(x, 3) + 330*std::pow(x, 2) - 40*x + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (2*x + 6*y - 2)*(715*std::pow(x, 4) - 880*std::pow(x, 3) + 330*std::pow(x, 2) - 40*x + 1) + (2860*std::pow(x, 3) - 2640*std::pow(x, 2) + 660*x - 40)*(std::pow(x, 2) + x*(6*y - 2) + 6*std::pow(y, 2) - 6*y + 1),
            (6*x + 12*y - 6)*(715*std::pow(x, 4) - 880*std::pow(x, 3) + 330*std::pow(x, 2) - 40*x + 1)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 24
template<>
struct DGBasis2D<24> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (286*std::pow(x, 3) - 198*std::pow(x, 2) + 33*x - 1)*(std::pow(x, 3) + 3*std::pow(x, 2)*(4*y - 1) + 3*x*(10*std::pow(y, 2) - 8*y + 1) + 20*std::pow(y, 3) - 30*std::pow(y, 2) + 12*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (858*std::pow(x, 2) - 396*x + 33)*(std::pow(x, 3) + 3*std::pow(x, 2)*(4*y - 1) + 3*x*(10*std::pow(y, 2) - 8*y + 1) + 20*std::pow(y, 3) - 30*std::pow(y, 2) + 12*y - 1) + (286*std::pow(x, 3) - 198*std::pow(x, 2) + 33*x - 1)*(3*std::pow(x, 2) + 6*x*(4*y - 1) + 30*std::pow(y, 2) - 24*y + 3),
            (286*std::pow(x, 3) - 198*std::pow(x, 2) + 33*x - 1)*(12*std::pow(x, 2) + 3*x*(20*y - 8) + 60*std::pow(y, 2) - 60*y + 12)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 25
template<>
struct DGBasis2D<25> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (78*std::pow(x, 2) - 24*x + 1)*(std::pow(x, 4) + 4*std::pow(x, 3)*(5*y - 1) + std::pow(x, 2)*(90*std::pow(y, 2) - 60*y + 6) + 4*x*(35*std::pow(y, 3) - 45*std::pow(y, 2) + 15*y - 1) + 70*std::pow(y, 4) - 140*std::pow(y, 3) + 90*std::pow(y, 2) - 20*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (156*x - 24)*(std::pow(x, 4) + 4*std::pow(x, 3)*(5*y - 1) + std::pow(x, 2)*(90*std::pow(y, 2) - 60*y + 6) + 4*x*(35*std::pow(y, 3) - 45*std::pow(y, 2) + 15*y - 1) + 70*std::pow(y, 4) - 140*std::pow(y, 3) + 90*std::pow(y, 2) - 20*y + 1) + (78*std::pow(x, 2) - 24*x + 1)*(4*std::pow(x, 3) + 12*std::pow(x, 2)*(5*y - 1) + 2*x*(90*std::pow(y, 2) - 60*y + 6) + 140*std::pow(y, 3) - 180*std::pow(y, 2) + 60*y - 4),
            (78*std::pow(x, 2) - 24*x + 1)*(20*std::pow(x, 3) + std::pow(x, 2)*(180*y - 60) + 4*x*(105*std::pow(y, 2) - 90*y + 15) + 280*std::pow(y, 3) - 420*std::pow(y, 2) + 180*y - 20)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 26
template<>
struct DGBasis2D<26> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (13*x - 1)*(std::pow(x, 5) + 5*std::pow(x, 4)*(6*y - 1) + 10*std::pow(x, 3)*(21*std::pow(y, 2) - 12*y + 1) + 10*std::pow(x, 2)*(56*std::pow(y, 3) - 63*std::pow(y, 2) + 18*y - 1) + 5*x*(126*std::pow(y, 4) - 224*std::pow(y, 3) + 126*std::pow(y, 2) - 24*y + 1) + 252*std::pow(y, 5) - 630*std::pow(y, 4) + 560*std::pow(y, 3) - 210*std::pow(y, 2) + 30*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            13*std::pow(x, 5) + 65*std::pow(x, 4)*(6*y - 1) + 130*std::pow(x, 3)*(21*std::pow(y, 2) - 12*y + 1) + 130*std::pow(x, 2)*(56*std::pow(y, 3) - 63*std::pow(y, 2) + 18*y - 1) + 65*x*(126*std::pow(y, 4) - 224*std::pow(y, 3) + 126*std::pow(y, 2) - 24*y + 1) + 3276*std::pow(y, 5) - 8190*std::pow(y, 4) + 7280*std::pow(y, 3) - 2730*std::pow(y, 2) + 390*y + (13*x - 1)*(5*std::pow(x, 4) + 20*std::pow(x, 3)*(6*y - 1) + 30*std::pow(x, 2)*(21*std::pow(y, 2) - 12*y + 1) + 20*x*(56*std::pow(y, 3) - 63*std::pow(y, 2) + 18*y - 1) + 630*std::pow(y, 4) - 1120*std::pow(y, 3) + 630*std::pow(y, 2) - 120*y + 5) - 13,
            (13*x - 1)*(30*std::pow(x, 4) + 10*std::pow(x, 3)*(42*y - 12) + 10*std::pow(x, 2)*(168*std::pow(y, 2) - 126*y + 18) + 5*x*(504*std::pow(y, 3) - 672*std::pow(y, 2) + 252*y - 24) + 1260*std::pow(y, 4) - 2520*std::pow(y, 3) + 1680*std::pow(y, 2) - 420*y + 30)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 27
template<>
struct DGBasis2D<27> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return std::pow(x, 6) + 6*std::pow(x, 5)*(7*y - 1) + 15*std::pow(x, 4)*(28*std::pow(y, 2) - 14*y + 1) + 20*std::pow(x, 3)*(84*std::pow(y, 3) - 84*std::pow(y, 2) + 21*y - 1) + 15*std::pow(x, 2)*(210*std::pow(y, 4) - 336*std::pow(y, 3) + 168*std::pow(y, 2) - 28*y + 1) + 6*x*(462*std::pow(y, 5) - 1050*std::pow(y, 4) + 840*std::pow(y, 3) - 280*std::pow(y, 2) + 35*y - 1) + 924*std::pow(y, 6) - 2772*std::pow(y, 5) + 3150*std::pow(y, 4) - 1680*std::pow(y, 3) + 420*std::pow(y, 2) - 42*y + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            6*std::pow(x, 5) + 30*std::pow(x, 4)*(7*y - 1) + 60*std::pow(x, 3)*(28*std::pow(y, 2) - 14*y + 1) + 60*std::pow(x, 2)*(84*std::pow(y, 3) - 84*std::pow(y, 2) + 21*y - 1) + 30*x*(210*std::pow(y, 4) - 336*std::pow(y, 3) + 168*std::pow(y, 2) - 28*y + 1) + 2772*std::pow(y, 5) - 6300*std::pow(y, 4) + 5040*std::pow(y, 3) - 1680*std::pow(y, 2) + 210*y - 6,
            42*std::pow(x, 5) + 15*std::pow(x, 4)*(56*y - 14) + 20*std::pow(x, 3)*(252*std::pow(y, 2) - 168*y + 21) + 15*std::pow(x, 2)*(840*std::pow(y, 3) - 1008*std::pow(y, 2) + 336*y - 28) + 6*x*(2310*std::pow(y, 4) - 4200*std::pow(y, 3) + 2520*std::pow(y, 2) - 560*y + 35) + 5544*std::pow(y, 5) - 13860*std::pow(y, 4) + 12600*std::pow(y, 3) - 5040*std::pow(y, 2) + 840*y - 42
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 28
template<>
struct DGBasis2D<28> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return 6435*std::pow(x, 7) - 21021*std::pow(x, 6) + 27027*std::pow(x, 5) - 17325*std::pow(x, 4) + 5775*std::pow(x, 3) - 945*std::pow(x, 2) + 63*x - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            45045*std::pow(x, 6) - 126126*std::pow(x, 5) + 135135*std::pow(x, 4) - 69300*std::pow(x, 3) + 17325*std::pow(x, 2) - 1890*x + 63,
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
        return (x + 2*y - 1)*(5005*std::pow(x, 6) - 12012*std::pow(x, 5) + 10725*std::pow(x, 4) - 4400*std::pow(x, 3) + 825*std::pow(x, 2) - 60*x + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            5005*std::pow(x, 6) - 12012*std::pow(x, 5) + 10725*std::pow(x, 4) - 4400*std::pow(x, 3) + 825*std::pow(x, 2) - 60*x + (x + 2*y - 1)*(30030*std::pow(x, 5) - 60060*std::pow(x, 4) + 42900*std::pow(x, 3) - 13200*std::pow(x, 2) + 1650*x - 60) + 1,
            10010*std::pow(x, 6) - 24024*std::pow(x, 5) + 21450*std::pow(x, 4) - 8800*std::pow(x, 3) + 1650*std::pow(x, 2) - 120*x + 2
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 30
template<>
struct DGBasis2D<30> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (std::pow(x, 2) + x*(6*y - 2) + 6*std::pow(y, 2) - 6*y + 1)*(3003*std::pow(x, 5) - 5005*std::pow(x, 4) + 2860*std::pow(x, 3) - 660*std::pow(x, 2) + 55*x - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (2*x + 6*y - 2)*(3003*std::pow(x, 5) - 5005*std::pow(x, 4) + 2860*std::pow(x, 3) - 660*std::pow(x, 2) + 55*x - 1) + (std::pow(x, 2) + x*(6*y - 2) + 6*std::pow(y, 2) - 6*y + 1)*(15015*std::pow(x, 4) - 20020*std::pow(x, 3) + 8580*std::pow(x, 2) - 1320*x + 55),
            (6*x + 12*y - 6)*(3003*std::pow(x, 5) - 5005*std::pow(x, 4) + 2860*std::pow(x, 3) - 660*std::pow(x, 2) + 55*x - 1)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 31
template<>
struct DGBasis2D<31> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (1365*std::pow(x, 4) - 1456*std::pow(x, 3) + 468*std::pow(x, 2) - 48*x + 1)*(std::pow(x, 3) + 3*std::pow(x, 2)*(4*y - 1) + 3*x*(10*std::pow(y, 2) - 8*y + 1) + 20*std::pow(y, 3) - 30*std::pow(y, 2) + 12*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (5460*std::pow(x, 3) - 4368*std::pow(x, 2) + 936*x - 48)*(std::pow(x, 3) + 3*std::pow(x, 2)*(4*y - 1) + 3*x*(10*std::pow(y, 2) - 8*y + 1) + 20*std::pow(y, 3) - 30*std::pow(y, 2) + 12*y - 1) + (3*std::pow(x, 2) + 6*x*(4*y - 1) + 30*std::pow(y, 2) - 24*y + 3)*(1365*std::pow(x, 4) - 1456*std::pow(x, 3) + 468*std::pow(x, 2) - 48*x + 1),
            (12*std::pow(x, 2) + 3*x*(20*y - 8) + 60*std::pow(y, 2) - 60*y + 12)*(1365*std::pow(x, 4) - 1456*std::pow(x, 3) + 468*std::pow(x, 2) - 48*x + 1)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 32
template<>
struct DGBasis2D<32> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (455*std::pow(x, 3) - 273*std::pow(x, 2) + 39*x - 1)*(std::pow(x, 4) + 4*std::pow(x, 3)*(5*y - 1) + std::pow(x, 2)*(90*std::pow(y, 2) - 60*y + 6) + 4*x*(35*std::pow(y, 3) - 45*std::pow(y, 2) + 15*y - 1) + 70*std::pow(y, 4) - 140*std::pow(y, 3) + 90*std::pow(y, 2) - 20*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (1365*std::pow(x, 2) - 546*x + 39)*(std::pow(x, 4) + 4*std::pow(x, 3)*(5*y - 1) + std::pow(x, 2)*(90*std::pow(y, 2) - 60*y + 6) + 4*x*(35*std::pow(y, 3) - 45*std::pow(y, 2) + 15*y - 1) + 70*std::pow(y, 4) - 140*std::pow(y, 3) + 90*std::pow(y, 2) - 20*y + 1) + (455*std::pow(x, 3) - 273*std::pow(x, 2) + 39*x - 1)*(4*std::pow(x, 3) + 12*std::pow(x, 2)*(5*y - 1) + 2*x*(90*std::pow(y, 2) - 60*y + 6) + 140*std::pow(y, 3) - 180*std::pow(y, 2) + 60*y - 4),
            (455*std::pow(x, 3) - 273*std::pow(x, 2) + 39*x - 1)*(20*std::pow(x, 3) + std::pow(x, 2)*(180*y - 60) + 4*x*(105*std::pow(y, 2) - 90*y + 15) + 280*std::pow(y, 3) - 420*std::pow(y, 2) + 180*y - 20)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 33
template<>
struct DGBasis2D<33> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (105*std::pow(x, 2) - 28*x + 1)*(std::pow(x, 5) + 5*std::pow(x, 4)*(6*y - 1) + 10*std::pow(x, 3)*(21*std::pow(y, 2) - 12*y + 1) + 10*std::pow(x, 2)*(56*std::pow(y, 3) - 63*std::pow(y, 2) + 18*y - 1) + 5*x*(126*std::pow(y, 4) - 224*std::pow(y, 3) + 126*std::pow(y, 2) - 24*y + 1) + 252*std::pow(y, 5) - 630*std::pow(y, 4) + 560*std::pow(y, 3) - 210*std::pow(y, 2) + 30*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (210*x - 28)*(std::pow(x, 5) + 5*std::pow(x, 4)*(6*y - 1) + 10*std::pow(x, 3)*(21*std::pow(y, 2) - 12*y + 1) + 10*std::pow(x, 2)*(56*std::pow(y, 3) - 63*std::pow(y, 2) + 18*y - 1) + 5*x*(126*std::pow(y, 4) - 224*std::pow(y, 3) + 126*std::pow(y, 2) - 24*y + 1) + 252*std::pow(y, 5) - 630*std::pow(y, 4) + 560*std::pow(y, 3) - 210*std::pow(y, 2) + 30*y - 1) + (105*std::pow(x, 2) - 28*x + 1)*(5*std::pow(x, 4) + 20*std::pow(x, 3)*(6*y - 1) + 30*std::pow(x, 2)*(21*std::pow(y, 2) - 12*y + 1) + 20*x*(56*std::pow(y, 3) - 63*std::pow(y, 2) + 18*y - 1) + 630*std::pow(y, 4) - 1120*std::pow(y, 3) + 630*std::pow(y, 2) - 120*y + 5),
            (105*std::pow(x, 2) - 28*x + 1)*(30*std::pow(x, 4) + 10*std::pow(x, 3)*(42*y - 12) + 10*std::pow(x, 2)*(168*std::pow(y, 2) - 126*y + 18) + 5*x*(504*std::pow(y, 3) - 672*std::pow(y, 2) + 252*y - 24) + 1260*std::pow(y, 4) - 2520*std::pow(y, 3) + 1680*std::pow(y, 2) - 420*y + 30)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 34
template<>
struct DGBasis2D<34> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (15*x - 1)*(std::pow(x, 6) + 6*std::pow(x, 5)*(7*y - 1) + 15*std::pow(x, 4)*(28*std::pow(y, 2) - 14*y + 1) + 20*std::pow(x, 3)*(84*std::pow(y, 3) - 84*std::pow(y, 2) + 21*y - 1) + 15*std::pow(x, 2)*(210*std::pow(y, 4) - 336*std::pow(y, 3) + 168*std::pow(y, 2) - 28*y + 1) + 6*x*(462*std::pow(y, 5) - 1050*std::pow(y, 4) + 840*std::pow(y, 3) - 280*std::pow(y, 2) + 35*y - 1) + 924*std::pow(y, 6) - 2772*std::pow(y, 5) + 3150*std::pow(y, 4) - 1680*std::pow(y, 3) + 420*std::pow(y, 2) - 42*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            15*std::pow(x, 6) + 90*std::pow(x, 5)*(7*y - 1) + 225*std::pow(x, 4)*(28*std::pow(y, 2) - 14*y + 1) + 300*std::pow(x, 3)*(84*std::pow(y, 3) - 84*std::pow(y, 2) + 21*y - 1) + 225*std::pow(x, 2)*(210*std::pow(y, 4) - 336*std::pow(y, 3) + 168*std::pow(y, 2) - 28*y + 1) + 90*x*(462*std::pow(y, 5) - 1050*std::pow(y, 4) + 840*std::pow(y, 3) - 280*std::pow(y, 2) + 35*y - 1) + 13860*std::pow(y, 6) - 41580*std::pow(y, 5) + 47250*std::pow(y, 4) - 25200*std::pow(y, 3) + 6300*std::pow(y, 2) - 630*y + (15*x - 1)*(6*std::pow(x, 5) + 30*std::pow(x, 4)*(7*y - 1) + 60*std::pow(x, 3)*(28*std::pow(y, 2) - 14*y + 1) + 60*std::pow(x, 2)*(84*std::pow(y, 3) - 84*std::pow(y, 2) + 21*y - 1) + 30*x*(210*std::pow(y, 4) - 336*std::pow(y, 3) + 168*std::pow(y, 2) - 28*y + 1) + 2772*std::pow(y, 5) - 6300*std::pow(y, 4) + 5040*std::pow(y, 3) - 1680*std::pow(y, 2) + 210*y - 6) + 15,
            (15*x - 1)*(42*std::pow(x, 5) + 15*std::pow(x, 4)*(56*y - 14) + 20*std::pow(x, 3)*(252*std::pow(y, 2) - 168*y + 21) + 15*std::pow(x, 2)*(840*std::pow(y, 3) - 1008*std::pow(y, 2) + 336*y - 28) + 6*x*(2310*std::pow(y, 4) - 4200*std::pow(y, 3) + 2520*std::pow(y, 2) - 560*y + 35) + 5544*std::pow(y, 5) - 13860*std::pow(y, 4) + 12600*std::pow(y, 3) - 5040*std::pow(y, 2) + 840*y - 42)
        };
    }
    static constexpr uInt Order = 4;
};

// Basis 35
template<>
struct DGBasis2D<35> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return std::pow(x, 7) + 7*std::pow(x, 6)*(8*y - 1) + 21*std::pow(x, 5)*(36*std::pow(y, 2) - 16*y + 1) + 35*std::pow(x, 4)*(120*std::pow(y, 3) - 108*std::pow(y, 2) + 24*y - 1) + 35*std::pow(x, 3)*(330*std::pow(y, 4) - 480*std::pow(y, 3) + 216*std::pow(y, 2) - 32*y + 1) + 21*std::pow(x, 2)*(792*std::pow(y, 5) - 1650*std::pow(y, 4) + 1200*std::pow(y, 3) - 360*std::pow(y, 2) + 40*y - 1) + 7*x*(1716*std::pow(y, 6) - 4752*std::pow(y, 5) + 4950*std::pow(y, 4) - 2400*std::pow(y, 3) + 540*std::pow(y, 2) - 48*y + 1) + 3432*std::pow(y, 7) - 12012*std::pow(y, 6) + 16632*std::pow(y, 5) - 11550*std::pow(y, 4) + 4200*std::pow(y, 3) - 756*std::pow(y, 2) + 56*y - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            7*std::pow(x, 6) + 42*std::pow(x, 5)*(8*y - 1) + 105*std::pow(x, 4)*(36*std::pow(y, 2) - 16*y + 1) + 140*std::pow(x, 3)*(120*std::pow(y, 3) - 108*std::pow(y, 2) + 24*y - 1) + 105*std::pow(x, 2)*(330*std::pow(y, 4) - 480*std::pow(y, 3) + 216*std::pow(y, 2) - 32*y + 1) + 42*x*(792*std::pow(y, 5) - 1650*std::pow(y, 4) + 1200*std::pow(y, 3) - 360*std::pow(y, 2) + 40*y - 1) + 12012*std::pow(y, 6) - 33264*std::pow(y, 5) + 34650*std::pow(y, 4) - 16800*std::pow(y, 3) + 3780*std::pow(y, 2) - 336*y + 7,
            56*std::pow(x, 6) + 21*std::pow(x, 5)*(72*y - 16) + 35*std::pow(x, 4)*(360*std::pow(y, 2) - 216*y + 24) + 35*std::pow(x, 3)*(1320*std::pow(y, 3) - 1440*std::pow(y, 2) + 432*y - 32) + 21*std::pow(x, 2)*(3960*std::pow(y, 4) - 6600*std::pow(y, 3) + 3600*std::pow(y, 2) - 720*y + 40) + 7*x*(10296*std::pow(y, 5) - 23760*std::pow(y, 4) + 19800*std::pow(y, 3) - 7200*std::pow(y, 2) + 1080*y - 48) + 24024*std::pow(y, 6) - 72072*std::pow(y, 5) + 83160*std::pow(y, 4) - 46200*std::pow(y, 3) + 12600*std::pow(y, 2) - 1512*y + 56
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 36
template<>
struct DGBasis2D<36> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return 24310*std::pow(x, 8) - 91520*std::pow(x, 7) + 140140*std::pow(x, 6) - 112112*std::pow(x, 5) + 50050*std::pow(x, 4) - 12320*std::pow(x, 3) + 1540*std::pow(x, 2) - 80*x + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            194480*std::pow(x, 7) - 640640*std::pow(x, 6) + 840840*std::pow(x, 5) - 560560*std::pow(x, 4) + 200200*std::pow(x, 3) - 36960*std::pow(x, 2) + 3080*x - 80,
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
        return (x + 2*y - 1)*(19448*std::pow(x, 7) - 56056*std::pow(x, 6) + 63063*std::pow(x, 5) - 35035*std::pow(x, 4) + 10010*std::pow(x, 3) - 1386*std::pow(x, 2) + 77*x - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            19448*std::pow(x, 7) - 56056*std::pow(x, 6) + 63063*std::pow(x, 5) - 35035*std::pow(x, 4) + 10010*std::pow(x, 3) - 1386*std::pow(x, 2) + 77*x + (x + 2*y - 1)*(136136*std::pow(x, 6) - 336336*std::pow(x, 5) + 315315*std::pow(x, 4) - 140140*std::pow(x, 3) + 30030*std::pow(x, 2) - 2772*x + 77) - 1,
            38896*std::pow(x, 7) - 112112*std::pow(x, 6) + 126126*std::pow(x, 5) - 70070*std::pow(x, 4) + 20020*std::pow(x, 3) - 2772*std::pow(x, 2) + 154*x - 2
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 38
template<>
struct DGBasis2D<38> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (std::pow(x, 2) + x*(6*y - 2) + 6*std::pow(y, 2) - 6*y + 1)*(12376*std::pow(x, 6) - 26208*std::pow(x, 5) + 20475*std::pow(x, 4) - 7280*std::pow(x, 3) + 1170*std::pow(x, 2) - 72*x + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (2*x + 6*y - 2)*(12376*std::pow(x, 6) - 26208*std::pow(x, 5) + 20475*std::pow(x, 4) - 7280*std::pow(x, 3) + 1170*std::pow(x, 2) - 72*x + 1) + (std::pow(x, 2) + x*(6*y - 2) + 6*std::pow(y, 2) - 6*y + 1)*(74256*std::pow(x, 5) - 131040*std::pow(x, 4) + 81900*std::pow(x, 3) - 21840*std::pow(x, 2) + 2340*x - 72),
            (6*x + 12*y - 6)*(12376*std::pow(x, 6) - 26208*std::pow(x, 5) + 20475*std::pow(x, 4) - 7280*std::pow(x, 3) + 1170*std::pow(x, 2) - 72*x + 1)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 39
template<>
struct DGBasis2D<39> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (6188*std::pow(x, 5) - 9100*std::pow(x, 4) + 4550*std::pow(x, 3) - 910*std::pow(x, 2) + 65*x - 1)*(std::pow(x, 3) + 3*std::pow(x, 2)*(4*y - 1) + 3*x*(10*std::pow(y, 2) - 8*y + 1) + 20*std::pow(y, 3) - 30*std::pow(y, 2) + 12*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (3*std::pow(x, 2) + 6*x*(4*y - 1) + 30*std::pow(y, 2) - 24*y + 3)*(6188*std::pow(x, 5) - 9100*std::pow(x, 4) + 4550*std::pow(x, 3) - 910*std::pow(x, 2) + 65*x - 1) + (30940*std::pow(x, 4) - 36400*std::pow(x, 3) + 13650*std::pow(x, 2) - 1820*x + 65)*(std::pow(x, 3) + 3*std::pow(x, 2)*(4*y - 1) + 3*x*(10*std::pow(y, 2) - 8*y + 1) + 20*std::pow(y, 3) - 30*std::pow(y, 2) + 12*y - 1),
            (12*std::pow(x, 2) + 3*x*(20*y - 8) + 60*std::pow(y, 2) - 60*y + 12)*(6188*std::pow(x, 5) - 9100*std::pow(x, 4) + 4550*std::pow(x, 3) - 910*std::pow(x, 2) + 65*x - 1)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 40
template<>
struct DGBasis2D<40> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (2380*std::pow(x, 4) - 2240*std::pow(x, 3) + 630*std::pow(x, 2) - 56*x + 1)*(std::pow(x, 4) + 4*std::pow(x, 3)*(5*y - 1) + std::pow(x, 2)*(90*std::pow(y, 2) - 60*y + 6) + 4*x*(35*std::pow(y, 3) - 45*std::pow(y, 2) + 15*y - 1) + 70*std::pow(y, 4) - 140*std::pow(y, 3) + 90*std::pow(y, 2) - 20*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (9520*std::pow(x, 3) - 6720*std::pow(x, 2) + 1260*x - 56)*(std::pow(x, 4) + 4*std::pow(x, 3)*(5*y - 1) + std::pow(x, 2)*(90*std::pow(y, 2) - 60*y + 6) + 4*x*(35*std::pow(y, 3) - 45*std::pow(y, 2) + 15*y - 1) + 70*std::pow(y, 4) - 140*std::pow(y, 3) + 90*std::pow(y, 2) - 20*y + 1) + (2380*std::pow(x, 4) - 2240*std::pow(x, 3) + 630*std::pow(x, 2) - 56*x + 1)*(4*std::pow(x, 3) + 12*std::pow(x, 2)*(5*y - 1) + 2*x*(90*std::pow(y, 2) - 60*y + 6) + 140*std::pow(y, 3) - 180*std::pow(y, 2) + 60*y - 4),
            (2380*std::pow(x, 4) - 2240*std::pow(x, 3) + 630*std::pow(x, 2) - 56*x + 1)*(20*std::pow(x, 3) + std::pow(x, 2)*(180*y - 60) + 4*x*(105*std::pow(y, 2) - 90*y + 15) + 280*std::pow(y, 3) - 420*std::pow(y, 2) + 180*y - 20)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 41
template<>
struct DGBasis2D<41> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (680*std::pow(x, 3) - 360*std::pow(x, 2) + 45*x - 1)*(std::pow(x, 5) + 5*std::pow(x, 4)*(6*y - 1) + 10*std::pow(x, 3)*(21*std::pow(y, 2) - 12*y + 1) + 10*std::pow(x, 2)*(56*std::pow(y, 3) - 63*std::pow(y, 2) + 18*y - 1) + 5*x*(126*std::pow(y, 4) - 224*std::pow(y, 3) + 126*std::pow(y, 2) - 24*y + 1) + 252*std::pow(y, 5) - 630*std::pow(y, 4) + 560*std::pow(y, 3) - 210*std::pow(y, 2) + 30*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (2040*std::pow(x, 2) - 720*x + 45)*(std::pow(x, 5) + 5*std::pow(x, 4)*(6*y - 1) + 10*std::pow(x, 3)*(21*std::pow(y, 2) - 12*y + 1) + 10*std::pow(x, 2)*(56*std::pow(y, 3) - 63*std::pow(y, 2) + 18*y - 1) + 5*x*(126*std::pow(y, 4) - 224*std::pow(y, 3) + 126*std::pow(y, 2) - 24*y + 1) + 252*std::pow(y, 5) - 630*std::pow(y, 4) + 560*std::pow(y, 3) - 210*std::pow(y, 2) + 30*y - 1) + (680*std::pow(x, 3) - 360*std::pow(x, 2) + 45*x - 1)*(5*std::pow(x, 4) + 20*std::pow(x, 3)*(6*y - 1) + 30*std::pow(x, 2)*(21*std::pow(y, 2) - 12*y + 1) + 20*x*(56*std::pow(y, 3) - 63*std::pow(y, 2) + 18*y - 1) + 630*std::pow(y, 4) - 1120*std::pow(y, 3) + 630*std::pow(y, 2) - 120*y + 5),
            (680*std::pow(x, 3) - 360*std::pow(x, 2) + 45*x - 1)*(30*std::pow(x, 4) + 10*std::pow(x, 3)*(42*y - 12) + 10*std::pow(x, 2)*(168*std::pow(y, 2) - 126*y + 18) + 5*x*(504*std::pow(y, 3) - 672*std::pow(y, 2) + 252*y - 24) + 1260*std::pow(y, 4) - 2520*std::pow(y, 3) + 1680*std::pow(y, 2) - 420*y + 30)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 42
template<>
struct DGBasis2D<42> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (136*std::pow(x, 2) - 32*x + 1)*(std::pow(x, 6) + 6*std::pow(x, 5)*(7*y - 1) + 15*std::pow(x, 4)*(28*std::pow(y, 2) - 14*y + 1) + 20*std::pow(x, 3)*(84*std::pow(y, 3) - 84*std::pow(y, 2) + 21*y - 1) + 15*std::pow(x, 2)*(210*std::pow(y, 4) - 336*std::pow(y, 3) + 168*std::pow(y, 2) - 28*y + 1) + 6*x*(462*std::pow(y, 5) - 1050*std::pow(y, 4) + 840*std::pow(y, 3) - 280*std::pow(y, 2) + 35*y - 1) + 924*std::pow(y, 6) - 2772*std::pow(y, 5) + 3150*std::pow(y, 4) - 1680*std::pow(y, 3) + 420*std::pow(y, 2) - 42*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (272*x - 32)*(std::pow(x, 6) + 6*std::pow(x, 5)*(7*y - 1) + 15*std::pow(x, 4)*(28*std::pow(y, 2) - 14*y + 1) + 20*std::pow(x, 3)*(84*std::pow(y, 3) - 84*std::pow(y, 2) + 21*y - 1) + 15*std::pow(x, 2)*(210*std::pow(y, 4) - 336*std::pow(y, 3) + 168*std::pow(y, 2) - 28*y + 1) + 6*x*(462*std::pow(y, 5) - 1050*std::pow(y, 4) + 840*std::pow(y, 3) - 280*std::pow(y, 2) + 35*y - 1) + 924*std::pow(y, 6) - 2772*std::pow(y, 5) + 3150*std::pow(y, 4) - 1680*std::pow(y, 3) + 420*std::pow(y, 2) - 42*y + 1) + (136*std::pow(x, 2) - 32*x + 1)*(6*std::pow(x, 5) + 30*std::pow(x, 4)*(7*y - 1) + 60*std::pow(x, 3)*(28*std::pow(y, 2) - 14*y + 1) + 60*std::pow(x, 2)*(84*std::pow(y, 3) - 84*std::pow(y, 2) + 21*y - 1) + 30*x*(210*std::pow(y, 4) - 336*std::pow(y, 3) + 168*std::pow(y, 2) - 28*y + 1) + 2772*std::pow(y, 5) - 6300*std::pow(y, 4) + 5040*std::pow(y, 3) - 1680*std::pow(y, 2) + 210*y - 6),
            (136*std::pow(x, 2) - 32*x + 1)*(42*std::pow(x, 5) + 15*std::pow(x, 4)*(56*y - 14) + 20*std::pow(x, 3)*(252*std::pow(y, 2) - 168*y + 21) + 15*std::pow(x, 2)*(840*std::pow(y, 3) - 1008*std::pow(y, 2) + 336*y - 28) + 6*x*(2310*std::pow(y, 4) - 4200*std::pow(y, 3) + 2520*std::pow(y, 2) - 560*y + 35) + 5544*std::pow(y, 5) - 13860*std::pow(y, 4) + 12600*std::pow(y, 3) - 5040*std::pow(y, 2) + 840*y - 42)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 43
template<>
struct DGBasis2D<43> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (17*x - 1)*(std::pow(x, 7) + 7*std::pow(x, 6)*(8*y - 1) + 21*std::pow(x, 5)*(36*std::pow(y, 2) - 16*y + 1) + 35*std::pow(x, 4)*(120*std::pow(y, 3) - 108*std::pow(y, 2) + 24*y - 1) + 35*std::pow(x, 3)*(330*std::pow(y, 4) - 480*std::pow(y, 3) + 216*std::pow(y, 2) - 32*y + 1) + 21*std::pow(x, 2)*(792*std::pow(y, 5) - 1650*std::pow(y, 4) + 1200*std::pow(y, 3) - 360*std::pow(y, 2) + 40*y - 1) + 7*x*(1716*std::pow(y, 6) - 4752*std::pow(y, 5) + 4950*std::pow(y, 4) - 2400*std::pow(y, 3) + 540*std::pow(y, 2) - 48*y + 1) + 3432*std::pow(y, 7) - 12012*std::pow(y, 6) + 16632*std::pow(y, 5) - 11550*std::pow(y, 4) + 4200*std::pow(y, 3) - 756*std::pow(y, 2) + 56*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            17*std::pow(x, 7) + 119*std::pow(x, 6)*(8*y - 1) + 357*std::pow(x, 5)*(36*std::pow(y, 2) - 16*y + 1) + 595*std::pow(x, 4)*(120*std::pow(y, 3) - 108*std::pow(y, 2) + 24*y - 1) + 595*std::pow(x, 3)*(330*std::pow(y, 4) - 480*std::pow(y, 3) + 216*std::pow(y, 2) - 32*y + 1) + 357*std::pow(x, 2)*(792*std::pow(y, 5) - 1650*std::pow(y, 4) + 1200*std::pow(y, 3) - 360*std::pow(y, 2) + 40*y - 1) + 119*x*(1716*std::pow(y, 6) - 4752*std::pow(y, 5) + 4950*std::pow(y, 4) - 2400*std::pow(y, 3) + 540*std::pow(y, 2) - 48*y + 1) + 58344*std::pow(y, 7) - 204204*std::pow(y, 6) + 282744*std::pow(y, 5) - 196350*std::pow(y, 4) + 71400*std::pow(y, 3) - 12852*std::pow(y, 2) + 952*y + (17*x - 1)*(7*std::pow(x, 6) + 42*std::pow(x, 5)*(8*y - 1) + 105*std::pow(x, 4)*(36*std::pow(y, 2) - 16*y + 1) + 140*std::pow(x, 3)*(120*std::pow(y, 3) - 108*std::pow(y, 2) + 24*y - 1) + 105*std::pow(x, 2)*(330*std::pow(y, 4) - 480*std::pow(y, 3) + 216*std::pow(y, 2) - 32*y + 1) + 42*x*(792*std::pow(y, 5) - 1650*std::pow(y, 4) + 1200*std::pow(y, 3) - 360*std::pow(y, 2) + 40*y - 1) + 12012*std::pow(y, 6) - 33264*std::pow(y, 5) + 34650*std::pow(y, 4) - 16800*std::pow(y, 3) + 3780*std::pow(y, 2) - 336*y + 7) - 17,
            (17*x - 1)*(56*std::pow(x, 6) + 21*std::pow(x, 5)*(72*y - 16) + 35*std::pow(x, 4)*(360*std::pow(y, 2) - 216*y + 24) + 35*std::pow(x, 3)*(1320*std::pow(y, 3) - 1440*std::pow(y, 2) + 432*y - 32) + 21*std::pow(x, 2)*(3960*std::pow(y, 4) - 6600*std::pow(y, 3) + 3600*std::pow(y, 2) - 720*y + 40) + 7*x*(10296*std::pow(y, 5) - 23760*std::pow(y, 4) + 19800*std::pow(y, 3) - 7200*std::pow(y, 2) + 1080*y - 48) + 24024*std::pow(y, 6) - 72072*std::pow(y, 5) + 83160*std::pow(y, 4) - 46200*std::pow(y, 3) + 12600*std::pow(y, 2) - 1512*y + 56)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 44
template<>
struct DGBasis2D<44> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return std::pow(x, 8) + 8*std::pow(x, 7)*(9*y - 1) + 28*std::pow(x, 6)*(45*std::pow(y, 2) - 18*y + 1) + 56*std::pow(x, 5)*(165*std::pow(y, 3) - 135*std::pow(y, 2) + 27*y - 1) + 70*std::pow(x, 4)*(495*std::pow(y, 4) - 660*std::pow(y, 3) + 270*std::pow(y, 2) - 36*y + 1) + 56*std::pow(x, 3)*(1287*std::pow(y, 5) - 2475*std::pow(y, 4) + 1650*std::pow(y, 3) - 450*std::pow(y, 2) + 45*y - 1) + 28*std::pow(x, 2)*(3003*std::pow(y, 6) - 7722*std::pow(y, 5) + 7425*std::pow(y, 4) - 3300*std::pow(y, 3) + 675*std::pow(y, 2) - 54*y + 1) + 8*x*(6435*std::pow(y, 7) - 21021*std::pow(y, 6) + 27027*std::pow(y, 5) - 17325*std::pow(y, 4) + 5775*std::pow(y, 3) - 945*std::pow(y, 2) + 63*y - 1) + 12870*std::pow(y, 8) - 51480*std::pow(y, 7) + 84084*std::pow(y, 6) - 72072*std::pow(y, 5) + 34650*std::pow(y, 4) - 9240*std::pow(y, 3) + 1260*std::pow(y, 2) - 72*y + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            8*std::pow(x, 7) + 56*std::pow(x, 6)*(9*y - 1) + 168*std::pow(x, 5)*(45*std::pow(y, 2) - 18*y + 1) + 280*std::pow(x, 4)*(165*std::pow(y, 3) - 135*std::pow(y, 2) + 27*y - 1) + 280*std::pow(x, 3)*(495*std::pow(y, 4) - 660*std::pow(y, 3) + 270*std::pow(y, 2) - 36*y + 1) + 168*std::pow(x, 2)*(1287*std::pow(y, 5) - 2475*std::pow(y, 4) + 1650*std::pow(y, 3) - 450*std::pow(y, 2) + 45*y - 1) + 56*x*(3003*std::pow(y, 6) - 7722*std::pow(y, 5) + 7425*std::pow(y, 4) - 3300*std::pow(y, 3) + 675*std::pow(y, 2) - 54*y + 1) + 51480*std::pow(y, 7) - 168168*std::pow(y, 6) + 216216*std::pow(y, 5) - 138600*std::pow(y, 4) + 46200*std::pow(y, 3) - 7560*std::pow(y, 2) + 504*y - 8,
            72*std::pow(x, 7) + 28*std::pow(x, 6)*(90*y - 18) + 56*std::pow(x, 5)*(495*std::pow(y, 2) - 270*y + 27) + 70*std::pow(x, 4)*(1980*std::pow(y, 3) - 1980*std::pow(y, 2) + 540*y - 36) + 56*std::pow(x, 3)*(6435*std::pow(y, 4) - 9900*std::pow(y, 3) + 4950*std::pow(y, 2) - 900*y + 45) + 28*std::pow(x, 2)*(18018*std::pow(y, 5) - 38610*std::pow(y, 4) + 29700*std::pow(y, 3) - 9900*std::pow(y, 2) + 1350*y - 54) + 8*x*(45045*std::pow(y, 6) - 126126*std::pow(y, 5) + 135135*std::pow(y, 4) - 69300*std::pow(y, 3) + 17325*std::pow(y, 2) - 1890*y + 63) + 102960*std::pow(y, 7) - 360360*std::pow(y, 6) + 504504*std::pow(y, 5) - 360360*std::pow(y, 4) + 138600*std::pow(y, 3) - 27720*std::pow(y, 2) + 2520*y - 72
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 45
template<>
struct DGBasis2D<45> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return 92378*std::pow(x, 9) - 393822*std::pow(x, 8) + 700128*std::pow(x, 7) - 672672*std::pow(x, 6) + 378378*std::pow(x, 5) - 126126*std::pow(x, 4) + 24024*std::pow(x, 3) - 2376*std::pow(x, 2) + 99*x - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            831402*std::pow(x, 8) - 3150576*std::pow(x, 7) + 4900896*std::pow(x, 6) - 4036032*std::pow(x, 5) + 1891890*std::pow(x, 4) - 504504*std::pow(x, 3) + 72072*std::pow(x, 2) - 4752*x + 99,
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
        return (x + 2*y - 1)*(75582*std::pow(x, 8) - 254592*std::pow(x, 7) + 346528*std::pow(x, 6) - 244608*std::pow(x, 5) + 95550*std::pow(x, 4) - 20384*std::pow(x, 3) + 2184*std::pow(x, 2) - 96*x + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            75582*std::pow(x, 8) - 254592*std::pow(x, 7) + 346528*std::pow(x, 6) - 244608*std::pow(x, 5) + 95550*std::pow(x, 4) - 20384*std::pow(x, 3) + 2184*std::pow(x, 2) - 96*x + (x + 2*y - 1)*(604656*std::pow(x, 7) - 1782144*std::pow(x, 6) + 2079168*std::pow(x, 5) - 1223040*std::pow(x, 4) + 382200*std::pow(x, 3) - 61152*std::pow(x, 2) + 4368*x - 96) + 1,
            151164*std::pow(x, 8) - 509184*std::pow(x, 7) + 693056*std::pow(x, 6) - 489216*std::pow(x, 5) + 191100*std::pow(x, 4) - 40768*std::pow(x, 3) + 4368*std::pow(x, 2) - 192*x + 2
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 47
template<>
struct DGBasis2D<47> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (std::pow(x, 2) + x*(6*y - 2) + 6*std::pow(y, 2) - 6*y + 1)*(50388*std::pow(x, 7) - 129948*std::pow(x, 6) + 129948*std::pow(x, 5) - 63700*std::pow(x, 4) + 15925*std::pow(x, 3) - 1911*std::pow(x, 2) + 91*x - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (2*x + 6*y - 2)*(50388*std::pow(x, 7) - 129948*std::pow(x, 6) + 129948*std::pow(x, 5) - 63700*std::pow(x, 4) + 15925*std::pow(x, 3) - 1911*std::pow(x, 2) + 91*x - 1) + (std::pow(x, 2) + x*(6*y - 2) + 6*std::pow(y, 2) - 6*y + 1)*(352716*std::pow(x, 6) - 779688*std::pow(x, 5) + 649740*std::pow(x, 4) - 254800*std::pow(x, 3) + 47775*std::pow(x, 2) - 3822*x + 91),
            (6*x + 12*y - 6)*(50388*std::pow(x, 7) - 129948*std::pow(x, 6) + 129948*std::pow(x, 5) - 63700*std::pow(x, 4) + 15925*std::pow(x, 3) - 1911*std::pow(x, 2) + 91*x - 1)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 48
template<>
struct DGBasis2D<48> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (std::pow(x, 3) + 3*std::pow(x, 2)*(4*y - 1) + 3*x*(10*std::pow(y, 2) - 8*y + 1) + 20*std::pow(y, 3) - 30*std::pow(y, 2) + 12*y - 1)*(27132*std::pow(x, 6) - 51408*std::pow(x, 5) + 35700*std::pow(x, 4) - 11200*std::pow(x, 3) + 1575*std::pow(x, 2) - 84*x + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (3*std::pow(x, 2) + 6*x*(4*y - 1) + 30*std::pow(y, 2) - 24*y + 3)*(27132*std::pow(x, 6) - 51408*std::pow(x, 5) + 35700*std::pow(x, 4) - 11200*std::pow(x, 3) + 1575*std::pow(x, 2) - 84*x + 1) + (162792*std::pow(x, 5) - 257040*std::pow(x, 4) + 142800*std::pow(x, 3) - 33600*std::pow(x, 2) + 3150*x - 84)*(std::pow(x, 3) + 3*std::pow(x, 2)*(4*y - 1) + 3*x*(10*std::pow(y, 2) - 8*y + 1) + 20*std::pow(y, 3) - 30*std::pow(y, 2) + 12*y - 1),
            (12*std::pow(x, 2) + 3*x*(20*y - 8) + 60*std::pow(y, 2) - 60*y + 12)*(27132*std::pow(x, 6) - 51408*std::pow(x, 5) + 35700*std::pow(x, 4) - 11200*std::pow(x, 3) + 1575*std::pow(x, 2) - 84*x + 1)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 49
template<>
struct DGBasis2D<49> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (11628*std::pow(x, 5) - 15300*std::pow(x, 4) + 6800*std::pow(x, 3) - 1200*std::pow(x, 2) + 75*x - 1)*(std::pow(x, 4) + 4*std::pow(x, 3)*(5*y - 1) + std::pow(x, 2)*(90*std::pow(y, 2) - 60*y + 6) + 4*x*(35*std::pow(y, 3) - 45*std::pow(y, 2) + 15*y - 1) + 70*std::pow(y, 4) - 140*std::pow(y, 3) + 90*std::pow(y, 2) - 20*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (58140*std::pow(x, 4) - 61200*std::pow(x, 3) + 20400*std::pow(x, 2) - 2400*x + 75)*(std::pow(x, 4) + 4*std::pow(x, 3)*(5*y - 1) + std::pow(x, 2)*(90*std::pow(y, 2) - 60*y + 6) + 4*x*(35*std::pow(y, 3) - 45*std::pow(y, 2) + 15*y - 1) + 70*std::pow(y, 4) - 140*std::pow(y, 3) + 90*std::pow(y, 2) - 20*y + 1) + (11628*std::pow(x, 5) - 15300*std::pow(x, 4) + 6800*std::pow(x, 3) - 1200*std::pow(x, 2) + 75*x - 1)*(4*std::pow(x, 3) + 12*std::pow(x, 2)*(5*y - 1) + 2*x*(90*std::pow(y, 2) - 60*y + 6) + 140*std::pow(y, 3) - 180*std::pow(y, 2) + 60*y - 4),
            (11628*std::pow(x, 5) - 15300*std::pow(x, 4) + 6800*std::pow(x, 3) - 1200*std::pow(x, 2) + 75*x - 1)*(20*std::pow(x, 3) + std::pow(x, 2)*(180*y - 60) + 4*x*(105*std::pow(y, 2) - 90*y + 15) + 280*std::pow(y, 3) - 420*std::pow(y, 2) + 180*y - 20)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 50
template<>
struct DGBasis2D<50> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (3876*std::pow(x, 4) - 3264*std::pow(x, 3) + 816*std::pow(x, 2) - 64*x + 1)*(std::pow(x, 5) + 5*std::pow(x, 4)*(6*y - 1) + 10*std::pow(x, 3)*(21*std::pow(y, 2) - 12*y + 1) + 10*std::pow(x, 2)*(56*std::pow(y, 3) - 63*std::pow(y, 2) + 18*y - 1) + 5*x*(126*std::pow(y, 4) - 224*std::pow(y, 3) + 126*std::pow(y, 2) - 24*y + 1) + 252*std::pow(y, 5) - 630*std::pow(y, 4) + 560*std::pow(y, 3) - 210*std::pow(y, 2) + 30*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (15504*std::pow(x, 3) - 9792*std::pow(x, 2) + 1632*x - 64)*(std::pow(x, 5) + 5*std::pow(x, 4)*(6*y - 1) + 10*std::pow(x, 3)*(21*std::pow(y, 2) - 12*y + 1) + 10*std::pow(x, 2)*(56*std::pow(y, 3) - 63*std::pow(y, 2) + 18*y - 1) + 5*x*(126*std::pow(y, 4) - 224*std::pow(y, 3) + 126*std::pow(y, 2) - 24*y + 1) + 252*std::pow(y, 5) - 630*std::pow(y, 4) + 560*std::pow(y, 3) - 210*std::pow(y, 2) + 30*y - 1) + (3876*std::pow(x, 4) - 3264*std::pow(x, 3) + 816*std::pow(x, 2) - 64*x + 1)*(5*std::pow(x, 4) + 20*std::pow(x, 3)*(6*y - 1) + 30*std::pow(x, 2)*(21*std::pow(y, 2) - 12*y + 1) + 20*x*(56*std::pow(y, 3) - 63*std::pow(y, 2) + 18*y - 1) + 630*std::pow(y, 4) - 1120*std::pow(y, 3) + 630*std::pow(y, 2) - 120*y + 5),
            (3876*std::pow(x, 4) - 3264*std::pow(x, 3) + 816*std::pow(x, 2) - 64*x + 1)*(30*std::pow(x, 4) + 10*std::pow(x, 3)*(42*y - 12) + 10*std::pow(x, 2)*(168*std::pow(y, 2) - 126*y + 18) + 5*x*(504*std::pow(y, 3) - 672*std::pow(y, 2) + 252*y - 24) + 1260*std::pow(y, 4) - 2520*std::pow(y, 3) + 1680*std::pow(y, 2) - 420*y + 30)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 51
template<>
struct DGBasis2D<51> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (969*std::pow(x, 3) - 459*std::pow(x, 2) + 51*x - 1)*(std::pow(x, 6) + 6*std::pow(x, 5)*(7*y - 1) + 15*std::pow(x, 4)*(28*std::pow(y, 2) - 14*y + 1) + 20*std::pow(x, 3)*(84*std::pow(y, 3) - 84*std::pow(y, 2) + 21*y - 1) + 15*std::pow(x, 2)*(210*std::pow(y, 4) - 336*std::pow(y, 3) + 168*std::pow(y, 2) - 28*y + 1) + 6*x*(462*std::pow(y, 5) - 1050*std::pow(y, 4) + 840*std::pow(y, 3) - 280*std::pow(y, 2) + 35*y - 1) + 924*std::pow(y, 6) - 2772*std::pow(y, 5) + 3150*std::pow(y, 4) - 1680*std::pow(y, 3) + 420*std::pow(y, 2) - 42*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (2907*std::pow(x, 2) - 918*x + 51)*(std::pow(x, 6) + 6*std::pow(x, 5)*(7*y - 1) + 15*std::pow(x, 4)*(28*std::pow(y, 2) - 14*y + 1) + 20*std::pow(x, 3)*(84*std::pow(y, 3) - 84*std::pow(y, 2) + 21*y - 1) + 15*std::pow(x, 2)*(210*std::pow(y, 4) - 336*std::pow(y, 3) + 168*std::pow(y, 2) - 28*y + 1) + 6*x*(462*std::pow(y, 5) - 1050*std::pow(y, 4) + 840*std::pow(y, 3) - 280*std::pow(y, 2) + 35*y - 1) + 924*std::pow(y, 6) - 2772*std::pow(y, 5) + 3150*std::pow(y, 4) - 1680*std::pow(y, 3) + 420*std::pow(y, 2) - 42*y + 1) + (969*std::pow(x, 3) - 459*std::pow(x, 2) + 51*x - 1)*(6*std::pow(x, 5) + 30*std::pow(x, 4)*(7*y - 1) + 60*std::pow(x, 3)*(28*std::pow(y, 2) - 14*y + 1) + 60*std::pow(x, 2)*(84*std::pow(y, 3) - 84*std::pow(y, 2) + 21*y - 1) + 30*x*(210*std::pow(y, 4) - 336*std::pow(y, 3) + 168*std::pow(y, 2) - 28*y + 1) + 2772*std::pow(y, 5) - 6300*std::pow(y, 4) + 5040*std::pow(y, 3) - 1680*std::pow(y, 2) + 210*y - 6),
            (969*std::pow(x, 3) - 459*std::pow(x, 2) + 51*x - 1)*(42*std::pow(x, 5) + 15*std::pow(x, 4)*(56*y - 14) + 20*std::pow(x, 3)*(252*std::pow(y, 2) - 168*y + 21) + 15*std::pow(x, 2)*(840*std::pow(y, 3) - 1008*std::pow(y, 2) + 336*y - 28) + 6*x*(2310*std::pow(y, 4) - 4200*std::pow(y, 3) + 2520*std::pow(y, 2) - 560*y + 35) + 5544*std::pow(y, 5) - 13860*std::pow(y, 4) + 12600*std::pow(y, 3) - 5040*std::pow(y, 2) + 840*y - 42)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 52
template<>
struct DGBasis2D<52> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (171*std::pow(x, 2) - 36*x + 1)*(std::pow(x, 7) + 7*std::pow(x, 6)*(8*y - 1) + 21*std::pow(x, 5)*(36*std::pow(y, 2) - 16*y + 1) + 35*std::pow(x, 4)*(120*std::pow(y, 3) - 108*std::pow(y, 2) + 24*y - 1) + 35*std::pow(x, 3)*(330*std::pow(y, 4) - 480*std::pow(y, 3) + 216*std::pow(y, 2) - 32*y + 1) + 21*std::pow(x, 2)*(792*std::pow(y, 5) - 1650*std::pow(y, 4) + 1200*std::pow(y, 3) - 360*std::pow(y, 2) + 40*y - 1) + 7*x*(1716*std::pow(y, 6) - 4752*std::pow(y, 5) + 4950*std::pow(y, 4) - 2400*std::pow(y, 3) + 540*std::pow(y, 2) - 48*y + 1) + 3432*std::pow(y, 7) - 12012*std::pow(y, 6) + 16632*std::pow(y, 5) - 11550*std::pow(y, 4) + 4200*std::pow(y, 3) - 756*std::pow(y, 2) + 56*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (342*x - 36)*(std::pow(x, 7) + 7*std::pow(x, 6)*(8*y - 1) + 21*std::pow(x, 5)*(36*std::pow(y, 2) - 16*y + 1) + 35*std::pow(x, 4)*(120*std::pow(y, 3) - 108*std::pow(y, 2) + 24*y - 1) + 35*std::pow(x, 3)*(330*std::pow(y, 4) - 480*std::pow(y, 3) + 216*std::pow(y, 2) - 32*y + 1) + 21*std::pow(x, 2)*(792*std::pow(y, 5) - 1650*std::pow(y, 4) + 1200*std::pow(y, 3) - 360*std::pow(y, 2) + 40*y - 1) + 7*x*(1716*std::pow(y, 6) - 4752*std::pow(y, 5) + 4950*std::pow(y, 4) - 2400*std::pow(y, 3) + 540*std::pow(y, 2) - 48*y + 1) + 3432*std::pow(y, 7) - 12012*std::pow(y, 6) + 16632*std::pow(y, 5) - 11550*std::pow(y, 4) + 4200*std::pow(y, 3) - 756*std::pow(y, 2) + 56*y - 1) + (171*std::pow(x, 2) - 36*x + 1)*(7*std::pow(x, 6) + 42*std::pow(x, 5)*(8*y - 1) + 105*std::pow(x, 4)*(36*std::pow(y, 2) - 16*y + 1) + 140*std::pow(x, 3)*(120*std::pow(y, 3) - 108*std::pow(y, 2) + 24*y - 1) + 105*std::pow(x, 2)*(330*std::pow(y, 4) - 480*std::pow(y, 3) + 216*std::pow(y, 2) - 32*y + 1) + 42*x*(792*std::pow(y, 5) - 1650*std::pow(y, 4) + 1200*std::pow(y, 3) - 360*std::pow(y, 2) + 40*y - 1) + 12012*std::pow(y, 6) - 33264*std::pow(y, 5) + 34650*std::pow(y, 4) - 16800*std::pow(y, 3) + 3780*std::pow(y, 2) - 336*y + 7),
            (171*std::pow(x, 2) - 36*x + 1)*(56*std::pow(x, 6) + 21*std::pow(x, 5)*(72*y - 16) + 35*std::pow(x, 4)*(360*std::pow(y, 2) - 216*y + 24) + 35*std::pow(x, 3)*(1320*std::pow(y, 3) - 1440*std::pow(y, 2) + 432*y - 32) + 21*std::pow(x, 2)*(3960*std::pow(y, 4) - 6600*std::pow(y, 3) + 3600*std::pow(y, 2) - 720*y + 40) + 7*x*(10296*std::pow(y, 5) - 23760*std::pow(y, 4) + 19800*std::pow(y, 3) - 7200*std::pow(y, 2) + 1080*y - 48) + 24024*std::pow(y, 6) - 72072*std::pow(y, 5) + 83160*std::pow(y, 4) - 46200*std::pow(y, 3) + 12600*std::pow(y, 2) - 1512*y + 56)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 53
template<>
struct DGBasis2D<53> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (19*x - 1)*(std::pow(x, 8) + 8*std::pow(x, 7)*(9*y - 1) + 28*std::pow(x, 6)*(45*std::pow(y, 2) - 18*y + 1) + 56*std::pow(x, 5)*(165*std::pow(y, 3) - 135*std::pow(y, 2) + 27*y - 1) + 70*std::pow(x, 4)*(495*std::pow(y, 4) - 660*std::pow(y, 3) + 270*std::pow(y, 2) - 36*y + 1) + 56*std::pow(x, 3)*(1287*std::pow(y, 5) - 2475*std::pow(y, 4) + 1650*std::pow(y, 3) - 450*std::pow(y, 2) + 45*y - 1) + 28*std::pow(x, 2)*(3003*std::pow(y, 6) - 7722*std::pow(y, 5) + 7425*std::pow(y, 4) - 3300*std::pow(y, 3) + 675*std::pow(y, 2) - 54*y + 1) + 8*x*(6435*std::pow(y, 7) - 21021*std::pow(y, 6) + 27027*std::pow(y, 5) - 17325*std::pow(y, 4) + 5775*std::pow(y, 3) - 945*std::pow(y, 2) + 63*y - 1) + 12870*std::pow(y, 8) - 51480*std::pow(y, 7) + 84084*std::pow(y, 6) - 72072*std::pow(y, 5) + 34650*std::pow(y, 4) - 9240*std::pow(y, 3) + 1260*std::pow(y, 2) - 72*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            19*std::pow(x, 8) + 152*std::pow(x, 7)*(9*y - 1) + 532*std::pow(x, 6)*(45*std::pow(y, 2) - 18*y + 1) + 1064*std::pow(x, 5)*(165*std::pow(y, 3) - 135*std::pow(y, 2) + 27*y - 1) + 1330*std::pow(x, 4)*(495*std::pow(y, 4) - 660*std::pow(y, 3) + 270*std::pow(y, 2) - 36*y + 1) + 1064*std::pow(x, 3)*(1287*std::pow(y, 5) - 2475*std::pow(y, 4) + 1650*std::pow(y, 3) - 450*std::pow(y, 2) + 45*y - 1) + 532*std::pow(x, 2)*(3003*std::pow(y, 6) - 7722*std::pow(y, 5) + 7425*std::pow(y, 4) - 3300*std::pow(y, 3) + 675*std::pow(y, 2) - 54*y + 1) + 152*x*(6435*std::pow(y, 7) - 21021*std::pow(y, 6) + 27027*std::pow(y, 5) - 17325*std::pow(y, 4) + 5775*std::pow(y, 3) - 945*std::pow(y, 2) + 63*y - 1) + 244530*std::pow(y, 8) - 978120*std::pow(y, 7) + 1597596*std::pow(y, 6) - 1369368*std::pow(y, 5) + 658350*std::pow(y, 4) - 175560*std::pow(y, 3) + 23940*std::pow(y, 2) - 1368*y + (19*x - 1)*(8*std::pow(x, 7) + 56*std::pow(x, 6)*(9*y - 1) + 168*std::pow(x, 5)*(45*std::pow(y, 2) - 18*y + 1) + 280*std::pow(x, 4)*(165*std::pow(y, 3) - 135*std::pow(y, 2) + 27*y - 1) + 280*std::pow(x, 3)*(495*std::pow(y, 4) - 660*std::pow(y, 3) + 270*std::pow(y, 2) - 36*y + 1) + 168*std::pow(x, 2)*(1287*std::pow(y, 5) - 2475*std::pow(y, 4) + 1650*std::pow(y, 3) - 450*std::pow(y, 2) + 45*y - 1) + 56*x*(3003*std::pow(y, 6) - 7722*std::pow(y, 5) + 7425*std::pow(y, 4) - 3300*std::pow(y, 3) + 675*std::pow(y, 2) - 54*y + 1) + 51480*std::pow(y, 7) - 168168*std::pow(y, 6) + 216216*std::pow(y, 5) - 138600*std::pow(y, 4) + 46200*std::pow(y, 3) - 7560*std::pow(y, 2) + 504*y - 8) + 19,
            (19*x - 1)*(72*std::pow(x, 7) + 28*std::pow(x, 6)*(90*y - 18) + 56*std::pow(x, 5)*(495*std::pow(y, 2) - 270*y + 27) + 70*std::pow(x, 4)*(1980*std::pow(y, 3) - 1980*std::pow(y, 2) + 540*y - 36) + 56*std::pow(x, 3)*(6435*std::pow(y, 4) - 9900*std::pow(y, 3) + 4950*std::pow(y, 2) - 900*y + 45) + 28*std::pow(x, 2)*(18018*std::pow(y, 5) - 38610*std::pow(y, 4) + 29700*std::pow(y, 3) - 9900*std::pow(y, 2) + 1350*y - 54) + 8*x*(45045*std::pow(y, 6) - 126126*std::pow(y, 5) + 135135*std::pow(y, 4) - 69300*std::pow(y, 3) + 17325*std::pow(y, 2) - 1890*y + 63) + 102960*std::pow(y, 7) - 360360*std::pow(y, 6) + 504504*std::pow(y, 5) - 360360*std::pow(y, 4) + 138600*std::pow(y, 3) - 27720*std::pow(y, 2) + 2520*y - 72)
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 54
template<>
struct DGBasis2D<54> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return std::pow(x, 9) + std::pow(x, 8)*(90*y - 9) + 36*std::pow(x, 7)*(55*std::pow(y, 2) - 20*y + 1) + 84*std::pow(x, 6)*(220*std::pow(y, 3) - 165*std::pow(y, 2) + 30*y - 1) + 126*std::pow(x, 5)*(715*std::pow(y, 4) - 880*std::pow(y, 3) + 330*std::pow(y, 2) - 40*y + 1) + 126*std::pow(x, 4)*(2002*std::pow(y, 5) - 3575*std::pow(y, 4) + 2200*std::pow(y, 3) - 550*std::pow(y, 2) + 50*y - 1) + 84*std::pow(x, 3)*(5005*std::pow(y, 6) - 12012*std::pow(y, 5) + 10725*std::pow(y, 4) - 4400*std::pow(y, 3) + 825*std::pow(y, 2) - 60*y + 1) + 36*std::pow(x, 2)*(11440*std::pow(y, 7) - 35035*std::pow(y, 6) + 42042*std::pow(y, 5) - 25025*std::pow(y, 4) + 7700*std::pow(y, 3) - 1155*std::pow(y, 2) + 70*y - 1) + 9*x*(24310*std::pow(y, 8) - 91520*std::pow(y, 7) + 140140*std::pow(y, 6) - 112112*std::pow(y, 5) + 50050*std::pow(y, 4) - 12320*std::pow(y, 3) + 1540*std::pow(y, 2) - 80*y + 1) + 48620*std::pow(y, 9) - 218790*std::pow(y, 8) + 411840*std::pow(y, 7) - 420420*std::pow(y, 6) + 252252*std::pow(y, 5) - 90090*std::pow(y, 4) + 18480*std::pow(y, 3) - 1980*std::pow(y, 2) + 90*y - 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            9*std::pow(x, 8) + 8*std::pow(x, 7)*(90*y - 9) + 252*std::pow(x, 6)*(55*std::pow(y, 2) - 20*y + 1) + 504*std::pow(x, 5)*(220*std::pow(y, 3) - 165*std::pow(y, 2) + 30*y - 1) + 630*std::pow(x, 4)*(715*std::pow(y, 4) - 880*std::pow(y, 3) + 330*std::pow(y, 2) - 40*y + 1) + 504*std::pow(x, 3)*(2002*std::pow(y, 5) - 3575*std::pow(y, 4) + 2200*std::pow(y, 3) - 550*std::pow(y, 2) + 50*y - 1) + 252*std::pow(x, 2)*(5005*std::pow(y, 6) - 12012*std::pow(y, 5) + 10725*std::pow(y, 4) - 4400*std::pow(y, 3) + 825*std::pow(y, 2) - 60*y + 1) + 72*x*(11440*std::pow(y, 7) - 35035*std::pow(y, 6) + 42042*std::pow(y, 5) - 25025*std::pow(y, 4) + 7700*std::pow(y, 3) - 1155*std::pow(y, 2) + 70*y - 1) + 218790*std::pow(y, 8) - 823680*std::pow(y, 7) + 1261260*std::pow(y, 6) - 1009008*std::pow(y, 5) + 450450*std::pow(y, 4) - 110880*std::pow(y, 3) + 13860*std::pow(y, 2) - 720*y + 9,
            90*std::pow(x, 8) + 36*std::pow(x, 7)*(110*y - 20) + 84*std::pow(x, 6)*(660*std::pow(y, 2) - 330*y + 30) + 126*std::pow(x, 5)*(2860*std::pow(y, 3) - 2640*std::pow(y, 2) + 660*y - 40) + 126*std::pow(x, 4)*(10010*std::pow(y, 4) - 14300*std::pow(y, 3) + 6600*std::pow(y, 2) - 1100*y + 50) + 84*std::pow(x, 3)*(30030*std::pow(y, 5) - 60060*std::pow(y, 4) + 42900*std::pow(y, 3) - 13200*std::pow(y, 2) + 1650*y - 60) + 36*std::pow(x, 2)*(80080*std::pow(y, 6) - 210210*std::pow(y, 5) + 210210*std::pow(y, 4) - 100100*std::pow(y, 3) + 23100*std::pow(y, 2) - 2310*y + 70) + 9*x*(194480*std::pow(y, 7) - 640640*std::pow(y, 6) + 840840*std::pow(y, 5) - 560560*std::pow(y, 4) + 200200*std::pow(y, 3) - 36960*std::pow(y, 2) + 3080*y - 80) + 437580*std::pow(y, 8) - 1750320*std::pow(y, 7) + 2882880*std::pow(y, 6) - 2522520*std::pow(y, 5) + 1261260*std::pow(y, 4) - 360360*std::pow(y, 3) + 55440*std::pow(y, 2) - 3960*y + 90
        };
    }
    static constexpr uInt Order = 5;
};

// Basis 55
template<>
struct DGBasis2D<55> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return 352716*std::pow(x, 10) - 1679600*std::pow(x, 9) + 3401190*std::pow(x, 8) - 3818880*std::pow(x, 7) + 2598960*std::pow(x, 6) - 1100736*std::pow(x, 5) + 286650*std::pow(x, 4) - 43680*std::pow(x, 3) + 3510*std::pow(x, 2) - 120*x + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            3527160*std::pow(x, 9) - 15116400*std::pow(x, 8) + 27209520*std::pow(x, 7) - 26732160*std::pow(x, 6) + 15593760*std::pow(x, 5) - 5503680*std::pow(x, 4) + 1146600*std::pow(x, 3) - 131040*std::pow(x, 2) + 7020*x - 120,
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
        return (x + 2*y - 1)*(293930*std::pow(x, 9) - 1133730*std::pow(x, 8) + 1813968*std::pow(x, 7) - 1559376*std::pow(x, 6) + 779688*std::pow(x, 5) - 229320*std::pow(x, 4) + 38220*std::pow(x, 3) - 3276*std::pow(x, 2) + 117*x - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            293930*std::pow(x, 9) - 1133730*std::pow(x, 8) + 1813968*std::pow(x, 7) - 1559376*std::pow(x, 6) + 779688*std::pow(x, 5) - 229320*std::pow(x, 4) + 38220*std::pow(x, 3) - 3276*std::pow(x, 2) + 117*x + (x + 2*y - 1)*(2645370*std::pow(x, 8) - 9069840*std::pow(x, 7) + 12697776*std::pow(x, 6) - 9356256*std::pow(x, 5) + 3898440*std::pow(x, 4) - 917280*std::pow(x, 3) + 114660*std::pow(x, 2) - 6552*x + 117) - 1,
            587860*std::pow(x, 9) - 2267460*std::pow(x, 8) + 3627936*std::pow(x, 7) - 3118752*std::pow(x, 6) + 1559376*std::pow(x, 5) - 458640*std::pow(x, 4) + 76440*std::pow(x, 3) - 6552*std::pow(x, 2) + 234*x - 2
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 57
template<>
struct DGBasis2D<57> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (std::pow(x, 2) + x*(6*y - 2) + 6*std::pow(y, 2) - 6*y + 1)*(203490*std::pow(x, 8) - 620160*std::pow(x, 7) + 759696*std::pow(x, 6) - 479808*std::pow(x, 5) + 166600*std::pow(x, 4) - 31360*std::pow(x, 3) + 2940*std::pow(x, 2) - 112*x + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (2*x + 6*y - 2)*(203490*std::pow(x, 8) - 620160*std::pow(x, 7) + 759696*std::pow(x, 6) - 479808*std::pow(x, 5) + 166600*std::pow(x, 4) - 31360*std::pow(x, 3) + 2940*std::pow(x, 2) - 112*x + 1) + (std::pow(x, 2) + x*(6*y - 2) + 6*std::pow(y, 2) - 6*y + 1)*(1627920*std::pow(x, 7) - 4341120*std::pow(x, 6) + 4558176*std::pow(x, 5) - 2399040*std::pow(x, 4) + 666400*std::pow(x, 3) - 94080*std::pow(x, 2) + 5880*x - 112),
            (6*x + 12*y - 6)*(203490*std::pow(x, 8) - 620160*std::pow(x, 7) + 759696*std::pow(x, 6) - 479808*std::pow(x, 5) + 166600*std::pow(x, 4) - 31360*std::pow(x, 3) + 2940*std::pow(x, 2) - 112*x + 1)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 58
template<>
struct DGBasis2D<58> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (std::pow(x, 3) + 3*std::pow(x, 2)*(4*y - 1) + 3*x*(10*std::pow(y, 2) - 8*y + 1) + 20*std::pow(y, 3) - 30*std::pow(y, 2) + 12*y - 1)*(116280*std::pow(x, 7) - 271320*std::pow(x, 6) + 244188*std::pow(x, 5) - 107100*std::pow(x, 4) + 23800*std::pow(x, 3) - 2520*std::pow(x, 2) + 105*x - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (3*std::pow(x, 2) + 6*x*(4*y - 1) + 30*std::pow(y, 2) - 24*y + 3)*(116280*std::pow(x, 7) - 271320*std::pow(x, 6) + 244188*std::pow(x, 5) - 107100*std::pow(x, 4) + 23800*std::pow(x, 3) - 2520*std::pow(x, 2) + 105*x - 1) + (std::pow(x, 3) + 3*std::pow(x, 2)*(4*y - 1) + 3*x*(10*std::pow(y, 2) - 8*y + 1) + 20*std::pow(y, 3) - 30*std::pow(y, 2) + 12*y - 1)*(813960*std::pow(x, 6) - 1627920*std::pow(x, 5) + 1220940*std::pow(x, 4) - 428400*std::pow(x, 3) + 71400*std::pow(x, 2) - 5040*x + 105),
            (12*std::pow(x, 2) + 3*x*(20*y - 8) + 60*std::pow(y, 2) - 60*y + 12)*(116280*std::pow(x, 7) - 271320*std::pow(x, 6) + 244188*std::pow(x, 5) - 107100*std::pow(x, 4) + 23800*std::pow(x, 3) - 2520*std::pow(x, 2) + 105*x - 1)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 59
template<>
struct DGBasis2D<59> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (54264*std::pow(x, 6) - 93024*std::pow(x, 5) + 58140*std::pow(x, 4) - 16320*std::pow(x, 3) + 2040*std::pow(x, 2) - 96*x + 1)*(std::pow(x, 4) + 4*std::pow(x, 3)*(5*y - 1) + std::pow(x, 2)*(90*std::pow(y, 2) - 60*y + 6) + 4*x*(35*std::pow(y, 3) - 45*std::pow(y, 2) + 15*y - 1) + 70*std::pow(y, 4) - 140*std::pow(y, 3) + 90*std::pow(y, 2) - 20*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (325584*std::pow(x, 5) - 465120*std::pow(x, 4) + 232560*std::pow(x, 3) - 48960*std::pow(x, 2) + 4080*x - 96)*(std::pow(x, 4) + 4*std::pow(x, 3)*(5*y - 1) + std::pow(x, 2)*(90*std::pow(y, 2) - 60*y + 6) + 4*x*(35*std::pow(y, 3) - 45*std::pow(y, 2) + 15*y - 1) + 70*std::pow(y, 4) - 140*std::pow(y, 3) + 90*std::pow(y, 2) - 20*y + 1) + (4*std::pow(x, 3) + 12*std::pow(x, 2)*(5*y - 1) + 2*x*(90*std::pow(y, 2) - 60*y + 6) + 140*std::pow(y, 3) - 180*std::pow(y, 2) + 60*y - 4)*(54264*std::pow(x, 6) - 93024*std::pow(x, 5) + 58140*std::pow(x, 4) - 16320*std::pow(x, 3) + 2040*std::pow(x, 2) - 96*x + 1),
            (20*std::pow(x, 3) + std::pow(x, 2)*(180*y - 60) + 4*x*(105*std::pow(y, 2) - 90*y + 15) + 280*std::pow(y, 3) - 420*std::pow(y, 2) + 180*y - 20)*(54264*std::pow(x, 6) - 93024*std::pow(x, 5) + 58140*std::pow(x, 4) - 16320*std::pow(x, 3) + 2040*std::pow(x, 2) - 96*x + 1)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 60
template<>
struct DGBasis2D<60> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (20349*std::pow(x, 5) - 24225*std::pow(x, 4) + 9690*std::pow(x, 3) - 1530*std::pow(x, 2) + 85*x - 1)*(std::pow(x, 5) + 5*std::pow(x, 4)*(6*y - 1) + 10*std::pow(x, 3)*(21*std::pow(y, 2) - 12*y + 1) + 10*std::pow(x, 2)*(56*std::pow(y, 3) - 63*std::pow(y, 2) + 18*y - 1) + 5*x*(126*std::pow(y, 4) - 224*std::pow(y, 3) + 126*std::pow(y, 2) - 24*y + 1) + 252*std::pow(y, 5) - 630*std::pow(y, 4) + 560*std::pow(y, 3) - 210*std::pow(y, 2) + 30*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (101745*std::pow(x, 4) - 96900*std::pow(x, 3) + 29070*std::pow(x, 2) - 3060*x + 85)*(std::pow(x, 5) + 5*std::pow(x, 4)*(6*y - 1) + 10*std::pow(x, 3)*(21*std::pow(y, 2) - 12*y + 1) + 10*std::pow(x, 2)*(56*std::pow(y, 3) - 63*std::pow(y, 2) + 18*y - 1) + 5*x*(126*std::pow(y, 4) - 224*std::pow(y, 3) + 126*std::pow(y, 2) - 24*y + 1) + 252*std::pow(y, 5) - 630*std::pow(y, 4) + 560*std::pow(y, 3) - 210*std::pow(y, 2) + 30*y - 1) + (20349*std::pow(x, 5) - 24225*std::pow(x, 4) + 9690*std::pow(x, 3) - 1530*std::pow(x, 2) + 85*x - 1)*(5*std::pow(x, 4) + 20*std::pow(x, 3)*(6*y - 1) + 30*std::pow(x, 2)*(21*std::pow(y, 2) - 12*y + 1) + 20*x*(56*std::pow(y, 3) - 63*std::pow(y, 2) + 18*y - 1) + 630*std::pow(y, 4) - 1120*std::pow(y, 3) + 630*std::pow(y, 2) - 120*y + 5),
            (20349*std::pow(x, 5) - 24225*std::pow(x, 4) + 9690*std::pow(x, 3) - 1530*std::pow(x, 2) + 85*x - 1)*(30*std::pow(x, 4) + 10*std::pow(x, 3)*(42*y - 12) + 10*std::pow(x, 2)*(168*std::pow(y, 2) - 126*y + 18) + 5*x*(504*std::pow(y, 3) - 672*std::pow(y, 2) + 252*y - 24) + 1260*std::pow(y, 4) - 2520*std::pow(y, 3) + 1680*std::pow(y, 2) - 420*y + 30)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 61
template<>
struct DGBasis2D<61> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (5985*std::pow(x, 4) - 4560*std::pow(x, 3) + 1026*std::pow(x, 2) - 72*x + 1)*(std::pow(x, 6) + 6*std::pow(x, 5)*(7*y - 1) + 15*std::pow(x, 4)*(28*std::pow(y, 2) - 14*y + 1) + 20*std::pow(x, 3)*(84*std::pow(y, 3) - 84*std::pow(y, 2) + 21*y - 1) + 15*std::pow(x, 2)*(210*std::pow(y, 4) - 336*std::pow(y, 3) + 168*std::pow(y, 2) - 28*y + 1) + 6*x*(462*std::pow(y, 5) - 1050*std::pow(y, 4) + 840*std::pow(y, 3) - 280*std::pow(y, 2) + 35*y - 1) + 924*std::pow(y, 6) - 2772*std::pow(y, 5) + 3150*std::pow(y, 4) - 1680*std::pow(y, 3) + 420*std::pow(y, 2) - 42*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (23940*std::pow(x, 3) - 13680*std::pow(x, 2) + 2052*x - 72)*(std::pow(x, 6) + 6*std::pow(x, 5)*(7*y - 1) + 15*std::pow(x, 4)*(28*std::pow(y, 2) - 14*y + 1) + 20*std::pow(x, 3)*(84*std::pow(y, 3) - 84*std::pow(y, 2) + 21*y - 1) + 15*std::pow(x, 2)*(210*std::pow(y, 4) - 336*std::pow(y, 3) + 168*std::pow(y, 2) - 28*y + 1) + 6*x*(462*std::pow(y, 5) - 1050*std::pow(y, 4) + 840*std::pow(y, 3) - 280*std::pow(y, 2) + 35*y - 1) + 924*std::pow(y, 6) - 2772*std::pow(y, 5) + 3150*std::pow(y, 4) - 1680*std::pow(y, 3) + 420*std::pow(y, 2) - 42*y + 1) + (5985*std::pow(x, 4) - 4560*std::pow(x, 3) + 1026*std::pow(x, 2) - 72*x + 1)*(6*std::pow(x, 5) + 30*std::pow(x, 4)*(7*y - 1) + 60*std::pow(x, 3)*(28*std::pow(y, 2) - 14*y + 1) + 60*std::pow(x, 2)*(84*std::pow(y, 3) - 84*std::pow(y, 2) + 21*y - 1) + 30*x*(210*std::pow(y, 4) - 336*std::pow(y, 3) + 168*std::pow(y, 2) - 28*y + 1) + 2772*std::pow(y, 5) - 6300*std::pow(y, 4) + 5040*std::pow(y, 3) - 1680*std::pow(y, 2) + 210*y - 6),
            (5985*std::pow(x, 4) - 4560*std::pow(x, 3) + 1026*std::pow(x, 2) - 72*x + 1)*(42*std::pow(x, 5) + 15*std::pow(x, 4)*(56*y - 14) + 20*std::pow(x, 3)*(252*std::pow(y, 2) - 168*y + 21) + 15*std::pow(x, 2)*(840*std::pow(y, 3) - 1008*std::pow(y, 2) + 336*y - 28) + 6*x*(2310*std::pow(y, 4) - 4200*std::pow(y, 3) + 2520*std::pow(y, 2) - 560*y + 35) + 5544*std::pow(y, 5) - 13860*std::pow(y, 4) + 12600*std::pow(y, 3) - 5040*std::pow(y, 2) + 840*y - 42)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 62
template<>
struct DGBasis2D<62> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (1330*std::pow(x, 3) - 570*std::pow(x, 2) + 57*x - 1)*(std::pow(x, 7) + 7*std::pow(x, 6)*(8*y - 1) + 21*std::pow(x, 5)*(36*std::pow(y, 2) - 16*y + 1) + 35*std::pow(x, 4)*(120*std::pow(y, 3) - 108*std::pow(y, 2) + 24*y - 1) + 35*std::pow(x, 3)*(330*std::pow(y, 4) - 480*std::pow(y, 3) + 216*std::pow(y, 2) - 32*y + 1) + 21*std::pow(x, 2)*(792*std::pow(y, 5) - 1650*std::pow(y, 4) + 1200*std::pow(y, 3) - 360*std::pow(y, 2) + 40*y - 1) + 7*x*(1716*std::pow(y, 6) - 4752*std::pow(y, 5) + 4950*std::pow(y, 4) - 2400*std::pow(y, 3) + 540*std::pow(y, 2) - 48*y + 1) + 3432*std::pow(y, 7) - 12012*std::pow(y, 6) + 16632*std::pow(y, 5) - 11550*std::pow(y, 4) + 4200*std::pow(y, 3) - 756*std::pow(y, 2) + 56*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (3990*std::pow(x, 2) - 1140*x + 57)*(std::pow(x, 7) + 7*std::pow(x, 6)*(8*y - 1) + 21*std::pow(x, 5)*(36*std::pow(y, 2) - 16*y + 1) + 35*std::pow(x, 4)*(120*std::pow(y, 3) - 108*std::pow(y, 2) + 24*y - 1) + 35*std::pow(x, 3)*(330*std::pow(y, 4) - 480*std::pow(y, 3) + 216*std::pow(y, 2) - 32*y + 1) + 21*std::pow(x, 2)*(792*std::pow(y, 5) - 1650*std::pow(y, 4) + 1200*std::pow(y, 3) - 360*std::pow(y, 2) + 40*y - 1) + 7*x*(1716*std::pow(y, 6) - 4752*std::pow(y, 5) + 4950*std::pow(y, 4) - 2400*std::pow(y, 3) + 540*std::pow(y, 2) - 48*y + 1) + 3432*std::pow(y, 7) - 12012*std::pow(y, 6) + 16632*std::pow(y, 5) - 11550*std::pow(y, 4) + 4200*std::pow(y, 3) - 756*std::pow(y, 2) + 56*y - 1) + (1330*std::pow(x, 3) - 570*std::pow(x, 2) + 57*x - 1)*(7*std::pow(x, 6) + 42*std::pow(x, 5)*(8*y - 1) + 105*std::pow(x, 4)*(36*std::pow(y, 2) - 16*y + 1) + 140*std::pow(x, 3)*(120*std::pow(y, 3) - 108*std::pow(y, 2) + 24*y - 1) + 105*std::pow(x, 2)*(330*std::pow(y, 4) - 480*std::pow(y, 3) + 216*std::pow(y, 2) - 32*y + 1) + 42*x*(792*std::pow(y, 5) - 1650*std::pow(y, 4) + 1200*std::pow(y, 3) - 360*std::pow(y, 2) + 40*y - 1) + 12012*std::pow(y, 6) - 33264*std::pow(y, 5) + 34650*std::pow(y, 4) - 16800*std::pow(y, 3) + 3780*std::pow(y, 2) - 336*y + 7),
            (1330*std::pow(x, 3) - 570*std::pow(x, 2) + 57*x - 1)*(56*std::pow(x, 6) + 21*std::pow(x, 5)*(72*y - 16) + 35*std::pow(x, 4)*(360*std::pow(y, 2) - 216*y + 24) + 35*std::pow(x, 3)*(1320*std::pow(y, 3) - 1440*std::pow(y, 2) + 432*y - 32) + 21*std::pow(x, 2)*(3960*std::pow(y, 4) - 6600*std::pow(y, 3) + 3600*std::pow(y, 2) - 720*y + 40) + 7*x*(10296*std::pow(y, 5) - 23760*std::pow(y, 4) + 19800*std::pow(y, 3) - 7200*std::pow(y, 2) + 1080*y - 48) + 24024*std::pow(y, 6) - 72072*std::pow(y, 5) + 83160*std::pow(y, 4) - 46200*std::pow(y, 3) + 12600*std::pow(y, 2) - 1512*y + 56)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 63
template<>
struct DGBasis2D<63> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (210*std::pow(x, 2) - 40*x + 1)*(std::pow(x, 8) + 8*std::pow(x, 7)*(9*y - 1) + 28*std::pow(x, 6)*(45*std::pow(y, 2) - 18*y + 1) + 56*std::pow(x, 5)*(165*std::pow(y, 3) - 135*std::pow(y, 2) + 27*y - 1) + 70*std::pow(x, 4)*(495*std::pow(y, 4) - 660*std::pow(y, 3) + 270*std::pow(y, 2) - 36*y + 1) + 56*std::pow(x, 3)*(1287*std::pow(y, 5) - 2475*std::pow(y, 4) + 1650*std::pow(y, 3) - 450*std::pow(y, 2) + 45*y - 1) + 28*std::pow(x, 2)*(3003*std::pow(y, 6) - 7722*std::pow(y, 5) + 7425*std::pow(y, 4) - 3300*std::pow(y, 3) + 675*std::pow(y, 2) - 54*y + 1) + 8*x*(6435*std::pow(y, 7) - 21021*std::pow(y, 6) + 27027*std::pow(y, 5) - 17325*std::pow(y, 4) + 5775*std::pow(y, 3) - 945*std::pow(y, 2) + 63*y - 1) + 12870*std::pow(y, 8) - 51480*std::pow(y, 7) + 84084*std::pow(y, 6) - 72072*std::pow(y, 5) + 34650*std::pow(y, 4) - 9240*std::pow(y, 3) + 1260*std::pow(y, 2) - 72*y + 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            (420*x - 40)*(std::pow(x, 8) + 8*std::pow(x, 7)*(9*y - 1) + 28*std::pow(x, 6)*(45*std::pow(y, 2) - 18*y + 1) + 56*std::pow(x, 5)*(165*std::pow(y, 3) - 135*std::pow(y, 2) + 27*y - 1) + 70*std::pow(x, 4)*(495*std::pow(y, 4) - 660*std::pow(y, 3) + 270*std::pow(y, 2) - 36*y + 1) + 56*std::pow(x, 3)*(1287*std::pow(y, 5) - 2475*std::pow(y, 4) + 1650*std::pow(y, 3) - 450*std::pow(y, 2) + 45*y - 1) + 28*std::pow(x, 2)*(3003*std::pow(y, 6) - 7722*std::pow(y, 5) + 7425*std::pow(y, 4) - 3300*std::pow(y, 3) + 675*std::pow(y, 2) - 54*y + 1) + 8*x*(6435*std::pow(y, 7) - 21021*std::pow(y, 6) + 27027*std::pow(y, 5) - 17325*std::pow(y, 4) + 5775*std::pow(y, 3) - 945*std::pow(y, 2) + 63*y - 1) + 12870*std::pow(y, 8) - 51480*std::pow(y, 7) + 84084*std::pow(y, 6) - 72072*std::pow(y, 5) + 34650*std::pow(y, 4) - 9240*std::pow(y, 3) + 1260*std::pow(y, 2) - 72*y + 1) + (210*std::pow(x, 2) - 40*x + 1)*(8*std::pow(x, 7) + 56*std::pow(x, 6)*(9*y - 1) + 168*std::pow(x, 5)*(45*std::pow(y, 2) - 18*y + 1) + 280*std::pow(x, 4)*(165*std::pow(y, 3) - 135*std::pow(y, 2) + 27*y - 1) + 280*std::pow(x, 3)*(495*std::pow(y, 4) - 660*std::pow(y, 3) + 270*std::pow(y, 2) - 36*y + 1) + 168*std::pow(x, 2)*(1287*std::pow(y, 5) - 2475*std::pow(y, 4) + 1650*std::pow(y, 3) - 450*std::pow(y, 2) + 45*y - 1) + 56*x*(3003*std::pow(y, 6) - 7722*std::pow(y, 5) + 7425*std::pow(y, 4) - 3300*std::pow(y, 3) + 675*std::pow(y, 2) - 54*y + 1) + 51480*std::pow(y, 7) - 168168*std::pow(y, 6) + 216216*std::pow(y, 5) - 138600*std::pow(y, 4) + 46200*std::pow(y, 3) - 7560*std::pow(y, 2) + 504*y - 8),
            (210*std::pow(x, 2) - 40*x + 1)*(72*std::pow(x, 7) + 28*std::pow(x, 6)*(90*y - 18) + 56*std::pow(x, 5)*(495*std::pow(y, 2) - 270*y + 27) + 70*std::pow(x, 4)*(1980*std::pow(y, 3) - 1980*std::pow(y, 2) + 540*y - 36) + 56*std::pow(x, 3)*(6435*std::pow(y, 4) - 9900*std::pow(y, 3) + 4950*std::pow(y, 2) - 900*y + 45) + 28*std::pow(x, 2)*(18018*std::pow(y, 5) - 38610*std::pow(y, 4) + 29700*std::pow(y, 3) - 9900*std::pow(y, 2) + 1350*y - 54) + 8*x*(45045*std::pow(y, 6) - 126126*std::pow(y, 5) + 135135*std::pow(y, 4) - 69300*std::pow(y, 3) + 17325*std::pow(y, 2) - 1890*y + 63) + 102960*std::pow(y, 7) - 360360*std::pow(y, 6) + 504504*std::pow(y, 5) - 360360*std::pow(y, 4) + 138600*std::pow(y, 3) - 27720*std::pow(y, 2) + 2520*y - 72)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 64
template<>
struct DGBasis2D<64> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return (21*x - 1)*(std::pow(x, 9) + std::pow(x, 8)*(90*y - 9) + 36*std::pow(x, 7)*(55*std::pow(y, 2) - 20*y + 1) + 84*std::pow(x, 6)*(220*std::pow(y, 3) - 165*std::pow(y, 2) + 30*y - 1) + 126*std::pow(x, 5)*(715*std::pow(y, 4) - 880*std::pow(y, 3) + 330*std::pow(y, 2) - 40*y + 1) + 126*std::pow(x, 4)*(2002*std::pow(y, 5) - 3575*std::pow(y, 4) + 2200*std::pow(y, 3) - 550*std::pow(y, 2) + 50*y - 1) + 84*std::pow(x, 3)*(5005*std::pow(y, 6) - 12012*std::pow(y, 5) + 10725*std::pow(y, 4) - 4400*std::pow(y, 3) + 825*std::pow(y, 2) - 60*y + 1) + 36*std::pow(x, 2)*(11440*std::pow(y, 7) - 35035*std::pow(y, 6) + 42042*std::pow(y, 5) - 25025*std::pow(y, 4) + 7700*std::pow(y, 3) - 1155*std::pow(y, 2) + 70*y - 1) + 9*x*(24310*std::pow(y, 8) - 91520*std::pow(y, 7) + 140140*std::pow(y, 6) - 112112*std::pow(y, 5) + 50050*std::pow(y, 4) - 12320*std::pow(y, 3) + 1540*std::pow(y, 2) - 80*y + 1) + 48620*std::pow(y, 9) - 218790*std::pow(y, 8) + 411840*std::pow(y, 7) - 420420*std::pow(y, 6) + 252252*std::pow(y, 5) - 90090*std::pow(y, 4) + 18480*std::pow(y, 3) - 1980*std::pow(y, 2) + 90*y - 1);
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            21*std::pow(x, 9) + 21*std::pow(x, 8)*(90*y - 9) + 756*std::pow(x, 7)*(55*std::pow(y, 2) - 20*y + 1) + 1764*std::pow(x, 6)*(220*std::pow(y, 3) - 165*std::pow(y, 2) + 30*y - 1) + 2646*std::pow(x, 5)*(715*std::pow(y, 4) - 880*std::pow(y, 3) + 330*std::pow(y, 2) - 40*y + 1) + 2646*std::pow(x, 4)*(2002*std::pow(y, 5) - 3575*std::pow(y, 4) + 2200*std::pow(y, 3) - 550*std::pow(y, 2) + 50*y - 1) + 1764*std::pow(x, 3)*(5005*std::pow(y, 6) - 12012*std::pow(y, 5) + 10725*std::pow(y, 4) - 4400*std::pow(y, 3) + 825*std::pow(y, 2) - 60*y + 1) + 756*std::pow(x, 2)*(11440*std::pow(y, 7) - 35035*std::pow(y, 6) + 42042*std::pow(y, 5) - 25025*std::pow(y, 4) + 7700*std::pow(y, 3) - 1155*std::pow(y, 2) + 70*y - 1) + 189*x*(24310*std::pow(y, 8) - 91520*std::pow(y, 7) + 140140*std::pow(y, 6) - 112112*std::pow(y, 5) + 50050*std::pow(y, 4) - 12320*std::pow(y, 3) + 1540*std::pow(y, 2) - 80*y + 1) + 1021020*std::pow(y, 9) - 4594590*std::pow(y, 8) + 8648640*std::pow(y, 7) - 8828820*std::pow(y, 6) + 5297292*std::pow(y, 5) - 1891890*std::pow(y, 4) + 388080*std::pow(y, 3) - 41580*std::pow(y, 2) + 1890*y + (21*x - 1)*(9*std::pow(x, 8) + 8*std::pow(x, 7)*(90*y - 9) + 252*std::pow(x, 6)*(55*std::pow(y, 2) - 20*y + 1) + 504*std::pow(x, 5)*(220*std::pow(y, 3) - 165*std::pow(y, 2) + 30*y - 1) + 630*std::pow(x, 4)*(715*std::pow(y, 4) - 880*std::pow(y, 3) + 330*std::pow(y, 2) - 40*y + 1) + 504*std::pow(x, 3)*(2002*std::pow(y, 5) - 3575*std::pow(y, 4) + 2200*std::pow(y, 3) - 550*std::pow(y, 2) + 50*y - 1) + 252*std::pow(x, 2)*(5005*std::pow(y, 6) - 12012*std::pow(y, 5) + 10725*std::pow(y, 4) - 4400*std::pow(y, 3) + 825*std::pow(y, 2) - 60*y + 1) + 72*x*(11440*std::pow(y, 7) - 35035*std::pow(y, 6) + 42042*std::pow(y, 5) - 25025*std::pow(y, 4) + 7700*std::pow(y, 3) - 1155*std::pow(y, 2) + 70*y - 1) + 218790*std::pow(y, 8) - 823680*std::pow(y, 7) + 1261260*std::pow(y, 6) - 1009008*std::pow(y, 5) + 450450*std::pow(y, 4) - 110880*std::pow(y, 3) + 13860*std::pow(y, 2) - 720*y + 9) - 21,
            (21*x - 1)*(90*std::pow(x, 8) + 36*std::pow(x, 7)*(110*y - 20) + 84*std::pow(x, 6)*(660*std::pow(y, 2) - 330*y + 30) + 126*std::pow(x, 5)*(2860*std::pow(y, 3) - 2640*std::pow(y, 2) + 660*y - 40) + 126*std::pow(x, 4)*(10010*std::pow(y, 4) - 14300*std::pow(y, 3) + 6600*std::pow(y, 2) - 1100*y + 50) + 84*std::pow(x, 3)*(30030*std::pow(y, 5) - 60060*std::pow(y, 4) + 42900*std::pow(y, 3) - 13200*std::pow(y, 2) + 1650*y - 60) + 36*std::pow(x, 2)*(80080*std::pow(y, 6) - 210210*std::pow(y, 5) + 210210*std::pow(y, 4) - 100100*std::pow(y, 3) + 23100*std::pow(y, 2) - 2310*y + 70) + 9*x*(194480*std::pow(y, 7) - 640640*std::pow(y, 6) + 840840*std::pow(y, 5) - 560560*std::pow(y, 4) + 200200*std::pow(y, 3) - 36960*std::pow(y, 2) + 3080*y - 80) + 437580*std::pow(y, 8) - 1750320*std::pow(y, 7) + 2882880*std::pow(y, 6) - 2522520*std::pow(y, 5) + 1261260*std::pow(y, 4) - 360360*std::pow(y, 3) + 55440*std::pow(y, 2) - 3960*y + 90)
        };
    }
    static constexpr uInt Order = 6;
};

// Basis 65
template<>
struct DGBasis2D<65> {
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y) {
        return std::pow(x, 10) + 10*std::pow(x, 9)*(11*y - 1) + 45*std::pow(x, 8)*(66*std::pow(y, 2) - 22*y + 1) + 120*std::pow(x, 7)*(286*std::pow(y, 3) - 198*std::pow(y, 2) + 33*y - 1) + 210*std::pow(x, 6)*(1001*std::pow(y, 4) - 1144*std::pow(y, 3) + 396*std::pow(y, 2) - 44*y + 1) + 252*std::pow(x, 5)*(3003*std::pow(y, 5) - 5005*std::pow(y, 4) + 2860*std::pow(y, 3) - 660*std::pow(y, 2) + 55*y - 1) + 210*std::pow(x, 4)*(8008*std::pow(y, 6) - 18018*std::pow(y, 5) + 15015*std::pow(y, 4) - 5720*std::pow(y, 3) + 990*std::pow(y, 2) - 66*y + 1) + 120*std::pow(x, 3)*(19448*std::pow(y, 7) - 56056*std::pow(y, 6) + 63063*std::pow(y, 5) - 35035*std::pow(y, 4) + 10010*std::pow(y, 3) - 1386*std::pow(y, 2) + 77*y - 1) + 45*std::pow(x, 2)*(43758*std::pow(y, 8) - 155584*std::pow(y, 7) + 224224*std::pow(y, 6) - 168168*std::pow(y, 5) + 70070*std::pow(y, 4) - 16016*std::pow(y, 3) + 1848*std::pow(y, 2) - 88*y + 1) + 10*x*(92378*std::pow(y, 9) - 393822*std::pow(y, 8) + 700128*std::pow(y, 7) - 672672*std::pow(y, 6) + 378378*std::pow(y, 5) - 126126*std::pow(y, 4) + 24024*std::pow(y, 3) - 2376*std::pow(y, 2) + 99*y - 1) + 184756*std::pow(y, 10) - 923780*std::pow(y, 9) + 1969110*std::pow(y, 8) - 2333760*std::pow(y, 7) + 1681680*std::pow(y, 6) - 756756*std::pow(y, 5) + 210210*std::pow(y, 4) - 34320*std::pow(y, 3) + 2970*std::pow(y, 2) - 110*y + 1;
    }
    
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y) {
        return {
            10*std::pow(x, 9) + 90*std::pow(x, 8)*(11*y - 1) + 360*std::pow(x, 7)*(66*std::pow(y, 2) - 22*y + 1) + 840*std::pow(x, 6)*(286*std::pow(y, 3) - 198*std::pow(y, 2) + 33*y - 1) + 1260*std::pow(x, 5)*(1001*std::pow(y, 4) - 1144*std::pow(y, 3) + 396*std::pow(y, 2) - 44*y + 1) + 1260*std::pow(x, 4)*(3003*std::pow(y, 5) - 5005*std::pow(y, 4) + 2860*std::pow(y, 3) - 660*std::pow(y, 2) + 55*y - 1) + 840*std::pow(x, 3)*(8008*std::pow(y, 6) - 18018*std::pow(y, 5) + 15015*std::pow(y, 4) - 5720*std::pow(y, 3) + 990*std::pow(y, 2) - 66*y + 1) + 360*std::pow(x, 2)*(19448*std::pow(y, 7) - 56056*std::pow(y, 6) + 63063*std::pow(y, 5) - 35035*std::pow(y, 4) + 10010*std::pow(y, 3) - 1386*std::pow(y, 2) + 77*y - 1) + 90*x*(43758*std::pow(y, 8) - 155584*std::pow(y, 7) + 224224*std::pow(y, 6) - 168168*std::pow(y, 5) + 70070*std::pow(y, 4) - 16016*std::pow(y, 3) + 1848*std::pow(y, 2) - 88*y + 1) + 923780*std::pow(y, 9) - 3938220*std::pow(y, 8) + 7001280*std::pow(y, 7) - 6726720*std::pow(y, 6) + 3783780*std::pow(y, 5) - 1261260*std::pow(y, 4) + 240240*std::pow(y, 3) - 23760*std::pow(y, 2) + 990*y - 10,
            110*std::pow(x, 9) + 45*std::pow(x, 8)*(132*y - 22) + 120*std::pow(x, 7)*(858*std::pow(y, 2) - 396*y + 33) + 210*std::pow(x, 6)*(4004*std::pow(y, 3) - 3432*std::pow(y, 2) + 792*y - 44) + 252*std::pow(x, 5)*(15015*std::pow(y, 4) - 20020*std::pow(y, 3) + 8580*std::pow(y, 2) - 1320*y + 55) + 210*std::pow(x, 4)*(48048*std::pow(y, 5) - 90090*std::pow(y, 4) + 60060*std::pow(y, 3) - 17160*std::pow(y, 2) + 1980*y - 66) + 120*std::pow(x, 3)*(136136*std::pow(y, 6) - 336336*std::pow(y, 5) + 315315*std::pow(y, 4) - 140140*std::pow(y, 3) + 30030*std::pow(y, 2) - 2772*y + 77) + 45*std::pow(x, 2)*(350064*std::pow(y, 7) - 1089088*std::pow(y, 6) + 1345344*std::pow(y, 5) - 840840*std::pow(y, 4) + 280280*std::pow(y, 3) - 48048*std::pow(y, 2) + 3696*y - 88) + 10*x*(831402*std::pow(y, 8) - 3150576*std::pow(y, 7) + 4900896*std::pow(y, 6) - 4036032*std::pow(y, 5) + 1891890*std::pow(y, 4) - 504504*std::pow(y, 3) + 72072*std::pow(y, 2) - 4752*y + 99) + 1847560*std::pow(y, 9) - 8314020*std::pow(y, 8) + 15752880*std::pow(y, 7) - 16336320*std::pow(y, 6) + 10090080*std::pow(y, 5) - 3783780*std::pow(y, 4) + 840840*std::pow(y, 3) - 102960*std::pow(y, 2) + 5940*y - 110
        };
    }
    static constexpr uInt Order = 6;
};
