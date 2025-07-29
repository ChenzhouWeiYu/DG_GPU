// include/DG/Flux/RiemannSolvers.h
class RiemannSolvers {
public:
    enum Type { LaxFriedrichs, Roe, HLLC };

    static void setType(Type type) { s_type = type; }

    static DenseMatrix<5,3> compute(
        const DenseMatrix<5,1>& U_L,
        const DenseMatrix<5,1>& U_R,
        const vector3f& normal)
    {
        switch(s_type) {
            case LaxFriedrichs: return computeLaxFriedrichs(U_L, U_R, normal);
            case Roe: return computeRoe(U_L, U_R, normal);
            case HLLC: return computeHLLC(U_L, U_R, normal);
            default: throw std::invalid_argument("Unknown Riemann solver");
        }
    }

private:
    static inline Type s_type = LaxFriedrichs;
    
    // 各解算器实现
    static DenseMatrix<5,3> computeLaxFriedrichs(...);
    static DenseMatrix<5,3> computeRoe(...);
    static DenseMatrix<5,3> computeHLLC(...);
};