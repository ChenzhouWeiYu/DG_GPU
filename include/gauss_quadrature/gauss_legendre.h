#include "base/type.h"


template <uInt Order, typename DefaultQuad>
struct AutoQuadSelector;

namespace GaussLegendreQuad {

    struct Middle {
        public:
            static constexpr int num_points = 1;
            static constexpr std::array<std::array<Scalar,2>, num_points> points = {{
                {0., 0.}
            }};
            static constexpr std::array<Scalar, num_points> weights = {4.0};
    };

    #include "gauss_quadrature/quad_quad_lower_order.h"
    
};

namespace GaussLegendreTri {
    
    struct Auto {
        public:
            static constexpr int num_points = 0;
    };
    
    struct Middle {
        public:
            static constexpr int num_points = 1;
            static constexpr std::array<std::array<Scalar,2>, num_points> points = {{
                {1.0/3, 1.0/3}
            }};
            static constexpr std::array<Scalar, num_points> weights = {0.5};
    };

    #include "gauss_quadrature/quad_tri_lower_order.h"
    #include "gauss_quadrature/quad_tri_high_order.h"

}

namespace GaussLegendreTet {

    struct Auto {
        public:
            static constexpr int num_points = 0;
    };
    struct Middle {
        public:
            static constexpr int num_points = 1;
            static constexpr std::array<std::array<Scalar,3>, num_points> points = {{
                {1.0/4, 1.0/4, 1.0/4}
            }};
            static constexpr std::array<Scalar, num_points> weights = {1.0/6};
    };
    
    #include "gauss_quadrature/quad_tet_lower_order.h"
    #include "gauss_quadrature/quad_tet_high_order.h"
}



template <uInt Order, typename DefaultQuad>
struct AutoQuadSelector {
    using type = DefaultQuad;
};

template <uInt Order>
struct AutoQuadSelector<Order, GaussLegendreTet::Auto> {
    using type = GaussLegendreTet::Degree20Points448;
};

template <uInt Order>
struct AutoQuadSelector<Order, GaussLegendreTri::Auto> {
    using type = GaussLegendreTri::Degree23Points100;
};

#include "gauss_quadrature/auto_quad_selector_tet.h"
#include "gauss_quadrature/auto_quad_selector_tri.h"