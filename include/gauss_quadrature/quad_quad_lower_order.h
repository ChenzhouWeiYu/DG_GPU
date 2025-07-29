    struct Degree3Points4 {
        public:
            static constexpr int num_points = 4;
            
            HostDevice
            static constexpr auto get_points() {
                // constexpr Scalar M_SQRT3 = 1.732050807568877293527446341505872366943;
                constexpr Scalar M_SQRT1_3 = 0.5773502691896257645091487805019574556476;
                return std::array<std::array<Scalar, 2>, num_points>{{
                    {-M_SQRT1_3, -M_SQRT1_3}, 
                    { M_SQRT1_3, -M_SQRT1_3},
                    { M_SQRT1_3,  M_SQRT1_3},
                    {-M_SQRT1_3,  M_SQRT1_3}
                }};
            }

            HostDevice
            static constexpr auto get_weights() {
                return std::array<Scalar, num_points>{ 1.0, 1.0, 1.0, 1.0 };
            }
    };

    struct Degree5Points7 {
        public:
            static constexpr int num_points = 7;
            HostDevice
            static constexpr auto get_points() {
                constexpr Scalar r = 0.6831300510639732255480692453680701327157;
                constexpr Scalar s = 0.8906544217808369920783964022053703474661;
                constexpr Scalar t = 0.3742566422865147407211610155677170660217;
                return std::array<std::array<Scalar, 2>, num_points>{{
                    {  0,  0}, 
                    {  r,  r},
                    { -r, -r},
                    {  s, -t},
                    { -s,  t},
                    {  t, -s},
                    { -t,  s}
                }};
            }

            HostDevice
            static constexpr auto get_weights() {
                return std::array<Scalar, num_points>{
                    4 * 2.0/7.0, 
                    4 * 25.0/168.0, 
                    4 * 25.0/168.0, 
                    4 * 5.0/48.0, 
                    4 * 5.0/48.0, 
                    4 * 5.0/48.0, 
                    4 * 5.0/48.0
                };
            }
    };

    struct Degree7Points12 {
        private:
        public:
            static constexpr int num_points = 12;
            HostDevice
            static constexpr auto get_points() {
                constexpr Scalar r = 0.9258200997725514615665667765839995225293;
                constexpr Scalar s = 0.3805544332083156563791063590863941355001;
                constexpr Scalar t = 0.8059797829185987437078561813507442463004;
                return std::array<std::array<Scalar, 2>, num_points>{{
                    {  r,  0}, 
                    { -r,  0},
                    {  0,  r}, 
                    {  0, -r},
                    {  s,  s}, 
                    { -s,  s},
                    {  s, -s}, 
                    { -s, -s},
                    {  t,  t}, 
                    { -t,  t},
                    {  t, -t}, 
                    { -t, -t}
                }};
            }

            HostDevice
            static constexpr auto get_weights() {
                constexpr Scalar Bs = 0.1301482291668486142849798580116827915066;
                constexpr Scalar Bt = 0.05935794367265755855452631482782338133293;
                return std::array<Scalar, num_points>{
                    4 * 49.0/810.0, 
                    4 * 49.0/810.0, 
                    4 * 49.0/810.0, 
                    4 * 49.0/810.0, 
                    4 * Bs, 
                    4 * Bs, 
                    4 * Bs, 
                    4 * Bs, 
                    4 * Bt, 
                    4 * Bt, 
                    4 * Bt, 
                    4 * Bt
                };
            }
    };