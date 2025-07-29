    struct Degree5Points15 {
        public:
            static constexpr int num_points = 15;

            HostDevice
            static constexpr std::array<std::array<Scalar, 3>, num_points> get_points() {
                constexpr Scalar x0 = 0.25;

                constexpr Scalar x1 = 0.0919710780527230327888451353005;
                constexpr Scalar y1 = 0.724086765841830901633464594098;

                constexpr Scalar x2 = 0.319793627829629908387625452935;
                constexpr Scalar y2 = 0.0406191165111102748371236411957;

                constexpr Scalar x3 = 0.0563508326896291557410367300109;
                constexpr Scalar y3 = 0.443649167310370844258963269989;

                return {{
                    {x0, x0, x0},
                    {x1, x1, x1}, {x1, x1, y1}, {x1, y1, x1}, {y1, x1, x1},
                    {x2, x2, x2}, {x2, x2, y2}, {x2, y2, x2}, {y2, x2, x2},
                    {x3, x3, y3}, {x3, y3, x3}, {y3, x3, x3},
                    {y3, y3, x3}, {y3, x3, y3}, {x3, y3, y3}
                }};
            }

            HostDevice
            static constexpr std::array<Scalar, num_points> get_weights() {
                constexpr Scalar w0 = 0.0197530864197530864197530864198;
                constexpr Scalar w1 = 0.0119895139631697700017306424850;
                constexpr Scalar w2 = 0.0115113678710453975467702393492;
                constexpr Scalar w3 = 0.00881834215167548500881834215168;

                return {
                    w0,
                    w1, w1, w1, w1,
                    w2, w2, w2, w2,
                    w3, w3, w3, w3, w3, w3
                };
            }
    };
