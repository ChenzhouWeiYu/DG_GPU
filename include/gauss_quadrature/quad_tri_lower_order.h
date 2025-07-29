    struct Degree3Points4 {
        private:
        public:
            static constexpr int num_points = 4;
            HostDevice
            static constexpr std::array<std::array<Scalar,2>, num_points> get_points() {
                constexpr Scalar x1 = 0.07503111022260811817747559832460309763591;
                constexpr Scalar x2 = 0.1785587282636164231170351333742224289755;
                constexpr Scalar y1 = 0.2800199154990740720027959942048077631675;
                constexpr Scalar y2 = 0.6663902460147013867026932740963667102211;
                return {{
                        {x1,y1}, {y1,x1}, {x2,y2}, {y2,x2}
                    }};
            }

            HostDevice
            static constexpr std::array<Scalar, num_points> get_weights() {
                constexpr Scalar w1 = 0.09097930912801141530281549896241817511158;
                constexpr Scalar w2 = 0.1590206908719885846971845010375818248884;
                return {w1,w1,w2,w2};
            }
    };

    struct Degree3Points4_B {
        public:
            static constexpr int num_points = 4;

            HostDevice
            static constexpr std::array<std::array<Scalar, 2>, num_points> get_points() {
                constexpr Scalar x1 = 0.0;
                constexpr Scalar y1 = 0.8;
                constexpr Scalar x2 = 0.4339491425357193931250391681262089353371;
                constexpr Scalar x3 = 0.1755746669880901306844846413976005884725;

                return {{
                    {x1, y1}, {y1, x1}, {x2, x2}, {x3, x3}
                }};
            }

            HostDevice
            static constexpr std::array<Scalar, num_points> get_weights() {
                constexpr Scalar w1 = 0.06510416666666666666666666666666666666667;
                constexpr Scalar w2 = 0.1921911384555808477492202542009676977468;
                constexpr Scalar w3 = 0.1776005282110858189174464124656989689199;

                return { w1, w1, w2, w3 };
            }
    };

    struct Degree3Points6 {
        public:
            static constexpr int num_points = 6;

            HostDevice
            static constexpr std::array<std::array<Scalar, 2>, num_points> get_points() {
                constexpr Scalar x1 = 0.5;
                constexpr Scalar y1 = 0.0;
                constexpr Scalar x2 = 0.1666666666666666666666666666666666666666;
                constexpr Scalar y2 = 0.6666666666666666666666666666666666666666;

                return {{
                    {x1, x1}, {x1, y1}, {y1, x1},
                    {x2, x2}, {x2, y2}, {y2, x2}
                }};
            }

            HostDevice
            static constexpr std::array<Scalar, num_points> get_weights() {
                constexpr Scalar w1 = 0.5 * 0.0333333333333333333333333333333333333333;
                constexpr Scalar w2 = 0.5 * 0.3;

                return { w1, w1, w1, w2, w2, w2 };
            }
    };

    struct Degree4Points7 {
        public:
            static constexpr int num_points = 7;

            HostDevice
            static constexpr std::array<std::array<Scalar, 2>, num_points> get_points() {
                constexpr Scalar x0 = 1.0/3;
                constexpr Scalar x1 = 0.1012865073234563388009873619151238280556;
                constexpr Scalar y1 = 0.7974269853530873223980252761697523438888;
                constexpr Scalar x2 = 0.4701420641051150897704412095134476005159;
                constexpr Scalar y2 = 0.05971587178976982045911758097310479896829;

                return {{
                    {x0, x0},
                    {x1, x1}, {x1, y1}, {y1, x1},
                    {x2, x2}, {x2, y2}, {y2, x2}
                }};
            }

            HostDevice
            static constexpr std::array<Scalar, num_points> get_weights() {
                constexpr Scalar w0 = 0.5 * 0.225;
                constexpr Scalar w1 = 0.5 * 0.1259391805448271525956839455001813336576;
                constexpr Scalar w2 = 0.5 * 0.1323941527885061807376493878331519996757;

                return { w0, w1, w1, w1, w2, w2, w2 };
            }
    };
