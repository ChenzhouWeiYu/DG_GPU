template <>
struct AutoQuadSelector<0, GaussLegendreTri::Auto> {
    using type = GaussLegendreTri::Degree4Points7;
};

template <>
struct AutoQuadSelector<1, GaussLegendreTri::Auto> {
    using type = GaussLegendreTri::Degree4Points7;
};

template <>
struct AutoQuadSelector<2, GaussLegendreTri::Auto> {
    using type = GaussLegendreTri::Degree10Points24;
};
template <>
struct AutoQuadSelector<3, GaussLegendreTri::Auto> {
    using type = GaussLegendreTri::Degree10Points24;
};
template <>
struct AutoQuadSelector<4, GaussLegendreTri::Auto> {
    using type = GaussLegendreTri::Degree10Points24;
};

template <>
struct AutoQuadSelector<5, GaussLegendreTri::Auto> {
    using type = GaussLegendreTri::Degree13Points36;
};
template <>
struct AutoQuadSelector<6, GaussLegendreTri::Auto> {
    using type = GaussLegendreTri::Degree13Points36;
};

template <>
struct AutoQuadSelector<7, GaussLegendreTri::Auto> {
    using type = GaussLegendreTri::Degree17Points57;
};
template <>
struct AutoQuadSelector<8, GaussLegendreTri::Auto> {
    using type = GaussLegendreTri::Degree17Points57;
};

template <>
struct AutoQuadSelector<9, GaussLegendreTri::Auto> {
    using type = GaussLegendreTri::Degree21Points85;
};
