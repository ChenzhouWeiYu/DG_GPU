
template <>
struct AutoQuadSelector<0, GaussLegendreTet::Auto> {
    using type = GaussLegendreTet::Degree5Points15;
};

template <>
struct AutoQuadSelector<1, GaussLegendreTet::Auto> {
    using type = GaussLegendreTet::Degree5Points15;
};

template <>
struct AutoQuadSelector<2, GaussLegendreTet::Auto> {
    using type = GaussLegendreTet::Degree5Points15;
};

template <>
struct AutoQuadSelector<3, GaussLegendreTet::Auto> {
    using type = GaussLegendreTet::Degree7Points31;
};

template <>
struct AutoQuadSelector<4, GaussLegendreTet::Auto> {
    using type = GaussLegendreTet::Degree9Points57;
};

template <>
struct AutoQuadSelector<5, GaussLegendreTet::Auto> {
    using type = GaussLegendreTet::Degree11Points94;
};

template <>
struct AutoQuadSelector<6, GaussLegendreTet::Auto> {
    using type = GaussLegendreTet::Degree13Points144;
};

template <>
struct AutoQuadSelector<7, GaussLegendreTet::Auto> {
    using type = GaussLegendreTet::Degree15Points207;
};

template <>
struct AutoQuadSelector<8, GaussLegendreTet::Auto> {
    using type = GaussLegendreTet::Degree17Points288;
};

template <>
struct AutoQuadSelector<9, GaussLegendreTet::Auto> {
    using type = GaussLegendreTet::Degree19Points390;
};