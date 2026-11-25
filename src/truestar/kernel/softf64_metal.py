"""IEEE 754 binary64 arithmetic on 64 bit integers, in Metal Shading Language.

Apple GPUs have no float64. Round once math (docs/ROUND_ONCE.md) is a fixed sequence of IEEE
float64 operations, so on Metal each operation is done here on the bits: add, subtract, multiply,
divide and square root rounded to nearest even (subnormal results included, no flush), compare,
floor, and the exact conversions from float32 and to float32 (rounded to nearest even). A float64
value is a ``ulong`` holding its bits. Every NaN result is the quiet NaN 0x7FF8000000000000; round
once makes the final NaN canonical anyway, and only NaN tests read a NaN.

Written for Truestar from the IEEE 754 definitions; tests/test_roundonce.py checks every operation
against the CPU's float64 on edge cases and on random operands, on the Mac's GPU.
"""

SOURCE = r"""
// ---- binary64 on ulong bits (truestar.kernel.softf64_metal)
#define TS_DSIGN 0x8000000000000000ul
#define TS_DEXP 0x7FF0000000000000ul
#define TS_DMANT 0x000FFFFFFFFFFFFFul
#define TS_DQNAN 0x7FF8000000000000ul
static inline bool ts_disnan(ulong a) { return (a & ~TS_DSIGN) > TS_DEXP; }
static inline bool ts_disinf(ulong a) { return (a & ~TS_DSIGN) == TS_DEXP; }
static inline bool ts_diszero(ulong a) { return (a & ~TS_DSIGN) == 0ul; }
static inline int ts_clz64(ulong x) { return x == 0ul ? 64 : (int)clz(x); }

// value = sig * 2^e with sig != 0, normalized so that bit 62 is the leading one
static inline void ts_dunpack(ulong a, thread ulong& sig, thread int& e) {
    int E = (int)((a >> 52) & 0x7FFul);
    ulong f = a & TS_DMANT;
    if (E == 0) { sig = f; e = -1074; } else { sig = f | (1ul << 52); e = E - 1075; }
    int s = ts_clz64(sig) - 1;
    sig <<= s;
    e -= s;
}

// round sig * 2^e (bit 62 the leading one, bit 0 sticky) to nearest even and pack
static inline ulong ts_dpack(ulong sign, int e, ulong sig) {
    int E = e + 62 + 1023;
    if (E >= 2047) return sign | TS_DEXP;
    ulong m;
    if (E <= 0) {
        int sh = 1 - E;
        sig = sh > 63 ? (sig != 0ul ? 1ul : 0ul) : ((sig >> sh) | ((sig & ((1ul << sh) - 1ul)) != 0ul ? 1ul : 0ul));
        E = 0;
    }
    ulong rb = sig & 0x3FFul;
    m = sig >> 10;
    if (rb > 0x200ul || (rb == 0x200ul && (m & 1ul) != 0ul)) m += 1ul;
    ulong bits = E == 0 ? m : ((ulong)(E - 1) << 52) + m;
    return sign | bits;
}

static inline ulong ts_dadd(ulong a, ulong b) {
    if (ts_disnan(a) || ts_disnan(b)) return TS_DQNAN;
    if (ts_disinf(a)) return (ts_disinf(b) && ((a ^ b) & TS_DSIGN) != 0ul) ? TS_DQNAN : a;
    if (ts_disinf(b)) return b;
    if (ts_diszero(a) && ts_diszero(b)) return a & b;      // -0 only when both are -0
    if (ts_diszero(a)) return b;
    if (ts_diszero(b)) return a;
    ulong sa, sb; int ea, eb;
    ts_dunpack(a, sa, ea);
    ts_dunpack(b, sb, eb);
    ulong sga = a & TS_DSIGN, sgb = b & TS_DSIGN;
    if (eb > ea || (eb == ea && sb > sa)) {
        ulong ts = sa; sa = sb; sb = ts;
        int te = ea; ea = eb; eb = te;
        ulong tg = sga; sga = sgb; sgb = tg;
    }
    int d = ea - eb;
    if (d > 0) sb = d > 63 ? 1ul : ((sb >> d) | ((sb & ((1ul << d) - 1ul)) != 0ul ? 1ul : 0ul));
    if (sga == sgb) {
        ulong s = sa + sb;
        if ((s >> 63) != 0ul) { s = (s >> 1) | (s & 1ul); ea += 1; }
        return ts_dpack(sga, ea, s);
    }
    ulong s = sa - sb;
    if (s == 0ul) return 0ul;
    int sh = ts_clz64(s) - 1;
    return ts_dpack(sga, ea - sh, s << sh);
}
static inline ulong ts_dneg(ulong a) { return a ^ TS_DSIGN; }
static inline ulong ts_dsub(ulong a, ulong b) { return ts_dadd(a, ts_dneg(b)); }

// 53 x 53 bit product as hi (bits 64..105) and lo (bits 0..63), from 32 bit limbs
static inline void ts_mul64(ulong x, ulong y, thread ulong& hi, thread ulong& lo) {
    ulong x0 = x & 0xFFFFFFFFul, x1 = x >> 32, y0 = y & 0xFFFFFFFFul, y1 = y >> 32;
    ulong p00 = x0 * y0, p01 = x0 * y1, p10 = x1 * y0, p11 = x1 * y1;
    ulong mid = (p00 >> 32) + (p01 & 0xFFFFFFFFul) + (p10 & 0xFFFFFFFFul);
    lo = (p00 & 0xFFFFFFFFul) | (mid << 32);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

static inline ulong ts_dmul(ulong a, ulong b) {
    ulong sign = (a ^ b) & TS_DSIGN;
    if (ts_disnan(a) || ts_disnan(b)) return TS_DQNAN;
    if (ts_disinf(a) || ts_disinf(b)) return (ts_diszero(a) || ts_diszero(b)) ? TS_DQNAN : (sign | TS_DEXP);
    if (ts_diszero(a) || ts_diszero(b)) return sign;
    ulong sa, sb; int ea, eb;
    ts_dunpack(a, sa, ea);
    ts_dunpack(b, sb, eb);
    sa >>= 10; ea += 10;                  // 53 bit significands, leading one at bit 52
    sb >>= 10; eb += 10;
    ulong hi, lo;
    ts_mul64(sa, sb, hi, lo);             // product < 2^106, leading one at bit 104 or 105
    ulong p = (hi << 22) | (lo >> 42);
    ulong st = (lo & ((1ul << 42) - 1ul)) != 0ul ? 1ul : 0ul;
    int e = ea + eb + 42;
    if ((p >> 63) != 0ul) { st |= p & 1ul; p >>= 1; e += 1; }
    return ts_dpack(sign, e, p | st);
}

static inline ulong ts_ddiv(ulong a, ulong b) {
    ulong sign = (a ^ b) & TS_DSIGN;
    if (ts_disnan(a) || ts_disnan(b)) return TS_DQNAN;
    if (ts_disinf(a)) return ts_disinf(b) ? TS_DQNAN : (sign | TS_DEXP);
    if (ts_disinf(b)) return sign;
    if (ts_diszero(b)) return ts_diszero(a) ? TS_DQNAN : (sign | TS_DEXP);
    if (ts_diszero(a)) return sign;
    ulong sa, sb; int ea, eb;
    ts_dunpack(a, sa, ea);
    ts_dunpack(b, sb, eb);
    sa >>= 10; ea += 10;
    sb >>= 10; eb += 10;
    if (sa < sb) { sa <<= 1; ea -= 1; }
    ulong q = 0ul, rem = sa;
    for (int i = 0; i < 63; ++i) {       // 63 quotient bits, the first one is 1
        q <<= 1;
        if (rem >= sb) { rem -= sb; q |= 1ul; }
        rem <<= 1;
    }
    return ts_dpack(sign, ea - eb - 62, q | (rem != 0ul ? 1ul : 0ul));
}

static inline ulong ts_dsqrt(ulong a) {
    if (ts_disnan(a)) return TS_DQNAN;
    if (ts_diszero(a)) return a;
    if ((a & TS_DSIGN) != 0ul) return TS_DQNAN;
    if (ts_disinf(a)) return a;
    ulong s; int e;
    ts_dunpack(a, s, e);
    s >>= 10; e += 10;                    // 53 bits, leading one at bit 52
    if ((e & 1) != 0) { s <<= 1; e -= 1; }
    // radicand R = s * 2^72 as (rh, rl): in [2^124, 2^126)
    ulong rh = s << 8, rl = 0ul;
    ulong res = 0ul, remh = 0ul, reml = 0ul;
    for (int i = 63; i >= 0; --i) {
        ulong two = i >= 32 ? (rh >> (2 * (i - 32))) & 3ul : (rl >> (2 * i)) & 3ul;
        remh = (remh << 2) | (reml >> 62);
        reml = (reml << 2) | two;
        ulong th = res >> 62, tl = (res << 2) | 1ul;      // trial = 4 res + 1
        if (remh > th || (remh == th && reml >= tl)) {
            ulong nl = reml - tl;
            remh = remh - th - (reml < tl ? 1ul : 0ul);
            reml = nl;
            res = (res << 1) | 1ul;
        } else {
            res <<= 1;
        }
    }
    return ts_dpack(0ul, (e - 72) / 2, res | ((remh | reml) != 0ul ? 1ul : 0ul));
}

static inline bool ts_deq(ulong a, ulong b) {
    if (ts_disnan(a) || ts_disnan(b)) return false;
    return a == b || (ts_diszero(a) && ts_diszero(b));
}
static inline bool ts_dne(ulong a, ulong b) { return !ts_deq(a, b); }
static inline ulong ts_dkey(ulong a) { return (a & TS_DSIGN) != 0ul ? ~a : (a | TS_DSIGN); }
static inline bool ts_dlt(ulong a, ulong b) {
    if (ts_disnan(a) || ts_disnan(b) || (ts_diszero(a) && ts_diszero(b))) return false;
    return ts_dkey(a) < ts_dkey(b);
}
static inline bool ts_dle(ulong a, ulong b) { return ts_dlt(a, b) || ts_deq(a, b); }
static inline ulong ts_dabs(ulong a) { return a & ~TS_DSIGN; }
static inline bool ts_dsignbit(ulong a) { return (a & TS_DSIGN) != 0ul; }

static inline ulong ts_dfloor(ulong a) {
    if (ts_disnan(a) || ts_disinf(a) || ts_diszero(a)) return a;
    int E = (int)((a >> 52) & 0x7FFul);
    bool neg = (a & TS_DSIGN) != 0ul;
    if (E < 1023) return neg ? 0xBFF0000000000000ul : 0ul;
    if (E >= 1075) return a;
    int fb = 1075 - E;
    ulong mask = (1ul << fb) - 1ul;
    if ((a & mask) == 0ul) return a;
    return neg ? ((a & ~mask) + (1ul << fb)) : (a & ~mask);
}

// an integer valued binary64 (|k| < 2^31) as int, exactly
static inline int ts_dtoint(ulong k) {
    if (ts_diszero(k)) return 0;
    int E = (int)((k >> 52) & 0x7FFul);
    ulong m = (k & TS_DMANT) | (1ul << 52);
    int sh = 1075 - E;
    long v = sh >= 64 ? 0l : (long)(sh >= 0 ? (m >> sh) : (m << (-sh)));
    return (k & TS_DSIGN) != 0ul ? -(int)v : (int)v;
}
static inline ulong ts_dfromint(int i) {
    if (i == 0) return 0ul;
    ulong sign = i < 0 ? TS_DSIGN : 0ul;
    ulong m = (ulong)(i < 0 ? -(long)i : (long)i);
    int s = ts_clz64(m) - 1;
    return ts_dpack(sign, -s, m << s);
}
static inline ulong ts_dpow2(ulong k) { return ((ulong)(ts_dtoint(k) + 1023)) << 52; }
static inline ulong ts_dfrexp_m(ulong x) { return (x & TS_DMANT) | 0x3FF0000000000000ul; }
static inline ulong ts_dfrexp_e(ulong x) { return ts_dfromint((int)((x >> 52) & 0x7FFul) - 1023); }

static inline ulong ts_dfromf32(float x) {
    uint b = as_type<uint>(x);
    ulong sign = ((ulong)(b >> 31)) << 63;
    int E = (int)((b >> 23) & 0xFFu);
    ulong m = (ulong)(b & 0x7FFFFFu);
    if (E == 255) return m != 0ul ? TS_DQNAN : (sign | TS_DEXP);
    if (E == 0) {
        if (m == 0ul) return sign;
        int s = ts_clz64(m) - 1;
        return ts_dpack(sign, -149 - s, m << s);
    }
    return sign | ((ulong)(E - 127 + 1023) << 52) | (m << 29);
}

// to float32, rounded to nearest even; NaN becomes the canonical 0x7FC00000
static inline float ts_dtof32(ulong a) {
    uint sign = (uint)(a >> 32) & 0x80000000u;
    if (ts_disnan(a)) return as_type<float>(0x7FC00000u);
    if (ts_disinf(a)) return as_type<float>(sign | 0x7F800000u);
    if (ts_diszero(a)) return as_type<float>(sign);
    ulong s; int e;
    ts_dunpack(a, s, e);
    int E = e + 62 + 127;
    if (E >= 255) return as_type<float>(sign | 0x7F800000u);
    if (E <= 0) {
        int sh = 1 - E;
        s = sh > 63 ? (s != 0ul ? 1ul : 0ul) : ((s >> sh) | ((s & ((1ul << sh) - 1ul)) != 0ul ? 1ul : 0ul));
        E = 0;
    }
    ulong rb = s & ((1ul << 39) - 1ul);
    ulong m = s >> 39;
    if (rb > (1ul << 38) || (rb == (1ul << 38) && (m & 1ul) != 0ul)) m += 1ul;
    uint bits = E == 0 ? (uint)m : ((uint)(E - 1) << 23) + (uint)m;
    return as_type<float>(sign | bits);
}
"""
