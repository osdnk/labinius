use std::arch::x86_64::*;

#[inline(always)]
unsafe fn x3(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    _mm512_ternarylogic_epi64::<0x96>(a, b, c)
}

#[inline(always)]
unsafe fn idx_lo() -> __m512i {
    _mm512_setr_epi64(0, 8, 2, 10, 4, 12, 6, 14)
}

#[inline(always)]
unsafe fn idx_hi() -> __m512i {
    _mm512_setr_epi64(1, 9, 3, 11, 5, 13, 7, 15)
}

#[inline(always)]
unsafe fn prods_soa8(a: [__m512i; 3], b: [__m512i; 3]) -> [__m512i; 12] {
    let sa01 = _mm512_xor_si512(a[0], a[1]);
    let sa02 = _mm512_xor_si512(a[0], a[2]);
    let sa12 = _mm512_xor_si512(a[1], a[2]);
    let sb01 = _mm512_xor_si512(b[0], b[1]);
    let sb02 = _mm512_xor_si512(b[0], b[2]);
    let sb12 = _mm512_xor_si512(b[1], b[2]);
    [
        _mm512_clmulepi64_epi128::<0x00>(a[0], b[0]),
        _mm512_clmulepi64_epi128::<0x11>(a[0], b[0]),
        _mm512_clmulepi64_epi128::<0x00>(a[1], b[1]),
        _mm512_clmulepi64_epi128::<0x11>(a[1], b[1]),
        _mm512_clmulepi64_epi128::<0x00>(a[2], b[2]),
        _mm512_clmulepi64_epi128::<0x11>(a[2], b[2]),
        _mm512_clmulepi64_epi128::<0x00>(sa01, sb01),
        _mm512_clmulepi64_epi128::<0x11>(sa01, sb01),
        _mm512_clmulepi64_epi128::<0x00>(sa02, sb02),
        _mm512_clmulepi64_epi128::<0x11>(sa02, sb02),
        _mm512_clmulepi64_epi128::<0x00>(sa12, sb12),
        _mm512_clmulepi64_epi128::<0x11>(sa12, sb12),
    ]
}

#[inline(always)]
pub unsafe fn mac_soa8(acc: &mut [__m512i; 12], a: [__m512i; 3], b: [__m512i; 3]) {
    let p = prods_soa8(a, b);
    for i in 0..12 {
        acc[i] = _mm512_xor_si512(acc[i], p[i]);
    }
}

#[inline(always)]
pub unsafe fn reduce_soa8(p: [__m512i; 12]) -> [__m512i; 3] {
    let m17 = _mm512_set1_epi64(((1u64 << 17) - 1) as i64);
    let m34 = _mm512_set1_epi64(((1u64 << 34) - 1) as i64);
    let (m0e, m0o, m1e, m1o, m2e, m2o) = (p[0], p[1], p[2], p[3], p[4], p[5]);
    let (k01e, k01o, k02e, k02o, k12e, k12o) = (p[6], p[7], p[8], p[9], p[10], p[11]);

    let d1e = x3(k01e, m0e, m1e);
    let d1o = x3(k01o, m0o, m1o);
    let d3e = x3(k12e, m1e, m2e);
    let d3o = x3(k12o, m1o, m2o);
    let d2e = _mm512_xor_si512(x3(k02e, m0e, m2e), m1e);
    let d2o = _mm512_xor_si512(x3(k02o, m0o, m2o), m1o);

    let lo = idx_lo();
    let hi = idx_hi();
    let p0 = _mm512_permutex2var_epi64(m0e, lo, m0o);
    let p1 = _mm512_xor_si512(
        _mm512_permutex2var_epi64(m0e, hi, m0o),
        _mm512_permutex2var_epi64(d1e, lo, d1o),
    );
    let p2 = _mm512_xor_si512(
        _mm512_permutex2var_epi64(d1e, hi, d1o),
        _mm512_permutex2var_epi64(d2e, lo, d2o),
    );
    let p3 = _mm512_xor_si512(
        _mm512_permutex2var_epi64(d2e, hi, d2o),
        _mm512_permutex2var_epi64(d3e, lo, d3o),
    );
    let p4 = _mm512_xor_si512(
        _mm512_permutex2var_epi64(d3e, hi, d3o),
        _mm512_permutex2var_epi64(m2e, lo, m2o),
    );
    let p5 = _mm512_permutex2var_epi64(m2e, hi, m2o);

    let h00 = _mm512_shrdi_epi64::<34>(p2, p3);
    let h01 = _mm512_and_si512(_mm512_srli_epi64::<34>(p3), m17);
    let h10 = _mm512_shrdi_epi64::<51>(p3, p4);
    let h11 = _mm512_shrdi_epi64::<51>(p4, p5);

    let r0 = x3(p0, h00, h10);
    let r1 = _mm512_xor_si512(x3(p1, h01, h11), _mm512_slli_epi64::<17>(h00));
    let r2 = _mm512_xor_si512(
        _mm512_and_si512(p2, m34),
        _mm512_shldi_epi64::<17>(h01, h00),
    );
    [r0, r1, r2]
}

#[inline(always)]
pub unsafe fn mul_soa8(a: [__m512i; 3], b: [__m512i; 3]) -> [__m512i; 3] {
    reduce_soa8(prods_soa8(a, b))
}
