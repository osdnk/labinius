//! Pointwise (slot-wise) products in the NTT domain on the vertical `Batch32` layout, signed
//! Montgomery. Not optimised (the task is the NTT); used by the multiplication tests/benches.
//!
//! Two families, selected by the form the NTT outputs are in:
//!
//! * **plain** (`ntt_bin_batch32`, `ntt_gen_batch32`): `v[j] = a(psi^u_j)`. A slot product needs
//!   `mont(a, b)` *and* a second multiplication by `R^2 = 2^32 mod q` to cancel the Montgomery
//!   factor — 8 multiply-port uops per slot for batch x batch (`mul_batch_batch`).
//! * **Montgomery** (`ntt_bin_batch32_mont`, `ntt_gen_batch32_mont`): `v[j] = R a(psi^u_j)` with
//!   R = 2^16 mod q. Then one signed Montgomery multiplication is already the answer,
//!   `mont(Ra, Rb) = R a b`: 4 multiply-port uops per slot for batch x batch
//!   (`mul_batch_batch_mont`), 3 for batch x element (`mul_batch_element` with
//!   `MontElement::new_mont`, whose companion is precomputed). The result is again Montgomery
//!   form, so products chain, and the single accumulated `R^-1` is undone by whoever leaves the
//!   NTT domain (`scalar::intt_mont`, together with the 1/648).
use crate::params::{center, Params, N};
use crate::types::{Batch32, Representation, RingElement};
use core::arch::x86_64::*;

/// Lane-wise signed Montgomery product a * b * 2^-16 mod q for arbitrary i16 lanes a, b: result in
/// (-q, q). (mullo, mulhi, mullo, mulhi, sub = 4 multiply uops; the twiddle form with a
/// precomputed b*qinv needs only 3.)
#[inline(always)]
pub unsafe fn mont_mul_epi16<const Q: u16>(a: __m512i, b: __m512i) -> __m512i {
    let q = _mm512_set1_epi16(Q as i16);
    let qinv = _mm512_set1_epi16(Params::<Q>::QINV as i16);
    let lo = _mm512_mullo_epi16(a, b);
    let m = _mm512_mullo_epi16(lo, qinv);
    let hi = _mm512_mulhi_epi16(a, b);
    let t = _mm512_mulhi_epi16(m, q);
    _mm512_sub_epi16(hi, t)
}

/// out[j][p] = a[j][p] * b[j][p] mod q, exactly (the Montgomery factor is removed by a second
/// multiplication with R^2 = 2^32 mod q). Inputs: any i16 lanes; output in (-q, q).
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn mul_batch_batch<const Q: u16>(a: &Batch32, b: &Batch32, out: &mut Batch32) {
    debug_assert_eq!(a.representation, Representation::Ntt);
    debug_assert_eq!(b.representation, Representation::Ntt);
    let r2 = _mm512_set1_epi16(Params::<Q>::to_mont(Params::<Q>::R));
    for j in 0..N {
        let x = _mm512_load_si512(a.v[j].as_ptr() as *const __m512i);
        let y = _mm512_load_si512(b.v[j].as_ptr() as *const __m512i);
        let p = mont_mul_epi16::<Q>(mont_mul_epi16::<Q>(x, y), r2);
        _mm512_store_si512(out.v[j].as_mut_ptr() as *mut __m512i, p);
    }
    out.representation = Representation::Ntt;
}

/// out[j][p] = a[j][p] * b[j][p] * 2^-16 mod q for **Montgomery-form** inputs: with
/// `a = R x`, `b = R y` the result is `R x y`, again Montgomery form, in one signed Montgomery
/// multiplication (4 multiply-port uops per slot instead of the 8 of `mul_batch_batch`).
/// Output in (-q, q).
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn mul_batch_batch_mont<const Q: u16>(a: &Batch32, b: &Batch32, out: &mut Batch32) {
    debug_assert_eq!(a.representation, Representation::Ntt);
    debug_assert_eq!(b.representation, Representation::Ntt);
    for j in 0..N {
        let x = _mm512_load_si512(a.v[j].as_ptr() as *const __m512i);
        let y = _mm512_load_si512(b.v[j].as_ptr() as *const __m512i);
        _mm512_store_si512(out.v[j].as_mut_ptr() as *mut __m512i, mont_mul_epi16::<Q>(x, y));
    }
    out.representation = Representation::Ntt;
}

/// A single NTT-domain element premultiplied by 2^16 (Montgomery form) with its q^-1 companion,
/// so that multiplying a batch by it costs 3 multiply uops per slot and yields the exact product.
/// Each i16 is stored duplicated in a u32 so that the broadcast is a plain `vpbroadcastd` load
/// (a `vpbroadcastw` from memory would cost a shuffle-port uop per slot).
pub struct MontElement {
    pub w: [u32; N],
    pub w_pre: [u32; N],
}

impl MontElement {
    pub fn new<const Q: u16>(e: &RingElement) -> Self {
        debug_assert_eq!(e.representation, Representation::Ntt);
        let mut w = [0u32; N];
        let mut w_pre = [0u32; N];
        for j in 0..N {
            let x = (e.v[j] as i32).rem_euclid(Q as i32) as u16;
            let wm = Params::<Q>::to_mont(x);
            let wp = Params::<Q>::mont_pre(wm);
            w[j] = (wm as u16 as u32) * 0x0001_0001;
            w_pre[j] = (wp as u16 as u32) * 0x0001_0001;
        }
        MontElement { w, w_pre }
    }

    /// Same, for an element that is **already** in Montgomery form (`e.v[j] = R x_j`, the output
    /// of `ntt_bin_batch32_mont` / `ntt_gen_batch32_mont`): the twiddle *is* `e` itself, centered,
    /// so no multiplication by R is needed to build it. `mul_batch_element` then maps a
    /// Montgomery-form batch `R a` to `R a x` — Montgomery form again, 3 multiply-port uops per
    /// slot.
    pub fn new_mont<const Q: u16>(e: &RingElement) -> Self {
        debug_assert_eq!(e.representation, Representation::Ntt);
        let mut w = [0u32; N];
        let mut w_pre = [0u32; N];
        for j in 0..N {
            let wm = center((e.v[j] as i32).rem_euclid(Q as i32) as u64, Q as u64);
            let wp = Params::<Q>::mont_pre(wm);
            w[j] = (wm as u16 as u32) * 0x0001_0001;
            w_pre[j] = (wp as u16 as u32) * 0x0001_0001;
        }
        MontElement { w, w_pre }
    }
}

/// out[j][p] = a[j][p] * e[j] mod q, output in (-q, q).
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn mul_batch_element<const Q: u16>(a: &Batch32, e: &MontElement, out: &mut Batch32) {
    debug_assert_eq!(a.representation, Representation::Ntt);
    let q = _mm512_set1_epi16(Q as i16);
    for j in 0..N {
        let x = _mm512_load_si512(a.v[j].as_ptr() as *const __m512i);
        let w = _mm512_set1_epi32(e.w[j] as i32);
        let wp = _mm512_set1_epi32(e.w_pre[j] as i32);
        let m = _mm512_mullo_epi16(x, wp);
        let hi = _mm512_mulhi_epi16(x, w);
        let t = _mm512_mulhi_epi16(m, q);
        _mm512_store_si512(out.v[j].as_mut_ptr() as *mut __m512i, _mm512_sub_epi16(hi, t));
    }
    out.representation = Representation::Ntt;
}

/// `mul_batch_element` for Montgomery-form inputs: `a[j][p] = R x`, `e` built with
/// [`MontElement::new_mont`] from a Montgomery-form element `R y`, result `R x y` in (-q, q).
/// It is bit for bit the same kernel as [`mul_batch_element`] (3 multiply-port uops per slot);
/// only the meaning of the operands differs, and it is named separately so call sites document
/// which form they are in.
///
/// # Safety
/// See [`mul_batch_element`].
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn mul_batch_element_mont<const Q: u16>(
    a: &Batch32,
    e: &MontElement,
    out: &mut Batch32,
) {
    mul_batch_element::<Q>(a, e, out)
}
