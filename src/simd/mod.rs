//! AVX-512 kernels. Each variant is a self-contained module exposing the same shape of API:
//! a per-batch forward NTT (batch of 32 polynomials for the vertical layouts, 4 for the horizontal
//! one) plus a driver over many polynomials. All kernels assume the host has AVX-512 F/BW/VBMI/
//! VBMI2/VNNI/GFNI (the target machine is an i7-11850H, Tiger Lake); they are compiled with
//! `-C target-cpu=native` (see `.cargo/config.toml`).

pub mod commit;
pub mod commit_h;
pub mod horizontal_gen;
pub mod ntt_f162;
pub mod pointwise;
pub mod transpose;
pub mod transpose_f162;
pub mod vertical_bin;
pub mod vertical_bin_asm;
pub mod vertical_bin_quad;
pub mod vertical_gen;
pub mod vertical_gen_quad;
