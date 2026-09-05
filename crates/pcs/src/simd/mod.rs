//! AVX-512 kernels. Each module is self-contained: a per-batch forward NTT over a batch of 32
//! polynomials in the vertical layout, plus what the commitment needs to consume it block by
//! block. All kernels assume the host has AVX-512 F/BW/VBMI/VBMI2/VNNI/GFNI (the target machine
//! is an i7-11850H, Tiger Lake); they are compiled with `-C target-cpu=native` (see
//! `.cargo/config.toml`).

pub mod bd;
pub mod commit;
pub mod norm;
pub mod ntt;
pub mod slots;
pub mod transpose32;
pub mod transpose_f162;
