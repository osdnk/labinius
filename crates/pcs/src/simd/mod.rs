//! AVX-512 kernels. Each module is self-contained: a per-batch forward NTT over a batch of 32
//! polynomials in the vertical layout, plus what the commitment needs to consume it block by
//! block. The kernels are gated on AVX-512 F/BW/VL/VBMI/VBMI2/VNNI/GFNI, the feature set of the
//! target machine (an i7-11850H, Tiger Lake), and are compiled with `-C target-cpu=native` (see
//! `.cargo/config.toml`). What they actually emit is narrower: the transforms use F/BW/VL and
//! the VBMI byte permutes (`vpermb`, `vpmultishiftqb`), the commitment F/BW, the `F162`
//! transpose ([`transpose_f162`]) GFNI. VBMI2 and VNNI appear only in the gate: the packed
//! accumulator in [`commit`] replaced the one `vpdpwssd` design (see its header).

pub mod bd;
pub mod commit;
pub mod norm;
pub mod ntt;
pub mod slots;
pub mod transpose32;
pub mod transpose_f162;
