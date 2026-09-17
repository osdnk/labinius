pub mod bin_asm;
pub mod bin_large;
pub mod bin_quad;
pub mod bin_small;
pub mod gen_large;
pub mod gen_quad;
pub mod gen_small;

macro_rules! r3_twiddles {
    ($n:literal, $pair:path, $zetas:expr) => {{
        let mut t = [[0u32; 4]; $n];
        let mut i = 0;
        while i < $n {
            t[i] = $pair($zetas[i]);
            i += 1;
        }
        t
    }};
}
pub(crate) use r3_twiddles;

macro_rules! bar_switch {
    ($f:ident, $bar:expr, $($arg:expr),* $(,)?) => {
        if $bar {
            $f::<true>($($arg),*)
        } else {
            $f::<false>($($arg),*)
        }
    };
}
pub(crate) use bar_switch;
