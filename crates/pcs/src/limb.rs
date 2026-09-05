use crate::params::quadratic_slots;
use crate::simd::vertical_bin_large::is_large;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Class {
    SplitSmall,
    SplitLarge,
    Quad,
}

pub const fn class(q: u16) -> Class {
    if quadratic_slots(q) {
        Class::Quad
    } else if is_large(q) {
        Class::SplitLarge
    } else {
        Class::SplitSmall
    }
}

pub fn is_quad(q: u16) -> bool {
    matches!(class(q), Class::Quad)
}

macro_rules! limb_classes {
    (items [$($m:tt)*]) => {
        $($m)*! { [3889, 9721] [17497, 19441] [2917, 4861, 12637] }
    };
    ([$($m:tt)*] $($rest:tt)*) => {
        $($m)*!([3889, 9721] [17497, 19441] [2917, 4861, 12637] $($rest)*)
    };
}
pub(crate) use limb_classes;

macro_rules! limb_consts {
    ([$($s:literal),*] [$($l:literal),*] [$($d:literal),*]) => {
        pub const LIMBS: usize = [$($s,)* $($l,)* $($d,)*].len();
        pub const PRIMES: [u16; LIMBS] = [$($s,)* $($l,)* $($d,)*];

        const _: () = {
            let mut i = 0;
            $(
                assert!(matches!(class($s), Class::SplitSmall));
                i += 1;
            )*
            $(
                assert!(matches!(class($l), Class::SplitLarge));
                i += 1;
            )*
            $(
                assert!(matches!(class($d), Class::Quad));
                i += 1;
            )*
            assert!(i == LIMBS);
        };
    };
}
limb_classes!(items [limb_consts]);

macro_rules! limb_match {
    ([$($s:literal),*] [$($l:literal),*] [$($d:literal),*] $q:expr,
     |$qa:ident| $a:expr, |$qb:ident| $b:expr, |$qc:ident| $c:expr) => {
        match $q {
            $($s => { #[allow(dead_code)] const $qa: u16 = $s; $a })*
            $($l => { #[allow(dead_code)] const $qb: u16 = $l; $b })*
            $($d => { #[allow(dead_code)] const $qc: u16 = $d; $c })*
            _ => unreachable!("no limb with q = {}", $q),
        }
    };
}
pub(crate) use limb_match;

macro_rules! dispatch_limb {
    ($q:expr, |$s:ident| $body:expr $(,)?) => {
        $crate::limb::limb_classes!([$crate::limb::limb_match] $q, |$s| $body, |$s| $body, |$s| $body)
    };
    ($q:expr, split |$s:ident| $sb:expr, quad |$d:ident| $db:expr $(,)?) => {
        $crate::limb::limb_classes!([$crate::limb::limb_match] $q, |$s| $sb, |$s| $sb, |$d| $db)
    };
    ($q:expr, small |$s:ident| $sb:expr, large |$l:ident| $lb:expr, quad |$d:ident| $db:expr $(,)?) => {
        $crate::limb::limb_classes!([$crate::limb::limb_match] $q, |$s| $sb, |$l| $lb, |$d| $db)
    };
    ($q:expr, split |$s:ident| $sb:expr $(,)?) => {
        $crate::limb::limb_classes!([$crate::limb::limb_match] $q, |$s| $sb, |$s| $sb,
            |$s| unreachable!("q = {} is not a splitting limb", $q))
    };
    ($q:expr, quad |$d:ident| $db:expr $(,)?) => {
        $crate::limb::limb_classes!([$crate::limb::limb_match] $q,
            |$d| unreachable!("q = {} is not a quadratic-slot limb", $q),
            |$d| unreachable!("q = {} is not a quadratic-slot limb", $q), |$d| $db)
    };
}
pub(crate) use dispatch_limb;
