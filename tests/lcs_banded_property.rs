use hcp_dp::{
    problems::{lcs::LcsProblem, lcs_banded::LcsBandedProblem},
    HcpEngine,
};
use proptest::prelude::*;

fn banded_len(s: &[u8], t: &[u8], band: usize) -> u32 {
    let (len, _) = HcpEngine::new(LcsBandedProblem::new(s, t, band)).run();
    len
}

fn full_len(s: &[u8], t: &[u8]) -> u32 {
    let (len, _) = HcpEngine::new(LcsProblem::new(s, t)).run();
    len
}

proptest! {
    #[test]
    fn band_matches_full_when_wide(a in "[ACGT]{0,10}", b in "[ACGT]{0,10}", slack in 0usize..3) {
        let s = a.as_bytes();
        let t = b.as_bytes();
        let band = (s.len().max(t.len()) + slack).max(1);
        prop_assert_eq!(banded_len(s, t, band), full_len(s, t));
    }

    #[test]
    fn banded_never_exceeds_full(a in "[ACGT]{0,12}", b in "[ACGT]{0,12}", band in 0usize..6) {
        let s = a.as_bytes();
        let t = b.as_bytes();
        prop_assert!(banded_len(s, t, band) <= full_len(s, t));
    }
}

#[cfg(feature = "heavy")]
#[test]
fn heavy_near_diagonal_stress() {
    fn make_seq(len: usize, drift: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(len);
        for i in 0..len {
            let ch = match (i / drift) % 4 {
                0 => b'A',
                1 => b'C',
                2 => b'G',
                _ => b'T',
            };
            v.push(ch);
        }
        v
    }
    let len = 30_000;
    let drift = 200;
    let s = make_seq(len, drift);
    let mut t = s.clone();
    // introduce small drift near diagonal
    for i in (0..len).step_by(drift * 5) {
        if i < len {
            t[i] = b'G';
        }
    }
    let band = 256;
    let band_len = banded_len(&s, &t, band);
    let expected_min = (len - len / (drift * 5) - 500) as u32;
    assert!(
        band_len >= expected_min,
        "banded alignment degraded too much: {band_len} < {expected_min}"
    );
    assert!(band_len <= len as u32);
}
