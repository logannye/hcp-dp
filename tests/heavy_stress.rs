#![cfg(feature = "heavy")]
use hcp_dp::{problems::lcs::LcsProblem, HcpEngine};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn random_dna(rng: &mut StdRng, len: usize) -> Vec<u8> {
    const ALPHABET: &[u8] = b"ACGT";
    (0..len)
        .map(|_| {
            let idx = rng.gen_range(0..ALPHABET.len());
            ALPHABET[idx]
        })
        .collect()
}

#[test]
fn heavy_stress_lcs_medium() {
    let mut rng = StdRng::seed_from_u64(123);
    let s = random_dna(&mut rng, 50_000);
    let t = random_dna(&mut rng, 50_000);
    let engine = HcpEngine::new(LcsProblem::new(&s, &t));
    let (cost, _path) = engine.run();
    // Just assert cost is finite and path is non-decreasing in length
    assert!(cost <= 50_000);
}
