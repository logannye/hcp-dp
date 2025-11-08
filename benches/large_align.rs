//! Benchmark: large-scale LCS / alignment with height-compressed DP.
//!
//! Run with:
//! `cargo bench`
//!
//! This is mainly to sanity-check overheads and verify that we can handle
//! large instances without materializing quadratic DP tables.

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
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

fn bench_lcs_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("lcs_large_height_compressed");

    // Example sizes; tune as needed for your machine.
    for &len in &[5_000usize, 10_000, 20_000] {
        group.bench_function(format!("lcs_len_{len}"), |b| {
            b.iter_batched(
                || {
                    let mut rng = StdRng::seed_from_u64(42);
                    let s = random_dna(&mut rng, len);
                    let t = random_dna(&mut rng, len);
                    (s, t)
                },
                |(s, t)| {
                    let problem = LcsProblem::new(&s, &t);
                    let engine = HcpEngine::new(problem);
                    let (cost, _path) = engine.run();
                    criterion::black_box(cost);
                },
                BatchSize::PerIteration,
            )
        });
    }

    group.finish();
}

criterion_group!(benches, bench_lcs_large);
criterion_main!(benches);
