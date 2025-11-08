use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use hcp_dp::{problems::lcs::LcsProblem, HcpEngine};
use rand::{rngs::StdRng, Rng, SeedableRng};
use sysinfo::{get_current_pid, ProcessRefreshKind, System};

fn random_dna(rng: &mut StdRng, len: usize) -> Vec<u8> {
    const ALPHABET: &[u8] = b"ACGT";
    (0..len)
        .map(|_| {
            let idx = rng.gen_range(0..ALPHABET.len());
            ALPHABET[idx]
        })
        .collect()
}

fn rss_kib() -> u64 {
    let mut sys = System::new();
    sys.refresh_processes_specifics(ProcessRefreshKind::new());
    if let Some(p) = sys.process(get_current_pid().unwrap()) {
        p.memory() // KiB on supported platforms
    } else {
        0
    }
}

fn bench_lcs_perf(c: &mut Criterion) {
    let mut group = c.benchmark_group("lcs_perf_height_compressed");
    for &len in &[1_000usize, 5_000, 10_000] {
        group.bench_function(format!("lcs_len_{len}"), |b| {
            b.iter_batched(
                || {
                    let mut rng = StdRng::seed_from_u64(42);
                    let s = random_dna(&mut rng, len);
                    let t = random_dna(&mut rng, len);
                    (s, t)
                },
                |(s, t)| {
                    let before = rss_kib();
                    let problem = LcsProblem::new(&s, &t);
                    let engine = HcpEngine::new(problem);
                    let (cost, _path) = engine.run();
                    let after = rss_kib();
                    criterion::black_box(cost);
                    // record memory delta to stderr to avoid criterion noise
                    eprintln!(
                        "RSS KiB delta (lcs {len}): {}",
                        after.saturating_sub(before)
                    );
                },
                BatchSize::PerIteration,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_lcs_perf);
criterion_main!(benches);
