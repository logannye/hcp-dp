use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use hcp_dp::{
    problems::{
        nw_affine::NwAffineProblem, nw_align::NwProblem, smith_waterman::SmithWatermanProblem,
    },
    HcpEngine,
};
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
        p.memory()
    } else {
        0
    }
}

fn bench_nw_perf(c: &mut Criterion) {
    let mut group = c.benchmark_group("nw_perf_height_compressed");
    for &len in &[1_000usize, 5_000] {
        group.bench_function(format!("nw_len_{len}"), |b| {
            b.iter_batched(
                || {
                    let mut rng = StdRng::seed_from_u64(43);
                    let s = random_dna(&mut rng, len);
                    let t = random_dna(&mut rng, len);
                    (s, t)
                },
                |(s, t)| {
                    let before = rss_kib();
                    let problem = NwProblem::new(&s, &t, 1, 1, -1);
                    let engine = HcpEngine::new(problem);
                    let (score, _path) = engine.run();
                    let after = rss_kib();
                    criterion::black_box(score);
                    eprintln!("RSS KiB delta (nw {len}): {}", after.saturating_sub(before));
                },
                BatchSize::PerIteration,
            )
        });
    }
    group.finish();
}

fn bench_affine_nw_perf(c: &mut Criterion) {
    let mut group = c.benchmark_group("nw_affine_perf_height_compressed");
    for &len in &[500usize, 1_000] {
        group.bench_function(format!("nw_affine_len_{len}"), |b| {
            b.iter_batched(
                || {
                    let mut rng = StdRng::seed_from_u64(44);
                    let s = random_dna(&mut rng, len);
                    let t = random_dna(&mut rng, len);
                    (s, t)
                },
                |(s, t)| {
                    let before = rss_kib();
                    let problem = NwAffineProblem::new(&s, &t, 2, 1, -3, -1);
                    let engine = HcpEngine::new(problem);
                    let (score, _path) = engine.run();
                    let after = rss_kib();
                    criterion::black_box(score);
                    eprintln!(
                        "RSS KiB delta (affine nw {len}): {}",
                        after.saturating_sub(before)
                    );
                },
                BatchSize::PerIteration,
            )
        });
    }
    group.finish();
}

fn bench_smith_waterman_perf(c: &mut Criterion) {
    let mut group = c.benchmark_group("smith_waterman_perf_height_compressed");
    for &len in &[500usize, 1_000] {
        group.bench_function(format!("smith_waterman_len_{len}"), |b| {
            b.iter_batched(
                || {
                    let mut rng = StdRng::seed_from_u64(45);
                    let s = random_dna(&mut rng, len);
                    let t = random_dna(&mut rng, len);
                    (s, t)
                },
                |(s, t)| {
                    let before = rss_kib();
                    let problem = SmithWatermanProblem::new(&s, &t, 2, 1, -2);
                    let engine = HcpEngine::new(problem);
                    let (score, _path) = engine.run();
                    let after = rss_kib();
                    criterion::black_box(score);
                    eprintln!(
                        "RSS KiB delta (smith-waterman {len}): {}",
                        after.saturating_sub(before)
                    );
                },
                BatchSize::PerIteration,
            )
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_nw_perf,
    bench_smith_waterman_perf,
    bench_affine_nw_perf
);
criterion_main!(benches);
