use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use hcp_dp::problems::viterbi::{Hmm, ViterbiProblem};
use hcp_dp::HcpEngine;
use rand::{rngs::StdRng, Rng, SeedableRng};
use sysinfo::{get_current_pid, ProcessRefreshKind, System};

fn random_obs(rng: &mut StdRng, len: usize, vocab: usize) -> Vec<usize> {
    (0..len).map(|_| rng.gen_range(0..vocab)).collect()
}

fn demo_hmm() -> Hmm {
    Hmm {
        n_states: 2,
        log_pi: vec![(0.5f64).ln(), (0.5f64).ln()],
        log_a: vec![
            vec![(0.9f64).ln(), (0.1f64).ln()],
            vec![(0.2f64).ln(), (0.8f64).ln()],
        ],
        log_b: vec![
            vec![(0.8f64).ln(), (0.2f64).ln()],
            vec![(0.3f64).ln(), (0.7f64).ln()],
        ],
    }
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

fn bench_viterbi_perf(c: &mut Criterion) {
    let mut group = c.benchmark_group("viterbi_perf_height_compressed");
    for &len in &[10_000usize, 50_000] {
        group.bench_function(format!("viterbi_len_{len}"), |b| {
            b.iter_batched(
                || {
                    let mut rng = StdRng::seed_from_u64(44);
                    random_obs(&mut rng, len, 2)
                },
                |obs| {
                    let before = rss_kib();
                    let problem = ViterbiProblem::new(demo_hmm(), obs);
                    let engine = HcpEngine::new(problem);
                    let (logp, _path) = engine.run();
                    let after = rss_kib();
                    criterion::black_box(logp);
                    eprintln!(
                        "RSS KiB delta (viterbi {len}): {}",
                        after.saturating_sub(before)
                    );
                },
                BatchSize::PerIteration,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_viterbi_perf);
criterion_main!(benches);
