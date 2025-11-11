use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hcp_dp::{
    problems::{
        dag_sp::DagLayered,
        lcs::LcsProblem,
        nw_align::NwProblem,
        viterbi::{Hmm, ViterbiProblem},
    },
    traits::SummaryApply,
    HcpProblem,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::Arc;

fn leak_vec(data: Vec<u8>) -> &'static [u8] {
    Box::leak(data.into_boxed_slice())
}

fn random_dna(rng: &mut StdRng, len: usize) -> Vec<u8> {
    const ALPHABET: &[u8] = b"ACGT";
    (0..len)
        .map(|_| {
            let idx = rng.gen_range(0..ALPHABET.len());
            ALPHABET[idx]
        })
        .collect()
}

fn bench_lcs_summary(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0x5155AA55);
    let len = 2048;
    let s = leak_vec(random_dna(&mut rng, len));
    let t = leak_vec(random_dna(&mut rng, len));
    let problem = Arc::new(LcsProblem::new(s, t));

    let frontier_a = problem.init_frontier();
    let mid = len / 2;
    let (frontier_b, sigma_left) = problem.summarize_block(0, mid, &frontier_a);
    let (_, sigma_right) = problem.summarize_block(mid, len, &frontier_b);

    let frontier_for_apply = frontier_a.clone();
    let sigma_for_apply = sigma_left.clone();
    let sigma_left_for_merge = sigma_left.clone();
    let sigma_right_for_merge = sigma_right.clone();
    let problem_for_merge = problem.clone();

    let mut group = c.benchmark_group("summary_ops_lcs");
    group.bench_function("apply", move |b| {
        b.iter(|| {
            let res = sigma_for_apply.apply(black_box(&frontier_for_apply));
            black_box(res);
        });
    });
    group.bench_function("merge", move |b| {
        b.iter(|| {
            let merged =
                problem_for_merge.merge_summary(&sigma_left_for_merge, &sigma_right_for_merge);
            black_box(merged);
        });
    });
    group.finish();
}

fn bench_nw_summary(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0xA1B2C3D4);
    let len = 1536;
    let s = leak_vec(random_dna(&mut rng, len));
    let t = leak_vec(random_dna(&mut rng, len));
    let ms = 2;
    let mm = 1;
    let gp = -1;
    let problem = Arc::new(NwProblem::new(s, t, ms, mm, gp));

    let frontier_a = problem.init_frontier();
    let mid = len / 2;
    let (frontier_b, sigma_left) = problem.summarize_block(0, mid, &frontier_a);
    let (_, sigma_right) = problem.summarize_block(mid, len, &frontier_b);

    let frontier_for_apply = frontier_a.clone();
    let sigma_for_apply = sigma_left.clone();
    let sigma_left_for_merge = sigma_left.clone();
    let sigma_right_for_merge = sigma_right.clone();
    let problem_for_merge = problem.clone();

    let mut group = c.benchmark_group("summary_ops_nw");
    group.bench_function("apply", move |b| {
        b.iter(|| {
            let res = sigma_for_apply.apply(black_box(&frontier_for_apply));
            black_box(res);
        });
    });
    group.bench_function("merge", move |b| {
        b.iter(|| {
            let merged =
                problem_for_merge.merge_summary(&sigma_left_for_merge, &sigma_right_for_merge);
            black_box(merged);
        });
    });
    group.finish();
}

fn bench_viterbi_summary(c: &mut Criterion) {
    let hmm = Hmm {
        n_states: 4,
        log_pi: vec![(0.3f64).ln(), (0.3f64).ln(), (0.2f64).ln(), (0.2f64).ln()],
        log_a: vec![
            vec![(0.7f64).ln(), (0.1f64).ln(), (0.1f64).ln(), (0.1f64).ln()],
            vec![(0.2f64).ln(), (0.5f64).ln(), (0.2f64).ln(), (0.1f64).ln()],
            vec![
                (0.25f64).ln(),
                (0.25f64).ln(),
                (0.25f64).ln(),
                (0.25f64).ln(),
            ],
            vec![(0.3f64).ln(), (0.2f64).ln(), (0.3f64).ln(), (0.2f64).ln()],
        ],
        log_b: vec![
            vec![(0.6f64).ln(), (0.4f64).ln()],
            vec![(0.3f64).ln(), (0.7f64).ln()],
            vec![(0.5f64).ln(), (0.5f64).ln()],
            vec![(0.4f64).ln(), (0.6f64).ln()],
        ],
    };
    let obs: Vec<usize> = (0..512).map(|i| (i % 2) as usize).collect();
    let problem = Arc::new(ViterbiProblem::new(hmm, obs));

    let frontier_a = problem.init_frontier();
    let mid = problem.num_layers() / 2;
    let (frontier_b, sigma_left) = problem.summarize_block(0, mid, &frontier_a);
    let (_, sigma_right) = problem.summarize_block(mid, problem.num_layers(), &frontier_b);

    let frontier_for_apply = frontier_a.clone();
    let sigma_for_apply = sigma_left.clone();
    let sigma_left_for_merge = sigma_left.clone();
    let sigma_right_for_merge = sigma_right.clone();
    let problem_for_merge = problem.clone();

    let mut group = c.benchmark_group("summary_ops_viterbi");
    group.bench_function("apply", move |b| {
        b.iter(|| {
            let res = sigma_for_apply.apply(black_box(&frontier_for_apply));
            black_box(res);
        });
    });
    group.bench_function("merge", move |b| {
        b.iter(|| {
            let merged =
                problem_for_merge.merge_summary(&sigma_left_for_merge, &sigma_right_for_merge);
            black_box(merged);
        });
    });
    group.finish();
}

fn bench_dag_summary(c: &mut Criterion) {
    let widths = vec![4, 4, 4, 4];
    let adjacency: Vec<Vec<Vec<(usize, i64)>>> = (0..widths.len() - 1)
        .map(|layer| {
            (0..widths[layer])
                .map(|u| {
                    (0..widths[layer + 1])
                        .map(|v| {
                            let weight = ((layer + u + v) % 7 + 1) as i64;
                            (v, weight)
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    let problem = Arc::new(DagLayered::new(adjacency, widths));

    let frontier_a = problem.init_frontier();
    let mid = problem.num_layers() / 2;
    let (frontier_b, sigma_left) = problem.summarize_block(0, mid, &frontier_a);
    let (_, sigma_right) = problem.summarize_block(mid, problem.num_layers(), &frontier_b);

    let frontier_for_apply = frontier_a.clone();
    let sigma_for_apply = sigma_left.clone();
    let sigma_left_for_merge = sigma_left.clone();
    let sigma_right_for_merge = sigma_right.clone();
    let problem_for_merge = problem.clone();

    let mut group = c.benchmark_group("summary_ops_dag");
    group.bench_function("apply", move |b| {
        b.iter(|| {
            let res = sigma_for_apply.apply(black_box(&frontier_for_apply));
            black_box(res);
        });
    });
    group.bench_function("merge", move |b| {
        b.iter(|| {
            let merged =
                problem_for_merge.merge_summary(&sigma_left_for_merge, &sigma_right_for_merge);
            black_box(merged);
        });
    });
    group.finish();
}

fn bench_summary_ops(c: &mut Criterion) {
    bench_lcs_summary(c);
    bench_nw_summary(c);
    bench_viterbi_summary(c);
    bench_dag_summary(c);
}

criterion_group!(benches, bench_summary_ops);
criterion_main!(benches);
