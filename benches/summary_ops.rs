use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hcp_dp::{
    problems::{
        edit_distance::EditDistanceProblem, lcs::LcsProblem, nw_affine::NwAffineProblem,
        nw_align::NwProblem, semiglobal::SemiGlobalProblem, smith_waterman::SmithWatermanProblem,
    },
    HcpProblem, SummaryApply,
};
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

fn bench_lcs_summary(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0x5155AA55);
    let len = 1024;
    let s = random_dna(&mut rng, len);
    let t = random_dna(&mut rng, len);
    let problem = LcsProblem::new(&s, &t);
    let frontier = problem.init_frontier();
    let sigma_left = problem.summarize_interval(0, len / 2);
    let sigma_right = problem.summarize_interval(len / 2, len);

    let mut group = c.benchmark_group("summary_ops_lcs");
    group.bench_function("apply", |b| {
        b.iter(|| black_box(sigma_left.apply(black_box(&frontier))));
    });
    group.bench_function("merge", |b| {
        b.iter(|| black_box(problem.merge_summary(&sigma_left, &sigma_right)));
    });
    group.finish();
}

fn bench_nw_summary(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0xA1B2C3D4);
    let len = 768;
    let s = random_dna(&mut rng, len);
    let t = random_dna(&mut rng, len);
    let problem = NwProblem::new(&s, &t, 2, 1, -2);
    let frontier = problem.init_frontier();
    let sigma_left = problem.summarize_interval(0, len / 2);
    let sigma_right = problem.summarize_interval(len / 2, len);

    let mut group = c.benchmark_group("summary_ops_nw");
    group.bench_function("apply", |b| {
        b.iter(|| black_box(sigma_left.apply(black_box(&frontier))));
    });
    group.bench_function("merge", |b| {
        b.iter(|| black_box(problem.merge_summary(&sigma_left, &sigma_right)));
    });
    group.finish();
}

fn bench_affine_nw_summary(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0xA4419E);
    let len = 512;
    let s = random_dna(&mut rng, len);
    let t = random_dna(&mut rng, len);
    let problem = NwAffineProblem::new(&s, &t, 2, 1, -3, -1);
    let frontier = problem.init_frontier();
    let sigma_left = problem.summarize_interval(0, len / 2);
    let sigma_right = problem.summarize_interval(len / 2, len);

    let mut group = c.benchmark_group("summary_ops_nw_affine");
    group.bench_function("apply", |b| {
        b.iter(|| black_box(sigma_left.apply(black_box(&frontier))));
    });
    group.bench_function("merge", |b| {
        b.iter(|| black_box(problem.merge_summary(&sigma_left, &sigma_right)));
    });
    group.finish();
}

fn bench_smith_waterman_summary(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0x5A117E);
    let len = 768;
    let s = random_dna(&mut rng, len);
    let t = random_dna(&mut rng, len);
    let problem = SmithWatermanProblem::new(&s, &t, 2, 1, -2);
    let frontier = problem.init_frontier();
    let sigma_left = problem.summarize_interval(0, len / 2);
    let sigma_right = problem.summarize_interval(len / 2, len);

    let mut group = c.benchmark_group("summary_ops_smith_waterman");
    group.bench_function("apply", |b| {
        b.iter(|| black_box(sigma_left.apply(black_box(&frontier))));
    });
    group.bench_function("merge", |b| {
        b.iter(|| black_box(problem.merge_summary(&sigma_left, &sigma_right)));
    });
    group.finish();
}

fn bench_edit_distance_summary(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0xED17);
    let len = 768;
    let s = random_dna(&mut rng, len);
    let t = random_dna(&mut rng, len);
    let problem = EditDistanceProblem::new(&s, &t);
    let frontier = problem.init_frontier();
    let sigma_left = problem.summarize_interval(0, len / 2);
    let sigma_right = problem.summarize_interval(len / 2, len);

    let mut group = c.benchmark_group("summary_ops_edit_distance");
    group.bench_function("apply", |b| {
        b.iter(|| black_box(sigma_left.apply(black_box(&frontier))));
    });
    group.bench_function("merge", |b| {
        b.iter(|| black_box(problem.merge_summary(&sigma_left, &sigma_right)));
    });
    group.finish();
}

fn bench_semiglobal_summary(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0x5E61910B);
    let len = 768;
    let s = random_dna(&mut rng, len);
    let t = random_dna(&mut rng, len + len / 4);
    let problem = SemiGlobalProblem::new(&s, &t, 2, 1, -2);
    let frontier = problem.init_frontier();
    let sigma_left = problem.summarize_interval(0, len / 2);
    let sigma_right = problem.summarize_interval(len / 2, len);

    let mut group = c.benchmark_group("summary_ops_semiglobal");
    group.bench_function("apply", |b| {
        b.iter(|| black_box(sigma_left.apply(black_box(&frontier))));
    });
    group.bench_function("merge", |b| {
        b.iter(|| black_box(problem.merge_summary(&sigma_left, &sigma_right)));
    });
    group.finish();
}

fn bench_summary_ops(c: &mut Criterion) {
    bench_lcs_summary(c);
    bench_nw_summary(c);
    bench_smith_waterman_summary(c);
    bench_affine_nw_summary(c);
    bench_edit_distance_summary(c);
    bench_semiglobal_summary(c);
}

criterion_group!(benches, bench_summary_ops);
criterion_main!(benches);
