use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hcp_dp::{
    records_from_json, solve, verify, AlignmentMode, ProblemSpec, ScoringScheme, SolveOptions,
    SolveRequest, VerificationLevel, VerificationRequest, DEFAULT_MAX_CELLS,
};

fn request(len: usize) -> SolveRequest {
    let query: Vec<u8> = (0..len).map(|i| b"ACGT"[i % 4]).collect();
    let mut target = query.clone();
    for i in (31..len).step_by(97) {
        target[i] = b"TGCA"[i % 4];
    }
    SolveRequest {
        pair_index: 0,
        query_id: "query".to_string(),
        target_id: "target".to_string(),
        query,
        target,
        problem: ProblemSpec {
            mode: AlignmentMode::EditDistance,
            scoring: ScoringScheme::EditDistance,
        },
        options: SolveOptions::default(),
    }
}

fn oracle_benches(c: &mut Criterion) {
    let solve_request = request(1_000);
    c.bench_function("oracle_solve/edit_distance_len_1000", |b| {
        b.iter(|| solve(black_box(solve_request.clone())).expect("solve benchmark"))
    });

    let record = solve(solve_request.clone()).expect("verification fixture");
    let verify_request = VerificationRequest {
        record: record.clone(),
        query: solve_request.query.clone(),
        target: solve_request.target.clone(),
        level: VerificationLevel::Optimal,
        max_cells: DEFAULT_MAX_CELLS,
    };
    c.bench_function("oracle_verify/edit_distance_len_1000", |b| {
        b.iter(|| verify(black_box(verify_request.clone())))
    });

    let json = serde_json::to_string(&record).expect("JSON fixture");
    c.bench_function("oracle_parse/json_alignment_record", |b| {
        b.iter(|| records_from_json(black_box(&json)).expect("parse benchmark"))
    });
}

criterion_group!(benches, oracle_benches);
criterion_main!(benches);
