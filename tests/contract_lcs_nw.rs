use hcp_dp::{
    contract::{
        assert_all_summary_laws, assert_engine_paths_for_all_block_sizes, reconstruct_top_split,
    },
    problems::{
        dtw::{DtwProblem, DtwState},
        edit_distance::{EditDistanceProblem, EditDistanceState},
        lcs::{LcsProblem, LcsState},
        nw_affine::{NwAffineProblem, NwAffineState},
        nw_align::{NwProblem, NwState},
        semiglobal::{SemiGlobalCell, SemiGlobalProblem},
        smith_waterman::{SmithWatermanProblem, SwCell},
    },
    HcpEngine,
};
use proptest::prelude::*;

fn assert_lcs_path(problem: &LcsProblem<'_>, cost: u32, path: &[LcsState]) {
    assert_eq!(cost, problem.full_table_len());
    assert_eq!(path.first(), Some(&(0, 0)));
    assert_eq!(path.last(), Some(&(problem.n(), problem.m())));
    assert_eq!(problem.score_path(path), Some(cost));
}

fn assert_nw_path(problem: &NwProblem<'_>, cost: i32, path: &[NwState]) {
    assert_eq!(cost, problem.full_table_score());
    assert_eq!(path.first(), Some(&(0, 0)));
    assert_eq!(path.last(), Some(&(problem.n(), problem.m())));
    assert_eq!(problem.score_path(path), Some(cost));
}

fn assert_affine_path(problem: &NwAffineProblem<'_>, cost: i32, path: &[NwAffineState]) {
    assert_eq!(cost, problem.full_table_score());
    assert_eq!(
        path.first().map(|state| (state.row, state.col)),
        Some((0, 0))
    );
    assert_eq!(
        path.last().map(|state| (state.row, state.col)),
        Some((problem.n(), problem.m()))
    );
    assert_eq!(problem.score_path(path), Some(cost));
}

fn assert_sw_path(problem: &SmithWatermanProblem<'_>, cost: i32, path: &[SwCell]) {
    assert_eq!(cost, problem.full_table_score());
    if cost == 0 {
        assert!(path.is_empty());
    } else {
        let (start, end) = problem
            .full_table_endpoints()
            .expect("positive SW score must have endpoints");
        assert_eq!(path.first(), Some(&start));
        assert_eq!(path.last(), Some(&end));
    }
    assert_eq!(problem.score_path(path), Some(cost));
}

fn assert_edit_path(problem: &EditDistanceProblem<'_>, cost: u32, path: &[EditDistanceState]) {
    assert_eq!(cost, problem.full_table_distance());
    assert_eq!(path.first(), Some(&(0, 0)));
    assert_eq!(path.last(), Some(&(problem.n(), problem.m())));
    assert_eq!(problem.score_path(path), Some(cost));
}

fn assert_dtw_path(problem: &DtwProblem<'_>, cost: u64, path: &[DtwState]) {
    assert_eq!(cost, problem.full_table_cost());
    assert_eq!(path.first(), Some(&(0, 0)));
    assert_eq!(path.last(), Some(&(problem.n(), problem.m())));
    assert_eq!(problem.score_path(path), Some(cost));
}

fn assert_semiglobal_path(problem: &SemiGlobalProblem<'_>, cost: i32, path: &[SemiGlobalCell]) {
    assert_eq!(cost, problem.full_table_score());
    let (target_start, target_end) = problem.full_table_target_interval();
    assert_eq!(
        path.first(),
        Some(&SemiGlobalCell {
            row: 0,
            col: target_start,
        })
    );
    assert_eq!(
        path.last(),
        Some(&SemiGlobalCell {
            row: problem.n(),
            col: target_end,
        })
    );
    assert_eq!(problem.score_path(path), Some(cost));
}

fn assert_lcs_split_contract(problem: &LcsProblem<'_>) {
    if let Some(path) = reconstruct_top_split(problem, problem.n() / 2) {
        assert_lcs_path(problem, problem.full_table_len(), &path);
    }
}

fn assert_nw_split_contract(problem: &NwProblem<'_>) {
    if let Some(path) = reconstruct_top_split(problem, problem.n() / 2) {
        assert_nw_path(problem, problem.full_table_score(), &path);
    }
}

fn assert_affine_split_contract(problem: &NwAffineProblem<'_>) {
    if let Some(path) = reconstruct_top_split(problem, problem.n() / 2) {
        assert_affine_path(problem, problem.full_table_score(), &path);
    }
}

fn assert_sw_split_contract(problem: &SmithWatermanProblem<'_>) {
    if let Some(path) = reconstruct_top_split(problem, problem.n() / 2) {
        assert_sw_path(problem, problem.full_table_score(), &path);
    }
}

fn assert_edit_split_contract(problem: &EditDistanceProblem<'_>) {
    if let Some(path) = reconstruct_top_split(problem, problem.n() / 2) {
        assert_edit_path(problem, problem.full_table_distance(), &path);
    }
}

fn assert_dtw_split_contract(problem: &DtwProblem<'_>) {
    if let Some(path) = reconstruct_top_split(problem, problem.n() / 2) {
        assert_dtw_path(problem, problem.full_table_cost(), &path);
    }
}

fn assert_semiglobal_split_contract(problem: &SemiGlobalProblem<'_>) {
    if let Some(path) = reconstruct_top_split(problem, problem.n() / 2) {
        assert_semiglobal_path(problem, problem.full_table_score(), &path);
    }
}

proptest! {
    #[test]
    fn lcs_contracts_hold(a in "[ACGT]{0,8}", b in "[ACGT]{0,8}") {
        let problem = LcsProblem::new(a.as_bytes(), b.as_bytes());
        assert_all_summary_laws(&problem);
        assert_lcs_split_contract(&problem);
        assert_engine_paths_for_all_block_sizes(problem.clone(), |cost, path| {
            assert_lcs_path(&problem, cost, path);
        });
    }

    #[test]
    fn nw_contracts_hold(a in "[ACGT]{0,7}", b in "[ACGT]{0,7}") {
        let problem = NwProblem::new(a.as_bytes(), b.as_bytes(), 2, 1, -2);
        assert_all_summary_laws(&problem);
        assert_nw_split_contract(&problem);
        assert_engine_paths_for_all_block_sizes(problem.clone(), |cost, path| {
            assert_nw_path(&problem, cost, path);
        });
    }

    #[test]
    fn affine_contracts_hold(a in "[ACGT]{0,6}", b in "[ACGT]{0,6}") {
        let problem = NwAffineProblem::new(a.as_bytes(), b.as_bytes(), 2, 1, -3, -1);
        assert_all_summary_laws(&problem);
        assert_affine_split_contract(&problem);
        assert_engine_paths_for_all_block_sizes(problem.clone(), |cost, path| {
            assert_affine_path(&problem, cost, path);
        });
    }

    #[test]
    fn smith_waterman_contracts_hold(a in "[ACGT]{0,6}", b in "[ACGT]{0,6}") {
        let problem = SmithWatermanProblem::new(a.as_bytes(), b.as_bytes(), 2, 1, -2);
        assert_all_summary_laws(&problem);
        assert_sw_split_contract(&problem);
        assert_engine_paths_for_all_block_sizes(problem.clone(), |cost, path| {
            assert_sw_path(&problem, cost, path);
        });
    }

    #[test]
    fn edit_distance_contracts_hold(a in "[ACGT]{0,7}", b in "[ACGT]{0,7}") {
        let problem = EditDistanceProblem::new(a.as_bytes(), b.as_bytes());
        assert_all_summary_laws(&problem);
        assert_edit_split_contract(&problem);
        assert_engine_paths_for_all_block_sizes(problem.clone(), |cost, path| {
            assert_edit_path(&problem, cost, path);
        });
    }

    #[test]
    fn dtw_contracts_hold(
        a in prop::collection::vec(-5i32..=5, 1..=6),
        b in prop::collection::vec(-5i32..=5, 1..=6)
    ) {
        let problem = DtwProblem::new(&a, &b);
        assert_all_summary_laws(&problem);
        assert_dtw_split_contract(&problem);
        assert_engine_paths_for_all_block_sizes(problem.clone(), |cost, path| {
            assert_dtw_path(&problem, cost, path);
        });
    }

    #[test]
    fn semiglobal_contracts_hold(a in "[ACGT]{0,6}", b in "[ACGT]{0,8}") {
        let problem = SemiGlobalProblem::new(a.as_bytes(), b.as_bytes(), 2, 1, -2);
        assert_all_summary_laws(&problem);
        assert_semiglobal_split_contract(&problem);
        assert_engine_paths_for_all_block_sizes(problem.clone(), |cost, path| {
            assert_semiglobal_path(&problem, cost, path);
        });
    }
}

#[test]
fn audit_regressions_are_fixed() {
    let lcs = LcsProblem::new(b"CCA", b"C");
    let (lcs_cost, lcs_path) = HcpEngine::new(lcs.clone()).run();
    assert_lcs_path(&lcs, lcs_cost, &lcs_path);

    let nw = NwProblem::new(b"CBA", b"ACC", 2, 1, -2);
    let (nw_cost, nw_path) = HcpEngine::new(nw.clone()).run();
    assert_nw_path(&nw, nw_cost, &nw_path);

    let affine = NwAffineProblem::new(b"ACB", b"A", 2, 1, -3, -1);
    let (affine_cost, affine_path) = HcpEngine::new(affine.clone()).run();
    assert_eq!(affine_cost, -3);
    assert_affine_path(&affine, affine_cost, &affine_path);

    let sw = SmithWatermanProblem::new(b"ACACACTA", b"AGCACACA", 2, 1, -2);
    let (sw_cost, sw_path) = HcpEngine::new(sw.clone()).run();
    assert_eq!(sw_cost, 10);
    assert_sw_path(&sw, sw_cost, &sw_path);

    let edit = EditDistanceProblem::new(b"kitten", b"sitting");
    let (edit_cost, edit_path) = HcpEngine::new(edit.clone()).run();
    assert_eq!(edit_cost, 3);
    assert_edit_path(&edit, edit_cost, &edit_path);

    let dtw = DtwProblem::new(&[1, 2, 3, 3, 7], &[1, 3, 4, 7]);
    let (dtw_cost, dtw_path) = HcpEngine::new(dtw.clone()).run();
    assert_dtw_path(&dtw, dtw_cost, &dtw_path);

    let semi = SemiGlobalProblem::new(b"ACGT", b"TTACGTTT", 2, 1, -2);
    let (semi_cost, semi_path) = HcpEngine::new(semi.clone()).run();
    assert_eq!(semi_cost, 8);
    assert_eq!(semi.full_table_target_interval(), (2, 6));
    assert_semiglobal_path(&semi, semi_cost, &semi_path);
}

#[test]
fn affine_edge_cases_hold() {
    let cases: &[(&[u8], &[u8])] = &[
        (b"", b"ABC"),
        (b"ABC", b""),
        (b"AAAAAA", b"A"),
        (b"AC", b"A"),
        (b"GGGG", b"TTTT"),
        (b"AAAA", b"AAAA"),
        (b"ACGTAC", b"AC"),
    ];

    for (s, t) in cases {
        let problem = NwAffineProblem::new(s, t, 2, 1, -3, -1);
        assert_engine_paths_for_all_block_sizes(problem.clone(), |cost, path| {
            assert_affine_path(&problem, cost, path);
        });
    }
}

#[test]
fn edit_distance_edge_cases_hold() {
    let long_insert = [b"ACGT".as_slice(), vec![b'A'; 64].as_slice(), b"ACGT"].concat();
    let long_delete = [b"ACGT".as_slice(), vec![b'C'; 64].as_slice(), b"ACGT"].concat();
    let cases: Vec<(&[u8], &[u8])> = vec![
        (b"", b""),
        (b"", b"ABC"),
        (b"ABC", b""),
        (b"AAAA", b"TTTT"),
        (&long_insert, b"ACGTACGT"),
        (b"ACGTACGT", &long_delete),
        (b"AAAAAAAA", b"AAAA"),
        (b"ATATATAT", b"TATATA"),
        (b"ACGTACGT", b"ACGTTCGT"),
    ];

    for (s, t) in cases {
        let problem = EditDistanceProblem::new(s, t);
        assert_all_summary_laws(&problem);
        assert_edit_split_contract(&problem);
        assert_engine_paths_for_all_block_sizes(problem.clone(), |cost, path| {
            assert_edit_path(&problem, cost, path);
        });
    }
}

#[test]
fn dtw_edge_cases_hold() {
    let long_plateau_a = vec![5; 32];
    let long_plateau_b = vec![5; 8];
    let cases: Vec<(&[i32], &[i32])> = vec![
        (&[7], &[7]),
        (&[1], &[4, 4, 4]),
        (&[4, 4, 4], &[1]),
        (&[1, 2, 3, 4], &[1, 1, 2, 3, 5]),
        (&[-3, -1, 0, 2], &[-2, 0, 1]),
        (&long_plateau_a, &long_plateau_b),
    ];

    for (query, target) in cases {
        let problem = DtwProblem::new(query, target);
        assert_all_summary_laws(&problem);
        assert_dtw_split_contract(&problem);
        assert_engine_paths_for_all_block_sizes(problem.clone(), |cost, path| {
            assert_dtw_path(&problem, cost, path);
        });
    }
}

#[test]
fn smith_waterman_edge_cases_hold() {
    type SwCase<'a> = (&'a [u8], &'a [u8], i32, i32, i32);
    let cases: &[SwCase<'_>] = &[
        (b"", b"ABC", 2, 1, -2),
        (b"ABC", b"", 2, 1, -2),
        (b"AAAA", b"TTTT", 1, 3, -2),
        (b"GGAC", b"TTGGA", 2, 1, -2),
        (b"ACGTAC", b"TAC", 2, 1, -2),
        (b"AAAAAA", b"AAA", 2, 1, -2),
        (b"ACGT", b"ACGT", 2, 1, -2),
    ];

    for (s, t, match_score, mismatch_penalty, gap_penalty) in cases {
        let problem =
            SmithWatermanProblem::new(s, t, *match_score, *mismatch_penalty, *gap_penalty);
        assert_engine_paths_for_all_block_sizes(problem.clone(), |cost, path| {
            assert_sw_path(&problem, cost, path);
        });
    }
}

#[test]
fn smith_waterman_optimized_split_regressions_hold() {
    type SwCase<'a> = (&'a [u8], &'a [u8], i32, i32, i32);
    let cases: &[SwCase<'_>] = &[
        (b"TTACGTAA", b"GGACGTCC", 2, 1, -2),
        (b"TTTTTACGT", b"ACGT", 2, 1, -2),
        (b"ACGTTTTTTT", b"ACGT", 2, 1, -2),
        (b"AAAAAA", b"AAA", 1, 1, -1),
    ];

    for (s, t, match_score, mismatch_penalty, gap_penalty) in cases {
        let problem =
            SmithWatermanProblem::new(s, t, *match_score, *mismatch_penalty, *gap_penalty);
        assert_engine_paths_for_all_block_sizes(problem.clone(), |cost, path| {
            assert_sw_path(&problem, cost, path);
        });
    }
}

#[test]
fn semiglobal_edge_cases_hold() {
    type SemiCase<'a> = (&'a [u8], &'a [u8], i32, i32, i32);
    let cases: &[SemiCase<'_>] = &[
        (b"", b"", 2, 1, -2),
        (b"", b"ABC", 2, 1, -2),
        (b"ABC", b"", 2, 1, -2),
        (b"ACGT", b"TTACGTTT", 2, 1, -2),
        (b"AAAA", b"TTAAAATT", 2, 1, -2),
        (b"ACGTAC", b"GGACGT", 2, 1, -2),
        (b"GGGG", b"TTTT", 1, 1, -2),
        (b"ACGT", b"ACGT", 2, 1, -2),
    ];

    for (s, t, match_score, mismatch_penalty, gap_penalty) in cases {
        let problem = SemiGlobalProblem::new(s, t, *match_score, *mismatch_penalty, *gap_penalty);
        assert_engine_paths_for_all_block_sizes(problem.clone(), |cost, path| {
            assert_semiglobal_path(&problem, cost, path);
        });
    }
}

#[test]
fn semiglobal_optimized_split_regressions_hold() {
    type SemiCase<'a> = (&'a [u8], &'a [u8], i32, i32, i32);
    let cases: &[SemiCase<'_>] = &[
        (b"ACGT", b"TTACGTTT", 2, 1, -2),
        (b"AAAAAACCCCCC", b"TTAAAAAATTT", 2, 1, -2),
        (b"ACGTACGT", b"TTACGTGGACGTTT", 2, 1, -2),
        (b"AAAAAA", b"TTAAAATT", 1, 1, -1),
    ];

    for (s, t, match_score, mismatch_penalty, gap_penalty) in cases {
        let problem = SemiGlobalProblem::new(s, t, *match_score, *mismatch_penalty, *gap_penalty);
        assert_engine_paths_for_all_block_sizes(problem.clone(), |cost, path| {
            assert_semiglobal_path(&problem, cost, path);
        });
    }
}
