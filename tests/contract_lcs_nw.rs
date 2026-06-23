use hcp_dp::{
    problems::{
        edit_distance::{EditDistanceProblem, EditDistanceState},
        lcs::{LcsProblem, LcsState},
        nw_affine::{NwAffineProblem, NwAffineState},
        nw_align::{NwProblem, NwState},
        semiglobal::{SemiGlobalCell, SemiGlobalProblem},
        smith_waterman::{SmithWatermanProblem, SwCell},
    },
    HcpEngine, HcpProblem, SummaryApply,
};
use proptest::prelude::*;
use std::fmt::Debug;

fn frontier_after<P>(problem: &P, a: usize) -> P::Frontier
where
    P: HcpProblem,
{
    let mut frontier = problem.init_frontier();
    for layer in 0..a {
        frontier = problem.forward_step(layer, &frontier);
    }
    frontier
}

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

fn assert_summary_contract<P>(problem: &P)
where
    P: HcpProblem,
    P::Frontier: Debug + PartialEq,
{
    let n = problem.num_layers();
    for a in 0..=n {
        let frontier_a = frontier_after(problem, a);

        for b in a..=n {
            let sigma_ab = problem.summarize_interval(a, b);
            let mut direct = frontier_a.clone();
            for layer in a..b {
                direct = problem.forward_step(layer, &direct);
            }
            assert_eq!(sigma_ab.apply(&frontier_a), direct);

            for c in b..=n {
                let sigma_bc = problem.summarize_interval(b, c);
                let merged = problem.merge_summary(&sigma_ab, &sigma_bc);
                let mut direct_ac = frontier_a.clone();
                for layer in a..c {
                    direct_ac = problem.forward_step(layer, &direct_ac);
                }
                assert_eq!(merged.apply(&frontier_a), direct_ac);
            }
        }
    }
}

fn assert_lcs_split_contract(problem: &LcsProblem<'_>) {
    let n = problem.n();
    if n < 2 {
        return;
    }
    let mid = n / 2;
    let beta_a = problem.initial_boundary();
    let frontier_t = frontier_after(problem, n);
    let beta_c = problem.terminal_boundary(&frontier_t);
    let sigma_left = problem.summarize_interval(0, mid);
    let sigma_right = problem.summarize_interval(mid, n);
    let beta_m = problem.choose_split(0, mid, n, &beta_a, &beta_c, &sigma_left, &sigma_right);
    let mut left = problem.reconstruct_leaf(0, mid, &beta_a, &beta_m);
    let right = problem.reconstruct_leaf(mid, n, &beta_m, &beta_c);
    assert_eq!(left.last(), right.first());
    left.extend_from_slice(&right[1..]);
    assert_lcs_path(problem, problem.full_table_len(), &left);
}

fn assert_nw_split_contract(problem: &NwProblem<'_>) {
    let n = problem.n();
    if n < 2 {
        return;
    }
    let mid = n / 2;
    let beta_a = problem.initial_boundary();
    let frontier_t = frontier_after(problem, n);
    let beta_c = problem.terminal_boundary(&frontier_t);
    let sigma_left = problem.summarize_interval(0, mid);
    let sigma_right = problem.summarize_interval(mid, n);
    let beta_m = problem.choose_split(0, mid, n, &beta_a, &beta_c, &sigma_left, &sigma_right);
    let mut left = problem.reconstruct_leaf(0, mid, &beta_a, &beta_m);
    let right = problem.reconstruct_leaf(mid, n, &beta_m, &beta_c);
    assert_eq!(left.last(), right.first());
    left.extend_from_slice(&right[1..]);
    assert_nw_path(problem, problem.full_table_score(), &left);
}

fn assert_affine_split_contract(problem: &NwAffineProblem<'_>) {
    let n = problem.n();
    if n < 2 {
        return;
    }
    let mid = n / 2;
    let beta_a = problem.initial_boundary();
    let frontier_t = frontier_after(problem, n);
    let beta_c = problem.terminal_boundary(&frontier_t);
    let sigma_left = problem.summarize_interval(0, mid);
    let sigma_right = problem.summarize_interval(mid, n);
    let beta_m = problem.choose_split(0, mid, n, &beta_a, &beta_c, &sigma_left, &sigma_right);
    let mut left = problem.reconstruct_leaf(0, mid, &beta_a, &beta_m);
    let right = problem.reconstruct_leaf(mid, n, &beta_m, &beta_c);
    assert_eq!(left.last(), right.first());
    left.extend_from_slice(&right[1..]);
    assert_affine_path(problem, problem.full_table_score(), &left);
}

fn assert_sw_split_contract(problem: &SmithWatermanProblem<'_>) {
    let n = problem.n();
    if n < 2 {
        return;
    }
    let mid = n / 2;
    let beta_a = problem.initial_boundary();
    let frontier_t = frontier_after(problem, n);
    let beta_c = problem.terminal_boundary(&frontier_t);
    let sigma_left = problem.summarize_interval(0, mid);
    let sigma_right = problem.summarize_interval(mid, n);
    let beta_m = problem.choose_split(0, mid, n, &beta_a, &beta_c, &sigma_left, &sigma_right);
    let left = problem.reconstruct_leaf(0, mid, &beta_a, &beta_m);
    let right = problem.reconstruct_leaf(mid, n, &beta_m, &beta_c);

    let mut joined = left;
    if joined.is_empty() {
        joined = right;
    } else if !right.is_empty() {
        assert_eq!(joined.last(), right.first());
        joined.extend_from_slice(&right[1..]);
    }
    assert_sw_path(problem, problem.full_table_score(), &joined);
}

fn assert_edit_split_contract(problem: &EditDistanceProblem<'_>) {
    let n = problem.n();
    if n < 2 {
        return;
    }
    let mid = n / 2;
    let beta_a = problem.initial_boundary();
    let frontier_t = frontier_after(problem, n);
    let beta_c = problem.terminal_boundary(&frontier_t);
    let sigma_left = problem.summarize_interval(0, mid);
    let sigma_right = problem.summarize_interval(mid, n);
    let beta_m = problem.choose_split(0, mid, n, &beta_a, &beta_c, &sigma_left, &sigma_right);
    let mut left = problem.reconstruct_leaf(0, mid, &beta_a, &beta_m);
    let right = problem.reconstruct_leaf(mid, n, &beta_m, &beta_c);
    assert_eq!(left.last(), right.first());
    left.extend_from_slice(&right[1..]);
    assert_edit_path(problem, problem.full_table_distance(), &left);
}

fn assert_semiglobal_split_contract(problem: &SemiGlobalProblem<'_>) {
    let n = problem.n();
    if n < 2 {
        return;
    }
    let mid = n / 2;
    let beta_a = problem.initial_boundary();
    let frontier_t = frontier_after(problem, n);
    let beta_c = problem.terminal_boundary(&frontier_t);
    let sigma_left = problem.summarize_interval(0, mid);
    let sigma_right = problem.summarize_interval(mid, n);
    let beta_m = problem.choose_split(0, mid, n, &beta_a, &beta_c, &sigma_left, &sigma_right);
    let mut left = problem.reconstruct_leaf(0, mid, &beta_a, &beta_m);
    let right = problem.reconstruct_leaf(mid, n, &beta_m, &beta_c);
    assert_eq!(left.last(), right.first());
    left.extend_from_slice(&right[1..]);
    assert_semiglobal_path(problem, problem.full_table_score(), &left);
}

proptest! {
    #[test]
    fn lcs_contracts_hold(a in "[ACGT]{0,8}", b in "[ACGT]{0,8}") {
        let problem = LcsProblem::new(a.as_bytes(), b.as_bytes());
        assert_summary_contract(&problem);
        assert_lcs_split_contract(&problem);
        for block_size in 1..=problem.n().max(1) {
            let (cost, path) = HcpEngine::with_block_size(problem.clone(), block_size).run();
            assert_lcs_path(&problem, cost, &path);
        }
    }

    #[test]
    fn nw_contracts_hold(a in "[ACGT]{0,7}", b in "[ACGT]{0,7}") {
        let problem = NwProblem::new(a.as_bytes(), b.as_bytes(), 2, 1, -2);
        assert_summary_contract(&problem);
        assert_nw_split_contract(&problem);
        for block_size in 1..=problem.n().max(1) {
            let (cost, path) = HcpEngine::with_block_size(problem.clone(), block_size).run();
            assert_nw_path(&problem, cost, &path);
        }
    }

    #[test]
    fn affine_contracts_hold(a in "[ACGT]{0,6}", b in "[ACGT]{0,6}") {
        let problem = NwAffineProblem::new(a.as_bytes(), b.as_bytes(), 2, 1, -3, -1);
        assert_summary_contract(&problem);
        assert_affine_split_contract(&problem);
        for block_size in 1..=problem.n().max(1) {
            let (cost, path) = HcpEngine::with_block_size(problem.clone(), block_size).run();
            assert_affine_path(&problem, cost, &path);
        }
    }

    #[test]
    fn smith_waterman_contracts_hold(a in "[ACGT]{0,6}", b in "[ACGT]{0,6}") {
        let problem = SmithWatermanProblem::new(a.as_bytes(), b.as_bytes(), 2, 1, -2);
        assert_summary_contract(&problem);
        assert_sw_split_contract(&problem);
        for block_size in 1..=problem.n().max(1) {
            let (cost, path) = HcpEngine::with_block_size(problem.clone(), block_size).run();
            assert_sw_path(&problem, cost, &path);
        }
    }

    #[test]
    fn edit_distance_contracts_hold(a in "[ACGT]{0,7}", b in "[ACGT]{0,7}") {
        let problem = EditDistanceProblem::new(a.as_bytes(), b.as_bytes());
        assert_summary_contract(&problem);
        assert_edit_split_contract(&problem);
        for block_size in 1..=problem.n().max(1) {
            let (cost, path) = HcpEngine::with_block_size(problem.clone(), block_size).run();
            assert_edit_path(&problem, cost, &path);
        }
    }

    #[test]
    fn semiglobal_contracts_hold(a in "[ACGT]{0,6}", b in "[ACGT]{0,8}") {
        let problem = SemiGlobalProblem::new(a.as_bytes(), b.as_bytes(), 2, 1, -2);
        assert_summary_contract(&problem);
        assert_semiglobal_split_contract(&problem);
        for block_size in 1..=problem.n().max(1) {
            let (cost, path) = HcpEngine::with_block_size(problem.clone(), block_size).run();
            assert_semiglobal_path(&problem, cost, &path);
        }
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
        for block_size in 1..=problem.n().max(1) {
            let (cost, path) = HcpEngine::with_block_size(problem.clone(), block_size).run();
            assert_affine_path(&problem, cost, &path);
        }
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
        assert_summary_contract(&problem);
        assert_edit_split_contract(&problem);
        for block_size in 1..=problem.n().max(1) {
            let (cost, path) = HcpEngine::with_block_size(problem.clone(), block_size).run();
            assert_edit_path(&problem, cost, &path);
        }
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
        for block_size in 1..=problem.n().max(1) {
            let (cost, path) = HcpEngine::with_block_size(problem.clone(), block_size).run();
            assert_sw_path(&problem, cost, &path);
        }
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
        for block_size in 1..=problem.n().max(1) {
            let (cost, path) = HcpEngine::with_block_size(problem.clone(), block_size).run();
            assert_sw_path(&problem, cost, &path);
        }
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
        for block_size in 1..=problem.n().max(1) {
            let (cost, path) = HcpEngine::with_block_size(problem.clone(), block_size).run();
            assert_semiglobal_path(&problem, cost, &path);
        }
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
        for block_size in 1..=problem.n().max(1) {
            let (cost, path) = HcpEngine::with_block_size(problem.clone(), block_size).run();
            assert_semiglobal_path(&problem, cost, &path);
        }
    }
}
