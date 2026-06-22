use hcp_dp::{
    problems::{
        lcs::{LcsProblem, LcsState},
        nw_align::{NwProblem, NwState},
    },
    HcpEngine, HcpProblem, SummaryApply,
};
use proptest::prelude::*;

fn lcs_frontier_after(problem: &LcsProblem<'_>, a: usize) -> Vec<u32> {
    let mut frontier = problem.init_frontier();
    for layer in 0..a {
        frontier = problem.forward_step(layer, &frontier);
    }
    frontier.scores
}

fn nw_frontier_after(problem: &NwProblem<'_>, a: usize) -> Vec<i32> {
    let mut frontier = problem.init_frontier();
    for layer in 0..a {
        frontier = problem.forward_step(layer, &frontier);
    }
    frontier.scores
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

fn assert_lcs_summary_contract(problem: &LcsProblem<'_>) {
    let n = problem.n();
    for a in 0..=n {
        let mut frontier_a = problem.init_frontier();
        for layer in 0..a {
            frontier_a = problem.forward_step(layer, &frontier_a);
        }

        for b in a..=n {
            let sigma_ab = problem.summarize_interval(a, b);
            let mut direct = frontier_a.clone();
            for layer in a..b {
                direct = problem.forward_step(layer, &direct);
            }
            assert_eq!(sigma_ab.apply(&frontier_a).scores, direct.scores);

            for c in b..=n {
                let sigma_bc = problem.summarize_interval(b, c);
                let merged = problem.merge_summary(&sigma_ab, &sigma_bc);
                let mut direct_ac = frontier_a.clone();
                for layer in a..c {
                    direct_ac = problem.forward_step(layer, &direct_ac);
                }
                assert_eq!(merged.apply(&frontier_a).scores, direct_ac.scores);
            }
        }
    }
}

fn assert_nw_summary_contract(problem: &NwProblem<'_>) {
    let n = problem.n();
    for a in 0..=n {
        let mut frontier_a = problem.init_frontier();
        for layer in 0..a {
            frontier_a = problem.forward_step(layer, &frontier_a);
        }

        for b in a..=n {
            let sigma_ab = problem.summarize_interval(a, b);
            let mut direct = frontier_a.clone();
            for layer in a..b {
                direct = problem.forward_step(layer, &direct);
            }
            assert_eq!(sigma_ab.apply(&frontier_a).scores, direct.scores);

            for c in b..=n {
                let sigma_bc = problem.summarize_interval(b, c);
                let merged = problem.merge_summary(&sigma_ab, &sigma_bc);
                let mut direct_ac = frontier_a.clone();
                for layer in a..c {
                    direct_ac = problem.forward_step(layer, &direct_ac);
                }
                assert_eq!(merged.apply(&frontier_a).scores, direct_ac.scores);
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
    let frontier_t_scores = lcs_frontier_after(problem, n);
    let beta_c = problem.terminal_boundary(&hcp_dp::problems::lcs::LcsFrontier {
        scores: frontier_t_scores,
    });
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
    let frontier_t_scores = nw_frontier_after(problem, n);
    let beta_c = problem.terminal_boundary(&hcp_dp::problems::nw_align::NwFrontier {
        scores: frontier_t_scores,
    });
    let sigma_left = problem.summarize_interval(0, mid);
    let sigma_right = problem.summarize_interval(mid, n);
    let beta_m = problem.choose_split(0, mid, n, &beta_a, &beta_c, &sigma_left, &sigma_right);
    let mut left = problem.reconstruct_leaf(0, mid, &beta_a, &beta_m);
    let right = problem.reconstruct_leaf(mid, n, &beta_m, &beta_c);
    assert_eq!(left.last(), right.first());
    left.extend_from_slice(&right[1..]);
    assert_nw_path(problem, problem.full_table_score(), &left);
}

proptest! {
    #[test]
    fn lcs_contracts_hold(a in "[ACGT]{0,8}", b in "[ACGT]{0,8}") {
        let problem = LcsProblem::new(a.as_bytes(), b.as_bytes());
        assert_lcs_summary_contract(&problem);
        assert_lcs_split_contract(&problem);
        for block_size in 1..=problem.n().max(1) {
            let (cost, path) = HcpEngine::with_block_size(problem.clone(), block_size).run();
            assert_lcs_path(&problem, cost, &path);
        }
    }

    #[test]
    fn nw_contracts_hold(a in "[ACGT]{0,7}", b in "[ACGT]{0,7}") {
        let problem = NwProblem::new(a.as_bytes(), b.as_bytes(), 2, 1, -2);
        assert_nw_summary_contract(&problem);
        assert_nw_split_contract(&problem);
        for block_size in 1..=problem.n().max(1) {
            let (cost, path) = HcpEngine::with_block_size(problem.clone(), block_size).run();
            assert_nw_path(&problem, cost, &path);
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
}
