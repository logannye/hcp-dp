//! Reusable correctness checks for [`crate::HcpProblem`] implementations.
//!
//! These helpers are intended for small bounded test cases. They panic on
//! contract violations, so they fit naturally inside unit tests and property
//! tests for custom dynamic-programming problems.

use crate::{HcpEngine, HcpProblem, SummaryApply};
use std::fmt::Debug;

/// Replay direct one-layer transitions from layer `0` through `layer`.
///
/// # Panics
/// Panics when `layer > problem.num_layers()`.
pub fn frontier_after<P>(problem: &P, layer: usize) -> P::Frontier
where
    P: HcpProblem,
{
    assert!(
        layer <= problem.num_layers(),
        "requested frontier layer must be within the problem"
    );
    let mut frontier = problem.init_frontier();
    for step in 0..layer {
        frontier = problem.forward_step(step, &frontier);
    }
    frontier
}

/// Replay direct one-layer transitions over interval `[a, b)`.
///
/// # Panics
/// Panics when the interval is outside `0..=problem.num_layers()`.
pub fn replay_interval<P>(problem: &P, a: usize, b: usize, frontier: &P::Frontier) -> P::Frontier
where
    P: HcpProblem,
{
    assert!(
        a <= b && b <= problem.num_layers(),
        "replay interval must be within the problem"
    );
    let mut replay = frontier.clone();
    for step in a..b {
        replay = problem.forward_step(step, &replay);
    }
    replay
}

/// Assert that `summarize_interval(a, b)` applies like direct replay.
///
/// This checks the core summary-operator law:
///
/// ```text
/// summary(a,b).apply(frontier_at_a) == replay(a,b, frontier_at_a)
/// ```
#[track_caller]
pub fn assert_summary_apply_matches_replay<P>(problem: &P, a: usize, b: usize)
where
    P: HcpProblem,
    P::Frontier: Debug + PartialEq,
{
    let frontier_a = frontier_after(problem, a);
    let summary = problem.summarize_interval(a, b);
    let direct = replay_interval(problem, a, b, &frontier_a);
    assert_eq!(
        summary.apply(&frontier_a),
        direct,
        "summary({a},{b}) must apply like direct replay"
    );
}

/// Assert that merging `[a, b)` and `[b, c)` applies like direct replay on `[a, c)`.
#[track_caller]
pub fn assert_summary_merge_matches_replay<P>(problem: &P, a: usize, b: usize, c: usize)
where
    P: HcpProblem,
    P::Frontier: Debug + PartialEq,
{
    assert!(
        a <= b && b <= c && c <= problem.num_layers(),
        "merge intervals must be adjacent and within the problem"
    );
    let frontier_a = frontier_after(problem, a);
    let left = problem.summarize_interval(a, b);
    let right = problem.summarize_interval(b, c);
    let merged = problem.merge_summary(&left, &right);
    let direct = replay_interval(problem, a, c, &frontier_a);
    assert_eq!(
        merged.apply(&frontier_a),
        direct,
        "merge(summary({a},{b}), summary({b},{c})) must apply like direct replay"
    );
}

/// Exhaustively assert summary apply and merge laws for one small problem instance.
///
/// This is `O(T^3)` in the number of layers, so use it on small generated or
/// hand-written cases. It is deliberately thorough because most broken HCP
/// implementations fail one of these laws before path reconstruction is tested.
#[track_caller]
pub fn assert_all_summary_laws<P>(problem: &P)
where
    P: HcpProblem,
    P::Frontier: Debug + PartialEq,
{
    let n = problem.num_layers();
    for a in 0..=n {
        for b in a..=n {
            assert_summary_apply_matches_replay(problem, a, b);
            for c in b..=n {
                assert_summary_merge_matches_replay(problem, a, b, c);
            }
        }
    }
}

/// Reconstruct and join the two halves around one top-level split.
///
/// Returns `None` when `m` is not an internal split for the problem. For local
/// alignment style problems, either half may reconstruct to an empty segment;
/// non-empty halves are still required to join exactly.
#[track_caller]
pub fn reconstruct_top_split<P>(problem: &P, m: usize) -> Option<Vec<P::State>>
where
    P: HcpProblem,
{
    let n = problem.num_layers();
    if m == 0 || m >= n {
        return None;
    }

    let beta_a = problem.initial_boundary();
    let frontier_t = frontier_after(problem, n);
    let beta_c = problem.terminal_boundary(&frontier_t);
    let sigma_left = problem.summarize_interval(0, m);
    let sigma_right = problem.summarize_interval(m, n);
    let beta_m = problem.choose_split(0, m, n, &beta_a, &beta_c, &sigma_left, &sigma_right);
    let left = problem.reconstruct_leaf(0, m, &beta_a, &beta_m);
    let right = problem.reconstruct_leaf(m, n, &beta_m, &beta_c);
    Some(join_segments(left, right))
}

/// Run one problem instance across all block sizes and validate each path.
///
/// This helper is for small bounded tests. `validate` should independently
/// compare the objective to a baseline and independently score the returned
/// path.
#[track_caller]
pub fn assert_engine_paths_for_all_block_sizes<P, F>(problem: P, mut validate: F)
where
    P: HcpProblem + Clone,
    F: FnMut(P::Cost, &[P::State]),
{
    for block_size in 1..=problem.num_layers().max(1) {
        let (cost, path) = HcpEngine::with_block_size(problem.clone(), block_size).run();
        validate(cost, &path);
    }
}

fn join_segments<S: Clone + PartialEq>(mut left: Vec<S>, right: Vec<S>) -> Vec<S> {
    if left.is_empty() {
        return right;
    }
    if right.is_empty() {
        return left;
    }
    assert!(
        left.last() == right.first(),
        "reconstructed path segments must join at the split boundary"
    );
    left.extend_from_slice(&right[1..]);
    left
}
