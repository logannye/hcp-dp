//! Dynamic time warping over integer-valued series.
//!
//! This module is intentionally small: it is the first non-sequence-alignment
//! DP exported by the crate, and exists to prove that the HCP contract applies
//! to frontier DPs beyond string grids.

use crate::traits::{HcpProblem, SummaryApply};

const DTW_INF: u64 = u64::MAX / 4;

#[derive(Clone)]
pub struct DtwProblem<'a> {
    pub query: &'a [i32],
    pub target: &'a [i32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DtwFrontier {
    pub costs: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct DtwSummary<'a> {
    query: &'a [i32],
    target: &'a [i32],
    start: usize,
    end: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DtwBoundary {
    pub row: usize,
    pub col: usize,
}

pub type DtwState = (usize, usize);

impl<'a> DtwProblem<'a> {
    /// Create a DTW instance.
    ///
    /// Standard DTW requires both series to contain at least one observation.
    /// Empty-series semantics are intentionally not invented here.
    pub fn new(query: &'a [i32], target: &'a [i32]) -> Self {
        assert!(
            !query.is_empty() && !target.is_empty(),
            "DTW requires non-empty query and target series"
        );
        Self { query, target }
    }

    pub fn n(&self) -> usize {
        self.query.len()
    }

    pub fn m(&self) -> usize {
        self.target.len()
    }

    pub fn score_path(&self, path: &[DtwState]) -> Option<u64> {
        let mut score = 0u64;
        for window in path.windows(2) {
            let (a, b) = (window[0], window[1]);
            let di = b.0.checked_sub(a.0)?;
            let dj = b.1.checked_sub(a.1)?;
            if !matches!((di, dj), (1, 0) | (0, 1) | (1, 1)) {
                return None;
            }
            if b.0 == 0 || b.1 == 0 || b.0 > self.n() || b.1 > self.m() {
                return None;
            }
            score = dtw_add(score, local_cost(self.query[b.0 - 1], self.target[b.1 - 1]));
        }
        Some(score)
    }

    pub fn full_table_cost(&self) -> u64 {
        dtw_constrained_score(self.query, self.target, 0, self.m())
    }
}

impl<'a> HcpProblem for DtwProblem<'a> {
    type State = DtwState;
    type Frontier = DtwFrontier;
    type Summary = DtwSummary<'a>;
    type Boundary = DtwBoundary;
    type Cost = u64;

    fn num_layers(&self) -> usize {
        self.n()
    }

    fn init_frontier(&self) -> Self::Frontier {
        let mut costs = vec![DTW_INF; self.m() + 1];
        costs[0] = 0;
        DtwFrontier { costs }
    }

    fn forward_step(&self, layer: usize, frontier: &Self::Frontier) -> Self::Frontier {
        assert_eq!(
            frontier.costs.len(),
            self.m() + 1,
            "DTW frontier width mismatch"
        );
        DtwFrontier {
            costs: dtw_next_row(self.query[layer], self.target, 0, self.m(), &frontier.costs),
        }
    }

    fn summarize_interval(&self, a: usize, b: usize) -> Self::Summary {
        assert!(a <= b && b <= self.n(), "invalid DTW interval");
        DtwSummary {
            query: self.query,
            target: self.target,
            start: a,
            end: b,
        }
    }

    fn merge_summary(&self, left: &Self::Summary, right: &Self::Summary) -> Self::Summary {
        assert_eq!(left.end, right.start, "DTW summaries must be adjacent");
        assert!(
            std::ptr::eq(left.query, right.query) && std::ptr::eq(left.target, right.target),
            "DTW summaries must belong to the same problem"
        );
        DtwSummary {
            query: self.query,
            target: self.target,
            start: left.start,
            end: right.end,
        }
    }

    fn initial_boundary(&self) -> Self::Boundary {
        DtwBoundary { row: 0, col: 0 }
    }

    fn terminal_boundary(&self, _frontier_t: &Self::Frontier) -> Self::Boundary {
        DtwBoundary {
            row: self.n(),
            col: self.m(),
        }
    }

    fn choose_split(
        &self,
        a: usize,
        m: usize,
        c: usize,
        beta_a: &Self::Boundary,
        beta_c: &Self::Boundary,
        _sigma_left: &Self::Summary,
        _sigma_right: &Self::Summary,
    ) -> Self::Boundary {
        assert_eq!(beta_a.row, a, "left boundary row must match interval start");
        assert_eq!(beta_c.row, c, "right boundary row must match interval end");
        assert!(a < m && m < c, "DTW split must be internal");

        let p = beta_a.col;
        let q = beta_c.col;
        assert!(p <= q && q <= self.m(), "invalid DTW boundary columns");

        let left = dtw_last_row_constrained(&self.query[a..m], self.target, p, q);
        let mut best_col = p;
        let mut best_cost = DTW_INF;
        for (offset, left_cost) in left.iter().copied().enumerate() {
            let split_col = p + offset;
            let right_cost = dtw_constrained_score(&self.query[m..c], self.target, split_col, q);
            let total = dtw_add(left_cost, right_cost);
            if total < best_cost {
                best_cost = total;
                best_col = split_col;
            }
        }
        assert!(best_cost < DTW_INF, "DTW split must be feasible");

        DtwBoundary {
            row: m,
            col: best_col,
        }
    }

    fn reconstruct_leaf(
        &self,
        a: usize,
        b: usize,
        beta_a: &Self::Boundary,
        beta_b: &Self::Boundary,
    ) -> Vec<Self::State> {
        assert_eq!(beta_a.row, a, "leaf start row must match beta_a");
        assert_eq!(beta_b.row, b, "leaf end row must match beta_b");
        let p = beta_a.col;
        let q = beta_b.col;
        assert!(p <= q && q <= self.m(), "invalid DTW leaf columns");

        let height = b - a;
        let width = q - p;
        let stride = width + 1;
        let mut dp = vec![DTW_INF; (height + 1) * stride];
        dp[0] = 0;

        for row in 1..=height {
            for rel_col in 0..=width {
                let abs_col = p + rel_col;
                if abs_col == 0 {
                    continue;
                }
                let cost = local_cost(self.query[a + row - 1], self.target[abs_col - 1]);
                let mut best = dp[(row - 1) * stride + rel_col];
                if rel_col > 0 {
                    best = best
                        .min(dp[(row - 1) * stride + rel_col - 1])
                        .min(dp[row * stride + rel_col - 1]);
                }
                dp[row * stride + rel_col] = dtw_add(best, cost);
            }
        }

        assert!(
            dp[height * stride + width] < DTW_INF,
            "DTW leaf endpoints must be feasible"
        );

        let mut row = height;
        let mut rel_col = width;
        let mut rev_path = Vec::with_capacity(height + width + 1);
        rev_path.push((a + row, p + rel_col));

        while row > 0 || rel_col > 0 {
            let here = dp[row * stride + rel_col];
            let abs_col = p + rel_col;
            assert!(
                row > 0 && abs_col > 0,
                "DTW traceback cannot enter row 0 or column 0"
            );
            let cost = local_cost(self.query[a + row - 1], self.target[abs_col - 1]);

            if row > 0 && rel_col > 0 && here == dtw_add(dp[(row - 1) * stride + rel_col - 1], cost)
            {
                row -= 1;
                rel_col -= 1;
            } else if row > 0 && here == dtw_add(dp[(row - 1) * stride + rel_col], cost) {
                row -= 1;
            } else {
                assert!(
                    rel_col > 0 && here == dtw_add(dp[row * stride + rel_col - 1], cost),
                    "DTW traceback must have an optimal predecessor"
                );
                rel_col -= 1;
            }
            rev_path.push((a + row, p + rel_col));
        }

        rev_path.reverse();
        rev_path
    }

    fn extract_cost(&self, frontier_t: &Self::Frontier, _beta_t: &Self::Boundary) -> Self::Cost {
        frontier_t.costs[self.m()]
    }
}

impl<'a> SummaryApply<DtwFrontier> for DtwSummary<'a> {
    fn apply(&self, frontier: &DtwFrontier) -> DtwFrontier {
        assert_eq!(
            frontier.costs.len(),
            self.target.len() + 1,
            "DTW frontier width mismatch"
        );
        let mut current = frontier.costs.clone();
        for layer in self.start..self.end {
            current = dtw_next_row(
                self.query[layer],
                self.target,
                0,
                self.target.len(),
                &current,
            );
        }
        DtwFrontier { costs: current }
    }
}

fn dtw_last_row_constrained(query: &[i32], target: &[i32], p: usize, q: usize) -> Vec<u64> {
    assert!(p <= q && q <= target.len(), "invalid DTW constrained range");
    let mut current = vec![DTW_INF; q - p + 1];
    current[0] = 0;
    for &value in query {
        current = dtw_next_row(value, target, p, q, &current);
    }
    current
}

fn dtw_constrained_score(query: &[i32], target: &[i32], p: usize, q: usize) -> u64 {
    dtw_last_row_constrained(query, target, p, q)
        .last()
        .copied()
        .unwrap_or(DTW_INF)
}

fn dtw_next_row(value: i32, target: &[i32], p: usize, q: usize, prev: &[u64]) -> Vec<u64> {
    assert_eq!(prev.len(), q - p + 1, "DTW previous-row width mismatch");
    let mut next = vec![DTW_INF; prev.len()];
    for rel_col in 0..prev.len() {
        let abs_col = p + rel_col;
        if abs_col == 0 {
            continue;
        }
        let cost = local_cost(value, target[abs_col - 1]);
        let mut best = prev[rel_col];
        if rel_col > 0 {
            best = best.min(prev[rel_col - 1]).min(next[rel_col - 1]);
        }
        next[rel_col] = dtw_add(best, cost);
    }
    next
}

fn local_cost(a: i32, b: i32) -> u64 {
    a.abs_diff(b) as u64
}

fn dtw_add(base: u64, delta: u64) -> u64 {
    if base >= DTW_INF / 2 {
        DTW_INF
    } else {
        base.saturating_add(delta).min(DTW_INF)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HcpEngine;

    #[test]
    fn simple_dtw_path_realizes_cost() {
        let problem = DtwProblem::new(&[1, 2, 3], &[2, 2, 4]);
        let (cost, path) = HcpEngine::new(problem.clone()).run();
        assert_eq!(cost, problem.full_table_cost());
        assert_eq!(path.first(), Some(&(0, 0)));
        assert_eq!(path.last(), Some(&(3, 3)));
        assert_eq!(problem.score_path(&path), Some(cost));
    }

    #[test]
    #[should_panic(expected = "DTW requires non-empty")]
    fn empty_series_are_rejected() {
        let _ = DtwProblem::new(&[], &[1]);
    }
}
