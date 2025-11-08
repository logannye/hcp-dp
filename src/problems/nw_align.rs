//! Needleman–Wunsch global alignment as an HCP-DP instance.
//!
//! This is a canonical example with strictly local dependencies and
//! associativity suitable for height compression. We implement:
//! - scoring with match/mismatch/gap penalties,
//! - exact optimal alignment path reconstruction.

use crate::traits::HcpProblem;

#[derive(Clone)]
pub struct NwProblem<'a> {
    pub s: &'a [u8],
    pub t: &'a [u8],
    pub match_score: i32,
    pub mismatch_penalty: i32,
    pub gap_penalty: i32,
}

#[derive(Clone, Debug)]
pub struct NwFrontier {
    pub scores: Vec<i32>,
}

#[derive(Clone, Debug)]
pub struct NwSummary {
    pub end_frontier: NwFrontier,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NwBoundary {
    pub row: usize,
    pub col: usize,
}

pub type NwState = (usize, usize);

impl<'a> NwProblem<'a> {
    pub fn new(
        s: &'a [u8],
        t: &'a [u8],
        match_score: i32,
        mismatch_penalty: i32,
        gap_penalty: i32,
    ) -> Self {
        Self {
            s,
            t,
            match_score,
            mismatch_penalty,
            gap_penalty,
        }
    }

    fn n(&self) -> usize {
        self.s.len()
    }

    fn m(&self) -> usize {
        self.t.len()
    }

    fn score_pair(&self, a: u8, b: u8) -> i32 {
        if a == b {
            self.match_score
        } else {
            -self.mismatch_penalty
        }
    }
}

impl<'a> HcpProblem for NwProblem<'a> {
    type State = NwState;
    type Frontier = NwFrontier;
    type Summary = NwSummary;
    type Boundary = NwBoundary;
    type Cost = i32;

    fn num_layers(&self) -> usize {
        self.n()
    }

    fn init_frontier(&self) -> Self::Frontier {
        // Row 0: gaps in s vs prefix of t.
        let m = self.m();
        let mut scores = Vec::with_capacity(m + 1);
        scores.push(0);
        for j in 1..=m {
            scores.push(scores[j - 1] + self.gap_penalty);
        }
        NwFrontier { scores }
    }

    fn forward_step(&self, layer: usize, f: &Self::Frontier) -> Self::Frontier {
        // Build row i+1 from row i.
        let i = layer;
        let m = self.m();
        let ch_s = self.s[i];
        let mut next = Vec::with_capacity(m + 1);
        next.push(f.scores[0] + self.gap_penalty); // gap in t

        for j in 1..=m {
            let diag = f.scores[j - 1] + self.score_pair(ch_s, self.t[j - 1]);
            let up = f.scores[j] + self.gap_penalty; // gap in t
            let left = next[j - 1] + self.gap_penalty; // gap in s
            next.push(diag.max(up).max(left));
        }

        NwFrontier { scores: next }
    }

    fn summarize_block(
        &self,
        a: usize,
        b: usize,
        frontier_a: &Self::Frontier,
    ) -> (Self::Frontier, Self::Summary) {
        let mut f = frontier_a.clone();
        for i in a..b {
            f = self.forward_step(i, &f);
        }
        (f.clone(), NwSummary { end_frontier: f })
    }

    fn merge_summary(&self, _left: &Self::Summary, right: &Self::Summary) -> Self::Summary {
        right.clone()
    }

    fn initial_boundary(&self) -> Self::Boundary {
        NwBoundary { row: 0, col: 0 }
    }

    fn terminal_boundary(&self, frontier_t: &Self::Frontier) -> Self::Boundary {
        // We align entire strings: end at (n,m).
        let _ = frontier_t; // frontier carries scores; we know coordinates.
        NwBoundary {
            row: self.n(),
            col: self.m(),
        }
    }

    fn choose_boundary(
        &self,
        a: usize,
        m: usize,
        c: usize,
        _sigma_left: &Self::Summary,
        _sigma_right: &Self::Summary,
        beta_a: &Self::Boundary,
        beta_c: &Self::Boundary,
    ) -> Self::Boundary {
        // Hirschberg-style mid column for Needleman–Wunsch on subproblem:
        // s[a..c], t[p..q].
        let p = beta_a.col;
        let q = beta_c.col;

        debug_assert!(a <= m && m <= c);
        debug_assert!(p <= q && q <= self.m());

        // Forward scores for s[a..m] vs t[p..q].
        let fwd = nw_last_row(
            &self.s[a..m],
            &self.t[p..q],
            self.match_score,
            self.mismatch_penalty,
            self.gap_penalty,
        );

        // Backward scores for reverse(s[m..c]) vs reverse(t[p..q]).
        let s_rev: Vec<u8> = self.s[m..c].iter().rev().copied().collect();
        let t_rev: Vec<u8> = self.t[p..q].iter().rev().copied().collect();
        let bwd = nw_last_row(
            &s_rev,
            &t_rev,
            self.match_score,
            self.mismatch_penalty,
            self.gap_penalty,
        );

        let len_t = q - p;
        let mut best_j = 0usize;
        let mut best_val = i32::MIN;

        for j in 0..=len_t {
            let v = fwd[j] + bwd[len_t - j];
            if v > best_val {
                best_val = v;
                best_j = j;
            }
        }

        NwBoundary {
            row: m,
            col: p + best_j,
        }
    }

    fn reconstruct_block(
        &self,
        a: usize,
        b: usize,
        beta_a: &Self::Boundary,
        beta_b: &Self::Boundary,
    ) -> Vec<Self::State> {
        // Local Needleman–Wunsch between (a,p) and (b,q).
        let p = beta_a.col;
        let q = beta_b.col;
        let s_sub = &self.s[a..b];
        let t_sub = &self.t[p..q];
        let n = s_sub.len();
        let m = t_sub.len();

        let mut dp = vec![vec![0i32; m + 1]; n + 1];

        // Initialize with linear gaps
        for i in 1..=n {
            dp[i][0] = dp[i - 1][0] + self.gap_penalty;
        }
        for j in 1..=m {
            dp[0][j] = dp[0][j - 1] + self.gap_penalty;
        }

        // Fill
        for i in 1..=n {
            for j in 1..=m {
                let diag = dp[i - 1][j - 1] + self.score_pair(s_sub[i - 1], t_sub[j - 1]);
                let up = dp[i - 1][j] + self.gap_penalty;
                let left = dp[i][j - 1] + self.gap_penalty;
                dp[i][j] = diag.max(up).max(left);
            }
        }

        // Backtrack
        let mut i = n;
        let mut j = m;
        let mut rev_path = Vec::with_capacity(n + m + 1);
        rev_path.push((a + i, p + j));

        while i > 0 || j > 0 {
            if i > 0
                && j > 0
                && dp[i][j] == dp[i - 1][j - 1] + self.score_pair(s_sub[i - 1], t_sub[j - 1])
            {
                i -= 1;
                j -= 1;
            } else if i > 0 && dp[i][j] == dp[i - 1][j] + self.gap_penalty {
                i -= 1;
            } else if j > 0 && dp[i][j] == dp[i][j - 1] + self.gap_penalty {
                j -= 1;
            } else {
                // Fallback to ensure termination
                if i > 0 {
                    i -= 1;
                } else {
                    j -= 1;
                }
            }
            rev_path.push((a + i, p + j));
        }

        rev_path.reverse();
        rev_path
    }

    fn extract_cost(&self, frontier_t: &Self::Frontier, _beta_t: &Self::Boundary) -> Self::Cost {
        *frontier_t.scores.last().unwrap_or(&0)
    }
}

/// Compute last row of Needleman–Wunsch DP for x vs y.
fn nw_last_row(
    x: &[u8],
    y: &[u8],
    match_score: i32,
    mismatch_penalty: i32,
    gap_penalty: i32,
) -> Vec<i32> {
    let m = y.len();
    let mut prev = Vec::with_capacity(m + 1);
    let mut curr = vec![0i32; m + 1];

    // row 0
    prev.push(0);
    for j in 1..=m {
        prev.push(prev[j - 1] + gap_penalty);
    }

    for &cx in x {
        curr[0] = prev[0] + gap_penalty;
        for j in 1..=m {
            let score = if cx == y[j - 1] {
                match_score
            } else {
                -mismatch_penalty
            };
            let diag = prev[j - 1] + score;
            let up = prev[j] + gap_penalty;
            let left = curr[j - 1] + gap_penalty;
            curr[j] = diag.max(up).max(left);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HcpEngine;

    fn valid_path(path: &[(usize, usize)], n: usize, m: usize) -> bool {
        if path.is_empty() {
            // For degenerate cases with zero layers, an empty path is acceptable.
            return n == 0 || m == 0;
        }
        if *path.first().unwrap() != (0, 0) {
            return false;
        }
        if *path.last().unwrap() != (n, m) {
            return false;
        }
        for w in path.windows(2) {
            let (a, b) = (w[0], w[1]);
            let di = b.0 as isize - a.0 as isize;
            let dj = b.1 as isize - a.1 as isize;
            match (di, dj) {
                (1, 0) | (0, 1) | (1, 1) => {}
                _ => return false,
            }
        }
        true
    }

    #[test]
    fn last_row_small_cases() {
        assert_eq!(nw_last_row(b"", b"", 1, 1, -1), vec![0]);
        assert_eq!(nw_last_row(b"A", b"", 1, 1, -1).len(), 1);
        assert_eq!(nw_last_row(b"", b"A", 1, 1, -1).len(), 2);
    }

    #[test]
    fn e2e_example_scores_and_path() {
        let s = b"GATTACA";
        let t = b"GCATGCU";
        let problem = NwProblem::new(s, t, 1, 1, -1);
        let engine = HcpEngine::new(problem);
        let (score, path) = engine.run();
        assert_eq!(score, 0);
        assert!(valid_path(&path, s.len(), t.len()));
    }

    #[test]
    fn edge_cases_empty_and_identical() {
        let s = b"";
        let t = b"ABC";
        let problem = NwProblem::new(s, t, 1, 1, -1);
        let engine = HcpEngine::new(problem);
        let (_score, path) = engine.run();
        assert!(valid_path(&path, 0, 3));

        let s = b"HELLO";
        let problem = NwProblem::new(s, s, 1, 1, -1);
        let engine = HcpEngine::new(problem);
        let (_score, path) = engine.run();
        assert!(valid_path(&path, s.len(), s.len()));
    }
}
