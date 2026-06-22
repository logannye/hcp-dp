//! Longest Common Subsequence as the canonical HCP-DP example.

use crate::traits::{HcpProblem, SummaryApply};

/// LCS problem over two byte strings.
#[derive(Clone)]
pub struct LcsProblem<'a> {
    pub s: &'a [u8],
    pub t: &'a [u8],
}

/// DP row over all prefixes of `t`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LcsFrontier {
    pub scores: Vec<u32>,
}

/// Boundary-independent summary for source interval `[start, end)`.
#[derive(Clone, Debug)]
pub struct LcsSummary<'a> {
    s: &'a [u8],
    t: &'a [u8],
    start: usize,
    end: usize,
}

/// Fixed DP cell boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LcsBoundary {
    pub row: usize,
    pub col: usize,
}

pub type LcsState = (usize, usize);

impl<'a> LcsProblem<'a> {
    pub fn new(s: &'a [u8], t: &'a [u8]) -> Self {
        Self { s, t }
    }

    pub fn n(&self) -> usize {
        self.s.len()
    }

    pub fn m(&self) -> usize {
        self.t.len()
    }

    /// Score a returned path by counting matching diagonal moves.
    pub fn score_path(&self, path: &[LcsState]) -> Option<u32> {
        let mut score = 0;
        for window in path.windows(2) {
            let (a, b) = (window[0], window[1]);
            let di = b.0.checked_sub(a.0)?;
            let dj = b.1.checked_sub(a.1)?;
            match (di, dj) {
                (1, 1) => {
                    if self.s.get(a.0)? != self.t.get(a.1)? {
                        return None;
                    }
                    score += 1;
                }
                (1, 0) | (0, 1) => {}
                _ => return None,
            }
        }
        Some(score)
    }

    pub fn full_table_len(&self) -> u32 {
        lcs_last_row(self.s, self.t).last().copied().unwrap_or(0)
    }
}

impl<'a> HcpProblem for LcsProblem<'a> {
    type State = LcsState;
    type Frontier = LcsFrontier;
    type Summary = LcsSummary<'a>;
    type Boundary = LcsBoundary;
    type Cost = u32;

    fn num_layers(&self) -> usize {
        self.n()
    }

    fn init_frontier(&self) -> Self::Frontier {
        LcsFrontier {
            scores: vec![0; self.m() + 1],
        }
    }

    fn forward_step(&self, layer: usize, frontier: &Self::Frontier) -> Self::Frontier {
        let ch = self.s[layer];
        let mut next = vec![0; self.m() + 1];
        for j in 1..=self.m() {
            let up = frontier.scores[j];
            let left = next[j - 1];
            let diag = frontier.scores[j - 1] + u32::from(self.t[j - 1] == ch);
            next[j] = up.max(left).max(diag);
        }
        LcsFrontier { scores: next }
    }

    fn summarize_interval(&self, a: usize, b: usize) -> Self::Summary {
        assert!(a <= b && b <= self.n(), "invalid LCS interval");
        LcsSummary {
            s: self.s,
            t: self.t,
            start: a,
            end: b,
        }
    }

    fn merge_summary(&self, left: &Self::Summary, right: &Self::Summary) -> Self::Summary {
        assert_eq!(left.end, right.start, "LCS summaries must be adjacent");
        assert!(
            std::ptr::eq(left.s, right.s) && std::ptr::eq(left.t, right.t),
            "LCS summaries must belong to the same problem"
        );
        LcsSummary {
            s: self.s,
            t: self.t,
            start: left.start,
            end: right.end,
        }
    }

    fn initial_boundary(&self) -> Self::Boundary {
        LcsBoundary { row: 0, col: 0 }
    }

    fn terminal_boundary(&self, _frontier_t: &Self::Frontier) -> Self::Boundary {
        LcsBoundary {
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
        assert!(a <= m && m <= c, "split must lie inside interval");

        let p = beta_a.col;
        let q = beta_c.col;
        assert!(p <= q && q <= self.m(), "invalid LCS boundary columns");

        let fwd = lcs_last_row(&self.s[a..m], &self.t[p..q]);
        let s_rev: Vec<u8> = self.s[m..c].iter().rev().copied().collect();
        let t_rev: Vec<u8> = self.t[p..q].iter().rev().copied().collect();
        let bwd = lcs_last_row(&s_rev, &t_rev);

        let width = q - p;
        let mut best_col = p;
        let mut best_score = 0;
        for j in 0..=width {
            let score = fwd[j] + bwd[width - j];
            if score > best_score {
                best_score = score;
                best_col = p + j;
            }
        }

        LcsBoundary {
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
        assert!(p <= q && q <= self.m(), "invalid LCS leaf columns");

        let s_sub = &self.s[a..b];
        let t_sub = &self.t[p..q];
        let n = s_sub.len();
        let m = t_sub.len();

        let mut dp = vec![vec![0; m + 1]; n + 1];
        for i in 1..=n {
            for j in 1..=m {
                let up = dp[i - 1][j];
                let left = dp[i][j - 1];
                let diag = dp[i - 1][j - 1] + u32::from(s_sub[i - 1] == t_sub[j - 1]);
                dp[i][j] = up.max(left).max(diag);
            }
        }

        let mut i = n;
        let mut j = m;
        let mut rev_path = Vec::with_capacity(n + m + 1);
        rev_path.push((a + i, p + j));

        while i > 0 || j > 0 {
            if i > 0 && j > 0 && s_sub[i - 1] == t_sub[j - 1] && dp[i][j] == dp[i - 1][j - 1] + 1 {
                i -= 1;
                j -= 1;
            } else if i > 0 && dp[i][j] == dp[i - 1][j] {
                i -= 1;
            } else {
                j -= 1;
            }
            rev_path.push((a + i, p + j));
        }

        rev_path.reverse();
        rev_path
    }

    fn extract_cost(&self, frontier_t: &Self::Frontier, _beta_t: &Self::Boundary) -> Self::Cost {
        frontier_t.scores.last().copied().unwrap_or(0)
    }
}

impl<'a> SummaryApply<LcsFrontier> for LcsSummary<'a> {
    fn apply(&self, frontier: &LcsFrontier) -> LcsFrontier {
        assert_eq!(
            frontier.scores.len(),
            self.t.len() + 1,
            "LCS frontier width mismatch"
        );
        let mut current = frontier.clone();
        for layer in self.start..self.end {
            let ch = self.s[layer];
            let mut next = vec![0; self.t.len() + 1];
            for j in 1..=self.t.len() {
                let up = current.scores[j];
                let left = next[j - 1];
                let diag = current.scores[j - 1] + u32::from(self.t[j - 1] == ch);
                next[j] = up.max(left).max(diag);
            }
            current = LcsFrontier { scores: next };
        }
        current
    }
}

fn lcs_last_row(x: &[u8], y: &[u8]) -> Vec<u32> {
    let mut prev = vec![0; y.len() + 1];
    let mut curr = vec![0; y.len() + 1];
    for &cx in x {
        curr[0] = 0;
        for j in 1..=y.len() {
            let up = prev[j];
            let left = curr[j - 1];
            let diag = prev[j - 1] + u32::from(cx == y[j - 1]);
            curr[j] = up.max(left).max(diag);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HcpEngine;

    #[test]
    fn audit_regression_path_realizes_cost() {
        let problem = LcsProblem::new(b"CCA", b"C");
        let (cost, path) = HcpEngine::new(problem.clone()).run();
        assert_eq!(cost, 1);
        assert_eq!(problem.score_path(&path), Some(cost));
    }

    #[test]
    fn empty_source_reconstructs_horizontal_path() {
        let problem = LcsProblem::new(b"", b"ABC");
        let (cost, path) = HcpEngine::new(problem.clone()).run();
        assert_eq!(cost, 0);
        assert_eq!(path.first(), Some(&(0, 0)));
        assert_eq!(path.last(), Some(&(0, 3)));
        assert_eq!(problem.score_path(&path), Some(0));
    }
}
