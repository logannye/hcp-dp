//! Banded Longest Common Subsequence (LCS).
//! Limits computation to a diagonal band of width `band` around the main diagonal.
//! Note: For too-narrow bands the result can be suboptimal; with wide enough
//! bands (>= |n - m| plus slack), it matches full LCS.

use crate::traits::{HcpProblem, SummaryApply};

#[derive(Clone)]
pub struct LcsBandedProblem<'a> {
    pub s: &'a [u8],
    pub t: &'a [u8],
    pub band: usize,
}

#[derive(Clone, Debug)]
pub struct LbFrontier {
    pub scores: Vec<u32>, // length = t.len() + 1
}

#[derive(Clone, Debug)]
pub struct LbSummary {
    pub end_frontier: LbFrontier,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LbBoundary {
    pub row: usize,
    pub col: usize,
}

pub type LbState = (usize, usize);

impl<'a> LcsBandedProblem<'a> {
    pub fn new(s: &'a [u8], t: &'a [u8], band: usize) -> Self {
        Self { s, t, band }
    }
    fn n(&self) -> usize {
        self.s.len()
    }
    fn m(&self) -> usize {
        self.t.len()
    }
}

impl<'a> HcpProblem for LcsBandedProblem<'a> {
    type State = LbState;
    type Frontier = LbFrontier;
    type Summary = LbSummary;
    type Boundary = LbBoundary;
    type Cost = u32;

    fn num_layers(&self) -> usize {
        self.n()
    }

    fn init_frontier(&self) -> Self::Frontier {
        LbFrontier {
            scores: vec![0; self.m() + 1],
        }
    }

    fn forward_step(&self, layer: usize, frontier_i: &Self::Frontier) -> Self::Frontier {
        let i = layer;
        let ch = self.s[i];
        let m = self.m();
        let mut next = frontier_i.scores.clone();

        // band centered around diagonal j ≈ i * m / n (approx), but for simplicity
        // we use main diagonal with width `band`.
        let center = (i as isize) * (m as isize) / (self.n().max(1) as isize);
        let start = ((center - self.band as isize).max(1)).min(m as isize) as usize;
        let end = ((center + self.band as isize).max(1)).min(m as isize) as usize;

        // compute within band using standard recurrence; outside, keep monotone max with left.
        for j in 1..=m {
            if j < start || j > end {
                let left = next[j - 1];
                let up = frontier_i.scores[j];
                next[j] = left.max(up);
            } else {
                let up = frontier_i.scores[j];
                let left = next[j - 1];
                let diag = frontier_i.scores[j - 1] + if self.t[j - 1] == ch { 1 } else { 0 };
                next[j] = up.max(left).max(diag);
            }
        }

        LbFrontier { scores: next }
    }

    fn summarize_block(
        &self,
        a: usize,
        b: usize,
        frontier_a: &Self::Frontier,
    ) -> (Self::Frontier, Self::Summary) {
        let mut f = frontier_a.clone();
        for layer in a..b {
            f = self.forward_step(layer, &f);
        }
        (f.clone(), LbSummary { end_frontier: f })
    }

    fn merge_summary(&self, _left: &Self::Summary, right: &Self::Summary) -> Self::Summary {
        right.clone()
    }

    fn initial_boundary(&self) -> Self::Boundary {
        LbBoundary { row: 0, col: 0 }
    }

    fn terminal_boundary(&self, _frontier_t: &Self::Frontier) -> Self::Boundary {
        LbBoundary {
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
        let p = beta_a.col;
        let q = beta_c.col;
        let fwd = lcs_last_row_banded(&self.s[a..m], &self.t[p..q], self.band);
        let s_rev: Vec<u8> = self.s[m..c].iter().rev().copied().collect();
        let t_rev: Vec<u8> = self.t[p..q].iter().rev().copied().collect();
        let bwd = lcs_last_row_banded(&s_rev, &t_rev, self.band);
        let len_t_sub = q - p;
        let mut best_j = 0usize;
        let mut best_val = 0u32;
        for j in 0..=len_t_sub {
            let v = fwd[j] + bwd[len_t_sub - j];
            if v > best_val {
                best_val = v;
                best_j = j;
            }
        }
        LbBoundary {
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
        // Use full-table reconstruction for robustness inside a block.
        let p = beta_a.col;
        let q = beta_b.col;
        let s_sub = &self.s[a..b];
        let t_sub = &self.t[p..q];
        let n = s_sub.len();
        let m = t_sub.len();
        let mut dp = vec![vec![0u32; m + 1]; n + 1];
        for i in 1..=n {
            for j in 1..=m {
                let up = dp[i - 1][j];
                let left = dp[i][j - 1];
                let diag = dp[i - 1][j - 1] + if s_sub[i - 1] == t_sub[j - 1] { 1 } else { 0 };
                dp[i][j] = up.max(left).max(diag);
            }
        }
        let mut i = n;
        let mut j = m;
        let mut rev = Vec::with_capacity(n + m + 1);
        rev.push((a + i, p + j));
        while i > 0 || j > 0 {
            if i > 0 && j > 0 && s_sub[i - 1] == t_sub[j - 1] && dp[i][j] == dp[i - 1][j - 1] + 1 {
                i -= 1;
                j -= 1;
            } else if i > 0 && dp[i][j] == dp[i - 1][j] {
                i -= 1;
            } else if j > 0 && dp[i][j] == dp[i][j - 1] {
                j -= 1;
            } else if i > 0 {
                i -= 1;
            } else {
                j -= 1;
            }
            rev.push((a + i, p + j));
        }
        rev.reverse();
        rev
    }

    fn extract_cost(&self, frontier_t: &Self::Frontier, _beta_t: &Self::Boundary) -> Self::Cost {
        *frontier_t.scores.last().unwrap_or(&0)
    }
}

impl SummaryApply<LbFrontier> for LbSummary {
    fn apply(&self, _frontier: &LbFrontier) -> LbFrontier {
        self.end_frontier.clone()
    }
}

fn lcs_last_row_banded(x: &[u8], y: &[u8], band: usize) -> Vec<u32> {
    let m = y.len();
    let mut prev = vec![0u32; m + 1];
    let mut curr = vec![0u32; m + 1];
    for (i, &cx) in x.iter().enumerate() {
        let center = (i as isize) * (m as isize) / (x.len().max(1) as isize);
        let start = ((center - band as isize).max(1)).min(m as isize) as usize;
        let end = ((center + band as isize).max(1)).min(m as isize) as usize;
        curr[0] = 0;
        for j in 1..=m {
            if j < start || j > end {
                let up = prev[j];
                let left = curr[j - 1];
                curr[j] = up.max(left);
            } else {
                let up = prev[j];
                let left = curr[j - 1];
                let diag = prev[j - 1] + if cx == y[j - 1] { 1 } else { 0 };
                curr[j] = up.max(left).max(diag);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problems::lcs::LcsProblem;
    use crate::HcpEngine;

    #[test]
    fn banded_matches_full_when_wide() {
        let s = b"ACCGGTCGAGTGCGCGGAAGCCGGCCGAA";
        let t = b"GTCGTTCGGAATGCCGTTGCTCTGTAAA";
        let full = {
            let (len, _) = HcpEngine::new(LcsProblem::new(s, t)).run();
            len
        };
        let band = 64;
        let (band_len, _path) = HcpEngine::new(LcsBandedProblem::new(s, t, band)).run();
        assert_eq!(band_len, full);
    }

    #[test]
    fn narrow_band_not_crashing() {
        let s = b"HELLO";
        let t = b"WORLD";
        let (_len, _path) = HcpEngine::new(LcsBandedProblem::new(s, t, 1)).run();
    }
}
