//! Levenshtein edit distance.

use crate::traits::{HcpProblem, SummaryApply};

#[derive(Clone)]
pub struct EditDistanceProblem<'a> {
    pub s: &'a [u8],
    pub t: &'a [u8],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EditDistanceFrontier {
    pub costs: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct EditDistanceSummary<'a> {
    s: &'a [u8],
    t: &'a [u8],
    start: usize,
    end: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EditDistanceBoundary {
    pub row: usize,
    pub col: usize,
}

pub type EditDistanceState = (usize, usize);

impl<'a> EditDistanceProblem<'a> {
    pub fn new(s: &'a [u8], t: &'a [u8]) -> Self {
        Self { s, t }
    }

    pub fn n(&self) -> usize {
        self.s.len()
    }

    pub fn m(&self) -> usize {
        self.t.len()
    }

    pub fn score_path(&self, path: &[EditDistanceState]) -> Option<u32> {
        let mut distance = 0;
        for window in path.windows(2) {
            let (a, b) = (window[0], window[1]);
            let di = b.0.checked_sub(a.0)?;
            let dj = b.1.checked_sub(a.1)?;
            match (di, dj) {
                (1, 1) => distance += u32::from(self.s.get(a.0)? != self.t.get(a.1)?),
                (1, 0) | (0, 1) => distance += 1,
                _ => return None,
            }
        }
        Some(distance)
    }

    pub fn full_table_distance(&self) -> u32 {
        edit_last_row(self.s, self.t).last().copied().unwrap_or(0)
    }
}

impl<'a> HcpProblem for EditDistanceProblem<'a> {
    type State = EditDistanceState;
    type Frontier = EditDistanceFrontier;
    type Summary = EditDistanceSummary<'a>;
    type Boundary = EditDistanceBoundary;
    type Cost = u32;

    fn num_layers(&self) -> usize {
        self.n()
    }

    fn init_frontier(&self) -> Self::Frontier {
        EditDistanceFrontier {
            costs: (0..=self.m() as u32).collect(),
        }
    }

    fn forward_step(&self, layer: usize, frontier: &Self::Frontier) -> Self::Frontier {
        let ch = self.s[layer];
        let mut next = Vec::with_capacity(self.m() + 1);
        next.push(frontier.costs[0] + 1);
        for col in 1..=self.m() {
            let subst = u32::from(ch != self.t[col - 1]);
            let diag = frontier.costs[col - 1] + subst;
            let delete = frontier.costs[col] + 1;
            let insert = next[col - 1] + 1;
            next.push(diag.min(delete).min(insert));
        }
        EditDistanceFrontier { costs: next }
    }

    fn summarize_interval(&self, a: usize, b: usize) -> Self::Summary {
        assert!(a <= b && b <= self.n(), "invalid edit-distance interval");
        EditDistanceSummary {
            s: self.s,
            t: self.t,
            start: a,
            end: b,
        }
    }

    fn merge_summary(&self, left: &Self::Summary, right: &Self::Summary) -> Self::Summary {
        assert_eq!(
            left.end, right.start,
            "edit-distance summaries must be adjacent"
        );
        assert!(
            std::ptr::eq(left.s, right.s) && std::ptr::eq(left.t, right.t),
            "edit-distance summaries must belong to the same problem"
        );
        EditDistanceSummary {
            s: self.s,
            t: self.t,
            start: left.start,
            end: right.end,
        }
    }

    fn initial_boundary(&self) -> Self::Boundary {
        EditDistanceBoundary { row: 0, col: 0 }
    }

    fn terminal_boundary(&self, _frontier_t: &Self::Frontier) -> Self::Boundary {
        EditDistanceBoundary {
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
        assert!(
            p <= q && q <= self.m(),
            "invalid edit-distance boundary columns"
        );

        let fwd = edit_last_row(&self.s[a..m], &self.t[p..q]);
        let s_rev: Vec<u8> = self.s[m..c].iter().rev().copied().collect();
        let t_rev: Vec<u8> = self.t[p..q].iter().rev().copied().collect();
        let bwd = edit_last_row(&s_rev, &t_rev);

        let width = q - p;
        let mut best_col = p;
        let mut best_cost = u32::MAX;
        for col in 0..=width {
            let cost = fwd[col] + bwd[width - col];
            if cost < best_cost {
                best_cost = cost;
                best_col = p + col;
            }
        }

        EditDistanceBoundary {
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
        assert!(
            p <= q && q <= self.m(),
            "invalid edit-distance leaf columns"
        );

        let s_sub = &self.s[a..b];
        let t_sub = &self.t[p..q];
        let height = s_sub.len();
        let width = t_sub.len();
        let mut dp = vec![vec![0; width + 1]; height + 1];
        for row in 1..=height {
            dp[row][0] = dp[row - 1][0] + 1;
        }
        for col in 1..=width {
            dp[0][col] = dp[0][col - 1] + 1;
        }

        for row in 1..=height {
            for col in 1..=width {
                let subst = u32::from(s_sub[row - 1] != t_sub[col - 1]);
                let diag = dp[row - 1][col - 1] + subst;
                let delete = dp[row - 1][col] + 1;
                let insert = dp[row][col - 1] + 1;
                dp[row][col] = diag.min(delete).min(insert);
            }
        }

        let mut row = height;
        let mut col = width;
        let mut rev_path = Vec::with_capacity(height + width + 1);
        rev_path.push((a + row, p + col));

        while row > 0 || col > 0 {
            if row > 0 && col > 0 {
                let subst = u32::from(s_sub[row - 1] != t_sub[col - 1]);
                if dp[row][col] == dp[row - 1][col - 1] + subst {
                    row -= 1;
                    col -= 1;
                    rev_path.push((a + row, p + col));
                    continue;
                }
            }

            if row > 0 && dp[row][col] == dp[row - 1][col] + 1 {
                row -= 1;
            } else {
                assert!(col > 0, "edit-distance traceback must have predecessor");
                col -= 1;
            }
            rev_path.push((a + row, p + col));
        }

        rev_path.reverse();
        rev_path
    }

    fn extract_cost(&self, frontier_t: &Self::Frontier, _beta_t: &Self::Boundary) -> Self::Cost {
        frontier_t.costs.last().copied().unwrap_or(0)
    }
}

impl<'a> SummaryApply<EditDistanceFrontier> for EditDistanceSummary<'a> {
    fn apply(&self, frontier: &EditDistanceFrontier) -> EditDistanceFrontier {
        assert_eq!(
            frontier.costs.len(),
            self.t.len() + 1,
            "edit-distance frontier width mismatch"
        );
        let mut current = frontier.clone();
        for layer in self.start..self.end {
            let ch = self.s[layer];
            let mut next = Vec::with_capacity(self.t.len() + 1);
            next.push(current.costs[0] + 1);
            for col in 1..=self.t.len() {
                let subst = u32::from(ch != self.t[col - 1]);
                let diag = current.costs[col - 1] + subst;
                let delete = current.costs[col] + 1;
                let insert = next[col - 1] + 1;
                next.push(diag.min(delete).min(insert));
            }
            current = EditDistanceFrontier { costs: next };
        }
        current
    }
}

fn edit_last_row(x: &[u8], y: &[u8]) -> Vec<u32> {
    let mut prev: Vec<u32> = (0..=y.len() as u32).collect();
    let mut curr = vec![0; y.len() + 1];
    for (row, &cx) in x.iter().enumerate() {
        curr[0] = row as u32 + 1;
        for col in 1..=y.len() {
            let subst = u32::from(cx != y[col - 1]);
            let diag = prev[col - 1] + subst;
            let delete = prev[col] + 1;
            let insert = curr[col - 1] + 1;
            curr[col] = diag.min(delete).min(insert);
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
    fn kitten_sitting_distance_is_three() {
        let problem = EditDistanceProblem::new(b"kitten", b"sitting");
        let (distance, path) = HcpEngine::new(problem.clone()).run();
        assert_eq!(distance, 3);
        assert_eq!(problem.score_path(&path), Some(distance));
    }

    #[test]
    fn empty_source_reconstructs_insertions() {
        let problem = EditDistanceProblem::new(b"", b"ABC");
        let (distance, path) = HcpEngine::new(problem.clone()).run();
        assert_eq!(distance, 3);
        assert_eq!(path.first(), Some(&(0, 0)));
        assert_eq!(path.last(), Some(&(0, 3)));
        assert_eq!(problem.score_path(&path), Some(distance));
    }
}
