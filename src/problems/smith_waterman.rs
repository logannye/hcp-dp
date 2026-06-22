//! Smith-Waterman local alignment with linear gap penalties.

use crate::traits::{HcpProblem, SummaryApply};

#[derive(Clone)]
pub struct SmithWatermanProblem<'a> {
    pub s: &'a [u8],
    pub t: &'a [u8],
    pub match_score: i32,
    pub mismatch_penalty: i32,
    pub gap_penalty: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwCell {
    pub row: usize,
    pub col: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwFrontier {
    pub scores: Vec<i32>,
    starts: Vec<SwCell>,
    best: SwSelection,
}

#[derive(Clone, Debug)]
pub struct SwSummary<'a> {
    s: &'a [u8],
    t: &'a [u8],
    start: usize,
    end: usize,
    match_score: i32,
    mismatch_penalty: i32,
    gap_penalty: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwPhase {
    Before,
    Active,
    After,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwBoundary {
    pub row: usize,
    pub col: usize,
    pub phase: SwPhase,
    selected: SwSelection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SwSelection {
    score: i32,
    start: SwCell,
    end: SwCell,
}

impl<'a> SmithWatermanProblem<'a> {
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

    pub fn n(&self) -> usize {
        self.s.len()
    }

    pub fn m(&self) -> usize {
        self.t.len()
    }

    pub fn score_path(&self, path: &[SwCell]) -> Option<i32> {
        if path.is_empty() {
            return Some(0);
        }

        let mut score = 0;
        for window in path.windows(2) {
            let (a, b) = (window[0], window[1]);
            let di = b.row.checked_sub(a.row)?;
            let dj = b.col.checked_sub(a.col)?;
            match (di, dj) {
                (1, 1) => score += self.score_pair(*self.s.get(a.row)?, *self.t.get(a.col)?),
                (1, 0) | (0, 1) => score += self.gap_penalty,
                _ => return None,
            }
        }
        Some(score)
    }

    pub fn full_table_score(&self) -> i32 {
        self.full_table_selection().score
    }

    pub fn full_table_endpoints(&self) -> Option<(SwCell, SwCell)> {
        let selection = self.full_table_selection();
        (selection.score > 0).then_some((selection.start, selection.end))
    }

    fn full_table_selection(&self) -> SwSelection {
        let mut frontier = self.init_frontier();
        for layer in 0..self.n() {
            frontier = self.forward_step(layer, &frontier);
        }
        frontier.best
    }

    fn score_pair(&self, a: u8, b: u8) -> i32 {
        if a == b {
            self.match_score
        } else {
            -self.mismatch_penalty
        }
    }

    fn selected_from_boundaries(&self, beta_a: &SwBoundary, beta_b: &SwBoundary) -> SwSelection {
        if beta_b.selected.score > 0 {
            beta_b.selected
        } else {
            beta_a.selected
        }
    }

    fn path_for_selection(&self, selection: SwSelection) -> Vec<SwCell> {
        if selection.score <= 0 {
            return Vec::new();
        }
        self.reconstruct_global_segment(selection.start, selection.end)
    }

    fn boundary_for_split(&self, row: usize, selection: SwSelection) -> SwBoundary {
        if selection.score <= 0 {
            return SwBoundary {
                row,
                col: 0,
                phase: SwPhase::Before,
                selected: selection,
            };
        }

        let path = self.path_for_selection(selection);
        if let Some(cell) = path.iter().find(|cell| cell.row == row) {
            return SwBoundary {
                row,
                col: cell.col,
                phase: SwPhase::Active,
                selected: selection,
            };
        }

        if row < selection.start.row {
            SwBoundary {
                row,
                col: selection.start.col,
                phase: SwPhase::Before,
                selected: selection,
            }
        } else {
            SwBoundary {
                row,
                col: selection.end.col,
                phase: SwPhase::After,
                selected: selection,
            }
        }
    }

    fn reconstruct_global_segment(&self, start: SwCell, end: SwCell) -> Vec<SwCell> {
        assert!(start.row <= end.row, "SW start row must precede end row");
        assert!(start.col <= end.col, "SW start col must precede end col");
        assert!(
            end.row <= self.n() && end.col <= self.m(),
            "SW endpoint out of bounds"
        );

        let height = end.row - start.row;
        let width = end.col - start.col;
        let cols = width + 1;
        let mut dp = vec![0; (height + 1) * cols];

        for i in 1..=height {
            dp[cell_idx(i, 0, cols)] = dp[cell_idx(i - 1, 0, cols)] + self.gap_penalty;
        }
        for j in 1..=width {
            dp[cell_idx(0, j, cols)] = dp[cell_idx(0, j - 1, cols)] + self.gap_penalty;
        }
        for i in 1..=height {
            for j in 1..=width {
                let pair = self.score_pair(self.s[start.row + i - 1], self.t[start.col + j - 1]);
                let diag = dp[cell_idx(i - 1, j - 1, cols)] + pair;
                let up = dp[cell_idx(i - 1, j, cols)] + self.gap_penalty;
                let left = dp[cell_idx(i, j - 1, cols)] + self.gap_penalty;
                dp[cell_idx(i, j, cols)] = diag.max(up).max(left);
            }
        }

        let mut i = height;
        let mut j = width;
        let mut rev_path = Vec::with_capacity(height + width + 1);
        rev_path.push(end);

        while i > 0 || j > 0 {
            if i > 0 && j > 0 {
                let pair = self.score_pair(self.s[start.row + i - 1], self.t[start.col + j - 1]);
                if dp[cell_idx(i, j, cols)] == dp[cell_idx(i - 1, j - 1, cols)] + pair {
                    i -= 1;
                    j -= 1;
                    rev_path.push(SwCell {
                        row: start.row + i,
                        col: start.col + j,
                    });
                    continue;
                }
            }

            if i > 0 && dp[cell_idx(i, j, cols)] == dp[cell_idx(i - 1, j, cols)] + self.gap_penalty
            {
                i -= 1;
            } else {
                assert!(j > 0, "SW traceback must have a horizontal predecessor");
                j -= 1;
            }
            rev_path.push(SwCell {
                row: start.row + i,
                col: start.col + j,
            });
        }

        rev_path.reverse();
        rev_path
    }
}

impl<'a> HcpProblem for SmithWatermanProblem<'a> {
    type State = SwCell;
    type Frontier = SwFrontier;
    type Summary = SwSummary<'a>;
    type Boundary = SwBoundary;
    type Cost = i32;

    fn num_layers(&self) -> usize {
        self.n()
    }

    fn init_frontier(&self) -> Self::Frontier {
        SwFrontier {
            scores: vec![0; self.m() + 1],
            starts: (0..=self.m()).map(|col| SwCell { row: 0, col }).collect(),
            best: SwSelection {
                score: 0,
                start: SwCell { row: 0, col: 0 },
                end: SwCell { row: 0, col: 0 },
            },
        }
    }

    fn forward_step(&self, layer: usize, frontier: &Self::Frontier) -> Self::Frontier {
        advance_sw_row(
            layer,
            self.s[layer],
            self.t,
            self.match_score,
            self.mismatch_penalty,
            self.gap_penalty,
            frontier,
        )
    }

    fn summarize_interval(&self, a: usize, b: usize) -> Self::Summary {
        assert!(a <= b && b <= self.n(), "invalid SW interval");
        SwSummary {
            s: self.s,
            t: self.t,
            start: a,
            end: b,
            match_score: self.match_score,
            mismatch_penalty: self.mismatch_penalty,
            gap_penalty: self.gap_penalty,
        }
    }

    fn merge_summary(&self, left: &Self::Summary, right: &Self::Summary) -> Self::Summary {
        assert_eq!(left.end, right.start, "SW summaries must be adjacent");
        assert!(
            std::ptr::eq(left.s, right.s) && std::ptr::eq(left.t, right.t),
            "SW summaries must belong to the same problem"
        );
        assert_eq!(
            (left.match_score, left.mismatch_penalty, left.gap_penalty),
            (right.match_score, right.mismatch_penalty, right.gap_penalty)
        );
        SwSummary {
            s: self.s,
            t: self.t,
            start: left.start,
            end: right.end,
            match_score: self.match_score,
            mismatch_penalty: self.mismatch_penalty,
            gap_penalty: self.gap_penalty,
        }
    }

    fn initial_boundary(&self) -> Self::Boundary {
        SwBoundary {
            row: 0,
            col: 0,
            phase: SwPhase::Before,
            selected: SwSelection {
                score: 0,
                start: SwCell { row: 0, col: 0 },
                end: SwCell { row: 0, col: 0 },
            },
        }
    }

    fn terminal_boundary(&self, frontier_t: &Self::Frontier) -> Self::Boundary {
        let selected = frontier_t.best;
        if selected.score <= 0 {
            SwBoundary {
                row: self.n(),
                col: 0,
                phase: SwPhase::Before,
                selected,
            }
        } else {
            SwBoundary {
                row: self.n(),
                col: selected.end.col,
                phase: SwPhase::After,
                selected,
            }
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
        let selection = self.selected_from_boundaries(beta_a, beta_c);
        self.boundary_for_split(m, selection)
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
        let selection = self.selected_from_boundaries(beta_a, beta_b);
        let path = self.path_for_selection(selection);
        if path.is_empty() {
            return Vec::new();
        }

        match (beta_a.phase, beta_b.phase) {
            (SwPhase::Before, SwPhase::Before) | (SwPhase::After, SwPhase::After) => {
                return Vec::new();
            }
            (SwPhase::After, _) | (_, SwPhase::Before) => return Vec::new(),
            _ => {}
        }

        let start_idx = boundary_path_index(&path, beta_a, true);
        let end_idx = boundary_path_index(&path, beta_b, false);
        if start_idx > end_idx {
            return Vec::new();
        }

        path[start_idx..=end_idx].to_vec()
    }

    fn extract_cost(&self, _frontier_t: &Self::Frontier, beta_t: &Self::Boundary) -> Self::Cost {
        beta_t.selected.score
    }
}

impl<'a> SummaryApply<SwFrontier> for SwSummary<'a> {
    fn apply(&self, frontier: &SwFrontier) -> SwFrontier {
        assert_eq!(
            frontier.scores.len(),
            self.t.len() + 1,
            "SW frontier width mismatch"
        );
        let mut current = frontier.clone();
        for layer in self.start..self.end {
            current = advance_sw_row(
                layer,
                self.s[layer],
                self.t,
                self.match_score,
                self.mismatch_penalty,
                self.gap_penalty,
                &current,
            );
        }
        current
    }
}

fn advance_sw_row(
    layer: usize,
    s_ch: u8,
    t: &[u8],
    match_score: i32,
    mismatch_penalty: i32,
    gap_penalty: i32,
    frontier: &SwFrontier,
) -> SwFrontier {
    assert_eq!(
        frontier.scores.len(),
        t.len() + 1,
        "SW frontier width mismatch"
    );
    assert_eq!(
        frontier.starts.len(),
        t.len() + 1,
        "SW frontier start width mismatch"
    );

    let row = layer + 1;
    let mut scores = vec![0; t.len() + 1];
    let mut starts = vec![SwCell { row, col: 0 }; t.len() + 1];
    let mut best = frontier.best;

    for col in 1..=t.len() {
        let pair = if s_ch == t[col - 1] {
            match_score
        } else {
            -mismatch_penalty
        };
        let reset = Candidate {
            score: 0,
            start: SwCell { row, col },
        };
        let diag = Candidate {
            score: frontier.scores[col - 1] + pair,
            start: frontier.starts[col - 1],
        };
        let up = Candidate {
            score: frontier.scores[col] + gap_penalty,
            start: frontier.starts[col],
        };
        let left = Candidate {
            score: scores[col - 1] + gap_penalty,
            start: starts[col - 1],
        };
        let chosen = choose_sw_cell(reset, diag, up, left);
        scores[col] = chosen.score;
        starts[col] = chosen.start;

        if chosen.score > best.score {
            best = SwSelection {
                score: chosen.score,
                start: chosen.start,
                end: SwCell { row, col },
            };
        }
    }

    SwFrontier {
        scores,
        starts,
        best,
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    score: i32,
    start: SwCell,
}

fn choose_sw_cell(reset: Candidate, diag: Candidate, up: Candidate, left: Candidate) -> Candidate {
    [diag, up, left]
        .into_iter()
        .filter(|candidate| candidate.score > 0)
        .max_by_key(|candidate| candidate.score)
        .unwrap_or(reset)
}

fn boundary_path_index(path: &[SwCell], boundary: &SwBoundary, start_side: bool) -> usize {
    match boundary.phase {
        SwPhase::Before => 0,
        SwPhase::After => path.len() - 1,
        SwPhase::Active => {
            let cell = SwCell {
                row: boundary.row,
                col: boundary.col,
            };
            if start_side {
                path.iter()
                    .position(|path_cell| *path_cell == cell)
                    .expect("SW active start boundary must lie on selected path")
            } else {
                path.iter()
                    .rposition(|path_cell| *path_cell == cell)
                    .expect("SW active end boundary must lie on selected path")
            }
        }
    }
}

fn cell_idx(row: usize, col: usize, cols: usize) -> usize {
    row * cols + col
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HcpEngine;

    #[test]
    fn classic_local_alignment_path_realizes_score() {
        let problem = SmithWatermanProblem::new(b"ACACACTA", b"AGCACACA", 2, 1, -2);
        let (score, path) = HcpEngine::new(problem.clone()).run();
        assert_eq!(score, problem.full_table_score());
        assert_eq!(problem.score_path(&path), Some(score));
        assert!(score > 0);
    }

    #[test]
    fn no_positive_alignment_returns_empty_path() {
        let problem = SmithWatermanProblem::new(b"AAAA", b"TTTT", 1, 3, -2);
        let (score, path) = HcpEngine::new(problem.clone()).run();
        assert_eq!(score, 0);
        assert!(path.is_empty());
        assert_eq!(problem.score_path(&path), Some(0));
    }
}
