//! Dense Viterbi dynamic program with exact traceback.

use crate::traits::{HcpProblem, SummaryApply};

const NEG_INF: i32 = i32::MIN / 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViterbiState {
    pub layer: usize,
    pub state: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViterbiBoundary {
    pub layer: usize,
    pub state: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViterbiFrontier {
    pub scores: Vec<i32>,
}

#[derive(Clone, Debug)]
pub struct ViterbiSummary {
    transitions: Vec<Vec<i32>>,
    emissions: Vec<Vec<i32>>,
    start: usize,
    end: usize,
}

#[derive(Clone, Debug)]
pub struct ViterbiProblem {
    initial_scores: Vec<i32>,
    transitions: Vec<Vec<i32>>,
    emissions: Vec<Vec<i32>>,
}

impl ViterbiProblem {
    pub fn new(
        initial_scores: Vec<i32>,
        transitions: Vec<Vec<i32>>,
        emissions: Vec<Vec<i32>>,
    ) -> Result<Self, String> {
        let state_count = initial_scores.len();
        if state_count == 0 {
            return Err("Viterbi problem needs at least one state".to_string());
        }
        if emissions.is_empty() {
            return Err("Viterbi problem needs at least one observation layer".to_string());
        }
        if transitions.len() != state_count
            || transitions.iter().any(|row| row.len() != state_count)
        {
            return Err("transition matrix must be square over all states".to_string());
        }
        if emissions.iter().any(|row| row.len() != state_count) {
            return Err("each emission row must have one score per state".to_string());
        }
        Ok(Self {
            initial_scores,
            transitions,
            emissions,
        })
    }

    pub fn state_count(&self) -> usize {
        self.initial_scores.len()
    }

    pub fn observation_count(&self) -> usize {
        self.emissions.len()
    }

    fn start_state(&self) -> usize {
        self.initial_scores
            .iter()
            .enumerate()
            .max_by_key(|(_, score)| *score)
            .map(|(state, _)| state)
            .unwrap_or(0)
    }

    pub fn full_table_score(&self) -> i32 {
        let mut frontier = self.init_frontier();
        for layer in 0..self.num_layers() {
            frontier = self.forward_step(layer, &frontier);
        }
        frontier.scores.into_iter().max().unwrap_or(NEG_INF)
    }

    pub fn score_path(&self, path: &[ViterbiState]) -> Option<i32> {
        if path.len() != self.num_layers() + 1 {
            return None;
        }
        let first = path.first()?;
        if first.layer != 0 || first.state >= self.state_count() {
            return None;
        }
        let mut score = *self.initial_scores.get(first.state)?;
        for window in path.windows(2) {
            let from = window[0];
            let to = window[1];
            if to.layer != from.layer + 1
                || from.state >= self.state_count()
                || to.state >= self.state_count()
            {
                return None;
            }
            score = score
                .saturating_add(self.transitions[from.state][to.state])
                .saturating_add(self.emissions[from.layer][to.state])
                .max(NEG_INF);
        }
        Some(score)
    }

    fn backward_scores(&self, a: usize, b: usize, boundary: &ViterbiBoundary) -> Vec<i32> {
        let mut next = vec![NEG_INF; self.state_count()];
        next[boundary.state] = 0;
        for layer in (a..b).rev() {
            let mut current = vec![NEG_INF; self.state_count()];
            for (from, current_score) in current.iter_mut().enumerate() {
                for (to, &next_score) in next.iter().enumerate() {
                    if next_score <= NEG_INF / 2 {
                        continue;
                    }
                    let candidate = self.transitions[from][to]
                        .saturating_add(self.emissions[layer][to])
                        .saturating_add(next_score)
                        .max(NEG_INF);
                    *current_score = (*current_score).max(candidate);
                }
            }
            next = current;
        }
        next
    }
}

impl HcpProblem for ViterbiProblem {
    type State = ViterbiState;
    type Frontier = ViterbiFrontier;
    type Summary = ViterbiSummary;
    type Boundary = ViterbiBoundary;
    type Cost = i32;

    fn num_layers(&self) -> usize {
        self.emissions.len()
    }

    fn init_frontier(&self) -> Self::Frontier {
        let mut scores = vec![NEG_INF; self.state_count()];
        let start_state = self.start_state();
        scores[start_state] = self.initial_scores[start_state];
        ViterbiFrontier { scores }
    }

    fn forward_step(&self, layer: usize, frontier: &Self::Frontier) -> Self::Frontier {
        advance_viterbi_layer(&self.transitions, &self.emissions[layer], frontier)
    }

    fn summarize_interval(&self, a: usize, b: usize) -> Self::Summary {
        assert!(a <= b && b <= self.num_layers(), "invalid Viterbi interval");
        ViterbiSummary {
            transitions: self.transitions.clone(),
            emissions: self.emissions.clone(),
            start: a,
            end: b,
        }
    }

    fn merge_summary(&self, left: &Self::Summary, right: &Self::Summary) -> Self::Summary {
        assert_eq!(left.end, right.start, "Viterbi summaries must be adjacent");
        assert_eq!(left.transitions, right.transitions);
        assert_eq!(left.emissions, right.emissions);
        ViterbiSummary {
            transitions: self.transitions.clone(),
            emissions: self.emissions.clone(),
            start: left.start,
            end: right.end,
        }
    }

    fn initial_boundary(&self) -> Self::Boundary {
        ViterbiBoundary {
            layer: 0,
            state: self.start_state(),
        }
    }

    fn terminal_boundary(&self, frontier_t: &Self::Frontier) -> Self::Boundary {
        let state = frontier_t
            .scores
            .iter()
            .enumerate()
            .max_by_key(|(_, score)| *score)
            .map(|(state, _)| state)
            .unwrap_or(0);
        ViterbiBoundary {
            layer: self.num_layers(),
            state,
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
        assert_eq!(beta_a.layer, a);
        assert_eq!(beta_c.layer, c);
        let mut start = vec![NEG_INF; self.state_count()];
        start[beta_a.state] = 0;
        let mut fwd = ViterbiFrontier { scores: start };
        for layer in a..m {
            fwd = self.forward_step(layer, &fwd);
        }
        let bwd = self.backward_scores(m, c, beta_c);
        let mut best_state = 0;
        let mut best_score = NEG_INF;
        for (state, &backward_score) in bwd.iter().enumerate() {
            let score = add_scores(fwd.scores[state], backward_score);
            if score > best_score {
                best_state = state;
                best_score = score;
            }
        }
        ViterbiBoundary {
            layer: m,
            state: best_state,
        }
    }

    fn reconstruct_leaf(
        &self,
        a: usize,
        b: usize,
        beta_a: &Self::Boundary,
        beta_b: &Self::Boundary,
    ) -> Vec<Self::State> {
        assert_eq!(beta_a.layer, a);
        assert_eq!(beta_b.layer, b);
        let width = self.state_count();
        let mut scores = vec![vec![NEG_INF; width]; b - a + 1];
        let mut predecessors = vec![vec![None; width]; b - a + 1];
        scores[0][beta_a.state] = 0;
        for layer in a..b {
            let local = layer - a;
            for from in 0..width {
                if scores[local][from] <= NEG_INF / 2 {
                    continue;
                }
                for to in 0..width {
                    let candidate = scores[local][from]
                        .saturating_add(self.transitions[from][to])
                        .saturating_add(self.emissions[layer][to])
                        .max(NEG_INF);
                    if candidate > scores[local + 1][to] {
                        scores[local + 1][to] = candidate;
                        predecessors[local + 1][to] = Some(from);
                    }
                }
            }
        }
        assert!(
            scores[b - a][beta_b.state] > NEG_INF / 2,
            "Viterbi leaf endpoint must be reachable"
        );
        let mut layer = b;
        let mut state = beta_b.state;
        let mut rev_path = Vec::with_capacity(b - a + 1);
        rev_path.push(ViterbiState { layer, state });
        while layer > a {
            let local = layer - a;
            state = predecessors[local][state].expect("Viterbi traceback must have predecessor");
            layer -= 1;
            rev_path.push(ViterbiState { layer, state });
        }
        rev_path.reverse();
        rev_path
    }

    fn extract_cost(&self, frontier_t: &Self::Frontier, beta_t: &Self::Boundary) -> Self::Cost {
        frontier_t.scores[beta_t.state]
    }
}

impl SummaryApply<ViterbiFrontier> for ViterbiSummary {
    fn apply(&self, frontier: &ViterbiFrontier) -> ViterbiFrontier {
        let mut current = frontier.clone();
        for layer in self.start..self.end {
            current = advance_viterbi_layer(&self.transitions, &self.emissions[layer], &current);
        }
        current
    }
}

fn advance_viterbi_layer(
    transitions: &[Vec<i32>],
    emissions: &[i32],
    frontier: &ViterbiFrontier,
) -> ViterbiFrontier {
    let width = frontier.scores.len();
    let mut next = vec![NEG_INF; width];
    for (from, transition_row) in transitions.iter().enumerate().take(width) {
        if frontier.scores[from] <= NEG_INF / 2 {
            continue;
        }
        for (to, &transition) in transition_row.iter().enumerate().take(width) {
            let candidate = frontier.scores[from]
                .saturating_add(transition)
                .saturating_add(emissions[to])
                .max(NEG_INF);
            next[to] = next[to].max(candidate);
        }
    }
    ViterbiFrontier { scores: next }
}

fn add_scores(left: i32, right: i32) -> i32 {
    if left <= NEG_INF / 2 || right <= NEG_INF / 2 {
        NEG_INF
    } else {
        left.saturating_add(right).max(NEG_INF)
    }
}

#[cfg(test)]
mod tests {
    use super::{ViterbiProblem, ViterbiState};
    use crate::contract::{assert_all_summary_laws, assert_engine_paths_for_all_block_sizes};
    use crate::HcpEngine;

    fn sample_problem() -> ViterbiProblem {
        ViterbiProblem::new(
            vec![0, -4],
            vec![vec![3, -2], vec![-1, 2]],
            vec![vec![2, -3], vec![-4, 3], vec![2, -2]],
        )
        .unwrap()
    }

    #[test]
    fn viterbi_engine_returns_exact_state_path() {
        let problem = sample_problem();
        let (score, path) = HcpEngine::new(problem.clone()).run();
        assert_eq!(score, 9);
        assert_eq!(path.first(), Some(&ViterbiState { layer: 0, state: 0 }));
        assert_eq!(path.last(), Some(&ViterbiState { layer: 3, state: 0 }));
        assert_eq!(problem.score_path(&path), Some(score));
    }

    #[test]
    fn viterbi_satisfies_contract_harness() {
        let problem = sample_problem();
        assert_all_summary_laws(&problem);
        assert_engine_paths_for_all_block_sizes(problem.clone(), |score, path| {
            assert_eq!(score, problem.full_table_score());
            assert_eq!(problem.score_path(path), Some(score));
        });
    }
}
