//! Example: implement a custom HCP-DP problem.
//!
//! Run with:
//! `cargo run --example custom_problem`
//!
//! This models a two-state layered shortest-path recurrence. Each layer has a
//! 2x2 transition-cost matrix. Interval summaries are min-plus transition
//! operators, so they apply to any compatible input frontier and compose by
//! min-plus matrix multiplication.

use hcp_dp::{
    contract::{
        assert_all_summary_laws, assert_engine_paths_for_all_block_sizes, reconstruct_top_split,
    },
    HcpEngine, HcpProblem, SummaryApply,
};

const STATES: usize = 2;
const INF: u32 = u32::MAX / 4;

#[derive(Clone)]
struct TwoStateShortestPath {
    steps: Vec<Step>,
}

#[derive(Clone, Copy)]
struct Step {
    cost: [[u32; STATES]; STATES],
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TwoStateSummary {
    cost: [[u32; STATES]; STATES],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TwoStateBoundary {
    layer: usize,
    state: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TwoStatePathState {
    layer: usize,
    state: usize,
}

impl TwoStateShortestPath {
    fn new(steps: Vec<Step>) -> Self {
        Self { steps }
    }

    fn full_table_cost(&self) -> u32 {
        let mut frontier = self.init_frontier();
        for layer in 0..self.num_layers() {
            frontier = self.forward_step(layer, &frontier);
        }
        frontier[0].min(frontier[1])
    }

    fn score_path(&self, path: &[TwoStatePathState]) -> Option<u32> {
        if path.len() != self.num_layers() + 1 {
            return None;
        }
        if path.first()?.layer != 0 || path.first()?.state != 0 {
            return None;
        }
        let mut cost = 0u32;
        for pair in path.windows(2) {
            let prev = pair[0];
            let next = pair[1];
            if next.layer != prev.layer + 1 || prev.state >= STATES || next.state >= STATES {
                return None;
            }
            cost = add_cost(cost, self.steps[prev.layer].cost[prev.state][next.state]);
        }
        Some(cost)
    }
}

impl SummaryApply<[u32; STATES]> for TwoStateSummary {
    fn apply(&self, frontier: &[u32; STATES]) -> [u32; STATES] {
        let mut out = [INF; STATES];
        for (to, out_cost) in out.iter_mut().enumerate() {
            for (from, frontier_cost) in frontier.iter().enumerate() {
                *out_cost = (*out_cost).min(add_cost(*frontier_cost, self.cost[from][to]));
            }
        }
        out
    }
}

impl HcpProblem for TwoStateShortestPath {
    type Boundary = TwoStateBoundary;
    type Cost = u32;
    type Frontier = [u32; STATES];
    type State = TwoStatePathState;
    type Summary = TwoStateSummary;

    fn num_layers(&self) -> usize {
        self.steps.len()
    }

    fn init_frontier(&self) -> Self::Frontier {
        [0, INF]
    }

    fn forward_step(&self, layer: usize, frontier: &Self::Frontier) -> Self::Frontier {
        step_apply(&self.steps[layer], frontier)
    }

    fn summarize_interval(&self, a: usize, b: usize) -> Self::Summary {
        let mut summary = identity_summary();
        for step in &self.steps[a..b] {
            summary = merge_matrices(&summary, &TwoStateSummary { cost: step.cost });
        }
        summary
    }

    fn merge_summary(&self, left: &Self::Summary, right: &Self::Summary) -> Self::Summary {
        merge_matrices(left, right)
    }

    fn initial_boundary(&self) -> Self::Boundary {
        TwoStateBoundary { layer: 0, state: 0 }
    }

    fn terminal_boundary(&self, frontier_t: &Self::Frontier) -> Self::Boundary {
        let state = if frontier_t[0] <= frontier_t[1] { 0 } else { 1 };
        TwoStateBoundary {
            layer: self.num_layers(),
            state,
        }
    }

    fn choose_split(
        &self,
        _a: usize,
        m: usize,
        _c: usize,
        beta_a: &Self::Boundary,
        beta_c: &Self::Boundary,
        sigma_left: &Self::Summary,
        sigma_right: &Self::Summary,
    ) -> Self::Boundary {
        let mut best_state = 0;
        let mut best_cost = INF;
        for state in 0..STATES {
            let cost = add_cost(
                sigma_left.cost[beta_a.state][state],
                sigma_right.cost[state][beta_c.state],
            );
            if cost < best_cost {
                best_cost = cost;
                best_state = state;
            }
        }
        TwoStateBoundary {
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
        if a == b {
            assert_eq!(beta_a.state, beta_b.state);
            return vec![TwoStatePathState {
                layer: a,
                state: beta_a.state,
            }];
        }

        let mut costs = [INF; STATES];
        costs[beta_a.state] = 0;
        let mut parents = Vec::with_capacity(b - a);

        for step in &self.steps[a..b] {
            let mut next = [INF; STATES];
            let mut parent = [0; STATES];
            for (from, from_cost) in costs.iter().enumerate() {
                for to in 0..STATES {
                    let candidate = add_cost(*from_cost, step.cost[from][to]);
                    if candidate < next[to] {
                        next[to] = candidate;
                        parent[to] = from;
                    }
                }
            }
            parents.push(parent);
            costs = next;
        }

        assert!(costs[beta_b.state] < INF, "endpoint must be reachable");

        let mut states = vec![0; b - a + 1];
        states[b - a] = beta_b.state;
        let mut current = beta_b.state;
        for (idx, parent) in parents.iter().enumerate().rev() {
            current = parent[current];
            states[idx] = current;
        }
        assert_eq!(states[0], beta_a.state);

        states
            .into_iter()
            .enumerate()
            .map(|(offset, state)| TwoStatePathState {
                layer: a + offset,
                state,
            })
            .collect()
    }

    fn extract_cost(&self, frontier_t: &Self::Frontier, beta_t: &Self::Boundary) -> Self::Cost {
        frontier_t[beta_t.state]
    }
}

fn main() {
    let problem = TwoStateShortestPath::new(vec![
        Step {
            cost: [[3, 1], [5, 4]],
        },
        Step {
            cost: [[2, 2], [4, 1]],
        },
        Step {
            cost: [[2, 5], [1, 3]],
        },
        Step {
            cost: [[1, 2], [4, 1]],
        },
    ]);

    assert_all_summary_laws(&problem);
    if let Some(path) = reconstruct_top_split(&problem, problem.num_layers() / 2) {
        assert_eq!(problem.score_path(&path), Some(problem.full_table_cost()));
    }
    assert_engine_paths_for_all_block_sizes(problem.clone(), |cost, path| {
        assert_eq!(cost, problem.full_table_cost());
        assert_eq!(problem.score_path(path), Some(cost));
    });

    let (cost, path) = HcpEngine::new(problem).run();
    println!("Two-state shortest path cost: {cost}");
    println!("Path: {}", format_path(&path));
}

fn step_apply(step: &Step, frontier: &[u32; STATES]) -> [u32; STATES] {
    let mut out = [INF; STATES];
    for (to, out_cost) in out.iter_mut().enumerate() {
        for (from, frontier_cost) in frontier.iter().enumerate() {
            *out_cost = (*out_cost).min(add_cost(*frontier_cost, step.cost[from][to]));
        }
    }
    out
}

fn identity_summary() -> TwoStateSummary {
    TwoStateSummary {
        cost: [[0, INF], [INF, 0]],
    }
}

fn merge_matrices(left: &TwoStateSummary, right: &TwoStateSummary) -> TwoStateSummary {
    let mut cost = [[INF; STATES]; STATES];
    for (from, row) in cost.iter_mut().enumerate() {
        for (to, cell) in row.iter_mut().enumerate() {
            for mid in 0..STATES {
                *cell = (*cell).min(add_cost(left.cost[from][mid], right.cost[mid][to]));
            }
        }
    }
    TwoStateSummary { cost }
}

fn add_cost(a: u32, b: u32) -> u32 {
    if a >= INF || b >= INF {
        INF
    } else {
        a.saturating_add(b).min(INF)
    }
}

fn format_path(path: &[TwoStatePathState]) -> String {
    path.iter()
        .map(|state| format!("{}:{}", state.layer, state_name(state.state)))
        .collect::<Vec<_>>()
        .join(" -> ")
}

fn state_name(state: usize) -> &'static str {
    match state {
        0 => "A",
        1 => "B",
        _ => "?",
    }
}
