//! Sparse layered-DAG longest-path problem.
//!
//! This module is a compact proof point that the HCP contract can operate over
//! sparse graph frontiers, not only dense rectangular sequence DP grids.

use crate::traits::{HcpProblem, SummaryApply};

const NEG_INF: i32 = i32::MIN / 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DagEdge {
    pub from: usize,
    pub to: usize,
    pub score: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DagState {
    pub layer: usize,
    pub node: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DagBoundary {
    pub layer: usize,
    pub node: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DagFrontier {
    pub scores: Vec<i32>,
}

#[derive(Clone, Debug)]
pub struct DagSummary {
    layer_widths: Vec<usize>,
    edges: Vec<Vec<DagEdge>>,
    start: usize,
    end: usize,
}

#[derive(Clone, Debug)]
pub struct LayeredDagProblem {
    layer_widths: Vec<usize>,
    edges: Vec<Vec<DagEdge>>,
    start_node: usize,
    end_node: usize,
}

impl LayeredDagProblem {
    pub fn new(
        layer_widths: Vec<usize>,
        edges: Vec<Vec<DagEdge>>,
        start_node: usize,
        end_node: usize,
    ) -> Result<Self, String> {
        if layer_widths.len() < 2 {
            return Err("layered DAG needs at least two layers".to_string());
        }
        if edges.len() + 1 != layer_widths.len() {
            return Err("edge layers must be one shorter than node layers".to_string());
        }
        if start_node >= layer_widths[0] {
            return Err("start node is outside layer 0".to_string());
        }
        let last_width = *layer_widths
            .last()
            .expect("validated DAG has a terminal layer");
        if end_node >= last_width {
            return Err("end node is outside the terminal layer".to_string());
        }
        for (layer, layer_edges) in edges.iter().enumerate() {
            let from_width = layer_widths[layer];
            let to_width = layer_widths[layer + 1];
            for edge in layer_edges {
                if edge.from >= from_width || edge.to >= to_width {
                    return Err(format!("edge in layer {layer} references an invalid node"));
                }
            }
        }
        Ok(Self {
            layer_widths,
            edges,
            start_node,
            end_node,
        })
    }

    pub fn layer_count(&self) -> usize {
        self.layer_widths.len()
    }

    pub fn full_table_score(&self) -> i32 {
        let mut frontier = self.init_frontier();
        for layer in 0..self.num_layers() {
            frontier = self.forward_step(layer, &frontier);
        }
        frontier.scores[self.end_node]
    }

    pub fn score_path(&self, path: &[DagState]) -> Option<i32> {
        if path.first()
            != Some(&DagState {
                layer: 0,
                node: self.start_node,
            })
        {
            return None;
        }
        if path.last()
            != Some(&DagState {
                layer: self.num_layers(),
                node: self.end_node,
            })
        {
            return None;
        }

        let mut score = 0;
        for window in path.windows(2) {
            let a = window[0];
            let b = window[1];
            if b.layer != a.layer + 1 {
                return None;
            }
            let edge_score = self.edges[a.layer]
                .iter()
                .filter(|edge| edge.from == a.node && edge.to == b.node)
                .map(|edge| edge.score)
                .max()?;
            score += edge_score;
        }
        Some(score)
    }

    fn frontier_from_boundary(&self, boundary: &DagBoundary) -> DagFrontier {
        let mut scores = vec![NEG_INF; self.layer_widths[boundary.layer]];
        scores[boundary.node] = 0;
        DagFrontier { scores }
    }

    fn replay(&self, a: usize, b: usize, frontier: &DagFrontier) -> DagFrontier {
        let mut current = frontier.clone();
        for layer in a..b {
            current =
                advance_sparse_layer(self.layer_widths[layer + 1], &self.edges[layer], &current);
        }
        current
    }

    fn backward_scores(&self, a: usize, b: usize, boundary: &DagBoundary) -> Vec<i32> {
        let mut next = vec![NEG_INF; self.layer_widths[b]];
        next[boundary.node] = 0;
        for layer in (a..b).rev() {
            let mut current = vec![NEG_INF; self.layer_widths[layer]];
            for edge in &self.edges[layer] {
                if next[edge.to] <= NEG_INF / 2 {
                    continue;
                }
                let candidate = next[edge.to].saturating_add(edge.score).max(NEG_INF);
                current[edge.from] = current[edge.from].max(candidate);
            }
            next = current;
        }
        next
    }
}

impl HcpProblem for LayeredDagProblem {
    type State = DagState;
    type Frontier = DagFrontier;
    type Summary = DagSummary;
    type Boundary = DagBoundary;
    type Cost = i32;

    fn num_layers(&self) -> usize {
        self.edges.len()
    }

    fn init_frontier(&self) -> Self::Frontier {
        let mut scores = vec![NEG_INF; self.layer_widths[0]];
        scores[self.start_node] = 0;
        DagFrontier { scores }
    }

    fn forward_step(&self, layer: usize, frontier: &Self::Frontier) -> Self::Frontier {
        advance_sparse_layer(self.layer_widths[layer + 1], &self.edges[layer], frontier)
    }

    fn summarize_interval(&self, a: usize, b: usize) -> Self::Summary {
        assert!(a <= b && b <= self.num_layers(), "invalid DAG interval");
        DagSummary {
            layer_widths: self.layer_widths.clone(),
            edges: self.edges.clone(),
            start: a,
            end: b,
        }
    }

    fn merge_summary(&self, left: &Self::Summary, right: &Self::Summary) -> Self::Summary {
        assert_eq!(left.end, right.start, "DAG summaries must be adjacent");
        assert_eq!(left.layer_widths, right.layer_widths);
        assert_eq!(left.edges, right.edges);
        DagSummary {
            layer_widths: self.layer_widths.clone(),
            edges: self.edges.clone(),
            start: left.start,
            end: right.end,
        }
    }

    fn initial_boundary(&self) -> Self::Boundary {
        DagBoundary {
            layer: 0,
            node: self.start_node,
        }
    }

    fn terminal_boundary(&self, _frontier_t: &Self::Frontier) -> Self::Boundary {
        DagBoundary {
            layer: self.num_layers(),
            node: self.end_node,
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
        let fwd = self.replay(a, m, &self.frontier_from_boundary(beta_a));
        let bwd = self.backward_scores(m, c, beta_c);
        let mut best_node = 0;
        let mut best_score = NEG_INF;
        for (node, &backward_score) in bwd.iter().enumerate().take(self.layer_widths[m]) {
            let score = add_scores(fwd.scores[node], backward_score);
            if score > best_score {
                best_node = node;
                best_score = score;
            }
        }
        DagBoundary {
            layer: m,
            node: best_node,
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
        let mut scores = vec![vec![]; b - a + 1];
        let mut predecessors = vec![vec![]; b - a + 1];
        scores[0] = vec![NEG_INF; self.layer_widths[a]];
        scores[0][beta_a.node] = 0;
        predecessors[0] = vec![None; self.layer_widths[a]];

        for layer in a..b {
            let local = layer - a;
            scores[local + 1] = vec![NEG_INF; self.layer_widths[layer + 1]];
            predecessors[local + 1] = vec![None; self.layer_widths[layer + 1]];
            for edge in &self.edges[layer] {
                let base = scores[local][edge.from];
                if base <= NEG_INF / 2 {
                    continue;
                }
                let candidate = base.saturating_add(edge.score).max(NEG_INF);
                if candidate > scores[local + 1][edge.to] {
                    scores[local + 1][edge.to] = candidate;
                    predecessors[local + 1][edge.to] = Some(edge.from);
                }
            }
        }

        assert!(
            scores[b - a][beta_b.node] > NEG_INF / 2,
            "DAG leaf endpoint must be reachable"
        );
        let mut layer = b;
        let mut node = beta_b.node;
        let mut rev_path = Vec::with_capacity(b - a + 1);
        rev_path.push(DagState { layer, node });
        while layer > a {
            let local = layer - a;
            node = predecessors[local][node].expect("DAG traceback must have predecessor");
            layer -= 1;
            rev_path.push(DagState { layer, node });
        }
        rev_path.reverse();
        rev_path
    }

    fn extract_cost(&self, frontier_t: &Self::Frontier, _beta_t: &Self::Boundary) -> Self::Cost {
        frontier_t.scores[self.end_node]
    }
}

impl SummaryApply<DagFrontier> for DagSummary {
    fn apply(&self, frontier: &DagFrontier) -> DagFrontier {
        let mut current = frontier.clone();
        for layer in self.start..self.end {
            current =
                advance_sparse_layer(self.layer_widths[layer + 1], &self.edges[layer], &current);
        }
        current
    }
}

fn advance_sparse_layer(
    next_width: usize,
    edges: &[DagEdge],
    frontier: &DagFrontier,
) -> DagFrontier {
    let mut next = vec![NEG_INF; next_width];
    for edge in edges {
        if frontier.scores[edge.from] <= NEG_INF / 2 {
            continue;
        }
        let candidate = frontier.scores[edge.from]
            .saturating_add(edge.score)
            .max(NEG_INF);
        next[edge.to] = next[edge.to].max(candidate);
    }
    DagFrontier { scores: next }
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
    use super::{DagEdge, DagState, LayeredDagProblem};
    use crate::contract::{assert_all_summary_laws, assert_engine_paths_for_all_block_sizes};
    use crate::HcpEngine;

    fn sample_problem() -> LayeredDagProblem {
        LayeredDagProblem::new(
            vec![1, 3, 2, 1],
            vec![
                vec![
                    DagEdge {
                        from: 0,
                        to: 0,
                        score: 1,
                    },
                    DagEdge {
                        from: 0,
                        to: 1,
                        score: 5,
                    },
                    DagEdge {
                        from: 0,
                        to: 2,
                        score: 2,
                    },
                ],
                vec![
                    DagEdge {
                        from: 0,
                        to: 0,
                        score: 2,
                    },
                    DagEdge {
                        from: 1,
                        to: 0,
                        score: 1,
                    },
                    DagEdge {
                        from: 1,
                        to: 1,
                        score: 4,
                    },
                    DagEdge {
                        from: 2,
                        to: 1,
                        score: 3,
                    },
                ],
                vec![
                    DagEdge {
                        from: 0,
                        to: 0,
                        score: 1,
                    },
                    DagEdge {
                        from: 1,
                        to: 0,
                        score: 2,
                    },
                ],
            ],
            0,
            0,
        )
        .unwrap()
    }

    #[test]
    fn layered_dag_engine_finds_sparse_longest_path() {
        let problem = sample_problem();
        let (score, path) = HcpEngine::new(problem.clone()).run();
        assert_eq!(score, 11);
        assert_eq!(path.first(), Some(&DagState { layer: 0, node: 0 }));
        assert_eq!(path.last(), Some(&DagState { layer: 3, node: 0 }));
        assert_eq!(problem.score_path(&path), Some(score));
    }

    #[test]
    fn layered_dag_satisfies_contract_harness() {
        let problem = sample_problem();
        assert_all_summary_laws(&problem);
        assert_engine_paths_for_all_block_sizes(problem.clone(), |score, path| {
            assert_eq!(score, problem.full_table_score());
            assert_eq!(problem.score_path(path), Some(score));
        });
    }
}
