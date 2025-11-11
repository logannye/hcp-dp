//! Layered DAG shortest path as a height-compressible DP.
//! Graph structure: layers 0..=T, vertices per layer with edges only from layer i to i+1.
//! Frontier: best distance per vertex in the current layer.

use crate::traits::{HcpProblem, SummaryApply};
use std::sync::Arc;

type SharedAdjacency = Arc<Vec<Vec<Vec<(usize, i64)>>>>;
type SharedWidths = Arc<Vec<usize>>;

#[derive(Clone)]
pub struct DagLayered {
    /// adjacency[i][u] = Vec<(v, weight)> edges from layer i vertex u to layer i+1 vertex v
    pub adjacency: SharedAdjacency,
    /// number of vertices per layer
    pub widths: SharedWidths,
}

#[derive(Clone, Debug)]
pub struct DagFrontier {
    pub dist: Vec<i64>, // length = widths[layer]
}

#[derive(Clone, Debug)]
pub struct DagSummary {
    adjacency: SharedAdjacency,
    widths: SharedWidths,
    start: usize,
    end: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DagBoundary {
    pub layer: usize,
    pub node: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DagState {
    pub layer: usize,
    pub node: usize,
}

impl DagLayered {
    pub fn new(adjacency: Vec<Vec<Vec<(usize, i64)>>>, widths: Vec<usize>) -> Self {
        assert_eq!(
            adjacency.len() + 1,
            widths.len(),
            "adjacency for T layers, widths for T+1 layers"
        );
        Self {
            adjacency: Arc::new(adjacency),
            widths: Arc::new(widths),
        }
    }
    fn t(&self) -> usize {
        self.adjacency.len()
    }
}

impl HcpProblem for DagLayered {
    type State = DagState;
    type Frontier = DagFrontier;
    type Summary = DagSummary;
    type Boundary = DagBoundary;
    type Cost = i64;

    fn num_layers(&self) -> usize {
        self.t()
    }

    fn init_frontier(&self) -> Self::Frontier {
        // Layer 0: source assumed to be node 0 with distance 0 by default
        let mut dist = vec![i64::MAX / 4; self.widths[0]];
        if !dist.is_empty() {
            dist[0] = 0;
        }
        DagFrontier { dist }
    }

    fn forward_step(&self, layer: usize, f: &Self::Frontier) -> Self::Frontier {
        let next_w = self.widths[layer + 1];
        let mut next = vec![i64::MAX / 4; next_w];
        for (u, &du) in f.dist.iter().enumerate() {
            if du >= i64::MAX / 8 {
                continue;
            }
            for &(v, w) in &self.adjacency[layer][u] {
                let cand = du.saturating_add(w);
                if cand < next[v] {
                    next[v] = cand;
                }
            }
        }
        DagFrontier { dist: next }
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
        (
            f.clone(),
            DagSummary {
                adjacency: Arc::clone(&self.adjacency),
                widths: Arc::clone(&self.widths),
                start: a,
                end: b,
            },
        )
    }

    fn merge_summary(&self, left: &Self::Summary, right: &Self::Summary) -> Self::Summary {
        debug_assert!(
            Arc::ptr_eq(&left.adjacency, &right.adjacency),
            "summaries must share adjacency"
        );
        debug_assert!(
            Arc::ptr_eq(&left.widths, &right.widths),
            "summaries must share widths"
        );

        DagSummary {
            adjacency: Arc::clone(&left.adjacency),
            widths: Arc::clone(&left.widths),
            start: left.start,
            end: right.end,
        }
    }

    fn initial_boundary(&self) -> Self::Boundary {
        DagBoundary { layer: 0, node: 0 }
    }

    fn terminal_boundary(&self, frontier_t: &Self::Frontier) -> Self::Boundary {
        // pick argmin at final layer
        let mut best = 0usize;
        let mut bestv = i64::MAX / 4;
        for (i, &v) in frontier_t.dist.iter().enumerate() {
            if v < bestv {
                bestv = v;
                best = i;
            }
        }
        DagBoundary {
            layer: self.t(),
            node: best,
        }
    }

    fn choose_boundary(
        &self,
        a: usize,
        m: usize,
        c: usize,
        _sigma_left: &Self::Summary,
        _sigma_right: &Self::Summary,
        _beta_a: &Self::Boundary,
        _beta_c: &Self::Boundary,
    ) -> Self::Boundary {
        // Forward from a..m to get distances at m; pick node with smallest potential to reach c.
        let _ = c;
        let mut f = self.init_frontier();
        for i in 0..a {
            f = self.forward_step(i, &f);
        }
        for i in a..m {
            f = self.forward_step(i, &f);
        }
        let mut best = 0usize;
        let mut bestv = i64::MAX / 4;
        for (i, &v) in f.dist.iter().enumerate() {
            if v < bestv {
                bestv = v;
                best = i;
            }
        }
        DagBoundary {
            layer: m,
            node: best,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn choose_boundary_with_frontiers(
        &self,
        _a: usize,
        m: usize,
        _c: usize,
        _frontier_a: &Self::Frontier,
        frontier_m_forward: &Self::Frontier,
        frontier_m_backward: &Self::Frontier,
        _frontier_c: &Self::Frontier,
        _sigma_left: &Self::Summary,
        _sigma_right: &Self::Summary,
        _beta_a: &Self::Boundary,
        _beta_c: &Self::Boundary,
    ) -> Self::Boundary {
        let mut best = i64::MAX / 4;
        let mut best_node = 0usize;
        for (idx, (&fwd, &bwd)) in frontier_m_forward
            .dist
            .iter()
            .zip(&frontier_m_backward.dist)
            .enumerate()
        {
            if fwd >= i64::MAX / 8 || bwd >= i64::MAX / 8 {
                continue;
            }
            let total = fwd.saturating_add(bwd);
            if total < best {
                best = total;
                best_node = idx;
            }
        }
        DagBoundary {
            layer: m,
            node: best_node,
        }
    }

    fn reconstruct_block(
        &self,
        a: usize,
        b: usize,
        beta_a: &Self::Boundary,
        beta_b: &Self::Boundary,
    ) -> Vec<Self::State> {
        // Standard DP with parent pointers inside [a,b].
        let mut f = self.init_frontier();
        for i in 0..a {
            f = self.forward_step(i, &f);
        }
        // parents per layer in [a+1..=b]
        let mut parents: Vec<Vec<usize>> = Vec::new();
        let mut current = f;
        for i in a..b {
            let next_w = self.widths[i + 1];
            let mut parent = vec![usize::MAX; next_w];
            let mut next = vec![i64::MAX / 4; next_w];
            for (u, &du) in current.dist.iter().enumerate() {
                if du >= i64::MAX / 8 {
                    continue;
                }
                for &(v, w) in &self.adjacency[i][u] {
                    let cand = du.saturating_add(w);
                    if cand < next[v] {
                        next[v] = cand;
                        parent[v] = u;
                    }
                }
            }
            parents.push(parent);
            current = DagFrontier { dist: next };
        }
        // backtrack from beta_b.node
        let mut path = Vec::new();
        let mut node = beta_b.node;
        for layer in (a + 1..=b).rev() {
            path.push(DagState { layer, node });
            let p = parents[layer - a - 1][node];
            if p == usize::MAX {
                break;
            }
            node = p;
        }
        // prepend start
        path.push(DagState {
            layer: a,
            node: beta_a.node,
        });
        path.reverse();
        path
    }

    fn extract_cost(&self, frontier_t: &Self::Frontier, beta_t: &Self::Boundary) -> Self::Cost {
        frontier_t.dist[beta_t.node]
    }
}

impl SummaryApply<DagFrontier> for DagSummary {
    fn apply(&self, frontier: &DagFrontier) -> DagFrontier {
        if self.start == self.end {
            return frontier.clone();
        }
        let mut current = frontier.dist.clone();
        for layer in self.start..self.end {
            let next_w = self.widths[layer + 1];
            let mut next = vec![i64::MAX / 4; next_w];
            for (u, &du) in current.iter().enumerate() {
                if du >= i64::MAX / 8 {
                    continue;
                }
                for &(v, w) in &self.adjacency[layer][u] {
                    let cand = du.saturating_add(w);
                    if cand < next[v] {
                        next[v] = cand;
                    }
                }
            }
            current = next;
        }
        DagFrontier { dist: current }
    }

    fn apply_reverse(&self, frontier: &DagFrontier) -> DagFrontier {
        if self.start == self.end {
            return frontier.clone();
        }
        let mut current = frontier.dist.clone();
        for layer in (self.start..self.end).rev() {
            let prev_w = self.widths[layer];
            let mut prev = vec![i64::MAX / 4; prev_w];
            for (u, edges) in self.adjacency[layer].iter().enumerate() {
                for &(v, w) in edges {
                    if v < current.len() && current[v] < i64::MAX / 8 {
                        let cand = current[v].saturating_add(w);
                        if cand < prev[u] {
                            prev[u] = cand;
                        }
                    }
                }
            }
            current = prev;
        }
        DagFrontier { dist: current }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HcpEngine;

    #[test]
    fn simple_dag_sp() {
        // 3 layers (0->1->2), widths [1,2,2]
        let adjacency = vec![
            vec![vec![(0, 1), (1, 5)]], // from layer0 node0 to layer1 nodes 0(cost1),1(cost5)
            vec![vec![(0, 1)], vec![(1, 1)]], // from layer1 node0->layer2 node0 (1), node1->layer2 node1(1)
        ];
        let widths = vec![1, 2, 2];
        let problem = DagLayered::new(adjacency, widths);
        let engine = HcpEngine::new(problem);
        let (cost, path) = engine.run();
        assert_eq!(cost, 2);
        assert!(!path.is_empty());
    }
}
