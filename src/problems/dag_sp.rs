//! Layered DAG shortest path as a height-compressible DP.
//! Graph structure: layers 0..=T, vertices per layer with edges only from layer i to i+1.
//! Frontier: best distance per vertex in the current layer.

use crate::traits::HcpProblem;

#[derive(Clone)]
pub struct DagLayered {
    /// adjacency[i][u] = Vec<(v, weight)> edges from layer i vertex u to layer i+1 vertex v
    pub adjacency: Vec<Vec<Vec<(usize, i64)>>>,
    /// number of vertices per layer
    pub widths: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct DagFrontier {
    pub dist: Vec<i64>, // length = widths[layer]
}

#[derive(Clone, Debug)]
pub struct DagSummary {
    pub end_frontier: DagFrontier,
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
        Self { adjacency, widths }
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
        (f.clone(), DagSummary { end_frontier: f })
    }

    fn merge_summary(&self, _left: &Self::Summary, right: &Self::Summary) -> Self::Summary {
        right.clone()
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
