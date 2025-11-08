//! Matrix-chain multiplication as an HCP-DP instance.
//!
//! Classic DP:
//! - Given dimensions p[0..n], matrices A_i of size p[i-1] x p[i],
//! - Find parenthesization minimizing scalar multiplications.
//!
//! Here we show how to wrap it as an `HcpProblem`. For simplicity and clarity,
//! we reconstruct a sequence of "split positions" as the path.

use crate::traits::HcpProblem;

/// Matrix-chain DP instance.
#[derive(Clone)]
pub struct MatrixChainProblem {
    /// Dimensions p[0..=n], with n matrices.
    pub p: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct McFrontier {
    // We encode DP for intervals ending at this "length".
    // For clarity, we won't aggressively compress this; it's a reference impl.
    pub costs: Vec<Vec<u64>>, // costs[i][j] for 0 <= i < j <= n
}

#[derive(Clone, Debug)]
pub struct McSummary {
    // In this simple example, we don't use sophisticated summaries.
    // A real implementation would store partial interval costs.
    pub dummy: (),
}

#[derive(Clone, Debug)]
pub struct McBoundary {
    pub start: usize,
    pub end: usize,
}

/// State along the optimal structure: record (i, j, k) split decisions.
#[derive(Clone, Debug, PartialEq)]
pub struct McState {
    pub i: usize,
    pub j: usize,
    pub k: usize,
}

impl MatrixChainProblem {
    pub fn new(p: Vec<usize>) -> Self {
        assert!(p.len() >= 2, "need at least one matrix");
        Self { p }
    }

    fn n(&self) -> usize {
        self.p.len() - 1
    }
}

impl HcpProblem for MatrixChainProblem {
    type State = McState;
    type Frontier = McFrontier;
    type Summary = McSummary;
    type Boundary = McBoundary;
    type Cost = u64;

    fn num_layers(&self) -> usize {
        // We'll use "length of chain" as a notion of layer; here we just
        // treat the entire problem as one block for simplicity.
        // A more advanced version would layer by interval length.
        1
    }

    fn init_frontier(&self) -> Self::Frontier {
        let n = self.n();
        let costs = vec![vec![0u64; n + 1]; n + 1];
        McFrontier { costs }
    }

    fn forward_step(&self, _layer: usize, frontier: &Self::Frontier) -> Self::Frontier {
        // Perform full O(n^3) DP; this is not height-compressed itself,
        // but demonstrates how to embed a DP in the interface.
        let n = self.n();
        let mut c = frontier.costs.clone();

        for len in 2..=n {
            for i in 1..=(n - len + 1) {
                let j = i + len - 1;
                c[i][j] = u64::MAX;
                for k in i..j {
                    let cost = c[i][k]
                        + c[k + 1][j]
                        + (self.p[i - 1] as u64) * (self.p[k] as u64) * (self.p[j] as u64);
                    if cost < c[i][j] {
                        c[i][j] = cost;
                    }
                }
            }
        }

        McFrontier { costs: c }
    }

    fn summarize_block(
        &self,
        a: usize,
        b: usize,
        frontier_a: &Self::Frontier,
    ) -> (Self::Frontier, Self::Summary) {
        // Since num_layers=1, we effectively just run forward_step once.
        debug_assert!(a == 0 && b == 1);
        let f = self.forward_step(0, frontier_a);
        (f, McSummary { dummy: () })
    }

    fn merge_summary(&self, left: &Self::Summary, _right: &Self::Summary) -> Self::Summary {
        // Only one block in this simple setup.
        McSummary { dummy: left.dummy }
    }

    fn initial_boundary(&self) -> Self::Boundary {
        McBoundary {
            start: 1,
            end: self.n(),
        }
    }

    fn terminal_boundary(&self, _frontier_t: &Self::Frontier) -> Self::Boundary {
        McBoundary {
            start: 1,
            end: self.n(),
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
        // For this reference impl, we don't do hierarchical splitting:
        // just propagate original boundary.
        let _ = (a, m, c, beta_a, beta_c);
        McBoundary {
            start: 1,
            end: self.n(),
        }
    }

    fn reconstruct_block(
        &self,
        _a: usize,
        _b: usize,
        _beta_a: &Self::Boundary,
        _beta_b: &Self::Boundary,
    ) -> Vec<Self::State> {
        // Standard reconstruction of split positions using full DP.
        let n = self.n();
        let mut c = vec![vec![0u64; n + 1]; n + 1];
        let mut split = vec![vec![0usize; n + 1]; n + 1];

        for len in 2..=n {
            for i in 1..=(n - len + 1) {
                let j = i + len - 1;
                c[i][j] = u64::MAX;
                for k in i..j {
                    let cost = c[i][k]
                        + c[k + 1][j]
                        + (self.p[i - 1] as u64) * (self.p[k] as u64) * (self.p[j] as u64);
                    if cost < c[i][j] {
                        c[i][j] = cost;
                        split[i][j] = k;
                    }
                }
            }
        }

        let mut states = Vec::new();
        fn collect(states: &mut Vec<McState>, split: &Vec<Vec<usize>>, i: usize, j: usize) {
            if i >= j {
                return;
            }
            let k = split[i][j];
            states.push(McState { i, j, k });
            collect(states, split, i, k);
            collect(states, split, k + 1, j);
        }
        collect(&mut states, &split, 1, n);
        states
    }

    fn extract_cost(&self, frontier_t: &Self::Frontier, _beta_t: &Self::Boundary) -> Self::Cost {
        let n = self.n();
        frontier_t.costs[1][n]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HcpEngine;

    #[test]
    fn clrs_example_cost() {
        let p = vec![30, 35, 15, 5, 10, 20, 25];
        let problem = MatrixChainProblem::new(p);
        let engine = HcpEngine::new(problem);
        let (cost, _states) = engine.run();
        assert_eq!(cost, 15125);
    }

    #[test]
    fn small_edges() {
        // Single matrix (n=1) -> cost 0
        let p = vec![10, 20];
        let problem = MatrixChainProblem::new(p);
        let engine = HcpEngine::new(problem);
        let (cost, states) = engine.run();
        assert_eq!(cost, 0);
        assert!(states.is_empty());

        // Two matrices (n=2) -> one multiplication cost
        let p = vec![10, 20, 30];
        let problem = MatrixChainProblem::new(p.clone());
        let engine = HcpEngine::new(problem);
        let (cost, _states) = engine.run();
        assert_eq!(cost, (p[0] as u64) * (p[1] as u64) * (p[2] as u64));
    }
}
