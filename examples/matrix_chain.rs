//! Example: Matrix-chain multiplication via HCP-DP.
//!
//! Run with:
//! `cargo run --example matrix_chain`

use hcp_dp::{problems::matrix_chain::MatrixChainProblem, HcpEngine};

fn main() {
    // Example from CLRS:
    // Matrices A1..A6 with dimensions:
    // 30x35, 35x15, 15x5, 5x10, 10x20, 20x25
    let p = vec![30, 35, 15, 5, 10, 20, 25];

    let problem = MatrixChainProblem::new(p);
    let engine = HcpEngine::new(problem);

    let (cost, states) = engine.run();

    println!("Optimal multiplication cost: {cost}");
    println!("Split decisions (i,j,k):");
    for s in states {
        println!("  Split A[{}..{}] at {}", s.i, s.j, s.k);
    }
}
