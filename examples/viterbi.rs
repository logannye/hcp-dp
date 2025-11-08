//! Example: Viterbi decoding via HCP-DP.
//!
//! Run with:
//! `cargo run --example viterbi`

use hcp_dp::{
    problems::viterbi::{Hmm, ViterbiProblem},
    HcpEngine,
};

fn main() {
    // Simple 2-state HMM demo:
    //
    // States: 0,1
    // Observations: 0,1
    //
    // This is purely illustrative; in real use you'd log-probs from data.

    let hmm = Hmm {
        n_states: 2,
        log_pi: vec![(0.5f64).ln(), (0.5f64).ln()],
        log_a: vec![
            vec![(0.9f64).ln(), (0.1f64).ln()],
            vec![(0.2f64).ln(), (0.8f64).ln()],
        ],
        log_b: vec![
            vec![(0.8f64).ln(), (0.2f64).ln()], // state 0 prefers obs 0
            vec![(0.3f64).ln(), (0.7f64).ln()], // state 1 prefers obs 1
        ],
    };

    // Observation sequence:
    let obs = vec![0, 0, 1, 1, 1, 0, 1];

    let problem = ViterbiProblem::new(hmm, obs);
    let engine = HcpEngine::new(problem);

    let (log_prob, path) = engine.run();

    println!("Best path log-probability: {log_prob}");
    println!("State sequence:");
    for s in &path {
        println!("  t = {:2}, state = {}", s.t, s.state);
    }
}
