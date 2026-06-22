use hcp_dp::{problems::smith_waterman::SmithWatermanProblem, HcpEngine};

fn main() {
    let problem = SmithWatermanProblem::new(b"ACACACTA", b"AGCACACA", 2, 1, -2);
    let (score, path) = HcpEngine::new(problem.clone()).run();

    println!("Local alignment score: {score}");
    println!("Path length: {}", path.len());
    assert_eq!(problem.score_path(&path), Some(score));
}
