use hcp_dp::{problems::lcs::LcsProblem, HcpEngine};

fn valid_path(path: &[(usize, usize)], n: usize, m: usize) -> bool {
    if path.is_empty() {
        return n == 0 && m == 0;
    }
    if *path.first().unwrap() != (0, 0) {
        return false;
    }
    if *path.last().unwrap() != (n, m) {
        return false;
    }
    for w in path.windows(2) {
        let (a, b) = (w[0], w[1]);
        let di = b.0 as isize - a.0 as isize;
        let dj = b.1 as isize - a.1 as isize;
        match (di, dj) {
            (1, 0) | (0, 1) | (1, 1) => {}
            _ => return false,
        }
    }
    true
}

#[test]
fn lcs_example_integration() {
    let s = b"ACCGGTCGAGTGCGCGGAAGCCGGCCGAA";
    let t = b"GTCGTTCGGAATGCCGTTGCTCTGTAAA";
    let problem = LcsProblem::new(s, t);
    let engine = HcpEngine::new(problem);
    let (len, path) = engine.run();
    assert_eq!(len, 20);
    assert!(valid_path(&path, s.len(), t.len()));
}
