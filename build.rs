use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=HCP_DP_GIT_COMMIT");
    println!("cargo:rerun-if-changed=.git/HEAD");
    if std::env::var_os("HCP_DP_GIT_COMMIT").is_some() {
        return;
    }
    let commit = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    if let Some(commit) = commit {
        println!("cargo:rustc-env=HCP_DP_GIT_COMMIT={commit}");
    }
}
