//! Minimizer seeding utilities for exact extension workflows.

use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Minimizer {
    pub hash: u64,
    pub position: usize,
    pub kmer: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeedHit {
    pub query_pos: usize,
    pub target_pos: usize,
    pub length: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeedChain {
    pub hits: Vec<SeedHit>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeededWindow {
    pub query_start: usize,
    pub query_end: usize,
    pub target_start: usize,
    pub target_end: usize,
    pub chain: SeedChain,
}

pub fn minimizers(sequence: &[u8], k: usize, window: usize) -> Result<Vec<Minimizer>, String> {
    validate_seed_params(k, window)?;
    if sequence.len() < k {
        return Ok(Vec::new());
    }

    let kmer_count = sequence.len() - k + 1;
    let window = window.min(kmer_count);
    let hashes: Vec<u64> = (0..kmer_count)
        .map(|pos| hash_kmer(&sequence[pos..pos + k]))
        .collect();
    let mut result = Vec::new();
    let mut last_position = None;

    for start in 0..=kmer_count - window {
        let end = start + window;
        let mut best_pos = start;
        let mut best_hash = hashes[start];
        for (offset, &hash) in hashes[start + 1..end].iter().enumerate() {
            let pos = start + 1 + offset;
            if hash < best_hash {
                best_hash = hash;
                best_pos = pos;
            }
        }
        if last_position == Some(best_pos) {
            continue;
        }
        last_position = Some(best_pos);
        result.push(Minimizer {
            hash: best_hash,
            position: best_pos,
            kmer: sequence[best_pos..best_pos + k].to_vec(),
        });
    }

    Ok(result)
}

pub fn minimizer_hits(
    query: &[u8],
    target: &[u8],
    k: usize,
    window: usize,
) -> Result<Vec<SeedHit>, String> {
    let query_minimizers = minimizers(query, k, window)?;
    let target_minimizers = minimizers(target, k, window)?;
    let mut target_index: HashMap<(u64, Vec<u8>), Vec<usize>> = HashMap::new();
    for minimizer in target_minimizers {
        target_index
            .entry((minimizer.hash, minimizer.kmer))
            .or_default()
            .push(minimizer.position);
    }

    let mut seen = HashSet::new();
    let mut hits = Vec::new();
    for minimizer in query_minimizers {
        if let Some(target_positions) = target_index.get(&(minimizer.hash, minimizer.kmer)) {
            for &target_pos in target_positions {
                if seen.insert((minimizer.position, target_pos)) {
                    hits.push(SeedHit {
                        query_pos: minimizer.position,
                        target_pos,
                        length: k,
                    });
                }
            }
        }
    }
    hits.sort_by_key(|hit| (hit.query_pos, hit.target_pos));
    Ok(hits)
}

pub fn chain_seed_hits(hits: &[SeedHit]) -> Option<SeedChain> {
    if hits.is_empty() {
        return None;
    }

    let mut sorted = hits.to_vec();
    sorted.sort_by_key(|hit| (hit.query_pos, hit.target_pos));
    let mut scores = vec![1usize; sorted.len()];
    let mut predecessors = vec![None; sorted.len()];

    for idx in 0..sorted.len() {
        for prev in 0..idx {
            if sorted[prev].query_pos < sorted[idx].query_pos
                && sorted[prev].target_pos < sorted[idx].target_pos
                && scores[prev] + 1 > scores[idx]
            {
                scores[idx] = scores[prev] + 1;
                predecessors[idx] = Some(prev);
            }
        }
    }

    let mut best = 0usize;
    for idx in 1..scores.len() {
        if scores[idx] > scores[best] {
            best = idx;
        }
    }

    let mut chain = Vec::new();
    let mut cursor = Some(best);
    while let Some(idx) = cursor {
        chain.push(sorted[idx]);
        cursor = predecessors[idx];
    }
    chain.reverse();
    Some(SeedChain { hits: chain })
}

pub fn seeded_window(
    query: &[u8],
    target: &[u8],
    k: usize,
    window: usize,
    flank: usize,
) -> Result<Option<SeededWindow>, String> {
    let hits = minimizer_hits(query, target, k, window)?;
    let Some(chain) = chain_seed_hits(&hits) else {
        return Ok(None);
    };
    let first = chain
        .hits
        .first()
        .expect("non-empty seed chain must have first hit");
    let last = chain
        .hits
        .last()
        .expect("non-empty seed chain must have last hit");

    Ok(Some(SeededWindow {
        query_start: first.query_pos.saturating_sub(flank),
        query_end: (last.query_pos + last.length + flank).min(query.len()),
        target_start: first.target_pos.saturating_sub(flank),
        target_end: (last.target_pos + last.length + flank).min(target.len()),
        chain,
    }))
}

fn validate_seed_params(k: usize, window: usize) -> Result<(), String> {
    if k == 0 {
        return Err("--seed-k must be positive".to_string());
    }
    if window == 0 {
        return Err("--seed-window must be positive".to_string());
    }
    Ok(())
}

fn hash_kmer(kmer: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in kmer {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::{chain_seed_hits, minimizer_hits, seeded_window, SeedHit};

    #[test]
    fn minimizer_hits_find_exact_shared_kmers() {
        let hits = minimizer_hits(b"TTACGTAA", b"GGACGTCC", 2, 2).unwrap();
        assert!(hits
            .iter()
            .any(|hit| hit.query_pos == 2 && hit.target_pos == 2));
    }

    #[test]
    fn seed_chain_is_monotone() {
        let hits = vec![
            SeedHit {
                query_pos: 0,
                target_pos: 5,
                length: 2,
            },
            SeedHit {
                query_pos: 2,
                target_pos: 1,
                length: 2,
            },
            SeedHit {
                query_pos: 4,
                target_pos: 3,
                length: 2,
            },
        ];
        let chain = chain_seed_hits(&hits).unwrap();
        assert_eq!(chain.hits.len(), 2);
        assert_eq!(chain.hits[0].query_pos, 2);
        assert_eq!(chain.hits[1].query_pos, 4);
    }

    #[test]
    fn seeded_window_wraps_chain_with_flank() {
        let window = seeded_window(b"TTACGTAA", b"GGACGTCC", 2, 2, 1)
            .unwrap()
            .unwrap();
        assert!(window.query_start <= 2);
        assert!(window.query_end >= 6);
        assert!(window.target_start <= 2);
        assert!(window.target_end >= 6);
        assert!(!window.chain.hits.is_empty());
    }

    #[test]
    fn seeded_window_returns_none_without_shared_seed() {
        assert!(seeded_window(b"AAAA", b"CCCC", 2, 2, 1).unwrap().is_none());
    }
}
