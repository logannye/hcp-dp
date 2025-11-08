use hcp_dp::problems::dag_sp::DagLayered;
use hcp_dp::HcpEngine;
use proptest::prelude::*;

fn topo_relax(adjacency: &[Vec<Vec<(usize, i64)>>], widths: &[usize]) -> Vec<i64> {
    let mut dist = vec![i64::MAX / 4; widths[0]];
    if !dist.is_empty() {
        dist[0] = 0;
    }
    for (layer, edges) in adjacency.iter().enumerate() {
        let mut next = vec![i64::MAX / 4; widths[layer + 1]];
        for (u, &du) in dist.iter().enumerate() {
            if du >= i64::MAX / 8 {
                continue;
            }
            for &(v, w) in &edges[u] {
                let cand = du.saturating_add(w);
                if cand < next[v] {
                    next[v] = cand;
                }
            }
        }
        dist = next;
    }
    dist
}

proptest! {
    #[test]
    fn randomized_dag_matches_topo(
        layers in 1usize..4,
        widths in prop::collection::vec(1usize..4, 0usize..5),
        weights in prop::collection::vec(-5i64..10, 0usize..60)
    ) {
        let mut widths = widths;
        if widths.len() < layers + 1 {
            widths.resize(layers + 1, 1);
        } else {
            widths.truncate(layers + 1);
        }
        let mut idx = 0;
        let mut adjacency = Vec::with_capacity(layers);
        for layer in 0..layers {
            let w = widths[layer];
            let next_w = widths[layer + 1];
            let mut layer_edges = Vec::with_capacity(w);
            for _ in 0..w {
                let mut edges = Vec::new();
                for v in 0..next_w {
                    let weight = if idx < weights.len() { weights[idx] } else { 1 };
                    idx += 1;
                    if idx % 2 == 0 {
                        edges.push((v, weight));
                    }
                }
                if edges.is_empty() {
                    edges.push((0, 1));
                }
                layer_edges.push(edges);
            }
            adjacency.push(layer_edges);
        }

        let baseline = topo_relax(&adjacency, &widths);
        let problem = DagLayered::new(adjacency.clone(), widths.clone());
        let (cost, path) = HcpEngine::new(problem).run();
        let best = *baseline.iter().min().unwrap();
        prop_assert_eq!(cost, best);
        if !path.is_empty() {
            prop_assert_eq!(path.first().unwrap().layer, 0);
            prop_assert_eq!(path.last().unwrap().layer, layers);
        }
    }
}
