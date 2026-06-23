# Problem Author Guide

This guide is for adding a new dynamic-programming problem to HCP-DP.

The bar is intentionally high: a problem should not be exported from
`problems` until it can report an exact objective and reconstruct an exact path
under the shared contract harness.

## The Contract In One Page

Model the DP as layers:

```text
layer 0 -> layer 1 -> ... -> layer T
```

At each layer boundary, the problem exposes a frontier. For a sequence-grid DP,
the frontier is usually one row. For another layered DP, it may be a vector of
state costs, sparse states, or another compact representation.

Your implementation must provide these pieces:

| Piece | Role |
|---|---|
| `State` | One point on the reconstructed path. |
| `Frontier` | DP values at one layer boundary. |
| `Summary` | A boundary-independent operator for an interval `[a, b)`. |
| `Boundary` | Endpoint constraint used during traceback. |
| `Cost` | Ordered objective value. |

The most important law is:

```text
summary(a,b).apply(frontier_at_a) == direct_replay(a,b, frontier_at_a)
```

Adjacent summaries must compose:

```text
merge(summary(a,b), summary(b,c)).apply(F)
  == summary(a,c).apply(F)
```

Reconstruction is endpoint constrained:

```text
reconstruct_leaf(a,b,beta_a,beta_b)
```

must return a segment that starts at `beta_a`, ends at `beta_b`, and realizes
the local optimum for those endpoints.

## Implementation Steps

1. Define the mathematical recurrence outside the engine.

   Decide whether the problem maximizes a score or minimizes a cost. Write down
   the base frontier, transition, legal path moves, and path scoring convention.
   Do this before writing `choose_split`.

2. Implement direct frontier replay.

   `init_frontier` and `forward_step` should be the simplest correct version of
   the recurrence. Contract tests use these as an independent replay oracle for
   summary application.

3. Implement interval summaries as operators.

   A summary may be lightweight. Many current modules store the problem slice
   and interval bounds, then replay the interval when `apply` is called. That is
   acceptable for correctness-first alpha code.

   Do not store "the output frontier from the build run" as the summary. That is
   not an operator; it only works for one input frontier.

4. Implement merge as interval composition.

   For lightweight summaries, merging usually means checking that intervals are
   adjacent and returning the wider interval.

5. Define boundary semantics explicitly.

   A boundary must contain everything needed to constrain both recursive halves.
   For simple alignment problems this is often `(row, col)`. For affine-gap
   alignment it must include gap state. For DTW, constrained subproblems must
   preserve the absolute target column because vertical continuation cost
   depends on the current target observation.

6. Implement `choose_split`.

   The split boundary must be feasible for both halves and compatible with an
   optimal path from `beta_a` to `beta_c`. A conservative implementation can use
   forward constrained scoring for the left half and backward or direct
   constrained scoring for the right half.

7. Implement leaf traceback.

   It is fine for leaves to use a dense local table. The table is bounded by the
   block size, and this keeps correctness easy to audit.

8. Add independent path scoring.

   The path scorer should not reuse the traceback logic. Tests and CLI/report
   validation use path scoring to prove that the returned path realizes the
   reported objective.

9. Add a baseline.

   Every exported problem should have a full-table or otherwise trusted baseline
   for bounded cases. Baselines are allowed to be slow.

10. Add contract tests before export.

    The problem should pass:

    - summary apply equals direct interval replay,
    - merged summary equals direct combined interval,
    - split boundary is feasible,
    - reconstructed path joins exactly,
    - independent path score equals reported objective,
    - baseline objective equals HCP objective.

## Boundary Pitfalls

Most correctness bugs in this architecture are boundary bugs.

Watch for these cases:

- A split boundary needs more than coordinates.
- A leaf path starts or ends one cell away from its requested boundary.
- A gap, run, or stateful transition crosses the split.
- A local-alignment path starts after the split or ends before the split.
- A free-end-gap mode accidentally charges or omits a terminal gap.
- A summary was built from one frontier and cannot apply to another.
- A path scorer silently accepts illegal moves.

When in doubt, put the missing information into `Boundary`, not into hidden
global state.

## Choosing A First Implementation Strategy

Prefer correctness over cleverness:

- summaries can replay intervals,
- split scoring can be direct and slow,
- leaves can use dense tables,
- tests should cover tiny exhaustive or property-generated inputs.

After correctness is established, optimize one piece at a time:

- specialize summary application,
- replace direct right-half split scoring with backward frontiers,
- add sparse or banded frontiers,
- add SIMD or bit-vector kernels,
- widen benchmark scenarios.

## Examples In This Repo

- `src/problems/edit_distance.rs`: simple minimization grid with exact traceback
  plus specialized score/path backends.
- `src/problems/nw_affine.rs`: stateful boundaries for affine gaps.
- `src/problems/smith_waterman.rs`: local alignment with variable start/end.
- `src/problems/semiglobal.rs`: free target prefix/suffix semantics.
- `src/problems/dtw.rs`: first non-sequence-alignment proof point.

Use these as templates, but keep the recurrence-specific boundary semantics
explicit. The engine can enforce joins; it cannot infer what information a
problem-specific boundary forgot to carry.
