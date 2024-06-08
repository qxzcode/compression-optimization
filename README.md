This repository contains experiments in using [CP-SAT from Google's `ortools`](https://developers.google.com/optimization/cp/cp_solver) with some custom lazy constraints to optimize text data for compressibility. One potential application would be reordering rules/attributes/functions/etc. in HTML/CSS/JS/SVG to minimize the gzip/brotli-compressed size.

# Here's the problem setup:

**Input:** a string of bytes containing groups of substrings that may be freely reordered. Example:
```rust
Sequence([
    Literal("<a"),
    Permutation([
        Literal(" href=\"https://example.com/\""),
        Literal(" id=\"example-link\""),
        Sequence([
            Literal(" class=\""),
            Permutation([
                Literal(" link"),
                Literal(" bg-blue"),
                Literal(" font-sm"),
            ]),
            Literal("\""),
        ]),
    ]),
    Literal(">Link text.</a>"),
])
```

Two possible permutations of these:
```html
<a href="https://example.com/" id="example-link" class=" link bg-blue font-sm">Link text.</a>
<a id="example-link" class=" link font-sm bg-blue" href="https://example.com/">Link text.</a>
```

**Output:** the ordering of each of the permutable groups that minimizes the **number of LZ "backreferences"** required to encode the string. (This is a crude approximation of the output size from an LZ-based compression algorithm such as gzip/DEFLATE or brotli.)

The example above is somewhat trivial on its own, but optimizing the attribute order of all elements on a page jointly *can* give a non-negligible reduction in the final compressed size.

Also, while I haven't completely thought it through – I'm pretty sure that this problem is NP-hard. Fun!


# The approach I experimented with

Simply sorting attributes by name is a good heuristic, but I wanted to see if I could obtain provably optimal results. Of course, for problem instances of practical interest, the number of possible permutations is far too big to try all of them exhaustively.

In addition to modelling the set of valid permutations as above, I model each character in the string as being associated with a boolean flag indicating whether that character is at the start of a new backreference/"copy span". The total number of copy spans, which I seek to minimize, is then the sum of these flags.

I start with no constraints on the flags, and loop the following process:

1. Use CP-SAT to find an optimial solution (permutations + copy start flags) to the current minimization problem.
2. Check if this solution is feasible (i.e. if the values of the copy start flags correspond to a valid LZ encoding of the string).
    - If it's not feasible, add constraints enforcing that any missing copy flags are set, conditional on any relevant aspects of the permutation (such as one symbol precedng another symbol, or two substrings being adjacent).  
    Intuitively, these constraints prevent the infeasible solution (and related ones) from being found by CP-SAT again, without eliminating any optimal solutions that are truly feasible. (The specifics of these constraints are toward the bottom of `src/model.rs` – feel free to reach out to me for a detailed explanation.)
    - If it is feasible (all copy flags are valid for the permutation identified), then this solution is provably optimal and we're done!

Each iteration reduces the search space by eliminating more and more infeasible solutions.
Since there are a strictly finite number of infeasible solutions, given unlimited time all the "super-optimal" but infeasible solutions are guaranteed to be ruled out – at which point the algorithm will terminate with a solution that is both optimal and feasible.


# Does it actually work? (Quickly?)

For smallish problems, yes. It's much faster than an exhaustive search.

The code here was able to find optimal solutions for some SVG files ranging up to a few kilobytes, taking seconds for small files and a few hours for the largest ones. Given the likely NP-hard nature of the problem, this code likely won't be able to solve an instance with thousands of attributes in a usable amount of time.

Also, minimizing the number of LZ backreferences does not directly correspond to minimizing, e.g., the brotli-compressed size. A heuristic hill-climbing approach (not included in this repo) was able to get nearly-as-good or slightly better compressed sizes than the "optimal" solutions found by this constraint programming approach, while running wayy faster.

Modeling brotli compression so it can be optimized within the CP framework is left to future work :)


# If you found this interesting

... or if you have other thoughts on this problem, let me know! (GitHub issue / [LinkedIn](https://www.linkedin.com/in/quinn-tucker) is fine)
