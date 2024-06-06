//! Data structures for representing combinatorial sets of symbol sequences ("strings").

/// A struct representing a finite (but potentially very large) set of strings.
/// By "string" we mean a finite sequence of u8 values.
pub enum StringSet {
    /// A set containing a single string.
    Literal(Vec<u8>),
    /// The concatenation of several string sets.
    Sequence(Vec<StringSet>),
    /// The set of all permutations of a collection of string sets.
    Permutation(Vec<StringSet>),
}
