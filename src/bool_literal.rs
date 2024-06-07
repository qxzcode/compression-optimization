use std::{
    collections::{hash_map::Entry, HashMap},
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Not},
};

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub enum BoolLiteral {
    Const(bool),
    Var { id: usize, negated: bool },
}

impl From<bool> for BoolLiteral {
    fn from(value: bool) -> Self {
        BoolLiteral::Const(value)
    }
}

impl BitAnd for BoolLiteral {
    type Output = Conjunction;

    fn bitand(self, other: Self) -> Conjunction {
        Conjunction::from(self) & other
    }
}

impl Not for BoolLiteral {
    type Output = BoolLiteral;

    fn not(self) -> BoolLiteral {
        match self {
            BoolLiteral::Const(value) => BoolLiteral::Const(!value),
            BoolLiteral::Var { id, negated } => BoolLiteral::Var {
                id,
                negated: !negated,
            },
        }
    }
}

/// A conjunction of boolean literals.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Conjunction {
    Const(bool),
    Vars(HashMap<usize, bool>),
}

impl Conjunction {
    pub fn literals(&self) -> Vec<BoolLiteral> {
        match self {
            Conjunction::Const(value) => vec![BoolLiteral::Const(*value)],
            Conjunction::Vars(vars) => vars
                .iter()
                .map(|(&id, &negated)| BoolLiteral::Var { id, negated })
                .collect(),
        }
    }
}

impl From<BoolLiteral> for Conjunction {
    fn from(literal: BoolLiteral) -> Self {
        match literal {
            BoolLiteral::Const(value) => Conjunction::Const(value),
            BoolLiteral::Var { id, negated } => {
                Conjunction::Vars(HashMap::from_iter([(id, negated)]))
            }
        }
    }
}

impl<T: Into<Conjunction>> FromIterator<T> for Conjunction {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        iter.into_iter()
            .fold(Conjunction::Const(true), |accum, item| accum & item.into())
    }
}

impl Not for Conjunction {
    type Output = Disjunction;

    fn not(self) -> Disjunction {
        match self {
            Conjunction::Const(value) => Disjunction::Const(!value),
            Conjunction::Vars(vars) => Disjunction::Vars(
                vars.into_iter()
                    .map(|(id, negated)| (id, !negated))
                    .collect(),
            ),
        }
    }
}

impl BitAnd for Conjunction {
    type Output = Conjunction;

    fn bitand(self, other: Self) -> Conjunction {
        match other {
            Conjunction::Const(true) => self,
            Conjunction::Const(false) => Conjunction::Const(false),
            Conjunction::Vars(vars) => vars.into_iter().fold(self, |accum, (id, negated)| {
                accum & BoolLiteral::Var { id, negated }
            }),
        }
    }
}

impl<T: Into<BoolLiteral>> BitAnd<T> for Conjunction {
    type Output = Conjunction;

    fn bitand(self, other: T) -> Conjunction {
        match (self, other.into()) {
            (Conjunction::Const(true), other) => other.into(),
            (self_, BoolLiteral::Const(true)) => self_,

            (Conjunction::Const(false), _) => Conjunction::Const(false),
            (_, BoolLiteral::Const(false)) => Conjunction::Const(false),

            (Conjunction::Vars(mut vars), BoolLiteral::Var { id, negated }) => {
                match vars.entry(id) {
                    Entry::Occupied(entry) => {
                        if *entry.get() == negated {
                            // The new literal is redundant.
                            Conjunction::Vars(vars)
                        } else {
                            // The new literal is the negation of an existing one.
                            Conjunction::Const(false)
                        }
                    }
                    Entry::Vacant(entry) => {
                        // The new literal is independent from all existing literals.
                        entry.insert(negated);
                        Conjunction::Vars(vars)
                    }
                }
            }
        }
    }
}

impl<T: Into<BoolLiteral>> BitAndAssign<T> for Conjunction {
    fn bitand_assign(&mut self, other: T) {
        let old_self = std::mem::replace(self, Conjunction::Const(true));
        *self = old_self & other;
    }
}

/// A disjunction of boolean literals.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Disjunction {
    Const(bool),
    Vars(HashMap<usize, bool>),
}

impl Disjunction {
    pub fn into_literals(self) -> Vec<BoolLiteral> {
        match self {
            Disjunction::Const(value) => vec![BoolLiteral::Const(value)],
            Disjunction::Vars(vars) => vars
                .into_iter()
                .map(|(id, negated)| BoolLiteral::Var { id, negated })
                .collect(),
        }
    }
}

impl From<BoolLiteral> for Disjunction {
    fn from(literal: BoolLiteral) -> Self {
        match literal {
            BoolLiteral::Const(value) => Disjunction::Const(value),
            BoolLiteral::Var { id, negated } => {
                Disjunction::Vars(HashMap::from_iter([(id, negated)]))
            }
        }
    }
}

impl<T: Into<Disjunction>> FromIterator<T> for Disjunction {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        iter.into_iter()
            .fold(Disjunction::Const(false), |accum, item| accum | item.into())
    }
}

impl Not for Disjunction {
    type Output = Conjunction;

    fn not(self) -> Conjunction {
        match self {
            Disjunction::Const(value) => Conjunction::Const(!value),
            Disjunction::Vars(vars) => Conjunction::Vars(
                vars.into_iter()
                    .map(|(id, negated)| (id, !negated))
                    .collect(),
            ),
        }
    }
}

impl BitOr for Disjunction {
    type Output = Disjunction;

    fn bitor(self, other: Self) -> Disjunction {
        match other {
            Disjunction::Const(true) => Disjunction::Const(true),
            Disjunction::Const(false) => self,
            Disjunction::Vars(vars) => vars.into_iter().fold(self, |accum, (id, negated)| {
                accum | BoolLiteral::Var { id, negated }
            }),
        }
    }
}

impl<T: Into<BoolLiteral>> BitOr<T> for Disjunction {
    type Output = Disjunction;

    fn bitor(self, other: T) -> Disjunction {
        match (self, other.into()) {
            (Disjunction::Const(false), other) => other.into(),
            (self_, BoolLiteral::Const(false)) => self_,

            (Disjunction::Const(true), _) => Disjunction::Const(true),
            (_, BoolLiteral::Const(true)) => Disjunction::Const(true),

            (Disjunction::Vars(mut vars), BoolLiteral::Var { id, negated }) => {
                match vars.entry(id) {
                    Entry::Occupied(entry) => {
                        if *entry.get() == negated {
                            // The new literal is redundant.
                            Disjunction::Vars(vars)
                        } else {
                            // The new literal is the negation of an existing one.
                            Disjunction::Const(true)
                        }
                    }
                    Entry::Vacant(entry) => {
                        // The new literal is independent from all existing literals.
                        entry.insert(negated);
                        Disjunction::Vars(vars)
                    }
                }
            }
        }
    }
}

impl<T: Into<BoolLiteral>> BitOrAssign<T> for Disjunction {
    fn bitor_assign(&mut self, other: T) {
        let old_self = std::mem::replace(self, Disjunction::Const(false));
        *self = old_self | other;
    }
}
