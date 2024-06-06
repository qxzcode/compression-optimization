use std::{
    collections::HashMap,
    ops::{Add, Mul, Sub},
};

#[derive(Clone, Debug)]
pub struct LinearExpr {
    pub terms: HashMap<usize, i64>,
    pub constant: i64,
}

pub fn var(id: usize) -> LinearExpr {
    LinearExpr {
        terms: HashMap::from_iter([(id, 1)]),
        constant: 0,
    }
}

pub fn constant(value: i64) -> LinearExpr {
    LinearExpr {
        terms: HashMap::new(),
        constant: value,
    }
}

impl From<i64> for LinearExpr {
    fn from(value: i64) -> Self {
        constant(value)
    }
}

impl Add for LinearExpr {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut terms = self.terms;
        for (id, coef2) in other.terms {
            let coef = terms.entry(id).or_default();
            *coef += coef2;
            if *coef == 0 {
                terms.remove(&id);
            }
        }
        LinearExpr {
            terms,
            constant: self.constant + other.constant,
        }
    }
}

impl Sub for LinearExpr {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut terms = self.terms;
        for (id, coef2) in other.terms {
            let coef = terms.entry(id).or_default();
            *coef -= coef2;
            if *coef == 0 {
                terms.remove(&id);
            }
        }
        LinearExpr {
            terms,
            constant: self.constant - other.constant,
        }
    }
}

impl Add<i64> for LinearExpr {
    type Output = Self;

    fn add(self, constant: i64) -> Self {
        LinearExpr {
            terms: self.terms,
            constant: self.constant + constant,
        }
    }
}

impl Sub<i64> for LinearExpr {
    type Output = Self;

    fn sub(self, constant: i64) -> Self {
        LinearExpr {
            terms: self.terms,
            constant: self.constant - constant,
        }
    }
}

impl Mul<i64> for LinearExpr {
    type Output = Self;

    fn mul(self, constant: i64) -> Self {
        if constant == 0 {
            0.into()
        } else {
            LinearExpr {
                terms: self
                    .terms
                    .into_iter()
                    .map(|(id, value)| (id, value * constant))
                    .collect(),
                constant: self.constant * constant,
            }
        }
    }
}
