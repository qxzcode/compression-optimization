//! Functionality for building CP-SAT models to optimize over `StringSet`s.

use std::{
    borrow::Borrow,
    cmp::Ordering,
    collections::HashMap,
    iter,
    ops::{Index, Range, RangeInclusive},
};

use pyo3::{prelude::*, py_run};

use crate::{
    bool_literal::{BoolLiteral, Conjunction, Disjunction},
    linear_expr::{var, LinearExpr},
    string_set::StringSet,
};

#[derive(Clone, Copy)]
enum IntValue {
    Constant(i32),
    Var(usize),
}

pub struct Symbol {
    label: u8,
    index: Vec<IntValue>,
    outgoing_arcs: HashMap<usize, BoolLiteral>,
    incoming_arcs: HashMap<usize, BoolLiteral>,
    copy_flags: HashMap<usize, (BoolLiteral, BeforeCondition)>,
}

#[derive(Default)]
struct VariableRegistry {
    ranges: Vec<RangeInclusive<i64>>,
    conjunction_vars: Vec<(usize, HashMap<usize, bool>)>,
    disjunctions: Vec<HashMap<usize, bool>>,
}

impl VariableRegistry {
    fn new_bool(&mut self) -> usize {
        self.ranges.push(0..=1);
        self.ranges.len() - 1
    }

    fn new_bool_literal(&mut self) -> BoolLiteral {
        BoolLiteral::Var {
            id: self.new_bool(),
            negated: false,
        }
    }

    #[allow(dead_code)]
    fn new_int(&mut self, min: i64, max: i64) -> usize {
        assert!(min <= max);
        self.ranges.push(min..=max);
        self.ranges.len() - 1
    }

    fn new_ints(&mut self, min: i64, max: i64, num_vars: usize) -> Range<usize> {
        assert!(min <= max);
        let start = self.ranges.len();
        self.ranges.extend(iter::repeat(min..=max).take(num_vars));
        start..self.ranges.len()
    }

    fn literal(&mut self, conjunction: Conjunction) -> BoolLiteral {
        match conjunction {
            Conjunction::Const(value) => BoolLiteral::Const(value),
            Conjunction::Vars(vars) => {
                if vars.is_empty() {
                    BoolLiteral::Const(true)
                } else if vars.len() == 1 {
                    let (id, negated) = vars.into_iter().next().unwrap();
                    BoolLiteral::Var { id, negated }
                } else {
                    let id = self.new_bool();
                    self.conjunction_vars.push((id, vars));
                    BoolLiteral::Var { id, negated: false }
                }
            }
        }
    }

    fn add_implication(
        &mut self,
        antecedent: impl Into<Conjunction>,
        consequent: impl Into<Disjunction>,
    ) {
        match !antecedent.into() | consequent.into() {
            Disjunction::Const(true) => {}
            Disjunction::Const(false) => panic!("add_implication: statically infeasible"),
            Disjunction::Vars(vars) => self.disjunctions.push(vars),
        }
    }
}

pub struct Model {
    pub symbols: Vec<Symbol>,
    symbols_by_label: HashMap<u8, Vec<usize>>,

    start_arcs: HashMap<usize, BoolLiteral>,

    permutation_index_sets: Vec<Range<usize>>,
    eq_indicator_vars: Vec<(usize, LinearExpr)>,
    copy_span_starts: Vec<BoolLiteral>,
    vars: VariableRegistry,
}

type ConditionalSymbolSet = Vec<(usize, BoolLiteral)>;

impl Model {
    pub fn build(string_set: StringSet) -> Model {
        let mut model = Model {
            symbols: Vec::new(),
            symbols_by_label: HashMap::new(),
            start_arcs: HashMap::new(),
            permutation_index_sets: Vec::new(),
            eq_indicator_vars: Vec::new(),
            copy_span_starts: Vec::new(),
            vars: VariableRegistry::default(),
        };
        let (start_arcs, _) = model.create_symbols(&string_set, &[]);
        model.start_arcs = HashMap::from_iter(start_arcs);
        model.create_copy_flags();
        model.create_copy_span_starts();
        model
    }

    fn new_symbol(&mut self, label: u8, index: Vec<IntValue>) -> (usize, &mut Symbol) {
        let symbol_index = self.symbols.len();
        self.symbols.push(Symbol {
            label,
            index,
            outgoing_arcs: HashMap::new(),
            incoming_arcs: HashMap::new(),
            copy_flags: HashMap::new(),
        });
        self.symbols_by_label
            .entry(label)
            .or_default()
            .push(symbol_index);
        (symbol_index, &mut self.symbols[symbol_index])
    }

    fn new_permutation_index_set(&mut self, size: usize) -> Range<usize> {
        let max_index = i64::try_from(size).unwrap() - 1;
        let index_set = self.vars.new_ints(0, max_index, size);
        self.permutation_index_sets.push(index_set.clone());
        index_set
    }

    fn new_eq_indicator(
        &mut self,
        expr1: impl Into<LinearExpr>,
        expr2: impl Into<LinearExpr>,
    ) -> BoolLiteral {
        let id = self.vars.new_bool();
        self.eq_indicator_vars
            .push((id, expr1.into() - expr2.into()));
        BoolLiteral::Var { id, negated: false }
    }

    /// Adds a conditional directed edge between the two symbols with the given IDs.
    fn connect_if(&mut self, id1: usize, id2: usize, condition: impl Into<BoolLiteral>) {
        let condition = condition.into();
        self.symbols[id1].outgoing_arcs.insert(id2, condition);
        self.symbols[id2].incoming_arcs.insert(id1, condition);
    }

    /// Adds conditional directed edges between the two sets of symbols with the given IDs.
    fn connect_all_if(
        &mut self,
        ids1: &[(usize, BoolLiteral)],
        ids2: &[(usize, BoolLiteral)],
        condition: impl Into<BoolLiteral>,
    ) {
        let condition = condition.into();
        for &(id1, condition1) in ids1 {
            for &(id2, condition2) in ids2 {
                let condition = self.vars.literal(condition1 & condition2 & condition);
                self.connect_if(id1, id2, condition);
            }
        }
    }

    /// Creates the `Symbol`s for the given `StringSet`.
    ///
    /// Returns the sets of (conditional) start and end symbols for the sequence.
    fn create_symbols(
        &mut self,
        string_set: &StringSet,
        parent_index: &[IntValue],
    ) -> (ConditionalSymbolSet, ConditionalSymbolSet) {
        match string_set {
            StringSet::Literal(string) => {
                assert!(!string.is_empty());

                let mut start_symbol = None;
                let mut prev_symbol = None;
                for (i, &label) in string.iter().enumerate() {
                    // Create the symbol for this character.
                    let index = append(parent_index, IntValue::Constant(i32::try_from(i).unwrap()));
                    let (id, _) = self.new_symbol(label, index);
                    if i == 0 {
                        start_symbol = Some(id);
                    }

                    // Connect the symbol to the previous symbol.
                    if let Some(prev_id) = prev_symbol {
                        self.connect_if(prev_id, id, true);
                    }
                    prev_symbol = Some(id);
                }

                (
                    vec![(start_symbol.unwrap(), true.into())],
                    vec![(prev_symbol.unwrap(), true.into())],
                )
            }
            StringSet::Sequence(items) => {
                assert!(!items.is_empty());

                let mut start_symbols = Vec::new();
                let mut prev_symbols = Vec::new();
                for (i, string_set) in items.iter().enumerate() {
                    // Recursively create symbols for the subsequence.
                    let (local_start_symbols, local_end_symbols) = self.create_symbols(
                        string_set,
                        &append(parent_index, IntValue::Constant(i32::try_from(i).unwrap())),
                    );

                    if i == 0 {
                        start_symbols = local_start_symbols;
                    } else {
                        // Connect the subsequence to the previous subsequence.
                        self.connect_all_if(&prev_symbols, &local_start_symbols, true);
                    }
                    prev_symbols = local_end_symbols;
                }

                (start_symbols, prev_symbols)
            }
            StringSet::Permutation(items) => {
                assert!(!items.is_empty());

                let index_set = self.new_permutation_index_set(items.len());
                let max_index = i64::try_from(items.len()).unwrap() - 1;

                let mut start_symbols = Vec::new();
                let mut end_symbols = Vec::new();
                let mut local_start_sets = Vec::new();
                let mut local_end_sets = Vec::new();
                for (index_var_id, string_set) in index_set.clone().zip(items) {
                    let (local_start_symbols, local_end_symbols) = self.create_symbols(
                        string_set,
                        &append(parent_index, IntValue::Var(index_var_id)),
                    );

                    let is_first = self.new_eq_indicator(var(index_var_id), 0);
                    for &(id, is_active) in &local_start_symbols {
                        start_symbols.push((id, self.vars.literal(is_first & is_active)));
                    }

                    let is_last = self.new_eq_indicator(var(index_var_id), max_index);
                    for &(id, is_active) in &local_end_symbols {
                        end_symbols.push((id, self.vars.literal(is_last & is_active)));
                    }

                    local_start_sets.push(local_start_symbols);
                    local_end_sets.push(local_end_symbols);
                }

                // Add connections between items, conditional on being adjacent in the permutation.
                for (index_var1_id, local_ends) in index_set.clone().zip(&local_end_sets) {
                    for (index_var2_id, local_starts) in index_set.clone().zip(&local_start_sets) {
                        if index_var1_id != index_var2_id {
                            let condition =
                                self.new_eq_indicator(var(index_var1_id) + 1, var(index_var2_id));
                            self.connect_all_if(local_ends, local_starts, condition);
                        }
                    }
                }

                (start_symbols, end_symbols)
            }
        }
    }
}

fn append<T: Clone>(list: &[T], item: T) -> Vec<T> {
    let mut new_list = Vec::with_capacity(list.len() + 1);
    new_list.extend_from_slice(list);
    new_list.push(item);
    new_list
}

#[derive(PartialEq, Eq, Clone, Copy, Hash, Debug)]
enum BeforeCondition {
    Never,
    Always,
    IfLess(usize, usize),
}

impl Model {
    fn is_before(&self, id1: usize, id2: usize) -> BeforeCondition {
        if id1 == id2 {
            return BeforeCondition::Never; // A symbol is never strictly before itself.
        }

        for (&i1, &i2) in iter::zip(&self.symbols[id1].index, &self.symbols[id2].index) {
            match (i1, i2) {
                (IntValue::Constant(i1), IntValue::Constant(i2)) => match i1.cmp(&i2) {
                    Ordering::Less => return BeforeCondition::Always,
                    Ordering::Greater => return BeforeCondition::Never,
                    Ordering::Equal => continue,
                },
                (IntValue::Var(id1), IntValue::Var(id2)) => {
                    if id1 == id2 {
                        continue;
                    } else {
                        return BeforeCondition::IfLess(id1, id2);
                    }
                }
                _ => unreachable!("tried to compare a constant index with a variable index"),
            }
        }

        // The two indices must be equal, so one does not come strictly before the other.
        BeforeCondition::Never
    }

    fn create_copy_flags(&mut self) {
        let mut total_num_flags = 0;

        for id in 0..self.symbols.len() {
            for &id2 in &self.symbols_by_label[&self.symbols[id].label] {
                let is_before_condition = self.is_before(id2, id);
                if is_before_condition != BeforeCondition::Never {
                    let flag = self.vars.new_bool_literal();
                    total_num_flags += 1;
                    self.symbols[id]
                        .copy_flags
                        .insert(id2, (flag, is_before_condition));
                }
            }
        }

        println!("Created {total_num_flags} copy flags.");
    }

    fn create_copy_span_starts(&mut self) {
        let mut total_num_literals = 0;

        for id in 0..self.symbols.len() {
            let symbol = &self.symbols[id];

            // is_css = true iff there is not any "parallel copy quadrilateral".
            let mut is_css = Conjunction::Const(true);
            for (&prev_id, &edge_flag) in &symbol.incoming_arcs {
                let prev_symbol = &self.symbols[prev_id];

                for (&id2, &(copy_flag, _)) in &symbol.copy_flags {
                    for (&prev_id2, &(prev_copy_flag, _)) in &prev_symbol.copy_flags {
                        let prev_symbol2 = &self.symbols[prev_id2];

                        if let Some(&edge_flag2) = prev_symbol2.outgoing_arcs.get(&id2) {
                            let parallel_copy = edge_flag & edge_flag2 & copy_flag & prev_copy_flag;

                            // Reduce the number of optimal solutions (while preserving the optimal value):
                            // If a parallel copy is possible, then mandate it.
                            self.vars.add_implication(
                                edge_flag & edge_flag2 & prev_copy_flag,
                                copy_flag,
                            );

                            is_css &= !self.vars.literal(parallel_copy);
                            total_num_literals += 1;
                        }
                    }
                }
            }

            self.copy_span_starts.push(self.vars.literal(is_css));
        }

        println!("Created copy span start flags. ({total_num_literals} literals)");
    }
}

struct VarMap<'py> {
    py: Python<'py>,
    map: HashMap<usize, Bound<'py, PyAny>>,
}

impl<'py> Index<usize> for VarMap<'py> {
    type Output = Bound<'py, PyAny>;

    fn index(&self, id: usize) -> &Self::Output {
        &self.map[&id]
    }
}

impl<'py> VarMap<'py> {
    fn literal(&self, literal: BoolLiteral) -> PyResult<Bound<'py, PyAny>> {
        match literal {
            BoolLiteral::Const(value) => Ok(value.into_py(self.py).into_bound(self.py)),
            BoolLiteral::Var { id, negated } => {
                let var = self[id].clone();
                if negated {
                    var.call_method0("Not")
                } else {
                    Ok(var)
                }
            }
        }
    }

    fn literals(
        &self,
        literals: impl IntoIterator<Item = impl Borrow<BoolLiteral>>,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        literals
            .into_iter()
            .map(|l| self.literal(*l.borrow()))
            .collect()
    }
}

impl Model {
    pub fn to_cp_sat<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // Create the model object.
        let model = py
            .import_bound("ortools.sat.python.cp_model")?
            .getattr("CpModel")?
            .call0()?;

        // Create the variables.
        let mut vars = HashMap::new();
        for (i, range) in self.vars.ranges.iter().enumerate() {
            let var = model.getattr("new_int_var")?.call1((
                *range.start(),
                *range.end(),
                format!("var{i}"),
            ))?;
            vars.insert(i, var);
        }
        let vars = VarMap { py, map: vars };

        let enforce_hints = false;
        if enforce_hints {
            // Create the hint enforcement constraints.
            for index_set in &self.permutation_index_sets {
                for (i, var_index) in index_set.clone().enumerate() {
                    model.call_method1("add", (vars[var_index].call_method1("__eq__", (i,))?,))?;
                }
            }
        }

        // Create the all_different constraints.
        for index_set in &self.permutation_index_sets {
            model.call_method1(
                "add_all_different",
                (index_set.clone().map(|i| &vars[i]).collect::<Vec<_>>(),),
            )?;
        }

        // Create the equality indicator constraints.
        for (id, expr) in &self.eq_indicator_vars {
            let mut sum = expr.constant.into_py(py).into_bound(py);
            for (&id, &coef) in &expr.terms {
                sum = sum.add(vars[id].mul(coef)?)?;
            }
            let is_eq = BoolLiteral::Var {
                id: *id,
                negated: false,
            };
            model
                .call_method1("add", (sum.call_method1("__eq__", (0,))?,))?
                .call_method1("only_enforce_if", (vars.literal(is_eq)?,))?;
            model
                .call_method1("add", (sum.call_method1("__ne__", (0,))?,))?
                .call_method1("only_enforce_if", (vars.literal(!is_eq)?,))?;
        }

        // Create the copy flag is_before and at_most_one constraints.
        for symbol in &self.symbols {
            for &(copy_flag, is_before) in symbol.copy_flags.values() {
                let is_before = match is_before {
                    BeforeCondition::Never => unreachable!(),
                    BeforeCondition::Always => continue,
                    BeforeCondition::IfLess(id1, id2) => {
                        vars[id1].call_method1("__lt__", (&vars[id2],))?
                    }
                };
                let copy_flag = vars.literal(copy_flag)?;
                model
                    .call_method1("add", (is_before,))?
                    .call_method1("only_enforce_if", (copy_flag,))?;
            }

            model.call_method1(
                "add_at_most_one",
                (vars.literals(symbol.copy_flags.values().map(|&(copy_flag, _)| copy_flag))?,),
            )?;
        }

        // Create the conjunction literal constraints.
        for (id, literals) in &self.vars.conjunction_vars {
            let literals = literals
                .iter()
                .map(|(&id, &negated)| BoolLiteral::Var { id, negated });
            model.call_method1("add_min_equality", (&vars[*id], vars.literals(literals)?))?;
        }

        // Create the disjunction constraints.
        for disjunction in &self.vars.disjunctions {
            let literals = disjunction
                .iter()
                .map(|(&id, &negated)| BoolLiteral::Var { id, negated });
            model.call_method1("add_bool_or", (vars.literals(literals)?,))?;
        }

        let csss = vars.literals(&self.copy_span_starts)?;

        py_run!(
            py,
            model csss,
            r#"
print()
print()
print()
from ortools.sat.python import cp_model

model.minimize(sum(csss))

# Solve
global solver
solver = cp_model.CpSolver()
solver.parameters.max_memory_in_mb = 1_000 // 2
solver.parameters.log_search_progress = True
# solver.parameters.max_time_in_seconds = 60.0 * 10
# solver.parameters.debug_crash_on_bad_hint = True

print("Solving...")
status = solver.solve(model)
print("Status:", solver.status_name())
            "#
        );

        let solver = py.eval_bound("solver", None, None)?;
        // let get_value = |var_id: usize| -> PyResult<i64> {
        //     let value = solver.call_method1("value", (&vars[var_id],))?;
        //     value.extract()
        // };
        let get_literal_value = |literal: BoolLiteral| -> PyResult<bool> {
            let value = solver.call_method1("value", (&vars.literal(literal)?,))?;
            Ok(match value.extract::<i64>()? {
                0 => false,
                1 => true,
                _ => panic!("unexpected boolean literal value: {}", value),
            })
        };
        let get_successor = |outgoing_arcs: &HashMap<usize, BoolLiteral>, id: Option<usize>| {
            let successors: Vec<usize> = outgoing_arcs
                .iter()
                .filter_map(|(&id, &cond)| {
                    get_literal_value(cond)
                        .map(|cond| cond.then_some(id))
                        .transpose()
                })
                .collect::<PyResult<_>>()?;
            if successors.is_empty() {
                PyResult::Ok(None)
            } else if successors.len() == 1 {
                PyResult::Ok(Some(*successors.first().unwrap()))
            } else {
                panic!(
                    "solution has {} successors for {}",
                    successors.len(),
                    match id {
                        Some(id) => format!(
                            "symbol ID {id:?} ({:?})",
                            char::from(self.symbols[id].label),
                        ),
                        None => "<start>".into(),
                    },
                );
            }
        };

        let mut output_ids = Vec::new();
        let mut next_id = get_successor(&self.start_arcs, None)?;
        assert!(next_id.is_some());
        while let Some(id) = next_id {
            output_ids.push(id);
            next_id = get_successor(&self.symbols[id].outgoing_arcs, Some(id))?;
        }

        let mut highlighted = true;
        for id in output_ids {
            highlighted = get_literal_value(self.copy_span_starts[id])?;
            if highlighted {
                print!("\x1b[38;5;198m\x1b[48;5;52m");
            } else {
                print!("\x1b[0m");
            }
            print!("{}", char::from(self.symbols[id].label));
        }
        if highlighted {
            print!("\x1b[0m");
        }
        println!();

        // Ok(model)
        Ok(py.None().into_bound(py))
    }
}
