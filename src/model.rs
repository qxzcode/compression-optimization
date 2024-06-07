//! Functionality for building CP-SAT models to optimize over `StringSet`s.

use std::{
    borrow::Borrow,
    cmp::Ordering,
    collections::{BTreeMap, HashMap, HashSet},
    iter,
    ops::{Index, Range, RangeInclusive},
};

use pyo3::prelude::*;

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
    copy_conditions: HashMap<usize, BeforeCondition>,
}

#[derive(Default)]
struct VariableRegistry {
    ranges: Vec<RangeInclusive<i64>>,
    conjunction_vars: HashMap<BTreeMap<usize, bool>, usize>,
    disjunctions: HashSet<BTreeMap<usize, bool>>,
    is_less_literals: HashMap<(usize, usize), BoolLiteral>, // (id1, id2) => is_less  means  is_less = (id1 < id2)
}

mod variable_registry_impl {
    use super::*;

    pub(super) trait IntoLiteral {
        fn into_literal(self, var_reg: &mut VariableRegistry) -> BoolLiteral;
    }

    impl<T: Into<BoolLiteral>> IntoLiteral for T {
        fn into_literal(self, _var_reg: &mut VariableRegistry) -> BoolLiteral {
            self.into()
        }
    }

    impl IntoLiteral for Conjunction {
        fn into_literal(self, var_reg: &mut VariableRegistry) -> BoolLiteral {
            match self {
                Conjunction::Const(value) => BoolLiteral::Const(value),
                Conjunction::Vars(vars) => {
                    if vars.is_empty() {
                        BoolLiteral::Const(true)
                    } else if vars.len() == 1 {
                        let (id, negated) = vars.into_iter().next().unwrap();
                        BoolLiteral::Var { id, negated }
                    } else {
                        let id = *var_reg
                            .conjunction_vars
                            .entry(BTreeMap::from_iter(vars))
                            .or_insert_with(|| new_bool(&mut var_reg.ranges));
                        BoolLiteral::Var { id, negated: false }
                    }
                }
            }
        }
    }

    impl IntoLiteral for BeforeCondition {
        fn into_literal(self, var_reg: &mut VariableRegistry) -> BoolLiteral {
            match self {
                BeforeCondition::Never => BoolLiteral::Const(false),
                BeforeCondition::Always => BoolLiteral::Const(true),
                BeforeCondition::IfLess(id1, id2) => *var_reg
                    .is_less_literals
                    .entry((id1, id2))
                    .or_insert_with(|| new_bool_literal(&mut var_reg.ranges)),
            }
        }
    }

    fn new_bool(ranges: &mut Vec<RangeInclusive<i64>>) -> usize {
        ranges.push(0..=1);
        ranges.len() - 1
    }

    fn new_bool_literal(ranges: &mut Vec<RangeInclusive<i64>>) -> BoolLiteral {
        BoolLiteral::Var {
            id: new_bool(ranges),
            negated: false,
        }
    }

    impl VariableRegistry {
        pub(super) fn new_bool(&mut self) -> usize {
            new_bool(&mut self.ranges)
        }

        pub(super) fn new_bool_literal(&mut self) -> BoolLiteral {
            new_bool_literal(&mut self.ranges)
        }

        #[allow(dead_code)]
        pub(super) fn new_int(&mut self, min: i64, max: i64) -> usize {
            assert!(min <= max);
            self.ranges.push(min..=max);
            self.ranges.len() - 1
        }

        pub(super) fn new_ints(&mut self, min: i64, max: i64, num_vars: usize) -> Range<usize> {
            assert!(min <= max);
            let start = self.ranges.len();
            self.ranges.extend(iter::repeat(min..=max).take(num_vars));
            start..self.ranges.len()
        }

        pub(super) fn literal(&mut self, value: impl IntoLiteral) -> BoolLiteral {
            value.into_literal(self)
        }

        pub(super) fn add_implication(
            &mut self,
            antecedent: impl Into<Conjunction>,
            consequent: impl Into<Disjunction>,
        ) {
            match !antecedent.into() | consequent.into() {
                Disjunction::Const(true) => {}
                Disjunction::Const(false) => panic!("add_implication: statically infeasible"),
                Disjunction::Vars(vars) => {
                    self.disjunctions.insert(BTreeMap::from_iter(vars));
                }
            }
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
        model.copy_span_starts = iter::repeat_with(|| model.vars.new_bool_literal())
            .take(model.symbols.len())
            .collect();
        model
    }

    fn new_symbol(&mut self, label: u8, index: Vec<IntValue>) -> (usize, &mut Symbol) {
        let symbol_index = self.symbols.len();
        self.symbols.push(Symbol {
            label,
            index,
            outgoing_arcs: HashMap::new(),
            incoming_arcs: HashMap::new(),
            copy_conditions: HashMap::new(),
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
                    total_num_flags += 1;
                    self.symbols[id]
                        .copy_conditions
                        .insert(id2, is_before_condition);
                }
            }
        }

        println!("Created {total_num_flags} copy flags.");
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

    fn literal_value(&self, solver: &Bound<'py, PyAny>, literal: BoolLiteral) -> PyResult<bool> {
        let value = solver.call_method1("value", (&self.literal(literal)?,))?;
        Ok(match value.extract::<i64>()? {
            0 => false,
            1 => true,
            _ => panic!("unexpected boolean literal value: {}", value),
        })
    }

    fn literal_values(
        &self,
        solver: &Bound<'py, PyAny>,
        literals: impl IntoIterator<Item = impl Borrow<BoolLiteral>>,
    ) -> PyResult<Vec<bool>> {
        literals
            .into_iter()
            .map(|l| self.literal_value(solver, *l.borrow()))
            .collect()
    }
}

impl Model {
    fn to_cp_sat<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, VarMap<'py>)> {
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

        // Create the conjunction literal constraints.
        for (literals, id) in &self.vars.conjunction_vars {
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

        // Create the "is less" indicator constraints.
        for ((id1, id2), is_less) in &self.vars.is_less_literals {
            let v1 = &vars[*id1];
            let v2 = &vars[*id2];
            model
                .call_method1("add", (v1.call_method1("__lt__", (v2,))?,))?
                .call_method1("only_enforce_if", (vars.literal(*is_less)?,))?;
            model
                .call_method1("add", (v1.call_method1("__ge__", (v2,))?,))?
                .call_method1("only_enforce_if", (vars.literal(!*is_less)?,))?;
        }

        let mut objective = 0i64.into_py(py).into_bound(py);
        for is_css in vars.literals(&self.copy_span_starts)? {
            objective = objective.add(is_css)?;
        }
        model.call_method1("minimize", (objective,))?;

        Ok((model, vars))
    }
}

struct Solution {
    symbol_sequence: Vec<usize>,
    copy_span_starts: Vec<bool>,
}

impl Solution {
    fn objective_value(&self) -> usize {
        self.copy_span_starts.iter().filter(|&&b| b).count()
    }

    fn print(&self, symbols: &[Symbol]) {
        println!("Objective value: {}", self.objective_value());

        let mut highlighted = true;
        for &id in &self.symbol_sequence {
            highlighted = self.copy_span_starts[id];
            if highlighted {
                print!("\x1b[38;5;198m\x1b[48;5;52m");
            } else {
                print!("\x1b[0m");
            }
            print!("{}", char::from(symbols[id].label));
        }
        if highlighted {
            print!("\x1b[0m");
        }
        println!();
    }
}

impl Model {
    pub fn solve<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        println!();

        for iter_num in 1..=10 {
            println!("====== Iteration {} ======", iter_num);
            let (model, vars) = self.to_cp_sat(py)?;

            let solver = py
                .import_bound("ortools.sat.python.cp_model")?
                .getattr("CpSolver")?
                .call0()?;
            let parameters = solver.getattr("parameters")?;
            // parameters.setattr("max_presolve_iterations", 30)?;
            parameters.setattr("max_memory_in_mb", 1_000 / 2)?;
            // parameters.setattr("log_search_progress", true)?;
            // parameters.setattr("max_time_in_seconds", 60.0 * 10.0)?;
            // parameters.setattr("debug_crash_on_bad_hint", true)?;

            let _status = solver.call_method1("solve", (&model,))?;
            println!("Status: {}", solver.call_method0("status_name")?);

            let mut solution = self.extract_solution(&solver, &vars)?;
            solution.print(&self.symbols);

            let mut total_new_literals = 0;
            let mut cur_span_end = 0;
            for (i, &id) in solution.symbol_sequence.iter().enumerate() {
                let is_css = &mut solution.copy_span_starts[id];
                if !*is_css && (i == 0 || i > cur_span_end) {
                    // This symbol is past the end of the span, so either:
                    //  - this symbol must start a new span, OR
                    //  - the symbol sequence needs to be different

                    // If there is not any "parallel copy quadrilateral", then is_css must be true.
                    let symbol = &self.symbols[id];
                    let mut is_any_parallel_copy_possible = Disjunction::Const(false);
                    for (&prev_id, &edge_flag) in &symbol.incoming_arcs {
                        let prev_symbol = &self.symbols[prev_id];

                        for (&copy_id, &can_copy) in &symbol.copy_conditions {
                            for (&prev_copy_id, &prev_can_copy) in &prev_symbol.copy_conditions {
                                let prev_copy_symbol = &self.symbols[prev_copy_id];

                                if let Some(&edge_flag2) =
                                    prev_copy_symbol.outgoing_arcs.get(&copy_id)
                                {
                                    let is_parallel_copy_possible = edge_flag
                                        & edge_flag2
                                        & self.vars.literal(can_copy)
                                        & self.vars.literal(prev_can_copy);

                                    // println!(
                                    //     "  {:?} possible = {is_parallel_copy_possible:?}",
                                    //     char::from(symbol.label)
                                    // );
                                    is_any_parallel_copy_possible |=
                                        self.vars.literal(is_parallel_copy_possible);
                                    total_new_literals += 1;
                                }
                            }
                        }
                    }

                    // println!(
                    //     "  {:?}   any possible = {is_any_parallel_copy_possible:?}",
                    //     char::from(symbol.label)
                    // );
                    self.vars
                        .add_implication(!is_any_parallel_copy_possible, self.copy_span_starts[id]);
                    *is_css = true;
                }

                if *is_css {
                    // Determine the maximum extent of the span starting at this symbol.
                    let max_match_len = (0..i)
                        .map(|mut j| {
                            let mut i2 = i;
                            while i2 < solution.symbol_sequence.len()
                                && self.symbols[solution.symbol_sequence[i2]].label
                                    == self.symbols[solution.symbol_sequence[j]].label
                            {
                                i2 += 1;
                                j += 1;
                            }
                            i2 - i
                        })
                        .max()
                        .unwrap_or(1)
                        .max(1);
                    cur_span_end = i + max_match_len - 1;
                    // println!("{max_match_len}  =>  {i}-{cur_span_end}");
                }
            }

            println!("Added {total_new_literals} new literals.");
            solution.print(&self.symbols);

            println!();
        }

        Ok(py.None().into_bound(py))
    }

    fn extract_solution<'py>(
        &self,
        solver: &Bound<'py, PyAny>,
        vars: &VarMap<'py>,
    ) -> PyResult<Solution> {
        let get_successor = |outgoing_arcs: &HashMap<usize, BoolLiteral>, id: Option<usize>| {
            let successors: Vec<usize> = outgoing_arcs
                .iter()
                .filter_map(|(&id, &cond)| {
                    vars.literal_value(solver, cond)
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

        let mut symbol_sequence = Vec::new();
        let mut next_id = get_successor(&self.start_arcs, None)?;
        assert!(next_id.is_some());
        while let Some(id) = next_id {
            symbol_sequence.push(id);
            next_id = get_successor(&self.symbols[id].outgoing_arcs, Some(id))?;
        }

        let copy_span_starts = vars.literal_values(solver, &self.copy_span_starts)?;
        Ok(Solution {
            symbol_sequence,
            copy_span_starts,
        })
    }
}
