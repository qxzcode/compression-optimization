pub mod bool_literal;
pub mod linear_expr;
pub mod model;
pub mod string_set;

use pyo3::prelude::*;

use crate::model::Model;

/// TODO: docstring
#[pyfunction]
fn solve() -> PyResult<Py<PyAny>> {
    use string_set::StringSet::*;

    let seq = Sequence(vec![
        Literal("<g".as_bytes().to_vec()),
        Permutation(vec![
            Literal(" stroke-width=\"6\"".as_bytes().to_vec()),
            Literal(" fill=\"red\"".as_bytes().to_vec()),
            Literal(" stroke=\"blue\"".as_bytes().to_vec()),
        ]),
        Literal(">filler content no. 66</g>".as_bytes().to_vec()),
        Literal("<g".as_bytes().to_vec()),
        Permutation(vec![
            Literal(" stroke=\"blue\"".as_bytes().to_vec()),
            Literal(" fill=\"red\"".as_bytes().to_vec()),
            Literal(" stroke-width=\"10\"".as_bytes().to_vec()),
        ]),
        Literal(">real content</g>".as_bytes().to_vec()),
        // Literal("<g".as_bytes().to_vec()),
        // Permutation(vec![
        //     Literal(" fill=\"green\"".as_bytes().to_vec()),
        //     Literal(" stroke-width=\"10\"".as_bytes().to_vec()),
        //     Literal(" stroke=\"blue\"".as_bytes().to_vec()),
        // ]),
        // Literal(">stuff</g>".as_bytes().to_vec()),
        // Literal("<g".as_bytes().to_vec()),
        // Permutation(vec![
        //     Literal(" stroke=\"blue\"".as_bytes().to_vec()),
        //     Literal(" stroke-width=\"10\"".as_bytes().to_vec()),
        //     Literal(" fill=\"reddish\"".as_bytes().to_vec()),
        // ]),
        // Literal(">things</g>".as_bytes().to_vec()),
        // Literal("<g".as_bytes().to_vec()),
        // Permutation(vec![
        //     Literal(" stroke=\"black\"".as_bytes().to_vec()),
        //     Literal(" fill=\"red\"".as_bytes().to_vec()),
        //     Literal(" stroke-width=\"10\"".as_bytes().to_vec()),
        // ]),
        // Literal(">wow #10!</g>".as_bytes().to_vec()),
        // Literal("<g".as_bytes().to_vec()),
        // Permutation(vec![
        //     Literal(" fill=\"red\"".as_bytes().to_vec()),
        //     Literal(" stroke=\"blue\"".as_bytes().to_vec()),
        //     Literal(" stroke-width=\"60\"".as_bytes().to_vec()),
        // ]),
        // Literal(">real content</g>".as_bytes().to_vec()),
        Literal(
            concat!(
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            )
            .as_bytes()
            .to_vec(),
        ),
    ]);

    println!("Building...");
    let model = Model::build(seq);
    println!("Model:");
    println!("    {} symbols", model.symbols.len());

    println!("Building cp_model.Model...");
    let cp_model = Python::with_gil(|py| model.to_cp_sat(py).map(|model| model.unbind()))?;
    println!("Done.");

    Ok(cp_model)
}

/// A Python module implemented in Rust.
#[pymodule]
fn compression_rs(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}
