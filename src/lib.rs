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
        //*
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
        Literal("<g".as_bytes().to_vec()),
        Permutation(vec![
            Literal(" fill=\"green\"".as_bytes().to_vec()),
            Literal(" stroke-width=\"10\"".as_bytes().to_vec()),
            Literal(" stroke=\"blue\"".as_bytes().to_vec()),
        ]),
        Literal(">stuff</g>".as_bytes().to_vec()),
        Literal("<g".as_bytes().to_vec()),
        Permutation(vec![
            Literal(" stroke=\"blue\"".as_bytes().to_vec()),
            Literal(" stroke-width=\"10\"".as_bytes().to_vec()),
            Literal(" fill=\"reddish\"".as_bytes().to_vec()),
        ]),
        Literal(">things</g>".as_bytes().to_vec()),
        Literal("<g".as_bytes().to_vec()),
        Permutation(vec![
            Literal(" stroke=\"black\"".as_bytes().to_vec()),
            Literal(" fill=\"red\"".as_bytes().to_vec()),
            Literal(" stroke-width=\"10\"".as_bytes().to_vec()),
        ]),
        Literal(">wow #10!</g>".as_bytes().to_vec()),
        Literal("<g".as_bytes().to_vec()),
        Permutation(vec![
            Literal(" fill=\"red\"".as_bytes().to_vec()),
            Literal(" stroke=\"blue\"".as_bytes().to_vec()),
            Literal(" stroke-width=\"60\"".as_bytes().to_vec()),
        ]),
        Literal(">real content</g>".as_bytes().to_vec()),
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
        // */
        //*
        Literal("<svg".as_bytes().to_vec()),
        Permutation(vec![
        Literal(" xmlns=\"http://www.w3.org/2000/svg\"".as_bytes().to_vec()),
        Literal(" width=\"700\"".as_bytes().to_vec()),
        Literal(" height=\"500\"".as_bytes().to_vec()),
        Literal(" font-family=\"\'Open Sans\',verdana,arial,sans-serif\"".as_bytes().to_vec()),
        ]),
        Literal(">\n<path".as_bytes().to_vec()),
        Permutation(vec![
        Literal(" d=\"M0 0h700v500H0z\"".as_bytes().to_vec()),
        Literal(" fill=\"#fff\"".as_bytes().to_vec()),
        ]),
        Literal("/>\n<path".as_bytes().to_vec()),
        Permutation(vec![
        Literal(" d=\"M80 100h490v320H80z\"".as_bytes().to_vec()),
        Literal(" fill=\"#e5ecf6\"".as_bytes().to_vec()),
        ]),
        Literal("/>\n<path".as_bytes().to_vec()),
        Permutation(vec![
        Literal(" d=\"M80 354.2h490M80 288.4h490M80 222.6h490M80 156.7h490\"".as_bytes().to_vec()),
        Literal(" stroke=\"#fff\"".as_bytes().to_vec()),
        ]),
        Literal("/>\n<path".as_bytes().to_vec()),
        Permutation(vec![
        Literal(" d=\"M89.8 420V158.9h78.4V420Zm98 0V167.2h78.4V420Zm98 0V168.7h78.4V420Zm98 0V158.9h78.4V420Zm98 0V166.8h78.4V420Zm124-283.9h-12v-12h12Z\"".as_bytes().to_vec()),
        Literal(" stroke=\"#e5ecf6\"".as_bytes().to_vec()),
        Literal(" stroke-width=\".5\"".as_bytes().to_vec()),
        Literal(" fill=\"#636efa\"".as_bytes().to_vec()),
        ]),
        Literal("/>\n<path".as_bytes().to_vec()),
        Permutation(vec![
        Literal(" d=\"M89.8 158.9v-5.4h78.4v5.5Zm98 8.3v-3.7h78.4v3.7Zm98 1.5v-4h78.4v4Zm98-9.8v-5.4h78.4v5.5Zm98 7.9v-5.4h78.4v5.4Zm124-11.7h-12v-12h12Z\"".as_bytes().to_vec()),
        Literal(" stroke=\"#e5ecf6\"".as_bytes().to_vec()),
        Literal(" stroke-width=\".5\"".as_bytes().to_vec()),
        Literal(" fill=\"#ef553b\"".as_bytes().to_vec()),
        ]),
        Literal("/>\n<path".as_bytes().to_vec()),
        Permutation(vec![
        Literal(" d=\"M89.8 153.5V116h78.4v37.5Zm98 10v-33h78.4v33Zm98 1.2V129h78.4v35.7Zm98-11.2V116h78.4v37.5Zm98 7.9V124h78.4v37.5Zm124 12.7h-12v-12h12Z\"".as_bytes().to_vec()),
        Literal(" stroke=\"#e5ecf6\"".as_bytes().to_vec()),
        Literal(" stroke-width=\".5\"".as_bytes().to_vec()),
        Literal(" fill=\"#00cc96\"".as_bytes().to_vec()),
        ]),
        Literal("/>\n<text".as_bytes().to_vec()),
        Permutation(vec![
        Literal(" x=\"129\"".as_bytes().to_vec()),
        Literal(" text-anchor=\"middle\"".as_bytes().to_vec()),
        Literal(" font-size=\"12\"".as_bytes().to_vec()),
        Literal(" fill=\"#2a3f5f\"".as_bytes().to_vec()),
        Literal(" y=\"433\"".as_bytes().to_vec()),
        ]),
        Literal(">\nuncompressed\n</text>\n<text".as_bytes().to_vec()),
        Permutation(vec![
        Literal(" x=\"227\"".as_bytes().to_vec()),
        Literal(" text-anchor=\"middle\"".as_bytes().to_vec()),
        Literal(" font-size=\"12\"".as_bytes().to_vec()),
        Literal(" fill=\"#2a3f5f\"".as_bytes().to_vec()),
        Literal(" y=\"433\"".as_bytes().to_vec()),
        ]),
        Literal(">\noptipng\n</text>\n<text".as_bytes().to_vec()),
        Permutation(vec![
        Literal(" x=\"325\"".as_bytes().to_vec()),
        Literal(" text-anchor=\"middle\"".as_bytes().to_vec()),
        Literal(" font-size=\"12\"".as_bytes().to_vec()),
        Literal(" fill=\"#2a3f5f\"".as_bytes().to_vec()),
        Literal(" y=\"433\"".as_bytes().to_vec()),
        ]),
        Literal(">\noxipng\n</text>\n<text".as_bytes().to_vec()),
        Permutation(vec![
        Literal(" x=\"423\"".as_bytes().to_vec()),
        Literal(" text-anchor=\"middle\"".as_bytes().to_vec()),
        Literal(" font-size=\"12\"".as_bytes().to_vec()),
        Literal(" fill=\"#2a3f5f\"".as_bytes().to_vec()),
        Literal(" y=\"433\"".as_bytes().to_vec()),
        ]),
        Literal(">\npngcrush\n</text>\n<text".as_bytes().to_vec()),
        Permutation(vec![
        Literal(" x=\"521\"".as_bytes().to_vec()),
        Literal(" text-anchor=\"middle\"".as_bytes().to_vec()),
        Literal(" font-size=\"12\"".as_bytes().to_vec()),
        Literal(" fill=\"#2a3f5f\"".as_bytes().to_vec()),
        Literal(" y=\"433\"".as_bytes().to_vec()),
        ]),
        Literal(">\npngquant\n</text>\n<text".as_bytes().to_vec()),
        Permutation(vec![
        Literal(" x=\"325\"".as_bytes().to_vec()),
        Literal(" text-anchor=\"middle\"".as_bytes().to_vec()),
        Literal(" font-size=\"14\"".as_bytes().to_vec()),
        Literal(" font-weight=\"400\"".as_bytes().to_vec()),
        Literal(" fill=\"#2a3f5f\"".as_bytes().to_vec()),
        Literal(" y=\"461.8\"".as_bytes().to_vec()),
        ]),
        Literal(">\nprogram\n</text>\n<text".as_bytes().to_vec()),
        Permutation(vec![
        Literal(" x=\"26.5\"".as_bytes().to_vec()),
        Literal(" text-anchor=\"middle\"".as_bytes().to_vec()),
        Literal(" font-size=\"14\"".as_bytes().to_vec()),
        Literal(" font-weight=\"400\"".as_bytes().to_vec()),
        Literal(" fill=\"#2a3f5f\"".as_bytes().to_vec()),
        Literal(" y=\"260\"".as_bytes().to_vec()),
        Literal(" transform=\"rotate(-90 26.5 260)\"".as_bytes().to_vec()),
        ]),
        Literal(">\nsize (bytes)\n</text>\n<text".as_bytes().to_vec()),
        Permutation(vec![
        Literal(" x=\"79\"".as_bytes().to_vec()),
        Literal(" text-anchor=\"end\"".as_bytes().to_vec()),
        Literal(" font-size=\"12\"".as_bytes().to_vec()),
        Literal(" fill=\"#2a3f5f\"".as_bytes().to_vec()),
        Literal(" y=\"424.2\"".as_bytes().to_vec()),
        ]),
        Literal(">\n0\n</text>\n<text".as_bytes().to_vec()),
        Permutation(vec![
        Literal(" x=\"79\"".as_bytes().to_vec()),
        Literal(" text-anchor=\"end\"".as_bytes().to_vec()),
        Literal(" font-size=\"12\"".as_bytes().to_vec()),
        Literal(" fill=\"#2a3f5f\"".as_bytes().to_vec()),
        Literal(" y=\"358.4\"".as_bytes().to_vec()),
        ]),
        Literal(">\n0.5M\n</text>\n<text".as_bytes().to_vec()),
        Permutation(vec![
        Literal(" x=\"79\"".as_bytes().to_vec()),
        Literal(" text-anchor=\"end\"".as_bytes().to_vec()),
        Literal(" font-size=\"12\"".as_bytes().to_vec()),
        Literal(" fill=\"#2a3f5f\"".as_bytes().to_vec()),
        Literal(" y=\"226.8\"".as_bytes().to_vec()),
        ]),
        Literal(">\n1.5M\n</text>\n<text".as_bytes().to_vec()),
        Permutation(vec![
        Literal(" x=\"79\"".as_bytes().to_vec()),
        Literal(" text-anchor=\"end\"".as_bytes().to_vec()),
        Literal(" font-size=\"12\"".as_bytes().to_vec()),
        Literal(" fill=\"#2a3f5f\"".as_bytes().to_vec()),
        Literal(" y=\"292.6\"".as_bytes().to_vec()),
        ]),
        Literal(">\n1M\n</text>\n<text".as_bytes().to_vec()),
        Permutation(vec![
        Literal(" x=\"79\"".as_bytes().to_vec()),
        Literal(" text-anchor=\"end\"".as_bytes().to_vec()),
        Literal(" font-size=\"12\"".as_bytes().to_vec()),
        Literal(" fill=\"#2a3f5f\"".as_bytes().to_vec()),
        Literal(" y=\"160.9\"".as_bytes().to_vec()),
        ]),
        Literal(">\n2M\n</text>\n<text".as_bytes().to_vec()),
        Permutation(vec![
        Literal(" x=\"581.8\"".as_bytes().to_vec()),
        Literal(" font-size=\"12\"".as_bytes().to_vec()),
        Literal(" fill=\"#2a3f5f\"".as_bytes().to_vec()),
        Literal(" y=\"115.6\"".as_bytes().to_vec()),
        ]),
        Literal(">\nimage\n</text>\n<text".as_bytes().to_vec()),
        Permutation(vec![
        Literal(" x=\"619.8\"".as_bytes().to_vec()),
        Literal(" font-size=\"12\"".as_bytes().to_vec()),
        Literal(" fill=\"#2a3f5f\"".as_bytes().to_vec()),
        Literal(" y=\"134.8\"".as_bytes().to_vec()),
        ]),
        Literal(">\ndesktop\n</text>\n<text".as_bytes().to_vec()),
        Permutation(vec![
        Literal(" x=\"619.8\"".as_bytes().to_vec()),
        Literal(" font-size=\"12\"".as_bytes().to_vec()),
        Literal(" fill=\"#2a3f5f\"".as_bytes().to_vec()),
        Literal(" y=\"153.8\"".as_bytes().to_vec()),
        ]),
        Literal(">\nquaternion\n</text>\n<text".as_bytes().to_vec()),
        Permutation(vec![
        Literal(" x=\"619.8\"".as_bytes().to_vec()),
        Literal(" font-size=\"12\"".as_bytes().to_vec()),
        Literal(" fill=\"#2a3f5f\"".as_bytes().to_vec()),
        Literal(" y=\"172.8\"".as_bytes().to_vec()),
        ]),
        Literal(">\nwikipedia\n</text>\n<text".as_bytes().to_vec()),
        Permutation(vec![
        Literal(" x=\"35\"".as_bytes().to_vec()),
        Literal(" font-size=\"17\"".as_bytes().to_vec()),
        Literal(" font-weight=\"400\"".as_bytes().to_vec()),
        Literal(" fill=\"#2a3f5f\"".as_bytes().to_vec()),
        Literal(" y=\"50\"".as_bytes().to_vec()),
        ]),
        Literal(">\nPNG Compressor Performance\n</text>\n</svg>".as_bytes().to_vec()),
        // */
    ]);

    println!("Building...");
    let mut model = Model::build(seq);
    println!("Model:");
    println!("    {} symbols", model.symbols.len());

    println!("Solving...");
    let result = Python::with_gil(|py| model.solve(py).map(|result| result.unbind()))?;
    println!("Done.");

    Ok(result)
}

/// A Python module implemented in Rust.
#[pymodule]
fn compression_rs(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}
