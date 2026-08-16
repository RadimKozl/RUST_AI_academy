use csv::{ReaderBuilder, WriterBuilder};
use ndarray::Array2;
use ndarray_csv::{Array2Reader, Array2Writer};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum IoError {
    #[error("I/O System error: {0}")]
    Io(#[from] std::io::Error),

    #[error("CSV SerDe error: {0}")]
    Csv(#[from] csv::Error),

    #[error("NdArray CSV Error: {0}")]
    NdArrayCsv(String),

    #[error("JSON SerDe error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Polars Engine error: {0}")]
    Polars(#[from] PolarsError),

    #[error("Shape mismatch or Parse error: {0}")]
    Parse(String),
}

// ==========================================
// 1. CSV I/O
// ==========================================
pub fn save_csv<P: AsRef<Path>>(array: &Array2<f64>, path: P) -> Result<(), IoError> {
    let file = File::create(path)?;
    let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
    writer
        .serialize_array2(array)
        .map_err(|e| IoError::NdArrayCsv(e.to_string()))?;
    Ok(())
}

pub fn load_csv<P: AsRef<Path>>(
    path: P,
    rows: usize,
    cols: usize,
) -> Result<Array2<f64>, IoError> {
    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let array: Array2<f64> = reader
        .deserialize_array2((rows, cols))
        .map_err(|e| IoError::NdArrayCsv(e.to_string()))?;
    Ok(array)
}

// ==========================================
// 2. JSON I/O
// ==========================================
#[derive(Serialize, Deserialize)]
struct MatrixData {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

pub fn save_json<P: AsRef<Path>>(array: &Array2<f64>, path: P) -> Result<(), IoError> {
    let payload = MatrixData {
        rows: array.nrows(),
        cols: array.ncols(),
        data: array.iter().copied().collect(),
    };
    let file = File::create(path)?;
    serde_json::to_writer_pretty(BufWriter::new(file), &payload)?;
    Ok(())
}

pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Array2<f64>, IoError> {
    let file = File::open(path)?;
    let payload: MatrixData = serde_json::from_reader(BufReader::new(file))?;
    Array2::from_shape_vec((payload.rows, payload.cols), payload.data)
        .map_err(|e| IoError::Parse(e.to_string()))
}

// ==========================================
// 3. TXT / DAT Plaintext I/O
// ==========================================
pub fn save_txt_dat<P: AsRef<Path>>(array: &Array2<f64>, path: P) -> Result<(), IoError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    for row in array.rows() {
        let line = row
            .iter()
            .map(|val| val.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        writeln!(writer, "{}", line)?;
    }
    writer.flush()?;
    Ok(())
}

pub fn load_txt_dat<P: AsRef<Path>>(path: P) -> Result<Array2<f64>, IoError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut rows_count = 0;
    let mut flat_data = Vec::new();

    for line in reader.lines() {
        let line_str = line?;
        if line_str.trim().is_empty() {
            continue;
        }
        let parsed_row: Vec<f64> = line_str
            .split_whitespace()
            .map(|s| s.parse::<f64>())
            .collect::<Result<_, _>>()
            .map_err(|e| IoError::Parse(e.to_string()))?;

        flat_data.extend(parsed_row);
        rows_count += 1;
    }

    if rows_count == 0 {
        return Err(IoError::Parse("Empty file".into()));
    }

    let cols_count = flat_data.len() / rows_count;
    Array2::from_shape_vec((rows_count, cols_count), flat_data)
        .map_err(|e| IoError::Parse(e.to_string()))
}

// ==========================================
// 4. Apache Parquet I/O (Columnar)
// ==========================================
pub fn save_parquet<P: AsRef<Path>>(array: &Array2<f64>, path: P) -> Result<(), IoError> {
    let mut columns = Vec::with_capacity(array.ncols());
    let height = array.nrows();

    for col_idx in 0..array.ncols() {
        let col_data: Vec<f64> = array.column(col_idx).iter().copied().collect();
        let series = Series::new(format!("col_{}", col_idx).into(), col_data);
        columns.push(Column::from(series));
    }

    let mut df = DataFrame::new(height, columns)?;
    let file = File::create(path)?;
    ParquetWriter::new(BufWriter::new(file)).finish(&mut df)?;
    Ok(())
}

pub fn load_parquet<P: AsRef<Path>>(path: P) -> Result<Array2<f64>, IoError> {
    let file = File::open(path)?;
    let df = ParquetReader::new(BufReader::new(file)).finish()?;
    let ndarray_matrix = df.to_ndarray::<Float64Type>(IndexOrder::C)?;
    Ok(ndarray_matrix)
}

// ==========================================
// 5. Python/NumPy C-ABI Zero-Copy Interop
// ==========================================
#[cfg(feature = "python-extension")]
pub mod numpy_interop {
    use super::*;
    use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
    use pyo3::prelude::*;

    #[pyfunction]
    pub fn scale_matrix_py<'py>(
        py: Python<'py>,
        input: PyReadonlyArray2<'py, f64>,
        factor: f64,
    ) -> Bound<'py, PyArray2<f64>> {
        let array_view = input.as_array();
        let scaled: Array2<f64> = array_view.mapv(|x| x * factor);
        scaled.into_pyarray(py)
    }

    #[pymodule]
    fn ndarray_io_formats_l18(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(scale_matrix_py, m)?)?;
        Ok(())
    }
}