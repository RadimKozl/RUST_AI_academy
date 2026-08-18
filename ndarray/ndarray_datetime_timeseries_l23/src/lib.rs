use chrono::{Datelike, Duration, NaiveDate};
use ndarray::Array1;

/// Generates a vector array of data in the given range (similar to np.arange for datetime64)
pub fn date_range(start: NaiveDate, days_count: usize) -> Array1<NaiveDate> {
    let mut dates = Vec::with_capacity(days_count);
    for i in 0..days_count {
        dates.push(start + Duration::days(i as i64));
    }
    Array1::from_vec(dates)
}

/// Vector arithmetic: Adds the specified number of days to all elements of an array
pub fn add_days(dates: &Array1<NaiveDate>, days: i64) -> Array1<NaiveDate> {
    dates.mapv(|d| d + Duration::days(days))
}

/// Vector calculation of the difference of two date fields (returns TimeDeltas in days)
pub fn date_difference_days(
    start_dates: &Array1<NaiveDate>,
    end_dates: &Array1<NaiveDate>,
) -> Array1<i64> {
    start_dates
        .iter()
        .zip(end_dates.iter())
        .map(|(s, e)| (*e - *s).num_days())
        .collect()
}

/// Masking/Indexing: Returns only those time series indices/values ​​that fall on weekdays (Mon-Fri)
pub fn filter_business_days(
    dates: &Array1<NaiveDate>,
    values: &Array1<f64>,
) -> (Array1<NaiveDate>, Array1<f64>) {
    let mut filtered_dates = Vec::new();
    let mut filtered_values = Vec::new();

    for (d, v) in dates.iter().zip(values.iter()) {
        let weekday = d.weekday();
        // Working days: Monday to Friday
        if weekday.number_from_monday() <= 5 {
            filtered_dates.push(*d);
            filtered_values.push(*v);
        }
    }

    (
        Array1::from_vec(filtered_dates),
        Array1::from_vec(filtered_values),
    )
}

/// Calculating a Moving Average over a time series
pub fn moving_average(values: &Array1<f64>, window_size: usize) -> Array1<f64> {
    if window_size == 0 || values.len() < window_size {
        return Array1::zeros(0);
    }

    let out_len = values.len() - window_size + 1;
    let mut result = Array1::zeros(out_len);

    for i in 0..out_len {
        let window = values.slice(ndarray::s![i..i + window_size]);
        result[i] = window.sum() / (window_size as f64);
    }

    result
}