use chrono::NaiveDate;
use ndarray::array;
use ndarray_datetime_timeseries_l23::{
    add_days, date_difference_days, date_range, filter_business_days, moving_average,
};

fn main() {
    println!("=== 1. Generating a Datetime Array (np.arange for dates) ===");
    let start_date = NaiveDate::from_ymd_opt(2026, 8, 1).unwrap();
    let dates = date_range(start_date, 10);
    println!("Generated date range:\n{:?}", dates);

    println!("\n=== 2. Vector Arithmetic with TimeDeltas ===");
    let future_dates = add_days(&dates, 7);
    println!("Dates shifted by +7 days:\n{:?}", future_dates);

    let diffs = date_difference_days(&dates, &future_dates);
    println!("Calculated intervals (in days):\n{}", diffs);

    println!("\n=== 3. Time Series: Working Days Filtering ===");
    // Simulated price data for 10 days
    let prices = array![100.0, 102.5, 101.0, 104.0, 105.0, 105.5, 106.0, 108.0, 107.5, 110.0];
    let (b_dates, b_prices) = filter_business_days(&dates, &prices);

    println!("Original number of days: {}", dates.len());
    println!("Number of working days: {}", b_dates.len());
    println!("Working dates:\n {:?}", b_dates);
    println!("Prices on weekdays:\n {}", b_prices);

    println!("\n=== 4. Moving Average ===");
    let window_size = 3;
    let ma = moving_average(&prices, window_size);
    println!("3-day Moving Average of prices:\n{:.2}", ma);
}