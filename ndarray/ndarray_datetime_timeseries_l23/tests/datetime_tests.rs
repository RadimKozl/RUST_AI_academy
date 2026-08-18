#[cfg(test)]
mod tests {
    use chrono::NaiveDate;
    use ndarray::array;
    use ndarray_datetime_timeseries_l23::{
        add_days, date_difference_days, date_range, filter_business_days, moving_average,
    };

    #[test]
    fn test_date_range_generation() {
        let start = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let dates = date_range(start, 3);

        assert_eq!(dates.len(), 3);
        assert_eq!(dates[0], NaiveDate::from_ymd_opt(2026, 1, 1).unwrap());
        assert_eq!(dates[1], NaiveDate::from_ymd_opt(2026, 1, 2).unwrap());
        assert_eq!(dates[2], NaiveDate::from_ymd_opt(2026, 1, 3).unwrap());
    }

    #[test]
    fn test_date_arithmetic() {
        let start = NaiveDate::from_ymd_opt(2026, 5, 10).unwrap();
        let dates = date_range(start, 2);
        let shifted = add_days(&dates, 5);

        assert_eq!(shifted[0], NaiveDate::from_ymd_opt(2026, 5, 15).unwrap());
        assert_eq!(shifted[1], NaiveDate::from_ymd_opt(2026, 5, 16).unwrap());

        let diffs = date_difference_days(&dates, &shifted);
        assert_eq!(diffs, array![5, 5]);
    }

    #[test]
    fn test_business_days_filter() {
        // 2026-08-01 is Saturday, 2026-08-02 is Sunday, 2026-08-03 is Monday
        let start = NaiveDate::from_ymd_opt(2026, 8, 1).unwrap();
        let dates = date_range(start, 3);
        let values = array![10.0, 20.0, 30.0];

        let (b_dates, b_values) = filter_business_days(&dates, &values);

        assert_eq!(b_dates.len(), 1);
        assert_eq!(b_values.len(), 1);
        assert_eq!(b_dates[0], NaiveDate::from_ymd_opt(2026, 8, 3).unwrap());
        assert_eq!(b_values[0], 30.0);
    }

    #[test]
    fn test_moving_average() {
        let values = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = moving_average(&values, 3);

        // [ (1+2+3)/3, (2+3+4)/3, (3+4+5)/3 ] = [2.0, 3.0, 4.0]
        assert_eq!(ma, array![2.0, 3.0, 4.0]);
    }
}