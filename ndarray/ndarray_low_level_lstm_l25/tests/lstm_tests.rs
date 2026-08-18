use ndarray_low_level_lstm_l25::LstmCell;
use ndarray::{array, Array1};

#[test]
fn test_lstm_cell_dimensions() {
    let input_size = 2;
    let hidden_size = 3;
    let cell = LstmCell::new(input_size, hidden_size);

    let x_t = array![1.0, -0.5];
    let h_prev = Array1::zeros(hidden_size);
    let c_prev = Array1::zeros(hidden_size);

    let (h_next, c_next) = cell.step(&x_t, &h_prev, &c_prev);

    assert_eq!(h_next.len(), hidden_size);
    assert_eq!(c_next.len(), hidden_size);
}

#[test]
fn test_forward_sequence_length() {
    let input_size = 1;
    let hidden_size = 4;
    let cell = LstmCell::new(input_size, hidden_size);

    let sequence = array![[0.1], [0.2], [0.3], [0.4], [0.5]];
    let hidden_states = cell.forward_sequence(&sequence);

    assert_eq!(hidden_states.len(), sequence.nrows());
    for h in hidden_states {
        assert_eq!(h.len(), hidden_size);
    }
}