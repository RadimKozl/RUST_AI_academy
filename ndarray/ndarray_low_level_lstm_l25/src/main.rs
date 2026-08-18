use ndarray_low_level_lstm_l25::LstmCell;
use ndarray::array;

fn main() {
    let input_size = 1;
    let hidden_size = 4;

    let cell = LstmCell::new(input_size, hidden_size);

    // Input sequence of numbers: [1.0, 2.0, 3.0, 4.0]
    // Shape: 4 steps (rows), 1 flag per step (cols)
    let sequence = array![[1.0], [2.0], [3.0], [4.0]];

    println!("--- Input sequence (seq_len=4, input_size=1) ---");
    println!("{:?}\n", sequence);
    println!("\nSequence shape: {:?}", sequence.shape());

    // Forward pass through the entire sequence
    let hidden_states = cell.forward_sequence(&sequence);

    println!("--- Output Hidden States for each step ---");
    for (t, h) in hidden_states.iter().enumerate() {
        println!("Step {}: h_t = {:?}", t + 1, h.as_slice().unwrap());
    }
}
