use ndarray_low_level_lstm_l25::LstmModel;
use ndarray::array;

fn main() {
    // Initialize an LSTM model with input_size = 1, hidden_size = 8, output_size = 1
    let mut model = LstmModel::new(1, 8, 1);

    // Raw sequence inputs: x_t = [1.0, 2.0, 3.0, 4.0]
    // Raw sequence targets: y_t = [2.0, 3.0, 4.0, 5.0]
    let raw_inputs = array![[1.0], [2.0], [3.0], [4.0]];
    let raw_targets = array![[2.0], [3.0], [4.0], [5.0]];

    // Compute sequence differences: Δx_t = Target - Input
    // For a linear progression with step +1.0, all target deltas equal 1.0.
    // Transforming the task to predict deltas bypasses the saturation issues 
    // caused by bounded activation functions (sigmoid/tanh) during extrapolation.
    let diff_targets = &raw_targets - &raw_inputs; // [1.0, 1.0, 1.0, 1.0]

    println!("--- Training LSTM to predict differences (Δx) ---");

    // Training loop executing BPTT over 3,000 epochs
    for epoch in 1..=30000 {
        // 1. Forward pass: compute predictions Δx_hat and activation caches
        let (preds, _, cache) = model.forward(&raw_inputs);
        
        // 2. Calculate Mean Squared Error (MSE) loss: (1/N) * Σ (pred - target)^2
        let loss: f64 = preds.iter().zip(diff_targets.rows())
            .map(|(p, t)| (p[0] - t[0]).powi(2))
            .sum::<f64>() / raw_inputs.nrows() as f64;

        // 3. Backward pass (BPTT): compute gradients for all weight matrices and biases
        let grads = model.backward(&cache, &diff_targets, &preds);

        // 4. Update parameters using Gradient Descent with gradient clipping (lr = 0.01)
        model.apply_gradients(&grads, 0.01);

        // Print training progress every 1,000 epochs
        if epoch % 1000 == 0 {
            println!("Epoch {:4}: Loss = {:.8}", epoch, loss);
        }
    }

    // Test sequence for linear extrapolation: [5.0, 6.0, 7.0]
    // Expected future targets: [6.0, 7.0, 8.0]
    let test_inputs = array![[5.0], [6.0], [7.0]];
    let (delta_preds, _, _) = model.forward(&test_inputs);
    
    println!("\nExtrapolation results (Input + Δ_pred):");
    for (i, delta) in delta_preds.iter().enumerate() {
        let x_t = test_inputs[[i, 0]];
        // Reconstruct the final absolute prediction by adding predicted delta to input x_t
        let y_hat = x_t + delta[0]; 
        println!("  Input: {:.1} -> Predicted: {:.4} (Expected: {:.1})", x_t, y_hat, x_t + 1.0);
    }
}