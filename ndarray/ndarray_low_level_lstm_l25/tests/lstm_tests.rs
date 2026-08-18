use ndarray_low_level_lstm_l25::LstmModel;
use ndarray::array;

#[test]
fn test_lstm_model_forward_dimensions() {
    let input_size = 2;
    let hidden_size = 3;
    let output_size = 1;
    let model = LstmModel::new(input_size, hidden_size, output_size);

    // Sequence of length 4 steps, each step has 2 flags
    let sequence = array![
        [1.0, -0.5],
        [0.2, 0.8],
        [-0.1, 0.5],
        [0.0, 1.2]
    ];

    let (predictions, hidden_states, cache) = model.forward(&sequence);

    // Check output dimensions
    assert_eq!(predictions.len(), sequence.nrows());
    assert_eq!(hidden_states.len(), sequence.nrows());
    assert_eq!(cache.len(), sequence.nrows());

    for pred in predictions {
        assert_eq!(pred.len(), output_size);
    }
    for h in hidden_states {
        assert_eq!(h.len(), hidden_size);
    }
}

#[test]
fn test_bptt_gradient_shapes() {
    let input_size = 1;
    let hidden_size = 4;
    let output_size = 1;
    let model = LstmModel::new(input_size, hidden_size, output_size);

    let inputs = array![[0.1], [0.2], [0.3]];
    let targets = array![[0.2], [0.3], [0.4]];

    let (predictions, _, cache) = model.forward(&inputs);
    let grads = model.backward(&cache, &targets, &predictions);

    let concat_size = hidden_size + input_size;

    // Verifying the correct dimensions of the gradient matrices
    assert_eq!(grads.dwf.dim(), (hidden_size, concat_size));
    assert_eq!(grads.dbf.dim(), hidden_size);
    assert_eq!(grads.dwi.dim(), (hidden_size, concat_size));
    assert_eq!(grads.dbi.dim(), hidden_size);
    assert_eq!(grads.dwc.dim(), (hidden_size, concat_size));
    assert_eq!(grads.dbc.dim(), hidden_size);
    assert_eq!(grads.dwo.dim(), (hidden_size, concat_size));
    assert_eq!(grads.dbo.dim(), hidden_size);
    assert_eq!(grads.dwy.dim(), (output_size, hidden_size));
    assert_eq!(grads.dby.dim(), output_size);
}

#[test]
fn test_loss_reduction() {
    let mut model = LstmModel::new(1, 8, 1);

    let inputs = array![[1.0], [2.0], [3.0]];
    let targets = array![[2.0], [3.0], [4.0]];

    let (preds_before, _, _) = model.forward(&inputs);
    let initial_loss: f64 = preds_before
        .iter()
        .zip(targets.rows())
        .map(|(p, t)| (p[0] - t[0]).powi(2))
        .sum::<f64>() / inputs.nrows() as f64;

    // Let's perform a few steps of Gradient Descent
    for _ in 0..50 {
        let (preds, _, cache) = model.forward(&inputs);
        let grads = model.backward(&cache, &targets, &preds);
        model.apply_gradients(&grads, 0.05);
    }

    let (preds_after, _, _) = model.forward(&inputs);
    let final_loss: f64 = preds_after
        .iter()
        .zip(targets.rows())
        .map(|(p, t)| (p[0] - t[0]).powi(2))
        .sum::<f64>() / inputs.nrows() as f64;

    // Loss after training must decrease
    assert!(
        final_loss < initial_loss,
        "Loss should decrease after optimization. Initial: {}, Final: {}",
        initial_loss,
        final_loss
    );
}