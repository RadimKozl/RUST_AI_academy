use linfa_trees::SplitQuality;
use lm_linfa_decision_tree::DecisionTreePipeline;
use std::path::Path;

#[test]
fn test_decision_tree_pipeline() {
    let (train, test) = DecisionTreePipeline::load_and_split_dataset(42);

    // Gini model test
    let (gini_eval, gini_model) = DecisionTreePipeline::train_and_eval(
        &train,
        &test,
        SplitQuality::Gini,
        "Gini",
        1.0,
        1.0,
    )
    .expect("Gini's training failed");

    assert!(gini_eval.accuracy > 70.0);

    // Test TikZ Export
    let test_path = Path::new("target/test_tree.tex");
    DecisionTreePipeline::export_tikz(&gini_model, test_path)
        .expect("Export to TikZ failed");
    assert!(test_path.exists());
}