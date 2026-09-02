use anyhow::Result;
use lm_linfa_ensemble::{run_adaboost, run_random_forest};
use std::io::{self, Write};

fn main() -> Result<()> {
    println!("=======================================");
    println!("    LINFA ENSEMBLE CLASSIFIER UI       ");
    println!("=======================================");
    println!("1. Run AdaBoost with Decision Stumps (Depth=1)");
    println!("2. Run AdaBoost with Shallow Trees (Depth=2)");
    println!("3. Run Random Forest");
    println!("4. Exit");
    print!("Choose an option (1-4): ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    match input.trim() {
        "1" => {
            println!("\nRunning AdaBoost with Stumps...");
            let res = run_adaboost(50, 1.0, 1)?;
            print_result(&res);
        }
        "2" => {
            println!("\nRunning AdaBoost with Shallow Trees...");
            let res = run_adaboost(50, 1.0, 2)?;
            print_result(&res);
        }
        "3" => {
            println!("\nRunning Random Forest...");
            let res = run_random_forest(100, 0.7, 0.5)?;
            print_result(&res);
        }
        "4" => println!("Exiting..."),
        _ => println!("Invalid selection."),
    }

    Ok(())
}

fn print_result(res: &lm_linfa_ensemble::EnsembleResult) {
    println!("\n--- Results ---");
    println!("Algorithm: {}", res.algorithm_name);
    println!("Accuracy: {:.2}%", res.accuracy);
    println!("Model serialized and saved to 'models/' directory.\n");
}
