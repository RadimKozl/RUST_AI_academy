fn main() {
    let state_code = "CZ";

    let state = match state_code {
        "CZ" => {
            println!("Match found for CZ");
            "Czech Republic"
        }
        "SK" => "Slovakia",
        "PL" => "Poland",
        "HU" => "Hungary",
        _ => "Unknown state",
    };

    println!("The name of the state is {}", state);
}