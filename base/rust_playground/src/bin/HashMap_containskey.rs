use std::collections::HashMap;
fn main() {
   let mut state_codes: HashMap<&str, &str> = HashMap::new();
   state_codes.insert("MT","Tomas");
   state_codes.insert("FJ","Jane");
   state_codes.insert("AJ","Alice");

   if state_codes.contains_key(&"AJ") {
      println!("found key");
   }
}