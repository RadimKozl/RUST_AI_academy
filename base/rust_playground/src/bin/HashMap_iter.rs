use std::collections::HashMap;
fn main() {
   let mut state_codes: HashMap<&str, &str> = HashMap::new();
   state_codes.insert("MT","Tomas");
   state_codes.insert("FJ","Jane");

   for (key, val) in state_codes.iter() {
      println!("key: {} val: {}", key, val);
   }
}