use std::collections::HashMap;
fn main() {
   let mut state_codes = HashMap::new();
   state_codes.insert("MT","Tomas");
   state_codes.insert("FJ","Jane");
   state_codes.insert("AJ","Alice");

   println!("length of the hashmap {}",state_codes.len());
   state_codes.remove(&"AJ");
   println!("length of the hashmap after remove() {}",state_codes.len());
}