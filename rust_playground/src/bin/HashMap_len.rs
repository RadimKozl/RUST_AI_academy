use std::collections::HashMap;
fn main(){
   let mut state_codes: HashMap<&str, &str> = HashMap::new();
   state_codes.insert("MT","Tomas");
   state_codes.insert("FJ","Jane");
   println!("size of map is {}",state_codes.len());
}