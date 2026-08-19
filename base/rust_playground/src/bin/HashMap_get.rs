use std::collections::HashMap;
fn main() {
   let mut state_codes: HashMap<&str, &str> = HashMap::new();
   state_codes.insert("MT","Tomas");
   state_codes.insert("FJ","Jane");
   println!("size of map is {}",state_codes.len());
   println!("{:?}",state_codes);

   match state_codes.get(&"MT") {
      Some(value)=> {
         println!("Value for key MT is {}",value);
      }
      None => {
         println!("nothing found");
      }
   }
}