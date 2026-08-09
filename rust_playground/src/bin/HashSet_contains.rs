use std::collections::HashSet;

fn main() {
   let mut names: HashSet<&str> = HashSet::new();
   names.insert("screw");
   names.insert("nut");
   names.insert("nail");

   if names.contains(&"nut") {
      println!("found name");
   }  
}