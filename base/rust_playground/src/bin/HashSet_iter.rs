use std::collections::HashSet;
fn main() {
   let mut names: HashSet<&str> = HashSet::new();
   names.insert("screw");
   names.insert("nut");
   names.insert("nail");
   names.insert("rivet");

   for name in names.iter() {
      println!("{}",name);
   }
}