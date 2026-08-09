use std::collections::HashSet;
fn main() {
   let mut names: HashSet<&str> = HashSet::new();
   names.insert("screw");
   names.insert("nut");
   names.insert("nail");
   println!("size of the set is {}",names.len());
}