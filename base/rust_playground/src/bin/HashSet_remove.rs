use std::collections::HashSet;

fn main() {
   let mut names: HashSet<&str> = HashSet::new();
   names.insert("screw");
   names.insert("nut");
   names.insert("nail");
   println!("length of the Hashset: {}",names.len());
   names.remove(&"nut");
   println!("length of the Hashset after remove() : {}",names.len());
}