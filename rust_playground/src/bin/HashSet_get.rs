use std::collections::HashSet;
fn main() {
   let mut names: HashSet<&str> = HashSet::new();
   names.insert("screw");
   names.insert("nut");
   names.insert("nail");
   names.insert("rivet");

   match names.get(&"rivet"){
      Some(value)=>{
         println!("found {}",value);
      }
      None =>{
         println!("not found");
      }
   }
   println!("{:?}",names);
}