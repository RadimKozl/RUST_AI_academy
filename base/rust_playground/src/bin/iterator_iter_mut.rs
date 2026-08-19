fn main() {
   let mut names: Vec<&str> = vec!["screw", "nut", "nail", "bolt"];
   for name in names.iter_mut() {
      match name {
         &mut "nail" => println!("The list includes fasteners for wood!"),
         _ => println!("Select for using {}", name),
      }
   }
   println!("{:?}",names);
   names.push("washer");
   println!("{:?}",names);
   //// reusing the collection after iteration
}