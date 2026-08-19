fn main() {
   let names: Vec<&str> = vec!["screw", "nut", "nail", "bolt"];
   for name in names.iter() {
      match name {
         &"nail" => println!("The list includes fasteners for wood!"),
         _ => println!("Select for using {}", name),
      }
   }
   println!("{:?}",names); 
   // reusing the collection after iteration
}