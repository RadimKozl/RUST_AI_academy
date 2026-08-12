fn main(){
   let names: Vec<&str> = vec!["screw", "nut", "nail", "bolt"];
   for name in names.into_iter() {
      match name {
         "nail" => println!("The list includes fasteners for wood!"),
         _ => println!("Select for using {}", name),
      }
   }
   // cannot reuse the collection after iteration
   //println!("{:?}",names); 
   //Error:Cannot access after ownership move
}