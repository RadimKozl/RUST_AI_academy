fn main() {
   let fullnametut = "RUST,base,tutorials";

   for token in fullnametut.split(","){
      println!("token is {}",token);
   }

   //store in a Vector
   println!("\n");
   let tokens:Vec<&str>= fullnametut.split(",").collect();
   println!("first word is {}",tokens[0]);
   println!("second word is {}",tokens[1]);
   println!("third word is {}",tokens[2]);
}