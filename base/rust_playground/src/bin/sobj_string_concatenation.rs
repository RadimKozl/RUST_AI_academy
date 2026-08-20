fn main(){
   let n1 = "RUST".to_string();
   let n2 = " Tutorials".to_string();

   let n3 = n1 + &n2; // n2 reference is passed
   println!("{}",n3);
}