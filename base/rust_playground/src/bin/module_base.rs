pub mod movies {
   pub fn play(name:String) {
      println!("Playing movie {}",name);
   }
}
fn main(){
   movies::play("Tomas and Jerry".to_string());
}