pub mod movies {
   pub mod english {
      pub mod comedy {
         pub fn play(name:String) {
            println!("Playing comedy movie {}",name);
         }
      }
   }
}
use movies::english::comedy::play; 
// importing a public module

fn main() {
   // short path syntax
   play("Tom and Jerry".to_string());
   play("Forrest Gump".to_string());

   //full path syntax
   movies::english::comedy::play("Wallace a Gromit  ".to_string());
}