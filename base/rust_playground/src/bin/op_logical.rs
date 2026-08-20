fn main() {
   let a = 20;
   let b = 30;
   let c = 0;
   let is_elder = false;

   println!("Inputs:\n a=20\n b=30\n c=0\n is_elder=false \n\n");
   
   if (a > 10) && (b > 10) {
      
      println!("IF a>10 AND b>10 :");
      println!("true");
   }

   if (c>10) || (b>10){

      println!("IF c>10 OR b>10 :");
      println!("true");
   }

   if !is_elder {
      println!("IF NOT is_elder :");
      println!("Not Elder");
   }
}