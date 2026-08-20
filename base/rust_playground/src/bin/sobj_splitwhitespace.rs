fn main(){
   let msg = "RUST Tutorials has base knowledges and skills.".to_string();
   let mut i = 1;
   
   for token in msg.split_whitespace(){
      println!("token {} {}",i,token);
      i+=1;
   }
}