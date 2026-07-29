fn main(){
   let name1 = "Hello RustTutorial , 
   Hello!".to_string();         //String object
   let name2 = name1.replace("Hello","Leo");    //find and replace
   println!("{}",name2);
}