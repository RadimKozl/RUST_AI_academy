fn main() {
   let var_i32: i32 = 5; 
   //stack
   let b: Box<i32> = Box::new(var_i32); 
   //heap
   println!("b = {}", b);
}