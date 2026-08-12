fn main() {
   let x: i32 = 5; 
   //value type variable
   let y: Box<i32> = Box::new(x); 
   //y points to a new value 5 in the heap

   println!("{}",5==x);
   println!("{}",5==*y); 
   //dereferencing y
}