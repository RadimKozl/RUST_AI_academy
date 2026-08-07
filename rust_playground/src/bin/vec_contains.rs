fn main() {
   let v: Vec<i32> = vec![10,20,30];
   if v.contains(&10) {
      println!("found 10");
   }
   println!("{:?}",v);
}