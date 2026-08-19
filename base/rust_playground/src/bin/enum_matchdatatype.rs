// The `derive` attribute automatically creates the implementation
// required to make this `enum` printable with `fmt::Debug`.
#[derive(Debug)]
enum GenderCategory {
   Name(String),UsrId(i32)
}
fn main() {
   let p1: GenderCategory = GenderCategory::Name(String::from("Peter"));
   let p2: GenderCategory = GenderCategory::UsrId(100);
   println!("{:?}",p1);
   println!("{:?}",p2);

   match p1 {
      GenderCategory::Name(val)=> {
         println!("{}",val);
      }
      GenderCategory::UsrId(val)=> {
         println!("{}",val);
      }
   }
}