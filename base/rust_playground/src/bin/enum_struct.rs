// The `derive` attribute automatically creates the implementation
// required to make this `enum` printable with `fmt::Debug`.

#[derive(Debug)]
enum GenderCategory {
   Male,Female
}

// The `derive` attribute automatically creates the implementation
// required to make this `struct` printable with `fmt::Debug`.
#[derive(Debug)]
#[allow(dead_code)]
struct Person {
   name:String,
   gender:GenderCategory
}

fn main() {
   let p1: Person = Person {
      name:String::from("Thomas"),
      gender:GenderCategory::Male
   };
   let p2: Person = Person {
      name:String::from("Sara"),
      gender:GenderCategory::Female
   };
   println!("{:?}",p1);
   println!("{:?}",p2);
}