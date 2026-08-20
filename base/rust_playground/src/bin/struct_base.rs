struct Employee {
   name:String,
   company:String,
   age:u32
}
fn main() {
   let emp1 = Employee {
      company:String::from("DigitalCore"),
      name:String::from("Thomas"),
      age:25
   };
   println!("Name is :{} company is {} age is {}",emp1.name,emp1.company,emp1.age);
}