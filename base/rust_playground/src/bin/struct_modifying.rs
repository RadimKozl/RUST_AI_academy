struct Employee {
   name:String,
   company:String,
   age:u32
}
fn main() {
   let mut emp1 = Employee {
      company:String::from("DigitalCore"),
      name:String::from("Thomas"),
      age:25
   };
   emp1.age = 37;
   emp1.company = String::from("Digitalhouse");
   println!("Name is :{} company is {} age is {}",emp1.name,emp1.company,emp1.age);
}