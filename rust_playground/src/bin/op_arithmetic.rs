fn main() {
   let num1 = 10 ;
   let num2 = 2;
   let mut res:i32;

   res = num1 + num2;
   println!("Sum: {} ",res);

   res = num1 - num2;
   println!("Difference: {} ",res);

   res = num1*num2 ;
   println!("Product: {} ",res) ;

   res = num1/num2 ;
   println!("Quotient: {} ",res);

   res = num1%num2 ;
   println!("Remainder: {} ",res);
}