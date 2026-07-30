fn main() {
   let fullname = " RUST Tutorials \r\n";
   println!("Before trim ");
   println!("length is {}",fullname.len());
   println!();
   println!("After trim ");
   println!("length is {}",fullname.trim().len());
}