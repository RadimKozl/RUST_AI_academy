use std::io::Write;
fn main() {
   let mut file = std::fs::File::create("./src/bin/files/data.txt").expect("create failed");
   file.write_all("Hello World".as_bytes()).expect("write failed");
   file.write_all("\nRUST Tutorials".as_bytes()).expect("write failed");
   println!("data written to file" );
}