use std::fs::OpenOptions;
use std::io::Write;

fn main() {
   let mut file = OpenOptions::new().append(true).open("./src/bin/files/data.txt").expect(
      "cannot open file");
   file.write_all("\nHello World".as_bytes()).expect("write failed");
   file.write_all("\nRUST Tutorials".as_bytes()).expect("write failed");
   println!("file append success");
}