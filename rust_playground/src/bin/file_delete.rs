use std::fs;
fn main() {
   fs::remove_file("./src/bin/files/data.txt").expect("could not remove file");
   println!("file is removed");
}