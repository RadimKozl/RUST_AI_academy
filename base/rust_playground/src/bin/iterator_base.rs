fn main() {
   //declare an array
   let a: [i32; 3] = [10,20,30];

   let mut iter: std::slice::Iter<'_, i32> = a.iter(); 
   // fetch an iterator object for the array
   println!("{:?}",iter);

   //fetch individual values from the iterator object
   println!("{:?}",iter.next());
   println!("{:?}",iter.next());
   println!("{:?}",iter.next());
   println!("{:?}",iter.next());

   // For cycle
   let b: [i32; 3] = [10,20,30];
   let iter: std::slice::Iter<'_, i32> = b.iter();
   for data in iter{
      print!("{}\t",data);
   }
}