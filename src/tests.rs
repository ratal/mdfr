
#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufReader;
    use std::io;
    use crate::mdfinfo;
    #[test]
    fn info_Test() -> io::Result<()>{
        let file_name ="/home/ratal/workspace/mdfr/test_files/Test.mf4";
        println!("reading {}", file_name);
        let f = File::open(file_name)?;
        let mut rdr = BufReader::new(f);
        let info = mdfinfo::mdfinfo(&mut rdr);
        println!("{:#?}", info);
        Ok(())
    }
}