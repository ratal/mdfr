
#[cfg(test)]
mod tests {
    use std::io;
    use crate::mdfinfo;
    #[test]
    fn info_Test() -> io::Result<()>{
        let mut file_name ="/home/ratal/workspace/mdfr/test_files/Test.mf4";
        println!("reading {}", file_name);
        let info = mdfinfo::mdfinfo(file_name);
        println!("{:#?}", info);
        file_name ="/home/ratal/workspace/mdfr/test_files/Mdf3_hiddenBytes_NotAlignedBytes.dat";
        println!("reading {}", file_name);
        let info = mdfinfo::mdfinfo(file_name);
        println!("{:#?}", info);
        Ok(())
    }
}