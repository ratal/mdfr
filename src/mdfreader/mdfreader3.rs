use crate::mdfinfo::mdfinfo3::MdfInfo3;
use std::fs::File;
use std::io::BufReader;

pub fn mdfreader3<'a>(rdr: &'a mut BufReader<&File>, info: &'a mut MdfInfo3) {
}
