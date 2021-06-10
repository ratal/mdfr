use dashmap::DashMap;

use crate::mdfinfo::mdfinfo3::MdfInfo3;
use std::{io::{BufReader, Cursor}, sync::Arc};
use std::fs::File;


pub fn mdfreader3<'a>(rdr: &'a mut BufReader<&File>, info: &'a mut MdfInfo3)  {

}