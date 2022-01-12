use std::io::BufWriter;

use crate::mdfinfo::{
    mdfinfo4::{Hd4, MdfInfo4, FhBlock, MDBlock},
    IdBlock,
};
use binrw::BinWriterExt;
use std::fs::File;

/// writes file on hard drive
pub fn mdfwriter4<'a>(writer: &'a mut BufWriter<&File>, info: &'a mut MdfInfo4) {
    let mut new_info = MdfInfo4::default();
    new_info.id_block = IdBlock::default();
    let mut pointer:i64 = 64;
    new_info.hd_block = Hd4::default();
    pointer += 104;
    new_info.hd_block.hd_fh_first = pointer;
    new_info.fh = Vec::new();
    let mut fh = FhBlock::default();
    pointer += 56;
    fh.fh_md_comment = pointer;
    let mut fh_comments_header = MDBlock::default();
    fh_comments_header.fh();
    pointer += fh_comments_header.md_len as i64;

    writer
        .write_le(&new_info.id_block)
        .expect("Could not write IdBlock");
    writer
        .write_le(&new_info.hd_block)
        .expect("Could not write HDBlock");
    writer
        .write_le(&fh)
        .expect("Could not write FHBlock");
    writer
        .write_le(&fh_comments_header)
        .expect("Could not write FH MD comments header");
}
