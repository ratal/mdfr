//! Data Stream Mode Composition Decoder
//!
//! This module decodes VLSD/VLSC data blobs that are linked to a DSBLOCK with `ds_mode = 0`
//! (data stream mode). It parses the composition hierarchy (CN, CL, CV, CU blocks) to extract
//! individual channel values from the raw data blob.
//!
//! Per MDF4.3 Section 6.25, data stream mode uses relative positioning where each channel's
//! offset depends on the previous channel's end position plus alignment rules.

use anyhow::{Result, bail};
use std::collections::HashMap;

use crate::mdfinfo::mdfinfo4::{
    CN_F_ALIGNMENT_RESET, Cl4Block, Cn4, CnType, Compo, Composition, Cu4Block, Cv4Block, Ds4Block,
};

/// Groups channel metadata and reverse lookup maps for O(1) channel resolution.
struct ChannelIndex<'a> {
    channels: &'a CnType,
    pos_to_rec: HashMap<i64, i32>,
    name_to_rec: HashMap<&'a str, i32>,
}

impl<'a> ChannelIndex<'a> {
    fn new(cg_channels: &'a CnType) -> Self {
        let pos_to_rec = cg_channels
            .iter()
            .map(|(rec_pos, cn)| (cn.block_position, *rec_pos))
            .collect();
        let name_to_rec = cg_channels
            .iter()
            .map(|(rec_pos, cn)| (cn.unique_name.as_str(), *rec_pos))
            .collect();
        Self {
            channels: cg_channels,
            pos_to_rec,
            name_to_rec,
        }
    }
}

/// Tracks position and alignment state within a data stream
#[derive(Debug, Clone)]
pub struct StreamState {
    /// Current bit position in the stream (P)
    pub bit_position: usize,
    /// Current alignment offset in bytes (A) - used for alignment calculations
    pub alignment_offset: usize,
}

impl StreamState {
    /// Creates a new StreamState starting at position 0
    pub fn new() -> Self {
        Self {
            bit_position: 0,
            alignment_offset: 0,
        }
    }

    /// Creates a new StreamState with a specified alignment start
    pub fn with_alignment_start(alignment_start: usize) -> Self {
        Self {
            bit_position: 0,
            alignment_offset: alignment_start,
        }
    }

    /// Returns the current byte position (ceiling of bit_position / 8)
    pub fn byte_position(&self) -> usize {
        self.bit_position.div_ceil(8)
    }

    /// Resets the alignment offset to the current byte position
    /// Called when CN_F_ALIGNMENT_RESET flag is set
    pub fn reset_alignment(&mut self) {
        self.alignment_offset = self.byte_position();
    }
}

impl Default for StreamState {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculates the aligned byte offset for a channel in data stream mode.
///
/// Per MDF4.3 Section 6.25.3, the alignment formula is:
/// - Find smallest `offset >= P` where `(offset - A) mod AV == 0`
/// - `AV = 2^cn_alignment` (alignment value in bytes)
/// - `A` = alignment_offset (from ds_cn_alignment_start or alignment reset)
/// - `P` = current_bit_pos converted to bytes (rounded up)
///
/// Special case: if `cn_alignment == 255`, it means bit-packed (no byte alignment),
/// return the current bit position as-is.
///
/// # Arguments
/// * `current_bit_pos` - Current position in the stream (in bits)
/// * `alignment_offset` - Alignment reference position (in bytes)
/// * `cn_alignment` - Channel alignment value (0-7 = 2^n bytes, 255 = bit-packed)
///
/// # Returns
/// The byte offset where the channel data starts (or bit offset if bit-packed)
pub fn calculate_aligned_offset(
    current_bit_pos: usize,
    alignment_offset: usize,
    cn_alignment: u8,
) -> usize {
    // Bit-packed mode (cn_alignment == 255): no byte alignment
    if cn_alignment == 255 {
        return current_bit_pos;
    }

    // Calculate current byte position (round up)
    let current_byte_pos = current_bit_pos.div_ceil(8);

    // cn_alignment of 0 means 1-byte aligned (2^0 = 1)
    if cn_alignment == 0 {
        return current_byte_pos;
    }

    // Calculate alignment value: AV = 2^cn_alignment
    let alignment_value = 1usize << cn_alignment;

    // Find smallest offset >= current_byte_pos where (offset - A) mod AV == 0
    // This means offset = A + k*AV for some k, and offset >= current_byte_pos

    // Calculate relative position from alignment offset
    if current_byte_pos <= alignment_offset {
        // Already at or before alignment offset, align to alignment_offset
        return alignment_offset;
    }

    let relative_pos = current_byte_pos - alignment_offset;
    let remainder = relative_pos % alignment_value;

    if remainder == 0 {
        // Already aligned
        current_byte_pos
    } else {
        // Need to advance to next aligned position
        current_byte_pos + (alignment_value - remainder)
    }
}

/// Calculates the final byte offset after applying byte/bit offsets to an aligned position.
///
/// When alignment is 255 (bit-packed), the aligned_offset is in bits and we combine
/// byte_offset * 8 + bit_offset in bits, then divide by 8 to get the final byte offset.
/// Otherwise, we simply add byte_offset to the aligned byte offset.
fn calculate_final_byte_offset(
    aligned_offset: usize,
    alignment: u8,
    byte_offset: u32,
    bit_offset: u8,
) -> usize {
    if alignment == 255 {
        (aligned_offset + byte_offset as usize * 8 + bit_offset as usize) / 8
    } else {
        aligned_offset + byte_offset as usize
    }
}

/// Extracts u64 values from a channel's already-read data.
///
/// This is used to get size values (for CLBLOCK) or discriminator values (for CVBLOCK)
/// from channels that have already been read during the fixed-length record processing.
///
/// # Arguments
/// * `cn` - The channel containing the data
///
/// # Returns
/// Vector of u64 values extracted from the channel data
pub fn extract_channel_values_as_u64(cn: &Cn4) -> Result<Vec<u64>> {
    cn.data.to_u64_vec().ok_or_else(|| {
        anyhow::anyhow!(
            "Cannot extract u64 values from channel data type: {:?}",
            std::mem::discriminant(&cn.data)
        )
    })
}

/// Reads a single primitive value from the data buffer at the given offset.
///
/// # Arguments
/// * `data` - The data buffer
/// * `offset` - Byte offset in the buffer
/// * `bit_count` - Number of bits to read
/// * `data_type` - MDF data type (0-5 for numeric types)
/// * `endian` - true for big endian, false for little endian
///
/// # Returns
/// The value as bytes (for storage in ChannelData)
fn read_primitive_value(
    data: &[u8],
    offset: usize,
    bit_count: u32,
    _data_type: u8,
    _endian: bool,
) -> Result<Vec<u8>> {
    let byte_count = bit_count.div_ceil(8) as usize;

    if offset + byte_count > data.len() {
        bail!(
            "Buffer underrun: need {} bytes at offset {}, but only {} bytes available",
            byte_count,
            offset,
            data.len()
        );
    }

    let bytes = &data[offset..offset + byte_count];
    Ok(bytes.to_vec())
}

/// Decodes a single channel value from the data stream.
///
/// This handles the basic data types (integers, floats, strings, byte arrays)
/// that can appear as leaf channels in a composition.
///
/// # Arguments
/// * `data` - The data buffer for this record
/// * `cn` - The channel block metadata
/// * `stream_state` - Current stream position tracking
///
/// # Returns
/// Tuple of (decoded bytes, bytes consumed)
pub fn decode_single_channel_value(
    data: &[u8],
    cn: &Cn4,
    stream_state: &mut StreamState,
) -> Result<(Vec<u8>, usize)> {
    // Handle alignment reset flag
    if (cn.block.cn_flags & CN_F_ALIGNMENT_RESET) != 0 {
        stream_state.reset_alignment();
    }

    // Calculate aligned offset
    let aligned_offset = calculate_aligned_offset(
        stream_state.bit_position,
        stream_state.alignment_offset,
        cn.block.cn_alignment,
    );

    // Apply cn_byte_offset and cn_bit_offset
    let final_byte_offset = calculate_final_byte_offset(
        aligned_offset,
        cn.block.cn_alignment,
        cn.block.cn_byte_offset,
        cn.block.cn_bit_offset,
    );

    let bit_count = cn.block.cn_bit_count;
    let byte_count = bit_count.div_ceil(8) as usize;

    // Handle variable-length types with cn_bit_count == 0
    if bit_count == 0 {
        // For VLSD (cn_type == 1) with flat storage: 4-byte length prefix + data
        if cn.block.cn_type == 1 {
            if final_byte_offset + 4 > data.len() {
                bail!("Buffer underrun reading VLSD length prefix");
            }
            let length =
                u32::from_le_bytes(data[final_byte_offset..final_byte_offset + 4].try_into()?)
                    as usize;
            let total_bytes = 4 + length;
            if final_byte_offset + total_bytes > data.len() {
                bail!("Buffer underrun reading VLSD data");
            }
            let value_bytes = data[final_byte_offset + 4..final_byte_offset + total_bytes].to_vec();
            stream_state.bit_position = (final_byte_offset + total_bytes) * 8;
            return Ok((value_bytes, total_bytes));
        }

        // For VLSC (cn_type == 7) with flat storage: size from size channel
        // This case should be handled by the caller who knows the size
        bail!("VLSC with cn_bit_count == 0 requires size channel value");
    }

    // Read fixed-length value
    let value_bytes = read_primitive_value(
        data,
        final_byte_offset,
        bit_count,
        cn.block.cn_data_type,
        cn.endian,
    )?;

    // Update stream position
    stream_state.bit_position = (final_byte_offset + byte_count) * 8;

    Ok((value_bytes, byte_count))
}

/// Decodes a channel union where all members share the same bytes.
///
/// # Arguments
/// * `data` - The data buffer for this record
/// * `cu_block` - The CUBLOCK metadata
/// * `cg_channels` - All channels in the channel group (to look up members)
/// * `stream_state` - Current stream position tracking
///
/// # Returns
/// HashMap of member channel name to decoded value bytes
pub fn decode_channel_union(
    data: &[u8],
    _cu_block: &Cu4Block,
    member_channels: &[&Cn4],
    stream_state: &mut StreamState,
) -> Result<HashMap<String, Vec<u8>>> {
    let mut result = HashMap::new();

    // All members start at the same position
    let start_position = stream_state.bit_position;
    let mut max_end_position = start_position;

    for cn in member_channels {
        // Reset to union start for each member
        stream_state.bit_position = start_position;

        let (value, _) = decode_single_channel_value(data, cn, stream_state)?;
        result.insert(cn.unique_name.clone(), value);

        // Track the furthest position reached
        if stream_state.bit_position > max_end_position {
            max_end_position = stream_state.bit_position;
        }
    }

    // Set stream position to end of largest member
    stream_state.bit_position = max_end_position;

    Ok(result)
}

/// Decodes a channel variant based on discriminator value.
///
/// # Arguments
/// * `data` - The data buffer for this record
/// * `cv_block` - The CVBLOCK metadata
/// * `option_channels` - The parsed option channels
/// * `discriminator_value` - The discriminator value for this record
/// * `stream_state` - Current stream position tracking
///
/// # Returns
/// Tuple of (option index, option channel name, decoded value bytes)
pub fn decode_channel_variant(
    data: &[u8],
    cv_block: &Cv4Block,
    option_channels: &[&Cn4],
    discriminator_value: u64,
    stream_state: &mut StreamState,
) -> Result<(usize, String, Vec<u8>)> {
    // Find matching option by discriminator value
    let option_index = cv_block
        .cv_option_val
        .iter()
        .position(|&val| val == discriminator_value);

    match option_index {
        Some(idx) if idx < option_channels.len() => {
            let cn = option_channels[idx];
            let (value, _) = decode_single_channel_value(data, cn, stream_state)?;
            Ok((idx, cn.unique_name.clone(), value))
        }
        _ => {
            // No matching option - this is a valid case per spec (data might be invalid)
            bail!(
                "No matching variant option for discriminator value {}. Available: {:?}",
                discriminator_value,
                cv_block.cv_option_val
            );
        }
    }
}

/// Decodes a channel list (variable-length array).
///
/// # Arguments
/// * `data` - The data buffer for this record
/// * `cl_block` - The CLBLOCK metadata
/// * `element_composition` - The composition defining each element (if any)
/// * `element_cn` - The template channel for list elements
/// * `size_value` - The size (element count or byte count) for this record
/// * `stream_state` - Current stream position tracking
///
/// # Returns
/// Vector of decoded element values
pub fn decode_channel_list(
    data: &[u8],
    cl_block: &Cl4Block,
    element_cn: &Cn4,
    size_value: u64,
    stream_state: &mut StreamState,
) -> Result<Vec<Vec<u8>>> {
    let mut elements = Vec::new();

    // Determine number of elements
    let element_count = if (cl_block.cl_flags & 0x01) != 0 {
        // Size is number of elements
        size_value as usize
    } else {
        // Size is number of bytes - need to calculate element count
        // For simple types, divide by element size
        let element_size = element_cn.block.cn_bit_count.div_ceil(8) as usize;
        if element_size > 0 {
            (size_value as usize) / element_size
        } else {
            0
        }
    };

    for i in 0..element_count {
        if i == 0 {
            // First element uses template channel's alignment
            let (value, _) = decode_single_channel_value(data, element_cn, stream_state)?;
            elements.push(value);
        } else {
            // Subsequent elements use CLBLOCK's alignment settings
            // Handle alignment reset flag from CLBLOCK
            if (cl_block.cl_flags & 0x02) != 0 {
                // Bit 1: Alignment reset flag
                stream_state.reset_alignment();
            }

            // Calculate aligned offset using CLBLOCK settings
            let aligned_offset = calculate_aligned_offset(
                stream_state.bit_position,
                stream_state.alignment_offset,
                cl_block.cl_alignment,
            );

            // Apply cl_byte_offset and cl_bit_offset
            let final_byte_offset = calculate_final_byte_offset(
                aligned_offset,
                cl_block.cl_alignment,
                cl_block.cl_byte_offset,
                cl_block.cl_bit_offset,
            );

            let bit_count = element_cn.block.cn_bit_count;
            let byte_count = bit_count.div_ceil(8) as usize;

            if final_byte_offset + byte_count > data.len() {
                bail!(
                    "Buffer underrun reading list element {}: need {} bytes at offset {}, have {}",
                    i,
                    byte_count,
                    final_byte_offset,
                    data.len()
                );
            }

            let value = data[final_byte_offset..final_byte_offset + byte_count].to_vec();
            elements.push(value);

            stream_state.bit_position = (final_byte_offset + byte_count) * 8;
        }
    }

    Ok(elements)
}

/// Main entry point: Decodes a data stream blob using its composition definition.
///
/// This function iterates through all records in the blob and decodes each one
/// according to the composition hierarchy.
///
/// # Arguments
/// * `data` - The raw data blob bytes (may contain multiple records)
/// * `ds_block` - The DSBLOCK metadata
/// * `composition` - The composition tree under the DSBLOCK
/// * `cg_channels` - Reference to all channels in the channel group
/// * `record_offsets` - Byte offsets to each record in the blob (from VLSD/VLSC)
/// * `record_sizes` - Sizes of each record (for VLSC, or calculated for VLSD)
///
/// # Returns
/// HashMap mapping child channel record positions to their decoded ChannelData
pub fn decode_datastream_blob(
    data: &[u8],
    ds_block: &Ds4Block,
    composition: &Composition,
    cg_channels: &CnType,
    record_offsets: &[u64],
    record_sizes: &[u64],
) -> Result<HashMap<i32, Vec<Vec<u8>>>> {
    let mut result: HashMap<i32, Vec<Vec<u8>>> = HashMap::new();
    let record_count = record_offsets.len();

    let index = ChannelIndex::new(cg_channels);

    // Get initial alignment start from ds_cn_alignment_start (if specified)
    let alignment_start = if ds_block.ds_cn_alignment_start() != 0 {
        if let Some(&rec) = index.pos_to_rec.get(&ds_block.ds_cn_alignment_start()) {
            if let Some(align_cn) = index.channels.get(&rec) {
                extract_channel_values_as_u64(align_cn)
                    .ok()
                    .and_then(|v| v.first().copied())
                    .unwrap_or(0) as usize
            } else {
                0
            }
        } else {
            0
        }
    } else {
        0
    };

    // Process each record
    for record_idx in 0..record_count {
        let record_offset = record_offsets[record_idx] as usize;
        let record_size = record_sizes[record_idx] as usize;

        if record_offset + record_size > data.len() {
            bail!(
                "Record {} extends beyond data buffer: offset={}, size={}, data_len={}",
                record_idx,
                record_offset,
                record_size,
                data.len()
            );
        }

        let record_data = &data[record_offset..record_offset + record_size];
        let mut stream_state = StreamState::with_alignment_start(alignment_start);

        // Decode the composition for this record
        decode_composition_record(
            record_data,
            composition,
            &index,
            &mut stream_state,
            record_idx,
            &mut result,
        )?;
    }

    Ok(result)
}

/// Recursively decodes a composition for a single record.
fn decode_composition_record(
    data: &[u8],
    composition: &Composition,
    index: &ChannelIndex,
    stream_state: &mut StreamState,
    record_idx: usize,
    result: &mut HashMap<i32, Vec<Vec<u8>>>,
) -> Result<()> {
    match &composition.block {
        Compo::CN(cn) => {
            // Decode this channel
            let (value, _) = decode_single_channel_value(data, cn, stream_state)?;

            // O(1) lookup via reverse map
            if let Some(&rec_pos) = index.pos_to_rec.get(&cn.block_position) {
                result.entry(rec_pos).or_default().push(value);
            }

            // Recursively decode nested composition
            if let Some(nested) = &composition.compo {
                decode_composition_record(data, nested, index, stream_state, record_idx, result)?;
            }
        }

        Compo::CL(cl_block) => {
            // Get size channel values via O(1) lookup
            let size_values = if let Some(&rec) = index.pos_to_rec.get(&cl_block.cl_cn_size) {
                if let Some(size_cn) = index.channels.get(&rec) {
                    extract_channel_values_as_u64(size_cn)?
                } else {
                    bail!(
                        "Size channel not found for CLBLOCK at position {}",
                        cl_block.cl_cn_size
                    );
                }
            } else {
                bail!(
                    "Size channel not found for CLBLOCK at position {}",
                    cl_block.cl_cn_size
                );
            };

            let size_value = size_values.get(record_idx).copied().unwrap_or(0);

            // Get element template from nested composition
            if let Some(nested) = &composition.compo
                && let Compo::CN(element_cn) = &nested.block
            {
                let elements =
                    decode_channel_list(data, cl_block, element_cn, size_value, stream_state)?;

                if let Some(&rec_pos) = index.pos_to_rec.get(&element_cn.block_position) {
                    for elem in elements {
                        result.entry(rec_pos).or_default().push(elem);
                    }
                }
            }
        }

        Compo::CV(cv_block) => {
            // Get discriminator channel values via O(1) lookup
            let discriminator_values =
                if let Some(&rec) = index.pos_to_rec.get(&cv_block.cv_cn_discriminator) {
                    if let Some(disc_cn) = index.channels.get(&rec) {
                        extract_channel_values_as_u64(disc_cn)?
                    } else {
                        bail!("Discriminator channel not found for CVBLOCK");
                    }
                } else {
                    bail!("Discriminator channel not found for CVBLOCK");
                };

            let disc_value = discriminator_values.get(record_idx).copied().unwrap_or(0);

            // Collect option channels via O(1) lookups
            let option_channels: Vec<&Cn4> = cv_block
                .cv_cn_option
                .iter()
                .filter_map(|pos| {
                    index
                        .pos_to_rec
                        .get(pos)
                        .and_then(|rec| index.channels.get(rec))
                })
                .collect();

            if let Ok((idx, _name, value)) =
                decode_channel_variant(data, cv_block, &option_channels, disc_value, stream_state)
                && idx < option_channels.len()
                && let Some(&rec_pos) =
                    index.pos_to_rec.get(&option_channels[idx].block_position)
            {
                result.entry(rec_pos).or_default().push(value);
            }
        }

        Compo::CU(cu_block) => {
            // Collect member channels via O(1) lookups
            let member_channels: Vec<&Cn4> = cu_block
                .cu_cn_member
                .iter()
                .filter_map(|pos| {
                    index
                        .pos_to_rec
                        .get(pos)
                        .and_then(|rec| index.channels.get(rec))
                })
                .collect();

            let decoded = decode_channel_union(data, cu_block, &member_channels, stream_state)?;

            // Store decoded values for all members via O(1) name lookup
            for (name, value) in decoded {
                if let Some(&rec_pos) = index.name_to_rec.get(name.as_str()) {
                    result.entry(rec_pos).or_default().push(value);
                }
            }
        }

        Compo::DS(_ds_block) => {
            // Nested DS block - recursively decode its composition
            if let Some(nested) = &composition.compo {
                decode_composition_record(data, nested, index, stream_state, record_idx, result)?;
            }
        }

        Compo::CA(_ca_block) => {
            // Channel array - not typically used in data stream mode
            if let Some(nested) = &composition.compo {
                decode_composition_record(data, nested, index, stream_state, record_idx, result)?;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_aligned_offset_no_alignment() {
        // cn_alignment = 0 means 1-byte aligned (2^0 = 1)
        assert_eq!(calculate_aligned_offset(0, 0, 0), 0);
        assert_eq!(calculate_aligned_offset(8, 0, 0), 1); // 8 bits = 1 byte
        assert_eq!(calculate_aligned_offset(9, 0, 0), 2); // 9 bits -> 2 bytes (rounded up)
        assert_eq!(calculate_aligned_offset(16, 0, 0), 2);
    }

    #[test]
    fn test_calculate_aligned_offset_word_alignment() {
        // cn_alignment = 1 means 2-byte (WORD) aligned (2^1 = 2)
        assert_eq!(calculate_aligned_offset(0, 0, 1), 0);
        assert_eq!(calculate_aligned_offset(8, 0, 1), 2); // 1 byte -> align to 2
        assert_eq!(calculate_aligned_offset(16, 0, 1), 2);
        assert_eq!(calculate_aligned_offset(24, 0, 1), 4); // 3 bytes -> align to 4
    }

    #[test]
    fn test_calculate_aligned_offset_dword_alignment() {
        // cn_alignment = 2 means 4-byte (DWORD) aligned (2^2 = 4)
        assert_eq!(calculate_aligned_offset(0, 0, 2), 0);
        assert_eq!(calculate_aligned_offset(8, 0, 2), 4); // 1 byte -> align to 4
        assert_eq!(calculate_aligned_offset(16, 0, 2), 4); // 2 bytes -> align to 4
        assert_eq!(calculate_aligned_offset(24, 0, 2), 4); // 3 bytes -> align to 4
        assert_eq!(calculate_aligned_offset(32, 0, 2), 4); // 4 bytes -> already aligned
        assert_eq!(calculate_aligned_offset(40, 0, 2), 8); // 5 bytes -> align to 8
    }

    #[test]
    fn test_calculate_aligned_offset_with_alignment_offset() {
        // With alignment offset = 2, cn_alignment = 2 (4-byte)
        // Valid positions: 2, 6, 10, 14, ...
        assert_eq!(calculate_aligned_offset(0, 2, 2), 2);
        assert_eq!(calculate_aligned_offset(8, 2, 2), 2); // 1 byte at pos 2
        assert_eq!(calculate_aligned_offset(24, 2, 2), 6); // 3 bytes -> align to 6
        assert_eq!(calculate_aligned_offset(48, 2, 2), 6); // 6 bytes -> already at 6
        assert_eq!(calculate_aligned_offset(56, 2, 2), 10); // 7 bytes -> align to 10
    }

    #[test]
    fn test_calculate_aligned_offset_bit_packed() {
        // cn_alignment = 255 means bit-packed
        assert_eq!(calculate_aligned_offset(0, 0, 255), 0);
        assert_eq!(calculate_aligned_offset(5, 0, 255), 5); // returns bit position as-is
        assert_eq!(calculate_aligned_offset(13, 0, 255), 13);
    }

    #[test]
    fn test_stream_state_basic() {
        let mut state = StreamState::new();
        assert_eq!(state.bit_position, 0);
        assert_eq!(state.byte_position(), 0);

        state.bit_position += 3 * 8;
        assert_eq!(state.bit_position, 24);
        assert_eq!(state.byte_position(), 3);

        state.bit_position += 5;
        assert_eq!(state.bit_position, 29);
        assert_eq!(state.byte_position(), 4); // ceil(29/8) = 4
    }

    #[test]
    fn test_stream_state_alignment_reset() {
        let mut state = StreamState::new();
        state.bit_position += 7 * 8;
        assert_eq!(state.alignment_offset, 0);

        state.reset_alignment();
        assert_eq!(state.alignment_offset, 7);
    }
}
