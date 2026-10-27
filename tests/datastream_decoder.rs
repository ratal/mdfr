//! Tests for datastream decoder functions and stream state machine.

use mdfr::mdfinfo::mdfinfo4::{Cn4, CN_F_ALIGNMENT_RESET};
use mdfr::mdfreader::datastream_decoder::*;


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

    // ── calculate_final_byte_offset tests ──

    #[test]
    fn test_final_byte_offset_normal_alignment() {
        // alignment != 255: result = aligned_offset + byte_offset (bit_offset ignored)
        assert_eq!(calculate_final_byte_offset(8, 0, 3, 0), 11);
        assert_eq!(calculate_final_byte_offset(8, 0, 3, 7), 11); // bit_offset ignored
        assert_eq!(calculate_final_byte_offset(0, 1, 5, 0), 5);
    }

    #[test]
    fn test_final_byte_offset_bit_packed() {
        // alignment == 255: result = (aligned_offset + byte_offset*8 + bit_offset) / 8
        assert_eq!(calculate_final_byte_offset(0, 255, 1, 4), 1); // (0 + 8 + 4) / 8 = 1
        assert_eq!(calculate_final_byte_offset(16, 255, 0, 4), 2); // (16 + 0 + 4) / 8 = 2
    }

    // ── decode_single_channel_value helper ──

    fn make_cn4(
        cn_type: u8,
        cn_bit_count: u32,
        cn_alignment: u8,
        cn_byte_offset: u32,
        cn_bit_offset: u8,
        cn_flags: u32,
    ) -> Cn4 {
        use mdfr::mdfinfo::mdfinfo4::Cn4Block;
        let mut block = Cn4Block::default();
        block.cn_type = cn_type;
        block.cn_bit_count = cn_bit_count;
        block.cn_alignment = cn_alignment;
        block.cn_byte_offset = cn_byte_offset;
        block.cn_bit_offset = cn_bit_offset;
        block.cn_flags = cn_flags;
        Cn4 {
            block,
            ..Default::default()
        }
    }

    // ── decode_single_channel_value tests ──

    #[test]
    fn test_decode_single_channel_byte_aligned() {
        let data = [0xABu8, 0xCD];
        let cn = make_cn4(0, 8, 0, 0, 0, 0);
        let mut stream_state = StreamState::new();
        let (bytes, consumed) = decode_single_channel_value(&data, &cn, &mut stream_state).unwrap();
        assert_eq!(bytes, vec![0xAB]);
        assert_eq!(consumed, 1);
        assert_eq!(stream_state.bit_position, 8);
    }

    #[test]
    fn test_decode_single_channel_with_byte_offset() {
        let data = [0x00u8, 0xAA, 0xBB];
        let cn = make_cn4(0, 8, 0, 1, 0, 0); // cn_byte_offset=1
        let mut stream_state = StreamState::new();
        let (bytes, consumed) = decode_single_channel_value(&data, &cn, &mut stream_state).unwrap();
        assert_eq!(bytes, vec![0xAA]);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_decode_single_channel_vlsd() {
        // 4-byte LE length = 3, then 3 bytes "ABC"
        let data = [3u8, 0, 0, 0, 0x41, 0x42, 0x43];
        let cn = make_cn4(1, 0, 0, 0, 0, 0); // cn_type=1 (VLSD)
        let mut stream_state = StreamState::new();
        let (bytes, consumed) = decode_single_channel_value(&data, &cn, &mut stream_state).unwrap();
        assert_eq!(bytes, vec![0x41, 0x42, 0x43]);
        assert_eq!(consumed, 7); // 4 (length prefix) + 3 (data)
        assert_eq!(stream_state.bit_position, 7 * 8);
    }

    #[test]
    fn test_decode_single_channel_vlsd_underrun() {
        // length=3 but only 1 byte of data
        let data = [3u8, 0, 0, 0, 0x41];
        let cn = make_cn4(1, 0, 0, 0, 0, 0);
        let mut stream_state = StreamState::new();
        let result = decode_single_channel_value(&data, &cn, &mut stream_state);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_single_channel_vlsc_bail() {
        // cn_type=7 (VLSC), cn_bit_count=0 -> should bail
        let data = [0u8; 16];
        let cn = make_cn4(7, 0, 0, 0, 0, 0);
        let mut stream_state = StreamState::new();
        let result = decode_single_channel_value(&data, &cn, &mut stream_state);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_single_channel_alignment_reset_flag() {
        // CN_F_ALIGNMENT_RESET = 1 << 18 = 0x40000
        let data = [0u8; 16];
        let cn = make_cn4(0, 8, 0, 0, 0, CN_F_ALIGNMENT_RESET);
        let mut stream_state = StreamState::new();
        // advance stream position to 6 bytes (48 bits)
        stream_state.bit_position = 48;
        stream_state.alignment_offset = 0;
        // When decode_single_channel_value is called, it should reset alignment_offset
        // to byte_position (= 6), then calculate the aligned offset from there
        let (bytes, _consumed) =
            decode_single_channel_value(&data, &cn, &mut stream_state).unwrap();
        assert_eq!(bytes.len(), 1);
        // alignment_offset should have been reset to 6 before calculation
        // aligned_offset for cn_alignment=0 = current_byte_pos = 6
        // final_byte_offset = 6 + 0 = 6
        // so bit_position = (6 + 1) * 8 = 56
        assert_eq!(stream_state.bit_position, 56);
    }

    // ── decode_channel_union tests ──

    #[test]
    fn test_decode_channel_union_two_members() {
        use mdfr::mdfinfo::mdfinfo4::Cu4Block;
        let data = [0x01u8, 0x00, 0x02, 0x00]; // 4 bytes
        let cu_block = Cu4Block::default();

        let mut cn_a = make_cn4(0, 16, 0, 0, 0, 0);
        cn_a.unique_name = "chan_a".to_string();
        let mut cn_b = make_cn4(0, 16, 0, 0, 0, 0);
        cn_b.unique_name = "chan_b".to_string();

        let member_channels: Vec<&Cn4> = vec![&cn_a, &cn_b];
        let mut stream_state = StreamState::new();

        let result =
            decode_channel_union(&data, &cu_block, &member_channels, &mut stream_state).unwrap();
        assert!(result.contains_key("chan_a"));
        assert!(result.contains_key("chan_b"));
        // Both start at 0, each consume 2 bytes (16 bits). max_end = 16 bits.
        assert_eq!(stream_state.bit_position, 16);
    }

    #[test]
    fn test_decode_channel_union_different_sizes() {
        use mdfr::mdfinfo::mdfinfo4::Cu4Block;
        let data = [0x01u8, 0x00, 0x02, 0x00, 0x03, 0x00]; // 6 bytes
        let cu_block = Cu4Block::default();

        let mut cn_small = make_cn4(0, 8, 0, 0, 0, 0); // 1 byte
        cn_small.unique_name = "small".to_string();
        let mut cn_large = make_cn4(0, 32, 0, 0, 0, 0); // 4 bytes
        cn_large.unique_name = "large".to_string();

        let member_channels: Vec<&Cn4> = vec![&cn_small, &cn_large];
        let mut stream_state = StreamState::new();

        let result =
            decode_channel_union(&data, &cu_block, &member_channels, &mut stream_state).unwrap();
        assert!(result.contains_key("small"));
        assert!(result.contains_key("large"));
        // max_end is 32 bits (4 bytes from cn_large)
        assert_eq!(stream_state.bit_position, 32);
    }

    // ── decode_channel_variant tests ──

    #[test]
    fn test_decode_channel_variant_match_first() {
        use mdfr::mdfinfo::mdfinfo4::Cv4Block;
        let data = [0xAAu8, 0xBB, 0xCC];
        let cv_block = Cv4Block {
            cv_option_val: vec![0, 1, 2],
            cv_option_count: 3,
            ..Default::default()
        };
        let mut cn_opt0 = make_cn4(0, 8, 0, 0, 0, 0);
        cn_opt0.unique_name = "opt0".to_string();
        let mut cn_opt1 = make_cn4(0, 8, 0, 0, 0, 0);
        cn_opt1.unique_name = "opt1".to_string();
        let mut cn_opt2 = make_cn4(0, 8, 0, 0, 0, 0);
        cn_opt2.unique_name = "opt2".to_string();

        let option_channels: Vec<&Cn4> = vec![&cn_opt0, &cn_opt1, &cn_opt2];
        let mut stream_state = StreamState::new();

        let (idx, name, bytes) =
            decode_channel_variant(&data, &cv_block, &option_channels, 0, &mut stream_state)
                .unwrap();
        assert_eq!(idx, 0);
        assert_eq!(name, "opt0");
        assert_eq!(bytes, vec![0xAA]);
    }

    #[test]
    fn test_decode_channel_variant_match_last() {
        use mdfr::mdfinfo::mdfinfo4::Cv4Block;
        let data = [0xAAu8, 0xBB, 0xCC];
        let cv_block = Cv4Block {
            cv_option_val: vec![0, 1, 2],
            cv_option_count: 3,
            ..Default::default()
        };
        let mut cn_opt0 = make_cn4(0, 8, 0, 0, 0, 0);
        cn_opt0.unique_name = "opt0".to_string();
        let mut cn_opt1 = make_cn4(0, 8, 0, 0, 0, 0);
        cn_opt1.unique_name = "opt1".to_string();
        let mut cn_opt2 = make_cn4(0, 8, 0, 0, 0, 0);
        cn_opt2.unique_name = "opt2".to_string();

        let option_channels: Vec<&Cn4> = vec![&cn_opt0, &cn_opt1, &cn_opt2];
        let mut stream_state = StreamState::new();

        let (idx, name, _bytes) =
            decode_channel_variant(&data, &cv_block, &option_channels, 2, &mut stream_state)
                .unwrap();
        assert_eq!(idx, 2);
        assert_eq!(name, "opt2");
    }

    #[test]
    fn test_decode_channel_variant_no_match() {
        use mdfr::mdfinfo::mdfinfo4::Cv4Block;
        let data = [0xAAu8, 0xBB, 0xCC];
        let cv_block = Cv4Block {
            cv_option_val: vec![0, 1, 2],
            cv_option_count: 3,
            ..Default::default()
        };
        let mut cn_opt0 = make_cn4(0, 8, 0, 0, 0, 0);
        cn_opt0.unique_name = "opt0".to_string();
        let option_channels: Vec<&Cn4> = vec![&cn_opt0];
        let mut stream_state = StreamState::new();

        let result =
            decode_channel_variant(&data, &cv_block, &option_channels, 99, &mut stream_state);
        assert!(result.is_err());
    }

    // ── decode_channel_list tests ──

    #[test]
    fn test_decode_channel_list_size_in_elements() {
        use mdfr::mdfinfo::mdfinfo4::Cl4Block;
        // cl_flags = 1: size_value is number of elements
        let cl_block = Cl4Block {
            cl_flags: 1,
            cl_alignment: 0,
            cl_bit_offset: 0,
            cl_byte_offset: 0,
            ..Default::default()
        };
        let data = [10u8, 20, 30, 40];
        let element_cn = make_cn4(0, 8, 0, 0, 0, 0);
        let mut stream_state = StreamState::new();

        let elements =
            decode_channel_list(&data, &cl_block, &element_cn, 3, &mut stream_state).unwrap();
        assert_eq!(elements.len(), 3);
        assert_eq!(elements[0], vec![10u8]);
        assert_eq!(elements[1], vec![20u8]);
        assert_eq!(elements[2], vec![30u8]);
    }

    #[test]
    fn test_decode_channel_list_size_in_bytes() {
        use mdfr::mdfinfo::mdfinfo4::Cl4Block;
        // cl_flags = 0: size_value is number of bytes; element is 1 byte → 4 elements
        let cl_block = Cl4Block {
            cl_flags: 0,
            cl_alignment: 0,
            cl_bit_offset: 0,
            cl_byte_offset: 0,
            ..Default::default()
        };
        let data = [10u8, 20, 30, 40];
        let element_cn = make_cn4(0, 8, 0, 0, 0, 0); // 1 byte per element
        let mut stream_state = StreamState::new();

        let elements =
            decode_channel_list(&data, &cl_block, &element_cn, 4, &mut stream_state).unwrap();
        assert_eq!(elements.len(), 4);
    }

    #[test]
    fn test_decode_channel_list_zero_elements() {
        use mdfr::mdfinfo::mdfinfo4::Cl4Block;
        let cl_block = Cl4Block {
            cl_flags: 1,
            ..Default::default()
        };
        let data = [10u8, 20, 30, 40];
        let element_cn = make_cn4(0, 8, 0, 0, 0, 0);
        let mut stream_state = StreamState::new();

        let elements =
            decode_channel_list(&data, &cl_block, &element_cn, 0, &mut stream_state).unwrap();
        assert_eq!(elements.len(), 0);
    }

    #[test]
    fn test_decode_channel_list_underrun() {
        use mdfr::mdfinfo::mdfinfo4::Cl4Block;
        // cl_flags = 1: size_value=5 elements, but data only has 2 bytes
        let cl_block = Cl4Block {
            cl_flags: 1,
            cl_alignment: 0,
            cl_bit_offset: 0,
            cl_byte_offset: 0,
            ..Default::default()
        };
        let data = [10u8, 20]; // only 2 bytes, not enough for 5 elements
        let element_cn = make_cn4(0, 8, 0, 0, 0, 0);
        let mut stream_state = StreamState::new();

        let result = decode_channel_list(&data, &cl_block, &element_cn, 5, &mut stream_state);
        assert!(result.is_err());
    }

    // ── extract_channel_values_as_u64 tests ──

    #[test]
    fn test_extract_u64_from_uint8() {
        use mdfr::data_holder::channel_data::ChannelData;
        use arrow::array::UInt8Builder;
        let mut builder = UInt8Builder::new();
        builder.append_value(1);
        builder.append_value(2);
        builder.append_value(3);
        let cn = Cn4 {
            data: ChannelData::UInt8(builder),
            ..Default::default()
        };
        let result = extract_channel_values_as_u64(&cn).unwrap();
        assert_eq!(result, vec![1u64, 2, 3]);
    }

    #[test]
    fn test_extract_u64_from_utf8_fails() {
        use mdfr::data_holder::channel_data::ChannelData;
        use arrow::array::LargeStringBuilder;
        let cn = Cn4 {
            data: ChannelData::Utf8(LargeStringBuilder::new()),
            ..Default::default()
        };
        let result = extract_channel_values_as_u64(&cn);
        assert!(result.is_err());
    }
