#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * reader buffer size, by default same as Rust BufReader
 */
#define DEFAULT_BUF_SIZE 8192

/**
 * The following constant represents the size of data chunk to be read and processed.
 * a big chunk will improve performance but consume more memory
 * a small chunk will not consume too much memory but will cause many read calls, penalising performance
 */
#define CHUNK_SIZE_READING_3 524288

/**
 * The following constant represents the size of data chunk to be read and processed.
 * a big chunk will improve performance but consume more memory
 * a small chunk will not consume too much memory but will cause many read calls, penalising performance
 */
#define CHUNK_SIZE_READING_4 524288

typedef struct At At;

typedef struct BTreeMap_i64__Dg4 BTreeMap_i64__Dg4;

typedef struct BTreeMap_u32__Dg3 BTreeMap_u32__Dg3;

typedef struct HashMap_String__ChannelId HashMap_String__ChannelId;

typedef struct HashMap_String__ChannelId3 HashMap_String__ChannelId3;

typedef struct HashMap_i64__Cc4Block HashMap_i64__Cc4Block;

typedef struct HashMap_i64__Ev4Block HashMap_i64__Ev4Block;

typedef struct HashMap_i64__MetaData HashMap_i64__MetaData;

typedef struct HashMap_i64__Si4Block HashMap_i64__Si4Block;

/**
 * HD3 strucutre
 */
typedef struct Hd3 Hd3;

/**
 * sharable blocks (most likely referenced multiple times and shared by several blocks)
 * that are in sharable fields and holds CC, CE, TX blocks
 */
typedef struct SharableBlocks3 SharableBlocks3;

typedef struct String String;

typedef struct Vec_FhBlock Vec_FhBlock;

/**
 * Common Id block structure for both versions 2 and 3
 */
typedef struct IdBlock {
  /**
   * "MDF
   */
  uint8_t id_file_id[8];
  /**
   * version in char
   */
  uint8_t id_vers[8];
  uint8_t id_prog[8];
  /**
   * 0 Little endian, >= 1 Big endian, only valid for 3.x
   */
  uint16_t id_default_byteorder;
  /**
   * default floating point number. 0: IEEE754, 1: G_Float, 2: D_Float, only valid for 3.x
   */
  uint16_t id_floatingpointformat;
  /**
   * version number, valid for both 3.x and 4.x
   */
  uint16_t id_ver;
  /**
   * code page (only for version 3.3)
   */
  uint16_t id_codepage;
  /**
   * check
   */
  uint8_t id_check[2];
  uint8_t id_fill[26];
  /**
   * only valid for 4.x but can exist in 3.x
   */
  uint16_t id_unfin_flags;
  /**
   * only valid for 4.x but can exist in 3.x
   */
  uint16_t id_custom_unfin_flags;
} IdBlock;

typedef struct HashMap_String__ChannelId3 ChannelNamesSet3;

/**
 * Specific to version 3.x mdf metadata structure
 */
typedef struct MdfInfo3 {
  /**
   * file name string
   */
  struct String file_name;
  /**
   * Identification block
   */
  struct IdBlock id_block;
  /**
   * code page
   */
  const Encoding *encoding;
  /**
   * Header block
   */
  struct Hd3 hd_block;
  /**
   * Header comments
   */
  struct String hd_comment;
  /**
   * data group block linking channel group/channel/conversion/..etc. and data block
   */
  struct BTreeMap_u32__Dg3 dg;
  /**
   * Conversion and CE blocks
   */
  struct SharableBlocks3 sharable;
  /**
   * set of all channel names
   */
  ChannelNamesSet3 channel_names_set;
} MdfInfo3;

/**
 * Hd4 (Header) block structure
 */
typedef struct Hd4 {
  /**
   * ##HD
   */
  uint8_t hd_id[4];
  /**
   * reserved
   */
  uint8_t hd_reserved[4];
  /**
   * Length of block in bytes
   */
  uint64_t hd_len;
  /**
   * # of links
   */
  uint64_t hd_link_counts;
  /**
   * Pointer to the first data group block (DGBLOCK) (can be NIL)
   */
  int64_t hd_dg_first;
  /**
   * Pointer to first file history block (FHBLOCK)
   * There must be at least one FHBLOCK with information about the application which created the MDF file.
   */
  int64_t hd_fh_first;
  /**
   * Pointer to first channel hierarchy block (CHBLOCK) (can be NIL).
   */
  int64_t hd_ch_first;
  /**
   * Pointer to first attachment block (ATBLOCK) (can be NIL)
   */
  int64_t hd_at_first;
  /**
   * Pointer to first event block (EVBLOCK) (can be NIL)
   */
  int64_t hd_ev_first;
  /**
   * Pointer to the measurement file comment (TXBLOCK or MDBLOCK) (can be NIL) For MDBLOCK contents, see Table 14.
   */
  int64_t hd_md_comment;
  /**
   * Data members
   * Time stamp in nanoseconds elapsed since 00:00:00 01.01.1970 (UTC time or local time, depending on "local time" flag)
   */
  uint64_t hd_start_time_ns;
  /**
   * Time zone offset in minutes. The value must be in range [-720,720], i.e. it can be negative! For example a value of 60 (min) means UTC+1 time zone = Central European Time (CET). Only valid if "time offsets valid" flag is set in time flags.
   */
  int16_t hd_tz_offset_min;
  /**
   * Daylight saving time (DST) offset in minutes for start time stamp. During the summer months, most regions observe a DST offset of 60 min (1 hour). Only valid if "time offsets valid" flag is set in time flags.
   */
  int16_t hd_dst_offset_min;
  /**
   * Time flags The value contains the following bit flags (see HD_TF_xxx)
   */
  uint8_t hd_time_flags;
  /**
   * Time quality class (see HD_TC[35, 35, 72, 68]_xxx)
   */
  uint8_t hd_time_class;
  /**
   * Flags The value contains the following bit flags (see HD_FL_xxx):
   */
  uint8_t hd_flags;
  /**
   * reserved
   */
  uint8_t hd_reserved2;
  /**
   * Start angle in radians at start of measurement (only for angle synchronous measurements) Only valid if "start angle valid" flag is set. All angle values for angle synchronized master channels or events are relative to this start angle.
   */
  double hd_start_angle_rad;
  /**
   * Start distance in meters at start of measurement (only for distance synchronous measurements) Only valid if "start distance valid" flag is set. All distance values for distance synchronized master channels or events are relative to this start distance.
   */
  double hd_start_distance_m;
} Hd4;

typedef struct Vec_FhBlock Fh;

/**
 * sharable blocks (most likely referenced multiple times and shared by several blocks)
 * that are in sharable fields and holds CC, SI, TX and MD blocks
 */
typedef struct SharableBlocks {
  struct HashMap_i64__MetaData md_tx;
  struct HashMap_i64__Cc4Block cc;
  struct HashMap_i64__Si4Block si;
} SharableBlocks;

typedef struct HashMap_String__ChannelId ChannelNamesSet;

/**
 * MdfInfo4 is the struct holding whole metadata of mdf4.x files
 * * blocks with unique links are at top level like attachment, events and file history
 * * sharable blocks (most likely referenced multiple times and shared by several blocks)
 *   that are in sharable fields and holds CC, SI, TX and MD blocks
 * * the dg fields nests cg itself nesting cn blocks and eventually compositions
 *   (other cn or ca blocks) and conversion
 * * channel_names_set is the complete set of channel names contained in file
 * * in general the blocks are contained in HashMaps with key corresponding
 *   to their position in the file
 */
typedef struct MdfInfo4 {
  /**
   * file name string
   */
  struct String file_name;
  /**
   * Identifier block
   */
  struct IdBlock id_block;
  /**
   * header block
   */
  struct Hd4 hd_block;
  /**
   * file history blocks
   */
  Fh fh;
  /**
   * attachment blocks
   */
  struct At at;
  /**
   * event blocks
   */
  struct HashMap_i64__Ev4Block ev;
  /**
   * data group block linking channel group/channel/conversion/compostion/..etc. and data block
   */
  struct BTreeMap_i64__Dg4 dg;
  /**
   * cc, md, tx and si blocks that can be referenced by several blocks
   */
  struct SharableBlocks sharable;
  /**
   * set of all channel names
   */
  ChannelNamesSet channel_names_set;
} MdfInfo4;

/**
 * joins mdf versions 3.x and 4.x
 */
typedef enum MdfInfo_Tag {
  V3,
  V4,
} MdfInfo_Tag;

typedef struct MdfInfo {
  MdfInfo_Tag tag;
  union {
    struct {
      struct MdfInfo3 *v3;
    };
    struct {
      struct MdfInfo4 *v4;
    };
  };
} MdfInfo;

/**
 * Main Mdf struct holding mdfinfo, arrow data and schema
 */
typedef struct Mdf {
  /**
   * MdfInfo enum
   */
  struct MdfInfo mdf_info;
} Mdf;

/**
 * create a new mdf from a file and its metadata
 */
struct Mdf *new_mdf(const char *file_name);

/**
 * returns mdf file version
 */
unsigned short get_version(const struct Mdf *mdf);

/**
 * returns channel's unit string
 * if no unit is existing for this channel, returns a null pointer
 */
const char *get_channel_unit(const struct Mdf *mdf, const char *channel_name);

/**
 * returns channel's description string
 * if no description is existing for this channel, returns null pointer
 */
const char *get_channel_desc(const struct Mdf *mdf, const char *channel_name);

/**
 * returns channel's associated master channel name string
 * if no master channel existing, returns null pointer
 */
const char *get_channel_master(const struct Mdf *mdf, const char *channel_name);

/**
 * returns channel's associated master channel type string
 * 0 = None (normal data channels), 1 = Time (seconds), 2 = Angle (radians),
 * 3 = Distance (meters), 4 = Index (zero-based index values)
 */
unsigned char get_channel_master_type(const struct Mdf *mdf, const char *channel_name);

/**
 * returns a sorted array of strings of all channel names contained in file
 */
char *const *get_channel_names_set(const struct Mdf *mdf);

/**
 * load all channels data in memory
 */
void load_all_channels_data_in_memory(struct Mdf *mdf);

/**
 * returns channel's arrow Array.
 * null pointer returned if not found
 */
const FFI_ArrowArray *get_channel_array(const struct Mdf *mdf, const char *channel_name);

void export_to_parquet(const struct Mdf *mdf, const char *file_name, const char *compression);

void export_to_hdf5(const struct Mdf *mdf, const char *file_name, const char *compression);
