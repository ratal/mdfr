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

/**
 * Main Mdf struct holding mdfinfo, arrow data and schema
 */
typedef struct Mdf Mdf;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

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
const ArrowArray *get_channel_array(const struct Mdf *mdf, const char *channel_name);

/**
 * returns channel's arrow Schema.
 * null pointer returned if not found
 */
const ArrowSchema *get_channel_schema(const struct Mdf *mdf, const char *channel_name);

/**
 * export to Parquet file
 * Compression can be one of the following strings
 * "snappy", "gzip", "lzo", "brotli", "lz4", "lz4raw"
 *  or null pointer if no compression wanted
 */
void export_to_parquet(const struct Mdf *mdf, const char *file_name, const char *compression);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
