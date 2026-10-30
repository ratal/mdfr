#ifndef MDFR_H
#define MDFR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "include/mdfr_logger.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* Arrow C Data Interface Specification                                      */
/* ========================================================================= */

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

#define ARROW_FLAG_DICTIONARY_ORDERED 1
#define ARROW_FLAG_NULLABLE 2
#define ARROW_FLAG_MAP_KEYS_SORTED 4

struct ArrowSchema {
  const char *format;
  const char *name;
  const char *metadata;
  int64_t flags;
  int64_t n_children;
  struct ArrowSchema **children;
  struct ArrowSchema *dictionary;
  void (*release)(struct ArrowSchema *);
  void *private_data;
};

struct ArrowArray {
  int64_t length;
  int64_t null_count;
  int64_t offset;
  int64_t n_buffers;
  int64_t n_children;
  const void **buffers;
  struct ArrowArray **children;
  struct ArrowArray *dictionary;
  void (*release)(struct ArrowArray *);
  void *private_data;
};

#endif /* ARROW_C_DATA_INTERFACE */

typedef struct ArrowArray FFI_ArrowArray;
typedef struct ArrowSchema FFI_ArrowSchema;

/* ========================================================================= */
/* Opaque Handles and Structs                                                */
/* ========================================================================= */

/**
 * Main Mdf struct holding file metadata, Arrow data, and schemas.
 * Created via `new_mdf()` and freed with `free_mdf()`.
 */
typedef struct Mdf Mdf;

/**
 * Uniquely identifies a channel by its file position.
 * `master_name` is heap-allocated; free with `free_channel_id_master()`.
 */
typedef struct ChannelIdC {
  char *master_name;
  int64_t dg_pos;
  int64_t cg_pos;
  uint64_t rec_id;
  int64_t cn_pos;
  int32_t rec_pos;
} ChannelIdC;

/* ========================================================================= */
/* Reader Lifecycle & General Information                                    */
/* ========================================================================= */

/**
 * Creates a new Mdf reader from a file path.
 * Returns a heap-allocated Mdf pointer, or NULL on error.
 * Caller must free the returned pointer with `free_mdf()`.
 */
Mdf *new_mdf(const char *file_name);

/**
 * Frees an Mdf object previously returned by `new_mdf()`.
 */
void free_mdf(Mdf *mdf);

/**
 * Returns the MDF file version (e.g. 410 for 4.10, 430 for 4.30).
 * Returns 0 on NULL input.
 */
unsigned short get_version(const Mdf *mdf);

/* ========================================================================= */
/* Channel Metadata & Access                                                 */
/* ========================================================================= */

/**
 * Returns the channel's unit string, or NULL if none exists.
 * The returned pointer is heap-allocated; caller must free with `free_c_string()`.
 */
const char *get_channel_unit(const Mdf *mdf, const char *channel_name);

/**
 * Returns the channel's description string, or NULL if none exists.
 * The returned pointer is heap-allocated; caller must free with `free_c_string()`.
 */
const char *get_channel_desc(const Mdf *mdf, const char *channel_name);

/**
 * Returns the channel's associated master channel name, or NULL if none exists.
 * The returned pointer is heap-allocated; caller must free with `free_c_string()`.
 */
const char *get_channel_master(const Mdf *mdf, const char *channel_name);

/**
 * Returns the channel's associated master channel type:
 * 0 = None, 1 = Time (s), 2 = Angle (rad), 3 = Distance (m), 4 = Index
 */
unsigned char get_channel_master_type(const Mdf *mdf, const char *channel_name);

/**
 * Returns a sorted array of all channel names in the file.
 * The returned pointer is heap-allocated; free with `free_channel_names_set(ptr, len)`.
 */
const char *const *get_channel_names_set(const Mdf *mdf);

/**
 * Frees a channel names array returned by `get_channel_names_set()`.
 */
void free_channel_names_set(char *const *ptr, size_t len);

/**
 * Frees a null-terminated C string allocated by any `get_*` function.
 */
void free_c_string(char *ptr);

/**
 * Preloads all channel raw and converted data into memory.
 */
void load_all_channels_data_in_memory(Mdf *mdf);

/**
 * Returns a channel's Arrow Array pointer.
 * Caller must free with `free_channel_array()`.
 */
FFI_ArrowArray *get_channel_array(const Mdf *mdf, const char *channel_name);

/**
 * Frees an Arrow Array returned by `get_channel_array()`.
 */
void free_channel_array(FFI_ArrowArray *array);

/**
 * Returns a channel's Arrow Schema pointer.
 * Caller must free with `free_channel_schema()`.
 */
FFI_ArrowSchema *get_channel_schema(const Mdf *mdf, const char *channel_name);

/**
 * Frees an Arrow Schema returned by `get_channel_schema()`.
 */
void free_channel_schema(FFI_ArrowSchema *schema);

/* ========================================================================= */
/* Export Functions                                                          */
/* ========================================================================= */

/**
 * Exports loaded channels to an Apache Parquet file.
 * `compression` can be "snappy", "gzip", "lz4", "zstd", etc., or NULL.
 */
void export_to_parquet(const Mdf *mdf, const char *file_name, const char *compression);

/**
 * Exports loaded channels to an HDF5 file.
 * `compression` can be "deflate", "lzf", or NULL.
 */
void export_to_hdf5(const Mdf *mdf, const char *file_name, const char *compression);

/* ========================================================================= */
/* MDF 4.3 Common Properties & Metadata                                      */
/* ========================================================================= */

bool get_recorder_sequence_index(const Mdf *mdf, uint64_t *out);
bool get_recorder_file_index(const Mdf *mdf, uint64_t *out);
bool get_recorder_file_last(const Mdf *mdf, unsigned char *out);
const char *get_recorder_uuid(const Mdf *mdf);
const char *get_measurement_uuid(const Mdf *mdf);

const char *get_author(const Mdf *mdf);
const char *get_department(const Mdf *mdf);
const char *get_project(const Mdf *mdf);
const char *get_subject(const Mdf *mdf);

bool get_channel_id(const Mdf *mdf, const char *channel_name, ChannelIdC *out);
void free_channel_id_master(ChannelIdC *id);

/* ========================================================================= */
/* Diagnostics, File History, Events, Hierarchy, Source Info                 */
/* ========================================================================= */

size_t get_file_history_count(const Mdf *mdf);
bool get_file_history_time_ns(const Mdf *mdf, size_t index, uint64_t *out);

size_t get_event_count(const Mdf *mdf);
uint8_t get_event_type(const Mdf *mdf, size_t index);

size_t get_channel_hierarchy_count(const Mdf *mdf);
const char *list_channel_hierarchy(const Mdf *mdf);

size_t get_source_information_count(const Mdf *mdf);
const char *list_source_information(const Mdf *mdf);

#ifdef __cplusplus
}
#endif

#endif /* MDFR_H */
