#ifndef MDFR_LOGGER_H
#define MDFR_LOGGER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes returned by mdfr_logger_* functions */
#define MDFR_OK                    0
#define MDFR_ERR_IO               -1
#define MDFR_ERR_BINRW            -2
#define MDFR_ERR_INVALID_SCHEMA   -3
#define MDFR_ERR_INVALID_HANDLE   -4
#define MDFR_ERR_UNSUPPORTED      -5
#define MDFR_ERR_COMPRESSION      -6
#define MDFR_ERR_RING_FULL        -7
#define MDFR_ERR_ALREADY_CLOSED   -8
#define MDFR_ERR_RECORD_SIZE      -9

/* Data layout values for mdfr_logger_builder_new */
#define MDFR_LAYOUT_SORTED         0
#define MDFR_LAYOUT_UNSORTED       1

/* Compression algorithm values for mdfr_logger_builder_new */
#define MDFR_COMPRESS_NONE       255
#define MDFR_COMPRESS_DEFLATE      0
#define MDFR_COMPRESS_DEFLATE_TR   1
#define MDFR_COMPRESS_ZSTD         2
#define MDFR_COMPRESS_ZSTD_TR      3
#define MDFR_COMPRESS_LZ4          4
#define MDFR_COMPRESS_LZ4_TR       5

/* ASAM MDF 4.3 CN data types (little-endian) */
#define MDFR_DTYPE_UINT_LE         0
#define MDFR_DTYPE_INT_LE          2
#define MDFR_DTYPE_FLOAT_LE        4
#define MDFR_DTYPE_STRING_UTF8     7
#define MDFR_DTYPE_BYTE_ARRAY     10

/* ASAM bus preset kinds for mdfr_logger_add_bus_preset */
#define MDFR_BUS_CAN               0
#define MDFR_BUS_CAN_FD            1
#define MDFR_BUS_LIN               2
#define MDFR_BUS_ETHERNET          3

typedef struct MdfrBuilderHandle MdfrBuilderHandle;
typedef struct MdfrLoggerHandle MdfrLoggerHandle;

typedef struct MdfrSinkVtable {
    void *ctx;
    int32_t (*write)(void *ctx, const uint8_t *buf, size_t len);
    int32_t (*seek)(void *ctx, uint64_t pos);
    int32_t (*flush)(void *ctx);
    int32_t (*close)(void *ctx);
} MdfrSinkVtable;

typedef struct MdfrClockVtable {
    void *ctx;
    uint64_t (*now_ticks)(void *ctx);
    uint64_t tick_ns;
    int32_t (*wall_clock_ns)(void *ctx, uint64_t *out_ns);
} MdfrClockVtable;

MdfrBuilderHandle *mdfr_logger_builder_new(uint8_t layout, uint8_t compression);
void mdfr_logger_builder_free(MdfrBuilderHandle *builder);

int32_t mdfr_logger_add_group(
    MdfrBuilderHandle *builder,
    const char *name,
    uint16_t *out_group
);

int32_t mdfr_logger_add_channel(
    MdfrBuilderHandle *builder,
    uint16_t group,
    const char *name,
    uint8_t data_type,
    uint32_t bit_count,
    uint32_t *out_ch
);

int32_t mdfr_logger_add_vlsd_channel(
    MdfrBuilderHandle *builder,
    uint16_t group,
    const char *name,
    uint8_t data_type,
    uint32_t *out_ch
);

int32_t mdfr_logger_set_conversion_linear(
    MdfrBuilderHandle *builder,
    uint32_t ch_packed,
    double p1,
    double p2
);

int32_t mdfr_logger_set_channel_unit(
    MdfrBuilderHandle *builder,
    uint32_t ch_packed,
    const char *unit
);

int32_t mdfr_logger_add_bus_preset(
    MdfrBuilderHandle *builder,
    uint8_t bus_kind,
    const char *bus_name,
    uint16_t *out_group
);

MdfrLoggerHandle *mdfr_logger_start(
    MdfrBuilderHandle *builder,
    MdfrSinkVtable sink,
    MdfrClockVtable clock,
    int32_t *out_err
);

int32_t mdfr_logger_push_raw(
    MdfrLoggerHandle *logger,
    uint16_t group,
    const uint8_t *record_bytes,
    size_t len
);

int32_t mdfr_logger_push_raw_at(
    MdfrLoggerHandle *logger,
    uint16_t group,
    uint64_t ticks,
    const uint8_t *record_bytes,
    size_t len
);

int32_t mdfr_logger_record_begin(MdfrLoggerHandle *logger, uint16_t group);
int32_t mdfr_logger_record_begin_at(MdfrLoggerHandle *logger, uint16_t group, uint64_t ticks);
int32_t mdfr_logger_record_set_u64(MdfrLoggerHandle *logger, uint32_t ch_packed, uint64_t val);
int32_t mdfr_logger_record_set_i64(MdfrLoggerHandle *logger, uint32_t ch_packed, int64_t val);
int32_t mdfr_logger_record_set_f64(MdfrLoggerHandle *logger, uint32_t ch_packed, double val);
int32_t mdfr_logger_record_set_bytes(
    MdfrLoggerHandle *logger,
    uint32_t ch_packed,
    const uint8_t *data,
    size_t len
);
int32_t mdfr_logger_record_commit(MdfrLoggerHandle *logger, uint16_t group);

int32_t mdfr_logger_checkpoint(MdfrLoggerHandle *logger);
int32_t mdfr_logger_rotate(MdfrLoggerHandle *logger, MdfrSinkVtable new_sink);
int32_t mdfr_logger_close(MdfrLoggerHandle *logger);

/* POSIX / std convenience helpers (available when linked with libmdfr) */
MdfrLoggerHandle *mdfr_logger_open(
    MdfrBuilderHandle *builder,
    const char *path,
    int32_t *out_err
);
int32_t mdfr_logger_rotate_path(MdfrLoggerHandle *logger, const char *new_path);
int32_t mdfr_finalize_file_in_place(const char *file_name);

#ifdef __cplusplus
}
#endif

#endif /* MDFR_LOGGER_H */

