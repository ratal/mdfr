//! ASAM MDF4 Bus Logging schema presets (CAN, CAN-FD, LIN, FlexRay, Ethernet).
//!
//! Creates ASAM-compliant channel groups with `SIBLOCK` (`si_type = 2` Bus) and
//! frame channels (`ID`, `DLC`, `DataLength`, `DataBytes`, `Dir`) that can be populated
//! via the dynamic record builder or `push_raw`.
use crate::blocks::cg::CG_F_EVENT_SIGNAL_GROUP;
use crate::blocks::cn::CnDataType;
use crate::error::Result;

use super::schema::{ChannelHandle, GroupHandle, LoggerBuilder, SourceDef};

/// Handles to the standard channels of a CAN / CAN-FD DataFrame bus logging group.
#[derive(Debug, Clone, Copy)]
pub struct CanGroupHandles {
    /// Channel group handle.
    pub group: GroupHandle,
    /// Bus channel number (`u8`, byte offset 8).
    pub bus_channel: ChannelHandle,
    /// CAN arbitration ID (`u32` LE, byte offset 9..13; bit 31 = extended IDE).
    pub can_id: ChannelHandle,
    /// Direction (`u8`, byte offset 13: 0 = Rx, 1 = Tx).
    pub dir: ChannelHandle,
    /// Data Length Code (`u8`, byte offset 14).
    pub dlc: ChannelHandle,
    /// Payload data length in bytes (`u8`, byte offset 15).
    pub data_length: ChannelHandle,
    /// Fixed-length payload bytes (`[u8; 8]` for classic CAN or `[u8; 64]` for CAN-FD, byte offset 16..)
    /// or VLSD payload channel when created with `add_can_fd_vlsd`.
    pub data_bytes: ChannelHandle,
}

/// Bus logging schema presets.
pub struct BusPreset;

impl BusPreset {
    /// Adds a classic CAN (8-byte fixed payload, 24-byte total record) group with `SIBLOCK(Bus, CAN)`.
    pub fn can(builder: &mut LoggerBuilder, bus_name: &str) -> Result<CanGroupHandles> {
        Self::can_with_payload(builder, bus_name, 8, false)
    }

    /// Adds a CAN-FD (64-byte fixed payload, 80-byte total record) group with `SIBLOCK(Bus, CAN)`.
    pub fn can_fd(builder: &mut LoggerBuilder, bus_name: &str) -> Result<CanGroupHandles> {
        Self::can_with_payload(builder, bus_name, 64, false)
    }

    /// Adds a CAN-FD group whose `DataBytes` channel uses VLSD (`SDBLOCK` / VLSD CG) for variable payload length.
    pub fn can_fd_vlsd(builder: &mut LoggerBuilder, bus_name: &str) -> Result<CanGroupHandles> {
        Self::can_with_payload(builder, bus_name, 64, true)
    }

    fn can_with_payload(
        builder: &mut LoggerBuilder,
        bus_name: &str,
        max_payload: u32,
        vlsd: bool,
    ) -> Result<CanGroupHandles> {
        let g_name = alloc::format!("{bus_name}_DataFrame");
        let group = builder.add_group(&g_name);
        let src = SourceDef {
            name: alloc::string::String::from(bus_name),
            path: Some(alloc::format!("{bus_name}/CAN_DataFrame")),
            si_type: 2,     // Bus
            si_bus_type: 2, // CAN
            si_flags: 0,
        };
        builder.set_group_source(group, src)?;
        let bus_channel =
            builder.add_channel(group, "CAN_DataFrame.BusChannel", CnDataType::UIntLE, 8)?;
        let can_id = builder.add_channel(group, "CAN_DataFrame.ID", CnDataType::UIntLE, 32)?;
        let dir = builder.add_channel(group, "CAN_DataFrame.Dir", CnDataType::UIntLE, 8)?;
        let dlc = builder.add_channel(group, "CAN_DataFrame.DLC", CnDataType::UIntLE, 8)?;
        let data_length =
            builder.add_channel(group, "CAN_DataFrame.DataLength", CnDataType::UIntLE, 8)?;
        let data_bytes = if vlsd {
            builder.add_vlsd_channel(group, "CAN_DataFrame.DataBytes", CnDataType::ByteArray)?
        } else {
            builder.add_channel(
                group,
                "CAN_DataFrame.DataBytes",
                CnDataType::ByteArray,
                max_payload * 8,
            )?
        };
        let _ = CG_F_EVENT_SIGNAL_GROUP;
        Ok(CanGroupHandles {
            group,
            bus_channel,
            can_id,
            dir,
            dlc,
            data_length,
            data_bytes,
        })
    }

    /// Adds a LIN frame group (`ID`, `DLC`, `DataLength`, 8-byte `DataBytes`) with `SIBLOCK(Bus, LIN)`.
    pub fn lin(builder: &mut LoggerBuilder, bus_name: &str) -> Result<GroupHandle> {
        let g_name = alloc::format!("{bus_name}_LIN_Frame");
        let group = builder.add_group(&g_name);
        builder.set_group_source(
            group,
            SourceDef {
                name: alloc::string::String::from(bus_name),
                path: Some(alloc::format!("{bus_name}/LIN_Frame")),
                si_type: 2,
                si_bus_type: 3, // LIN
                si_flags: 0,
            },
        )?;
        builder.add_channel(group, "LIN_Frame.ID", CnDataType::UIntLE, 8)?;
        builder.add_channel(group, "LIN_Frame.DLC", CnDataType::UIntLE, 8)?;
        builder.add_channel(group, "LIN_Frame.DataLength", CnDataType::UIntLE, 8)?;
        builder.add_channel(group, "LIN_Frame.DataBytes", CnDataType::ByteArray, 64)?;
        Ok(group)
    }

    /// Adds an Ethernet frame group (`Source` MAC, `Destination` MAC, `EtherType`, `DataLength`, VLSD `DataBytes`)
    /// with `SIBLOCK(Bus, Ethernet)`.
    pub fn ethernet(builder: &mut LoggerBuilder, bus_name: &str) -> Result<GroupHandle> {
        let g_name = alloc::format!("{bus_name}_ETH_Frame");
        let group = builder.add_group(&g_name);
        builder.set_group_source(
            group,
            SourceDef {
                name: alloc::string::String::from(bus_name),
                path: Some(alloc::format!("{bus_name}/ETH_Frame")),
                si_type: 2,
                si_bus_type: 7, // Ethernet
                si_flags: 0,
            },
        )?;
        builder.add_channel(group, "ETH_Frame.Source", CnDataType::ByteArray, 48)?;
        builder.add_channel(group, "ETH_Frame.Destination", CnDataType::ByteArray, 48)?;
        builder.add_channel(group, "ETH_Frame.EtherType", CnDataType::UIntBE, 16)?;
        builder.add_channel(group, "ETH_Frame.DataLength", CnDataType::UIntLE, 16)?;
        builder.add_vlsd_channel(group, "ETH_Frame.DataBytes", CnDataType::ByteArray)?;
        Ok(group)
    }
}
