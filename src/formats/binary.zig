const std = @import("std");
const lz4 = @import("lz4");

const Roblox = @import("../roblox.zig");
const Base64 = @import("../utils/base64.zig");

pub const BinaryChunkType = enum {
    INST,
    PROP,
    PRNT,
    META,
    SSTR,
    SIGN,
    END,
};

pub const BinaryMagicHeader = "<roblox!\x89\xff\x0d\x0a\x1a\x0a";

pub const BinaryChunkTypeMap = std.StaticStringMap(BinaryChunkType).initComptime(.{
    .{ "INST", .INST },
    .{ "PROP", .PROP },
    .{ "PRNT", .PRNT },
    .{ "META", .META },
    .{ "SSTR", .SSTR },
    .{ "SIGN", .SIGN },
    .{ &.{ 'E', 'N', 'D', 0 }, .END },
});

pub const DeserializeError = error{
    BadParentFormat,
    BadSharedStringFormat,
    BadDataNoEnd,
    BadCFrameRotationId,
};

pub const Chunk = struct {
    Type: BinaryChunkType,
    Data: []const u8,
    Size: u32,
    Reserved: u32,
    Allocated: bool = false,

    pub fn deinit(self: Chunk, allocator: std.mem.Allocator) void {
        if (self.Allocated) allocator.free(self.Data);
    }

    pub const Reader = struct {
        buffer: []const u8,
        pos: usize,

        const Self = @This();

        const ReadError = error{};

        const ThisReader = std.io.Reader(*Self, ReadError, read);

        pub fn reader(self: *Self) ThisReader {
            return .{ .context = self };
        }

        pub fn read(self: *Self, dest: []u8) ReadError!usize {
            const size = @min(dest.len, self.buffer.len - self.pos);
            const end = self.pos + size;

            @memcpy(dest[0..size], self.buffer[self.pos..end]);
            self.pos = end;

            return size;
        }

        pub inline fn readByte(self: *Self) !u8 {
            return try self.reader().readByte();
        }

        pub inline fn readVarInt(
            self: *Self,
            comptime ReturnType: type,
            endian: std.builtin.Endian,
            size: usize,
        ) !ReturnType {
            return self.reader().readVarInt(ReturnType, endian, size);
        }

        pub fn readChunk(self: *Self, amount: usize) []const u8 {
            const buf = self.buffer[self.pos .. self.pos + amount];
            self.pos += amount;
            return buf;
        }

        pub fn readf32(self: *Self) !f32 {
            return @bitCast(try self.readVarInt(u32, .little, 4));
        }
        pub fn readf64(self: *Self) !f64 {
            return @bitCast(try self.readVarInt(u64, .little, 8));
        }

        pub fn readAllocInterleavedu32(self: *Self, allocator: std.mem.Allocator, count: usize) ![]u32 {
            const chunk = self.readChunk(count * 4);
            return try ReadInterleavedu32(allocator, chunk, count);
        }
        pub fn readAllocInterleavedi32(self: *Self, allocator: std.mem.Allocator, count: usize) ![]i32 {
            const chunk = self.readChunk(count * 4);
            return try ReadInterleavedi32(allocator, chunk, count);
        }
        pub fn readAllocInstances(self: *Self, allocator: std.mem.Allocator, count: usize) ![]i32 {
            const chunk = self.readChunk(count * 4);
            const values = try ReadInterleavedi32(allocator, chunk, count);
            for (1..count) |i| {
                values[i] += values[i - 1];
            }
            return values;
        }
        pub fn readAllocInterleavedu64(self: *Self, allocator: std.mem.Allocator, count: usize) ![]u64 {
            const chunk = self.readChunk(count * 8);
            return try ReadInterleavedu64(allocator, chunk, count);
        }
        pub fn readAllocInterleavedi64(self: *Self, allocator: std.mem.Allocator, count: usize) ![]i64 {
            const chunk = self.readChunk(count * 8);
            return try ReadInterleavedi64(allocator, chunk, count);
        }

        pub fn readAllocInterleavedf32(self: *Self, allocator: std.mem.Allocator, count: usize) ![]f32 {
            const chunk = self.readChunk(count * 4);
            const list = try ReadInterleavedf32(allocator, chunk, count);
            return list;
        }

        pub fn readBuffer(self: *Self) ![]const u8 {
            const len = try self.readVarInt(u32, .little, 4);
            return self.readChunk(len);
        }
    };

    pub const Writer = struct {
        chunk_type: BinaryChunkType,
        allocator: std.mem.Allocator,
        list: std.ArrayList(u8),

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, chunk_type: BinaryChunkType) Self {
            return .{
                .chunk_type = chunk_type,
                .allocator = allocator,
                .list = std.ArrayList(u8).init(allocator),
            };
        }

        pub fn writer(self: *Self) std.ArrayList(u8).Writer {
            return self.list.writer();
        }

        pub inline fn writeAll(self: *Self, bytes: []const u8) !void {
            try writer(self).writeAll(bytes);
        }

        pub fn writeByte(self: *Self, byte: u8) !void {
            return self.list.append(byte);
        }

        pub inline fn writeIntType(self: *Self, comptime T: type, value: T) !void {
            try WriteIntType(T, writer(self), value);
        }

        pub inline fn writef32(self: *Self, value: f32) !void {
            try WriteIntType(u32, writer(self), @bitCast(value));
        }
        pub inline fn writef64(self: *Self, value: f64) !void {
            try WriteIntType(u64, writer(self), @bitCast(value));
        }

        pub fn writeBuffer(self: *Self, buf: []const u8) !void {
            var wr = writer(self);
            const string_len: [4]u8 = @bitCast(@as(u32, @intCast(buf.len)));
            try wr.writeAll(&string_len);
            try wr.writeAll(buf);
        }

        pub inline fn writeInterleavedu32(self: *Self, values: []const u32) !void {
            try WriteInterleavedu32(self.allocator, self.writer(), values);
        }
        pub inline fn writeInterleavedi32(self: *Self, values: []const i32) !void {
            try WriteInterleavedi32(self.allocator, self.writer(), values);
        }
        pub inline fn writeInterleavedu64(self: *Self, values: []const u64) !void {
            try WriteInterleavedu64(self.allocator, self.writer(), values);
        }
        pub inline fn writeInterleavedi64(self: *Self, values: []const i64) !void {
            try WriteInterleavedi64(self.allocator, self.writer(), values);
        }
        pub inline fn writeInterleavedf32(self: *Self, values: []const f32) !void {
            try WriteInterleavedf32(self.allocator, self.writer(), values);
        }
        pub inline fn writeInterleavedUniqueId(self: *Self, values: []const [16]u8) !void {
            try WriteInterleavedUniqueId(self.allocator, self.writer(), values);
        }

        pub fn finalize(self: *Self, wr: anytype, comptime compress: bool) !void {
            try wr.writeAll(switch (self.chunk_type) {
                .END => &.{ 'E', 'N', 'D', 0 },
                else => |t| @tagName(t),
            });

            const compressed = if (compress) try lz4.Standard.compress(self.allocator, self.list.items) else self.list.items;
            defer if (compress) self.allocator.free(compressed);

            try WriteIntType(u32, wr, if (compress) @as(u32, @intCast(compressed.len)) else 0);
            try WriteIntType(u32, wr, @as(u32, @intCast(self.list.items.len)));
            try WriteIntType(u32, wr, @as(u32, 0));

            try wr.writeAll(compressed);
        }

        pub fn deinit(self: *Self) void {
            self.list.deinit();
        }
    };
};

pub fn WriteIntType(comptime T: type, writer: anytype, value: T) !void {
    var buf: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &buf, value, .little);
    try writer.writeAll(&buf);
}

pub fn readArrayListSome(reader: anytype, arrayList: *std.ArrayList(u8), amount: usize) ![]const u8 {
    const before = arrayList.items.len;
    try arrayList.ensureTotalCapacityPrecise(before + amount);
    arrayList.expandToCapacity();
    const slice = arrayList.items[before..];
    const amount_read = try reader.readAll(slice);
    if (amount_read != amount) return error.Failed;
    return arrayList.items[before..];
}

pub fn readChunk(reader: anytype, bytes: *std.ArrayList(u8)) !Chunk {
    const data = try readArrayListSome(reader, bytes, 16);

    const chunk_type = data[0..4];
    const compressed_size = std.mem.readVarInt(u32, data[4..8], .little);
    const size = std.mem.readVarInt(u32, data[8..12], .little);
    const reserved = std.mem.readVarInt(u32, data[12..16], .little);

    var chunk = Chunk{
        .Type = BinaryChunkTypeMap.get(chunk_type) orelse return error.Fail,
        .Data = undefined,
        .Reserved = reserved,
        .Size = size,
    };

    if (compressed_size > 0) {
        const compressed = try readArrayListSome(reader, bytes, @intCast(compressed_size));
        if (compressed[0] == 0x58 or compressed[0] == 0x78) {
            std.debug.print("zlib: {s}\n", .{compressed});
            var writer = std.ArrayList(u8).init(bytes.allocator);
            var fixed_reader = std.io.fixedBufferStream(compressed);
            try std.compress.zlib.decompress(fixed_reader.reader(), writer.writer());
            chunk.Data = try writer.toOwnedSlice();
            chunk.Allocated = true;
        } else if (std.mem.eql(u8, compressed[1..3], &.{ 0xB5, 0x2F, 0xFD })) {
            std.debug.print("zstd: {s}\n", .{compressed});
            chunk.Data = try std.compress.zstd.decompress.decodeAlloc(bytes.allocator, compressed, true, 4096);
            chunk.Allocated = true;
        } else {
            chunk.Data = try lz4.Standard.decompress(bytes.allocator, compressed, @intCast(size));
            chunk.Allocated = true;
        }
    } else {
        chunk.Data = try readArrayListSome(reader, bytes, @intCast(size));
    }

    return chunk;
}

pub fn ReadInterleaved(comptime T: type, comptime transform: fn (*[@sizeOf(T)]u8) T) fn (allocator: std.mem.Allocator, buf: []const u8, count: usize) anyerror![]T {
    const size = @sizeOf(T);
    return struct {
        fn inner(allocator: std.mem.Allocator, buf: []const u8, count: usize) ![]T {
            const values = try allocator.alloc(T, count);
            errdefer allocator.free(values);
            if (buf.len < count * size) return error.Fail;

            for (0..count) |offset| {
                var bytes: [size]u8 = undefined;
                inline for (0..size) |i| bytes[i] = buf[(i * count) + offset];
                values[offset] = transform(&bytes);
            }

            return values;
        }
    }.inner;
}

pub fn WriteInterleaved(comptime T: type, comptime transform: fn (T) T) fn (allocator: std.mem.Allocator, writer: anytype, values: []const T) anyerror!void {
    const size = @sizeOf(T);
    const typeInfo = @typeInfo(T);
    return struct {
        fn inner(allocator: std.mem.Allocator, writer: anytype, values: []const T) !void {
            const count = values.len;
            const buf = try allocator.alloc(u8, count * size);
            defer allocator.free(buf);

            for (values, 0..) |v, i| {
                const value = transform(v);
                var bytes: [size]u8 = undefined;
                switch (typeInfo) {
                    .Int => std.mem.writeInt(T, &bytes, value, .big),
                    .Array => {
                        bytes = @bitCast(value);
                    },
                    .Float => switch (@divExact(typeInfo.Float.bits, 8)) {
                        4 => std.mem.writeInt(u32, &bytes, @bitCast(value), .big),
                        8 => std.mem.writeInt(i64, &bytes, @bitCast(value), .big),
                        else => @panic("Unsupported size"),
                    },
                    else => |u| std.debug.panic("Unsupported type: {}\n", .{u}),
                }
                const index = i * size;
                std.mem.copyForwards(u8, buf[index .. index + size], &bytes);
            }

            for (0..size) |i| {
                for (0..count) |c| try writer.writeByte(buf[(c * size) + i]);
            }
        }
    }.inner;
}

pub const ReadInterleavedu32 = ReadInterleaved(u32, tou32);
pub const ReadInterleavedi32 = ReadInterleaved(i32, rotatei32);
pub const ReadInterleavedu64 = ReadInterleaved(u64, tou64);
pub const ReadInterleavedi64 = ReadInterleaved(i64, rotatei64);
pub const ReadInterleavedf32 = ReadInterleaved(f32, rotatef32);

pub fn NoTransform(comptime T: type) fn (v: T) T {
    return struct {
        fn inner(v: T) T {
            return v;
        }
    }.inner;
}

pub const WriteInterleavedu32 = WriteInterleaved(u32, NoTransform(u32));
pub const WriteInterleavedu64 = WriteInterleaved(u64, NoTransform(u64));
pub const WriteInterleavedi32 = WriteInterleaved(i32, writeRotatei32);
pub fn writeRotatei32(value: i32) i32 {
    return (value << 1) ^ (value >> 31);
}
pub const WriteInterleavedi64 = WriteInterleaved(i64, writeRotatei64);
pub fn writeRotatei64(value: i64) i64 {
    return (value << 1) ^ (value >> 63);
}
pub const WriteInterleavedf32 = WriteInterleaved(f32, writeRotatef32);
pub fn writeRotatef32(value: f32) f32 {
    const uint: u32 = @bitCast(value);
    return @bitCast((uint << 1) | (uint >> 31));
}

pub const WriteInterleavedUniqueId = WriteInterleaved([16]u8, NoTransform([16]u8));

pub fn tou32(buf: *[4]u8) u32 {
    return std.mem.readVarInt(u32, buf, .big);
}
pub fn rotatei32(buf: *[4]u8) i32 {
    const v = std.mem.readVarInt(i32, buf, .big);
    return @as(i32, @intCast(@as(u32, @intCast(v)) >> 1)) ^ -(v & 1);
}
pub fn tou64(buf: *[8]u8) u64 {
    return std.mem.readVarInt(u64, buf, .big);
}
pub fn rotatei64(buf: *[8]u8) i64 {
    const v = std.mem.readVarInt(i64, buf, .big);
    return @as(i64, @intCast(@as(u64, @bitCast(v)) >> 1)) ^ -(v & 1);
}
pub fn rotatef32(buf: *[4]u8) f32 {
    const u = std.mem.readVarInt(u32, buf, .big);
    const i = (u >> 1) | (u << 31);
    return @bitCast(@as(i32, @bitCast(i)));
}

pub fn serialize(
    allocator: std.mem.Allocator,
    instances: std.ArrayList(Roblox.Instance),
    info: Roblox.DocumentInfo,
    metadata: std.StringHashMap([]const u8),
    sharedstrings: std.ArrayList(Roblox.SharedString),
    signatures: ?[]Roblox.Signature,
) ![]const u8 {
    var class_len: usize = 0;
    var instances_size: u32 = 0;

    var using_instances = std.ArrayList(Roblox.Instance).init(allocator);
    defer using_instances.deinit();

    var class_map = std.StringArrayHashMap(Roblox.Class).init(allocator);
    defer class_map.deinit();
    defer {
        var class_iter = class_map.iterator();
        while (class_iter.next()) |entry| {
            allocator.free(entry.value_ptr.instances);
        }
    }

    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    const bufwriter = buffer.writer();

    for (instances.items) |item| {
        if (item.Properties.get("Archivable")) |prop| switch (prop.Value) {
            .Bool => |b| if (b) continue,
            else => {},
        };

        var existing_class = class_map.getPtr(item.ClassName);
        if (existing_class == null) {
            var count: usize = 0;
            for (instances.items) |v| {
                if (std.mem.eql(u8, v.ClassName, item.ClassName)) count += 1;
            }

            try class_map.put(item.ClassName, Roblox.Class{
                .index = class_len,
                .name = item.ClassName,
                .is_service = item.IsService,
                .instances = try allocator.alloc(i32, count),
                .instances_len = 0,
            });

            existing_class = class_map.getPtr(item.ClassName) orelse unreachable;

            class_len += 1;
        }

        existing_class.?.instances[existing_class.?.instances_len] = item.Referent;
        existing_class.?.instances_len += 1;

        try using_instances.append(item);
        instances_size += 1;
    }

    try bufwriter.writeAll(BinaryMagicHeader);

    const version_buf: [2]u8 = @bitCast(info.version);
    try bufwriter.writeAll(&version_buf);

    try WriteIntType(u32, bufwriter, @as(u32, @intCast(class_len)));
    try WriteIntType(u32, bufwriter, @as(u32, @intCast(instances_size)));

    try bufwriter.writeAll(&.{ 0, 0, 0, 0, 0, 0, 0, 0 });

    if (metadata.count() > 0) {
        var chunk = Chunk.Writer.init(allocator, .META);
        defer chunk.deinit();

        try chunk.writeIntType(u32, @intCast(metadata.count()));

        var meta_iter = metadata.iterator();
        while (meta_iter.next()) |meta_entry| {
            try chunk.writeBuffer(meta_entry.key_ptr.*);
            try chunk.writeBuffer(meta_entry.value_ptr.*);
        }

        try chunk.finalize(bufwriter, true);
    }

    {
        var chunk = Chunk.Writer.init(allocator, .SSTR);
        defer chunk.deinit();

        try chunk.writeIntType(u32, 0);
        try chunk.writeIntType(u32, @intCast(sharedstrings.items.len));

        for (sharedstrings.items) |*shared| {
            try chunk.writeAll(&[_]u8{ 0, 0, 0, 0 } ** 4);

            if (shared.find(shared.Key)) |string| try chunk.writeBuffer(string) else @panic("Missing string");
        }

        try chunk.finalize(bufwriter, true);
    }

    if (signatures) |signs| {
        var chunk = Chunk.Writer.init(allocator, .SIGN);
        defer chunk.deinit();

        try chunk.writeIntType(u32, @intCast(signs.len));

        for (signs) |sign| {
            try chunk.writeIntType(i32, @intFromEnum(sign.Type));
            try chunk.writeIntType(i64, sign.PublicKey);
            try chunk.writeBuffer(sign.Value);
        }

        try chunk.finalize(bufwriter, true);
    }

    var class_iter = class_map.iterator();
    while (class_iter.next()) |entry| {
        if (entry.value_ptr.instances_len == 0) continue;

        var chunk = Chunk.Writer.init(allocator, .INST);
        defer chunk.deinit();

        try chunk.writeIntType(u32, @as(u32, @intCast(entry.value_ptr.index)));
        try chunk.writeBuffer(entry.value_ptr.name);
        try chunk.writeByte(if (entry.value_ptr.is_service) 1 else 0);
        try chunk.writeIntType(u32, @as(u32, @intCast(entry.value_ptr.instances_len)));

        const instance_ids = try allocator.alloc(i32, entry.value_ptr.instances_len);
        defer allocator.free(instance_ids);
        instance_ids[0] = entry.value_ptr.instances[0];
        for (1..entry.value_ptr.instances_len) |i| {
            instance_ids[i] = entry.value_ptr.instances[i];
            instance_ids[i] -= entry.value_ptr.instances[i - 1];
        }

        try chunk.writeInterleavedi32(instance_ids);

        if (entry.value_ptr.is_service) {
            for (entry.value_ptr.instances) |id| {
                const instance = instances.items[@intCast(id)];

                try chunk.writeByte(if (instance.Parent == null) 1 else 0);
            }
        }

        try chunk.finalize(bufwriter, true);
    }

    class_iter = class_map.iterator();
    while (class_iter.next()) |entry| {
        if (entry.value_ptr.instances_len == 0) continue;
        const instance = instances.items[@intCast(entry.value_ptr.instances[0])];
        var prop_iter = instance.Properties.iterator();
        while (prop_iter.next()) |prop_entry| {
            if (std.mem.eql(u8, prop_entry.key_ptr.*, "Archivable")) continue;
            if (std.mem.indexOfPosLinear(u8, prop_entry.key_ptr.*, 0, "__") != null) continue;

            var chunk = Chunk.Writer.init(allocator, .PROP);
            defer chunk.deinit();

            var properties = std.ArrayList(Roblox.Property).init(allocator);
            defer properties.deinit();

            try chunk.writeIntType(u32, @as(u32, @intCast(entry.value_ptr.index)));
            try chunk.writeBuffer(prop_entry.key_ptr.*);
            const prop_typeid: u8 = @intFromEnum(prop_entry.value_ptr.*.Type);
            try chunk.writeByte(if (prop_typeid >= 15) prop_typeid + 1 else prop_typeid);

            const property_values = try allocator.alloc(Roblox.Property, entry.value_ptr.instances_len);
            defer allocator.free(property_values);
            for (entry.value_ptr.instances, 0..) |id, o| {
                const sub_instance = instances.items[@intCast(id)];
                const prop_value = sub_instance.Properties.getPtr(prop_entry.key_ptr.*) orelse std.debug.panic("Missing property", .{});
                if (@intFromEnum(prop_value.*.Type) != prop_typeid) std.debug.panic("Invalid property type", .{});
                property_values[o] = prop_value.*;
            }

            switch (prop_entry.value_ptr.Type) {
                .String => {
                    for (property_values) |value| try chunk.writeBuffer(value.Value.String);
                },
                .Bool => {
                    for (property_values) |value| try chunk.writeByte(if (value.Value.Bool) 1 else 0);
                },
                .Int => {
                    const values = try allocator.alloc(i32, property_values.len);
                    defer allocator.free(values);
                    for (property_values, 0..) |value, i| values[i] = value.Value.Int;
                    try chunk.writeInterleavedi32(values);
                },
                .Float => {
                    const values = try allocator.alloc(f32, property_values.len);
                    defer allocator.free(values);
                    for (property_values, 0..) |value, i| values[i] = value.Value.Float;
                    try chunk.writeInterleavedf32(values);
                },
                .Double => {
                    for (property_values) |value| try chunk.writef64(value.Value.Double);
                },
                .UDim => {
                    const scale_values = try allocator.alloc(f32, property_values.len);
                    defer allocator.free(scale_values);
                    const offset_values = try allocator.alloc(i32, property_values.len);
                    defer allocator.free(offset_values);
                    for (property_values, 0..) |value, i| {
                        const scale, const offset = value.Value.UDim;
                        scale_values[i] = scale;
                        offset_values[i] = offset;
                    }
                    try chunk.writeInterleavedf32(scale_values);
                    try chunk.writeInterleavedi32(offset_values);
                },
                .UDim2 => {
                    const scale_x_values = try allocator.alloc(f32, property_values.len);
                    defer allocator.free(scale_x_values);
                    const scale_y_values = try allocator.alloc(f32, property_values.len);
                    defer allocator.free(scale_y_values);

                    const offset_x_values = try allocator.alloc(i32, property_values.len);
                    defer allocator.free(offset_x_values);
                    const offset_y_values = try allocator.alloc(i32, property_values.len);
                    defer allocator.free(offset_y_values);

                    for (property_values, 0..) |value, i| {
                        const scale_x, const offset_x, const scale_y, const offset_y = value.Value.UDim2;
                        scale_x_values[i] = scale_x;
                        scale_y_values[i] = scale_y;
                        offset_x_values[i] = offset_x;
                        offset_y_values[i] = offset_y;
                    }
                    try chunk.writeInterleavedf32(scale_x_values);
                    try chunk.writeInterleavedf32(scale_y_values);
                    try chunk.writeInterleavedi32(offset_x_values);
                    try chunk.writeInterleavedi32(offset_y_values);
                },
                .Ray => {
                    for (property_values) |value| {
                        const pos_x, const pos_y, const pos_z, const dir_x, const dir_y, const dir_z = value.Value.Ray;

                        try chunk.writef32(pos_x);
                        try chunk.writef32(pos_y);
                        try chunk.writef32(pos_z);
                        try chunk.writef32(dir_x);
                        try chunk.writef32(dir_y);
                        try chunk.writef32(dir_z);
                    }
                },
                .Faces => {
                    for (property_values) |value| try chunk.writeByte(value.Value.Faces);
                },
                .Axes => {
                    for (property_values) |value| try chunk.writeByte(value.Value.Axes);
                },
                .BrickColor => {
                    const values = try allocator.alloc(i32, property_values.len);
                    defer allocator.free(values);
                    for (property_values, 0..) |value, i| values[i] = value.Value.BrickColor;
                    try chunk.writeInterleavedi32(values);
                },
                .Color3 => {
                    const r_values = try allocator.alloc(f32, property_values.len);
                    defer allocator.free(r_values);
                    const g_values = try allocator.alloc(f32, property_values.len);
                    defer allocator.free(g_values);
                    const b_values = try allocator.alloc(f32, property_values.len);
                    defer allocator.free(b_values);
                    for (property_values, 0..) |value, i| {
                        const r, const g, const b = value.Value.Color3;
                        r_values[i] = r;
                        g_values[i] = g;
                        b_values[i] = b;
                    }
                    try chunk.writeInterleavedf32(r_values);
                    try chunk.writeInterleavedf32(g_values);
                    try chunk.writeInterleavedf32(b_values);
                },
                .Vector2 => {
                    const x_values = try allocator.alloc(f32, property_values.len);
                    defer allocator.free(x_values);
                    const y_values = try allocator.alloc(f32, property_values.len);
                    defer allocator.free(y_values);
                    for (property_values, 0..) |value, i| {
                        const x, const y = value.Value.Vector2;
                        x_values[i] = x;
                        y_values[i] = y;
                    }
                    try chunk.writeInterleavedf32(x_values);
                    try chunk.writeInterleavedf32(y_values);
                },
                .Vector3 => {
                    const x_values = try allocator.alloc(f32, property_values.len);
                    defer allocator.free(x_values);
                    const y_values = try allocator.alloc(f32, property_values.len);
                    defer allocator.free(y_values);
                    const z_values = try allocator.alloc(f32, property_values.len);
                    defer allocator.free(z_values);
                    for (property_values, 0..) |value, i| {
                        const x, const y, const z = value.Value.Vector3;
                        x_values[i] = x;
                        y_values[i] = y;
                        z_values[i] = z;
                    }
                    try chunk.writeInterleavedf32(x_values);
                    try chunk.writeInterleavedf32(y_values);
                    try chunk.writeInterleavedf32(z_values);
                },
                .CFrame, .Quaternion, .OptionalCFrame => |t| {
                    const x_values = try allocator.alloc(f32, property_values.len);
                    defer allocator.free(x_values);
                    const y_values = try allocator.alloc(f32, property_values.len);
                    defer allocator.free(y_values);
                    const z_values = try allocator.alloc(f32, property_values.len);
                    defer allocator.free(z_values);

                    if (t == .OptionalCFrame) try chunk.writeByte(@intFromEnum(Roblox.Property.Type.CFrame) + 1);

                    for (property_values, 0..) |value, i| {
                        const cf_value: struct { f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32 } = cf: {
                            if (t == .OptionalCFrame) {
                                if (value.Value.OptionalCFrame) |cf| break :cf cf;
                            } else break :cf value.Value.CFrame;
                            break :cf .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                        };
                        const x, const y, const z, const rx0, const ry0, const rz0, const rx1, const ry1, const rz1, const rx2, const ry2, const rz2 = cf_value;
                        x_values[i] = x;
                        y_values[i] = y;
                        z_values[i] = z;

                        try chunk.writeByte(0);

                        if (t == .Quaternion) {
                            const qt = (rx0 + ry1 + rz2);
                            if (qt > 0) {
                                const s = std.math.sqrt(qt + 1);
                                const r = 0.5 / s;

                                try chunk.writef32((ry2 - rz1) * r);
                                try chunk.writef32((rz0 - rx2) * r);
                                try chunk.writef32((rx1 - ry0) * r);
                                try chunk.writef32(s * 0.5);
                            } else {
                                const big = @max(@max(rx0, ry1), rz2);
                                if (big == rx0) {
                                    const s = std.math.sqrt(1 + rx0 - ry1 - rz2);
                                    const r = 0.5 / s;

                                    try chunk.writef32(0.5 * s);
                                    try chunk.writef32((rx1 + ry0) * r);
                                    try chunk.writef32((rz0 + rx2) * r);
                                    try chunk.writef32((ry2 - rz1) * r);
                                } else if (big == ry1) {
                                    const s = std.math.sqrt(1 - rx0 + ry1 - rz2);
                                    const r = 0.5 / s;

                                    try chunk.writef32((rx1 + ry0) * r);
                                    try chunk.writef32(0.5 * s);
                                    try chunk.writef32((ry2 + rz1) * r);
                                    try chunk.writef32((rz0 - rx2) * r);
                                } else if (big == rz2) {
                                    const s = std.math.sqrt(1 - rx0 - ry1 + rz2);
                                    const r = 0.5 / s;

                                    try chunk.writef32((rz0 + rx2) * r);
                                    try chunk.writef32((ry2 + rz1) * r);
                                    try chunk.writef32(0.5 * s);
                                    try chunk.writef32((rx1 - ry0) * r);
                                }
                            }
                        } else {
                            try chunk.writef32(rx0);
                            try chunk.writef32(ry0);
                            try chunk.writef32(rz0);
                            try chunk.writef32(rx1);
                            try chunk.writef32(ry1);
                            try chunk.writef32(rz1);
                            try chunk.writef32(rx2);
                            try chunk.writef32(ry2);
                            try chunk.writef32(rz2);
                        }
                    }

                    try chunk.writeInterleavedf32(x_values);
                    try chunk.writeInterleavedf32(y_values);
                    try chunk.writeInterleavedf32(z_values);

                    if (t == .OptionalCFrame) {
                        try chunk.writeByte(@intFromEnum(Roblox.Property.Type.Bool));
                        for (property_values) |value| try chunk.writeByte(if (value.Value.OptionalCFrame != null) 1 else 0);
                    }
                },
                .Enum => {
                    const values = try allocator.alloc(u32, property_values.len);
                    defer allocator.free(values);
                    for (property_values, 0..) |value, i| {
                        const val, _ = value.Value.Enum;
                        values[i] = val;
                    }
                    try chunk.writeInterleavedu32(values);
                },
                .Ref => {
                    const values = try allocator.alloc(i32, property_values.len);
                    defer allocator.free(values);
                    for (property_values, 0..) |value, i| {
                        if (value.Value.Ref) |ref| values[i] = @intCast(ref) else values[i] = -1;
                    }
                    const final_values = try allocator.dupe(i32, values);
                    defer allocator.free(final_values);
                    for (1..final_values.len) |i| final_values[i] -= values[i - 1];
                    try chunk.writeInterleavedi32(final_values);
                },
                .Vector3int16 => {
                    for (property_values) |value| {
                        const x, const y, const z = value.Value.Vector3int16;

                        try chunk.writeIntType(i16, x);
                        try chunk.writeIntType(i16, y);
                        try chunk.writeIntType(i16, z);
                    }
                },
                .NumberSequence => {
                    for (property_values) |prop| {
                        const points = prop.Value.NumberSequence;
                        try chunk.writeIntType(u32, @intCast(points.len));
                        for (points) |point| {
                            const time, const value, const envelope = point;
                            try chunk.writef32(time);
                            try chunk.writef32(value);
                            try chunk.writef32(envelope);
                        }
                    }
                },
                .ColorSequence => {
                    for (property_values) |prop| {
                        const points = prop.Value.ColorSequence;
                        try chunk.writeIntType(u32, @intCast(points.len));
                        for (points) |point| {
                            const time, const r, const g, const b, const envelope = point;
                            try chunk.writef32(time);
                            try chunk.writef32(r);
                            try chunk.writef32(g);
                            try chunk.writef32(b);
                            try chunk.writeIntType(i32, envelope);
                        }
                    }
                },
                .NumberRange => {
                    for (property_values) |prop| {
                        const min, const max = prop.Value.NumberRange;
                        try chunk.writef32(min);
                        try chunk.writef32(max);
                    }
                },
                .Rect => {
                    const x0_values = try allocator.alloc(f32, property_values.len);
                    defer allocator.free(x0_values);
                    const y0_values = try allocator.alloc(f32, property_values.len);
                    defer allocator.free(y0_values);
                    const x1_values = try allocator.alloc(f32, property_values.len);
                    defer allocator.free(x1_values);
                    const y1_values = try allocator.alloc(f32, property_values.len);
                    defer allocator.free(y1_values);
                    for (property_values, 0..) |value, i| {
                        const x0, const y0, const x1, const y1 = value.Value.Rect;
                        x0_values[i] = x0;
                        y0_values[i] = y0;
                        x1_values[i] = x1;
                        y1_values[i] = y1;
                    }
                    try chunk.writeInterleavedf32(x0_values);
                    try chunk.writeInterleavedf32(y0_values);
                    try chunk.writeInterleavedf32(x1_values);
                    try chunk.writeInterleavedf32(y1_values);
                },
                .PhysicalProperties => {
                    for (property_values) |value| {
                        if (value.Value.PhysicalProperties) |physical| {
                            try chunk.writeByte(1);
                            const density, const friction, const elasticity, const friction_weight, const elasticity_weight = physical;
                            try chunk.writef32(density);
                            try chunk.writef32(friction);
                            try chunk.writef32(elasticity);
                            try chunk.writef32(friction_weight);
                            try chunk.writef32(elasticity_weight);
                        } else try chunk.writeByte(0);
                    }
                },
                .Color3uint8 => {
                    const r_values = try allocator.alloc(u8, property_values.len);
                    defer allocator.free(r_values);
                    const g_values = try allocator.alloc(u8, property_values.len);
                    defer allocator.free(g_values);
                    const b_values = try allocator.alloc(u8, property_values.len);
                    defer allocator.free(b_values);
                    for (property_values, 0..) |value, i| {
                        const r, const g, const b = value.Value.Color3uint8;
                        r_values[i] = r;
                        g_values[i] = g;
                        b_values[i] = b;
                    }
                    try chunk.writeAll(r_values);
                    try chunk.writeAll(g_values);
                    try chunk.writeAll(b_values);
                },
                .Int64 => {
                    const values = try allocator.alloc(i64, property_values.len);
                    defer allocator.free(values);
                    for (property_values, 0..) |value, i| values[i] = value.Value.Int64;
                    try chunk.writeInterleavedi64(values);
                },
                .SharedString => {
                    const values = try allocator.alloc(u32, property_values.len);
                    defer allocator.free(values);
                    for (property_values, 0..) |value, i| values[i] = value.Value.SharedString;
                    try chunk.writeInterleavedu32(values);
                },
                .ProtectedString => {
                    for (property_values) |value| try chunk.writeBuffer(value.Value.ProtectedString);
                },
                .UniqueId => {
                    const values = try allocator.alloc([16]u8, property_values.len);
                    defer allocator.free(values);

                    for (property_values, 0..) |value, i| {
                        var ids: [16]u8 = undefined;
                        const random, const time, const index = value.Value.UniqueId;
                        @memcpy(ids[0..4], &@as([4]u8, @bitCast(random)));
                        @memcpy(ids[4..8], &@as([4]u8, @bitCast(time)));
                        @memcpy(ids[8..16], &@as([8]u8, @bitCast(index)));
                        values[i] = ids;
                    }

                    try chunk.writeInterleavedUniqueId(values);
                },
                .FontFace => {
                    for (property_values) |value| {
                        const family, const weight, const style, const cached = value.Value.FontFace;
                        try chunk.writeBuffer(family);
                        try chunk.writeIntType(u16, weight);
                        try chunk.writeByte(style);
                        try chunk.writeBuffer(cached);
                    }
                },
                .SecurityCapabilities => {
                    const values = try allocator.alloc(u64, property_values.len);
                    defer allocator.free(values);
                    for (property_values, 0..) |value, i| values[i] = value.Value.SecurityCapabilities;
                    try chunk.writeInterleavedu64(values);
                },
                else => |t| std.debug.panic("(serialize) Unhandled type: {}\n", .{t}),
            }

            try chunk.finalize(bufwriter, true);
        }
    }

    class_iter = class_map.iterator();
    {
        var chunk = Chunk.Writer.init(allocator, .PRNT);
        defer chunk.deinit();

        const child_ids = try allocator.alloc(i32, @intCast(using_instances.items.len));
        defer allocator.free(child_ids);
        const parent_ids = try allocator.alloc(i32, @intCast(using_instances.items.len));
        defer allocator.free(parent_ids);

        for (using_instances.items, 0..) |instance, i| {
            child_ids[i] = instance.Referent;
            if (instance.Parent) |parent| parent_ids[i] = @intCast(parent) else parent_ids[i] = -1;
        }

        const final_child_ids = try allocator.dupe(i32, child_ids);
        defer allocator.free(final_child_ids);
        const final_parent_ids = try allocator.dupe(i32, parent_ids);
        defer allocator.free(final_parent_ids);

        for (1..child_ids.len) |i| final_child_ids[i] -= child_ids[i - 1];
        for (1..parent_ids.len) |i| final_parent_ids[i] -= parent_ids[i - 1];

        try chunk.writeByte(0);
        try chunk.writeIntType(u32, @intCast(child_ids.len));

        try chunk.writeInterleavedi32(final_child_ids);
        try chunk.writeInterleavedi32(final_parent_ids);

        try chunk.finalize(bufwriter, true);
    }

    {
        var chunk = Chunk.Writer.init(allocator, .END);
        defer chunk.deinit();

        try chunk.writeAll("</roblox>");

        try chunk.finalize(bufwriter, false);
    }

    return try buffer.toOwnedSlice();
}

pub fn deserialize(allocator: std.mem.Allocator, reader: anytype) !Roblox.Document {
    var array = std.ArrayList(u8).init(allocator);

    defer array.deinit();

    const bdata = try readArrayListSome(reader, &array, 18);

    const version = std.mem.readVarInt(u16, bdata[0..2], .little);
    const classes = std.mem.readVarInt(u32, bdata[2..6], .little);
    const instances = std.mem.readVarInt(u32, bdata[6..10], .little);
    const reserved = std.mem.readVarInt(i64, bdata[10..18], .little);

    var doc = try Roblox.Document.init(allocator, .{
        .classes = classes,
        .version = version,
        .reserved = reserved,
        .instances = instances,
    });
    errdefer doc.deinit();

    // std.debug.print("doc: {any}\n", .{array.items[0..2]});
    // std.debug.print("doc: {}, {}, {}, {}\n", .{
    //     version,
    //     classes,
    //     instances,
    //     reserved,
    // });

    var ended = false;
    while (true) {
        const chunk = try readChunk(reader, &array);
        defer chunk.deinit(allocator);

        var chunk_reader = Chunk.Reader{
            .buffer = chunk.Data,
            .pos = 0,
        };
        switch (chunk.Type) {
            .INST => {
                const class_index = try chunk_reader.readVarInt(u32, .little, 4);
                const class_name_len = try chunk_reader.readVarInt(u32, .little, 4);
                const class_name = chunk_reader.readChunk(class_name_len);
                const is_service = try chunk_reader.readByte() != 0;

                const instances_len = try chunk_reader.readVarInt(u32, .little, 4);

                const list = try chunk_reader.readAllocInstances(allocator, instances_len);
                errdefer allocator.free(list);

                for (0..@intCast(instances_len)) |i| {
                    const id = list[i];
                    var inst = Roblox.Instance{
                        .Referent = id,
                        .IsService = is_service,
                        .ClassName = try allocator.dupe(u8, class_name),
                        .Parent = null,
                        .Childs = std.ArrayList(*Roblox.Instance).init(allocator),
                        .Properties = std.StringArrayHashMap(Roblox.Property).init(allocator),
                    };
                    errdefer allocator.free(inst.ClassName);
                    errdefer inst.Properties.deinit();
                    if (is_service) {
                        inst.Parent = if (try chunk_reader.readByte() != 0) null else null;
                    }
                    doc.instances.items[@intCast(id)] = inst;
                }

                doc.classes.items[class_index] = .{
                    .index = class_index,
                    .instances_len = instances_len,
                    .instances = list,
                    .name = try allocator.dupe(u8, class_name),
                    .is_service = is_service,
                };
            },
            .META => {
                for (0..@intCast(try chunk_reader.readVarInt(u32, .little, 4))) |_| {
                    const key = chunk_reader.readChunk(@intCast(try chunk_reader.readVarInt(u32, .little, 4)));
                    const value = chunk_reader.readChunk(@intCast(try chunk_reader.readVarInt(u32, .little, 4)));
                    const key_copy = try allocator.dupe(u8, key);
                    errdefer allocator.free(key_copy);
                    const value_copy = try allocator.dupe(u8, value);
                    errdefer allocator.free(value_copy);
                    try doc.metadata.put(key_copy, value_copy);
                }
            },
            .PRNT => {
                const format = try chunk_reader.readByte();
                const count: usize = @intCast(try chunk_reader.readVarInt(u32, .little, 4));

                if (format != 0) return DeserializeError.BadParentFormat;

                const childs_list = try chunk_reader.readAllocInstances(allocator, count);
                const parents_list = try chunk_reader.readAllocInstances(allocator, count);
                defer allocator.free(childs_list);
                defer allocator.free(parents_list);

                for (0..count) |i| {
                    const child_id = childs_list[i];
                    const parent_id = parents_list[i];
                    if (parent_id >= 0) doc.instances.items[@intCast(child_id)].Parent = @intCast(parent_id);
                }
            },
            .PROP => {
                const class_index = try chunk_reader.readVarInt(u32, .little, 4);
                const name = try chunk_reader.readBuffer();

                const class = doc.classes.items[class_index];

                const type_byte = try chunk_reader.readByte();
                const property_type: Roblox.Property.Type = @enumFromInt(if (type_byte > 15) type_byte - 1 else type_byte);

                const len = class.instances.len;
                const values = try allocator.alloc(Roblox.Value, len);
                defer allocator.free(values);
                errdefer for (values) |rbx_value| {
                    switch (rbx_value) {
                        .ProtectedString, .String => |s| allocator.free(s),
                        .FontFace => |a| {
                            const family, _, _, const cached = a;
                            allocator.free(family);
                            allocator.free(cached);
                        },
                        .ColorSequence => |a| allocator.free(a),
                        .NumberSequence => |a| allocator.free(a),
                        else => {},
                    }
                };

                switch (property_type) {
                    .String => {
                        for (0..len) |i| values[i] = .{ .String = try allocator.dupe(u8, try chunk_reader.readBuffer()) };
                    },
                    .Bool => {
                        for (0..len) |i| values[i] = .{ .Bool = try chunk_reader.readByte() != 0 };
                    },
                    .Int => {
                        const list = try chunk_reader.readAllocInterleavedi32(allocator, len);
                        defer allocator.free(list);
                        for (0..len) |i| values[i] = .{ .Int = list[i] };
                    },
                    .Float => {
                        const list = try chunk_reader.readAllocInterleavedf32(allocator, len);
                        defer allocator.free(list);
                        for (0..len) |i| values[i] = .{ .Float = list[i] };
                    },
                    .Double => {
                        for (0..len) |i| values[i] = .{ .Double = try chunk_reader.readf64() };
                    },
                    .UDim => {
                        const scales = try chunk_reader.readAllocInterleavedf32(allocator, len);
                        defer allocator.free(scales);
                        const offsets = try chunk_reader.readAllocInterleavedi32(allocator, len);
                        defer allocator.free(offsets);

                        for (0..len) |i| values[i] = .{ .UDim = .{ scales[i], offsets[i] } };
                    },
                    .UDim2 => {
                        const scales_x = try chunk_reader.readAllocInterleavedf32(allocator, len);
                        defer allocator.free(scales_x);
                        const scales_y = try chunk_reader.readAllocInterleavedf32(allocator, len);
                        defer allocator.free(scales_y);

                        const offsets_x = try chunk_reader.readAllocInterleavedi32(allocator, len);
                        defer allocator.free(offsets_x);
                        const offsets_y = try chunk_reader.readAllocInterleavedi32(allocator, len);
                        defer allocator.free(offsets_y);

                        for (0..len) |i| values[i] = .{ .UDim2 = .{ scales_x[i], offsets_x[i], scales_y[i], offsets_y[i] } };
                    },
                    .Ray => {
                        for (0..len) |i| {
                            const pos_x = try chunk_reader.readf32();
                            const pos_y = try chunk_reader.readf32();
                            const pos_z = try chunk_reader.readf32();

                            const dir_x = try chunk_reader.readf32();
                            const dir_y = try chunk_reader.readf32();
                            const dir_z = try chunk_reader.readf32();
                            values[i] = .{ .Ray = .{ pos_x, pos_y, pos_z, dir_x, dir_y, dir_z } };
                        }
                    },
                    .Faces => {
                        for (0..len) |i| values[i] = .{ .Faces = try chunk_reader.readByte() };
                    },
                    .Axes => {
                        for (0..len) |i| values[i] = .{ .Axes = try chunk_reader.readByte() };
                    },
                    .BrickColor => {
                        const list = try chunk_reader.readAllocInterleavedi32(allocator, len);
                        defer allocator.free(list);
                        for (0..len) |i| values[i] = .{ .BrickColor = list[i] };
                    },
                    .Color3 => {
                        const r = try chunk_reader.readAllocInterleavedf32(allocator, len);
                        defer allocator.free(r);
                        const g = try chunk_reader.readAllocInterleavedf32(allocator, len);
                        defer allocator.free(g);
                        const b = try chunk_reader.readAllocInterleavedf32(allocator, len);
                        defer allocator.free(b);
                        for (0..len) |i| values[i] = .{ .Color3 = .{ r[i], g[i], b[i] } };
                    },
                    .Vector2 => {
                        const x = try chunk_reader.readAllocInterleavedf32(allocator, len);
                        defer allocator.free(x);
                        const y = try chunk_reader.readAllocInterleavedf32(allocator, len);
                        defer allocator.free(y);
                        for (0..len) |i| values[i] = .{ .Vector2 = .{ x[i], y[i] } };
                    },
                    .Vector3 => {
                        const x = try chunk_reader.readAllocInterleavedf32(allocator, len);
                        defer allocator.free(x);
                        const y = try chunk_reader.readAllocInterleavedf32(allocator, len);
                        defer allocator.free(y);
                        const z = try chunk_reader.readAllocInterleavedf32(allocator, len);
                        defer allocator.free(z);
                        for (0..len) |i| values[i] = .{ .Vector3 = .{ x[i], y[i], z[i] } };
                    },
                    .CFrame, .Quaternion, .OptionalCFrame => |t| {
                        if (t == .OptionalCFrame and try chunk_reader.readByte() != @intFromEnum(Roblox.Property.Type.CFrame) + 1) {
                            for (0..len) |i| values[i] = .{ .OptionalCFrame = null };
                        } else {
                            const matrices = try allocator.alloc(struct { f32, f32, f32, f32, f32, f32, f32, f32, f32 }, len);
                            defer allocator.free(matrices);
                            for (0..len) |i| {
                                const orient_id: u8 = try chunk_reader.readByte();
                                if (orient_id > 0) {
                                    switch (orient_id) {
                                        0x02 => matrices[i] = .{ 1, 0, 0, 0, 1, 0, 0, 0, 1 },
                                        0x03 => matrices[i] = .{ 1, 0, 0, 0, 0, -1, 0, 1, 0 },
                                        0x05 => matrices[i] = .{ 1, 0, 0, 0, -1, 0, 0, 0, -1 },
                                        0x06 => matrices[i] = .{ 1, 0, 0, 0, 0, 1, 0, -1, 0 },
                                        0x07 => matrices[i] = .{ 0, 1, 0, 1, 0, 0, 0, 0, -1 },
                                        0x09 => matrices[i] = .{ 0, 0, 1, 1, 0, 0, 0, 1, 0 },
                                        0x0a => matrices[i] = .{ 0, -1, 0, 1, 0, 0, 0, 0, 1 },
                                        0x0c => matrices[i] = .{ 0, 0, -1, 1, 0, 0, 0, -1, 0 },
                                        0x0d => matrices[i] = .{ 0, 1, 0, 0, 0, 1, 1, 0, 0 },
                                        0x0e => matrices[i] = .{ 0, 0, -1, 0, 1, 0, 1, 0, 0 },
                                        0x10 => matrices[i] = .{ 0, -1, 0, 0, 0, -1, 1, 0, 0 },
                                        0x11 => matrices[i] = .{ 0, 0, 1, 0, -1, 0, 1, 0, 0 },
                                        0x14 => matrices[i] = .{ -1, 0, 0, 0, 1, 0, 0, 0, -1 },
                                        0x15 => matrices[i] = .{ -1, 0, 0, 0, 0, 1, 0, 1, 0 },
                                        0x17 => matrices[i] = .{ -1, 0, 0, 0, -1, 0, 0, 0, 1 },
                                        0x18 => matrices[i] = .{ -1, 0, 0, 0, 0, -1, 0, -1, 0 },
                                        0x19 => matrices[i] = .{ 0, 1, 0, -1, 0, 0, 0, 0, 1 },
                                        0x1b => matrices[i] = .{ 0, 0, 1, -1, 0, 0, 0, 1, 0 },
                                        0x1c => matrices[i] = .{ 0, -1, 0, -1, 0, 0, 0, 0, -1 },
                                        0x1e => matrices[i] = .{ 0, 0, -1, -1, 0, 0, 0, -1, 0 },
                                        0x1f => matrices[i] = .{ 0, 1, 0, 0, 0, -1, -1, 0, 0 },
                                        0x20 => matrices[i] = .{ 0, 0, 1, 0, 1, 0, -1, 0, 0 },
                                        0x22 => matrices[i] = .{ 0, -1, 0, 0, 0, 1, -1, 0, 0 },
                                        0x23 => matrices[i] = .{ 0, 0, -1, 0, -1, 0, -1, 0, 0 },
                                        else => return DeserializeError.BadCFrameRotationId,
                                    }
                                } else if (t == .Quaternion) {
                                    const qx = try chunk_reader.readf32();
                                    const qy = try chunk_reader.readf32();
                                    const qz = try chunk_reader.readf32();
                                    const qw = try chunk_reader.readf32();

                                    const xc = qx * 2;
                                    const yc = qy * 2;
                                    const zc = qz * 2;

                                    const xx = qx * xc;
                                    const xy = qx * yc;
                                    const xz = qx * zc;

                                    const wx = qw * xc;
                                    const wy = qw * yc;
                                    const wz = qw * zc;

                                    const yy = qy * yc;
                                    const yz = qy * zc;
                                    const zz = qz * zc;

                                    matrices[i] = .{ 1 - (yy + zz), xy - wz, xz + wy, xy + wz, 1 - (xx + zz), yz - wx, xz - wy, yz + wx, 1 - (xx + yy) };
                                } else {
                                    var matrix: [9]f32 = .{ 0, 0, 0 } ** 3;
                                    for (0..9) |m| matrix[m] = try chunk_reader.readf32();
                                    matrices[i] = .{ matrix[0], matrix[1], matrix[2], matrix[3], matrix[4], matrix[5], matrix[6], matrix[7], matrix[8] };
                                }
                            }

                            const cx = try chunk_reader.readAllocInterleavedf32(allocator, len);
                            defer allocator.free(cx);
                            const cy = try chunk_reader.readAllocInterleavedf32(allocator, len);
                            defer allocator.free(cy);
                            const cz = try chunk_reader.readAllocInterleavedf32(allocator, len);
                            defer allocator.free(cz);

                            const cframes = try allocator.alloc(struct { f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32 }, len);
                            defer allocator.free(cframes);

                            for (0..len) |i| {
                                const rx0, const ry0, const rz0, const rx1, const ry1, const rz1, const rx2, const ry2, const rz2 = matrices[i];

                                cframes[i] = .{ cx[i], cy[i], cz[i], rx0, ry0, rz0, rx1, ry1, rz1, rx2, ry2, rz2 };
                            }

                            if (t == .OptionalCFrame) {
                                if (try chunk_reader.readByte() != @intFromEnum(Roblox.Property.Type.Bool)) {
                                    for (0..len) |i| values[i] = .{ .OptionalCFrame = null };
                                } else {
                                    for (0..len) |i| {
                                        if (try chunk_reader.readByte() == 0) {
                                            values[i] = .{ .OptionalCFrame = null };
                                        } else {
                                            values[i] = .{ .OptionalCFrame = cframes[i] };
                                        }
                                    }
                                }
                            } else {
                                for (0..len) |i| values[i] = .{ .CFrame = cframes[i] };
                            }
                        }
                    },
                    .Enum => {
                        const enums = try chunk_reader.readAllocInterleavedu32(allocator, len);
                        defer allocator.free(enums);
                        for (0..len) |i| {
                            const value = enums[i];
                            values[i] = .{ .Enum = .{ value, "Blank" } };
                        }
                    },
                    .Ref => {
                        const list = try chunk_reader.readAllocInstances(allocator, len);
                        defer allocator.free(list);
                        for (0..len) |i| values[i] = .{ .Ref = if (list[i] > 0) @intCast(list[i]) else null };
                    },
                    .Vector3int16 => {
                        for (0..len) |i| {
                            const x = try chunk_reader.readVarInt(i16, .little, 2);
                            const y = try chunk_reader.readVarInt(i16, .little, 2);
                            const z = try chunk_reader.readVarInt(i16, .little, 2);
                            values[i] = .{ .Vector3int16 = .{ x, y, z } };
                        }
                    },
                    .NumberSequence => {
                        for (0..len) |i| {
                            const keys: usize = @intCast(try chunk_reader.readVarInt(u32, .little, 4));
                            const points = try allocator.alloc(struct { f32, f32, f32 }, keys);
                            errdefer allocator.free(points);

                            for (0..keys) |key| {
                                const time = try chunk_reader.readf32();
                                const value = try chunk_reader.readf32();
                                const envelope = try chunk_reader.readf32();
                                points[key] = .{ time, value, envelope };
                            }

                            values[i] = .{ .NumberSequence = points };
                        }
                    },
                    .ColorSequence => {
                        for (0..len) |i| {
                            const keys: usize = @intCast(try chunk_reader.readVarInt(u32, .little, 4));
                            const points = try allocator.alloc(struct { f32, f32, f32, f32, i32 }, keys);
                            errdefer allocator.free(points);

                            for (0..keys) |key| {
                                const time = try chunk_reader.readf32();
                                const r = try chunk_reader.readf32();
                                const g = try chunk_reader.readf32();
                                const b = try chunk_reader.readf32();
                                const envelope = try chunk_reader.readVarInt(i32, .little, 4);
                                points[key] = .{ time, r, g, b, envelope };
                            }

                            values[i] = .{ .ColorSequence = points };
                        }
                    },
                    .NumberRange => {
                        for (0..len) |i| {
                            const min = try chunk_reader.readf32();
                            const max = try chunk_reader.readf32();
                            values[i] = .{ .NumberRange = .{ min, max } };
                        }
                    },
                    .Rect => {
                        const x0 = try chunk_reader.readAllocInterleavedf32(allocator, len);
                        defer allocator.free(x0);
                        const y0 = try chunk_reader.readAllocInterleavedf32(allocator, len);
                        defer allocator.free(y0);
                        const x1 = try chunk_reader.readAllocInterleavedf32(allocator, len);
                        defer allocator.free(x1);
                        const y1 = try chunk_reader.readAllocInterleavedf32(allocator, len);
                        defer allocator.free(y1);
                        for (0..len) |i| values[i] = .{ .Rect = .{ x0[i], y0[i], x1[i], y1[i] } };
                    },
                    .PhysicalProperties => {
                        for (0..len) |i| {
                            values[i] = .{ .PhysicalProperties = if (try chunk_reader.readByte() != 0) .{
                                try chunk_reader.readf32(),
                                try chunk_reader.readf32(),
                                try chunk_reader.readf32(),
                                try chunk_reader.readf32(),
                                try chunk_reader.readf32(),
                            } else null };
                        }
                    },
                    .Color3uint8 => {
                        const r = chunk_reader.readChunk(len);
                        const g = chunk_reader.readChunk(len);
                        const b = chunk_reader.readChunk(len);
                        for (0..len) |i| values[i] = .{ .Color3uint8 = .{ r[i], g[i], b[i] } };
                    },
                    .Int64 => {
                        const list = try chunk_reader.readAllocInterleavedi64(allocator, len);
                        defer allocator.free(list);
                        for (0..len) |i| values[i] = .{ .Int64 = list[i] };
                    },
                    .SharedString => {
                        const keys = try chunk_reader.readAllocInterleavedu32(allocator, len);
                        defer allocator.free(keys);
                        for (0..len) |i| values[i] = .{ .SharedString = keys[i] };
                    },
                    .ProtectedString => {
                        for (0..len) |i| {
                            const size = try chunk_reader.readVarInt(u32, .little, 4);
                            const value = try allocator.dupe(u8, chunk_reader.readChunk(size));
                            values[i] = .{ .ProtectedString = value };
                        }
                    },
                    .UniqueId => {
                        const interleaved = ReadInterleaved(struct { u32, u32, i64 }, struct {
                            fn inner(buf: *[16]u8) struct { u32, u32, i64 } {
                                return .{
                                    std.mem.readVarInt(u32, buf[8..12], .little),
                                    std.mem.readVarInt(u32, buf[12..16], .little),
                                    std.mem.readVarInt(i64, buf[0..8], .little),
                                };
                            }
                        }.inner);
                        const ids = try interleaved(allocator, chunk_reader.readChunk(len * 16), len);
                        defer allocator.free(ids);
                        for (0..len) |i| values[i] = .{ .UniqueId = ids[i] };
                    },
                    .FontFace => {
                        for (0..len) |i| {
                            const family = try allocator.dupe(u8, try chunk_reader.readBuffer());
                            errdefer allocator.free(family);

                            const weight = try chunk_reader.readVarInt(u16, .little, 2);
                            const style = try chunk_reader.readByte();
                            const cached_face_id = try allocator.dupe(u8, try chunk_reader.readBuffer());
                            errdefer allocator.free(cached_face_id);

                            values[i] = .{ .FontFace = .{ family, weight, style, cached_face_id } };
                        }
                    },
                    .SecurityCapabilities => {
                        const capabilities = try chunk_reader.readAllocInterleavedu64(allocator, len);
                        defer allocator.free(capabilities);
                        for (0..len) |i| values[i] = .{ .SecurityCapabilities = capabilities[i] };
                    },
                    else => |t| std.debug.panic("(deserialize) Unhandled type: {}\n", .{t}),
                }

                for (0..len) |i| {
                    const id = class.instances[i];
                    const instance = &doc.instances.items[@intCast(id)];
                    if (instance.Properties.get(name) != null) return Roblox.DeserializeError.BadData;

                    const copy = try allocator.dupe(u8, name);
                    errdefer allocator.free(copy);
                    try instance.Properties.put(copy, .{
                        .Name = copy,
                        .Raw = null,
                        .Type = property_type,
                        .Value = values[i],
                    });
                }
            },
            .SIGN => {
                const sign_len = try chunk_reader.readVarInt(u32, .little, 4);
                const signs = try allocator.alloc(Roblox.Signature, sign_len);

                for (0..sign_len) |i| {
                    signs[i] = Roblox.Signature{
                        .Type = @enumFromInt(try chunk_reader.readVarInt(i32, .little, 4)),
                        .PublicKey = try chunk_reader.readVarInt(i64, .little, 8),
                        .Value = chunk_reader.readChunk(try chunk_reader.readVarInt(u32, .little, 4)),
                    };
                }

                doc.signatures = signs;
            },
            .SSTR => {
                const format = try chunk_reader.readVarInt(i32, .little, 4);
                const hash_len = try chunk_reader.readVarInt(u32, .little, 4);

                if (format != 0) return DeserializeError.BadSharedStringFormat;

                for (0..@intCast(hash_len)) |_| {
                    _ = chunk_reader.readChunk(16);
                    // const hash_base64 = try Base64.toBase64(allocator, hash);
                    // errdefer allocator.free(hash_base64);

                    const data_len = try chunk_reader.readVarInt(u32, .little, 4);
                    var shared = try Roblox.SharedString.init(allocator, chunk_reader.readChunk(data_len));
                    errdefer shared.deinit();

                    try doc.sharedstrings.append(shared);
                }
            },
            .END => {
                ended = true;
                break;
            },
        }
    }

    if (!ended) return DeserializeError.BadDataNoEnd;

    return doc;
}

fn testWriteFn(comptime T: type, allocator: std.mem.Allocator, writeFn: anytype, values: []const T, expected: []const u8) anyerror!void {
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    try writeFn(allocator, buffer.writer(), values);

    try std.testing.expectEqualSlices(u8, expected, buffer.items);
}

fn testReadFn(comptime T: type, allocator: std.mem.Allocator, readFn: anytype, buf: []const u8, expected: []const T) anyerror!void {
    const slices = try readFn(allocator, buf, expected.len);
    defer allocator.free(slices);

    try std.testing.expectEqualSlices(T, expected, slices);
}

test "Read/Write Chunk" {
    const allocator = std.testing.allocator;

    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    var chunk = Chunk.Writer.init(allocator, .META);
    defer chunk.deinit();

    try chunk.writeBuffer("Some string");
    try chunk.writeByte(255);
    try chunk.writef32(1.02);
    try chunk.writeIntType(u32, 20);
    try chunk.writeIntType(i32, -20);
    try chunk.writeIntType(u64, 9999);
    try chunk.writef64(9999.9999);
    try chunk.writeInterleavedi32(&.{ 25, -20, 10 });
    try chunk.writeInterleavedu32(&.{ 25, 20, 10 });
    try chunk.writeInterleavedi64(&.{ 2500, -2000, 1000 });
    try chunk.writeInterleavedu64(&.{ 2500, 2000, 1000 });
    try chunk.writeInterleavedf32(&.{ 1.23, -4.26, 55.55 });
    try chunk.writeInterleavedUniqueId(&[_][16]u8{.{ 4, 8, 16, 32, 2, 6, 14, 30, 1, 5, 13, 29, 14, 18, 26, 42 }});

    try chunk.finalize(buffer.writer(), true);

    var read_buffer = std.ArrayList(u8).init(allocator);
    defer read_buffer.deinit();

    var stream = std.io.fixedBufferStream(buffer.items);

    const read_chunk = try readChunk(stream.reader().any(), &read_buffer);
    defer read_chunk.deinit(allocator);
    try std.testing.expectEqual(.META, read_chunk.Type);
    try std.testing.expectEqual(0, read_chunk.Reserved);
    try std.testing.expectEqual(144, read_chunk.Size);

    // zig fmt: off
    try std.testing.expectEqualSlices(u8, &.{
        11,0,0,0,83,111,109,101,32,115,116,114,105,110,103,255,
        92,143,130,63,20,0,0,0,236,255,255,255,15,39,0,0,
        0,0,0,0,163,35,185,252,255,135,195,64,0,0,0,0,
        0,0,0,0,0,50,39,20,0,0,0,0,0,0,0,0,
        0,25,20,10,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,19,15,7,136,159,208,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,7,
        3,196,208,232,127,129,132,58,16,188,225,163,102,72,217,102,
        4,8,16,32,2,6,14,30,1,5,13,29,14,18,26,42
    }, read_chunk.Data);
    // zig fmt: on

    var chunk_reader = Chunk.Reader{
        .buffer = read_chunk.Data,
        .pos = 0,
    };

    try std.testing.expectEqualStrings("Some string", try chunk_reader.readBuffer());
    try std.testing.expectEqual(255, try chunk_reader.readByte());
    try std.testing.expectEqual(1.02, try chunk_reader.readf32());
    try std.testing.expectEqual(20, try chunk_reader.readVarInt(u32, .little, 4));
    try std.testing.expectEqual(-20, try chunk_reader.readVarInt(i32, .little, 4));
    try std.testing.expectEqual(9999, try chunk_reader.readVarInt(u64, .little, 8));
    try std.testing.expectEqual(9999.9999, try chunk_reader.readf64());
    {
        const values = try chunk_reader.readAllocInterleavedi32(allocator, 3);
        defer allocator.free(values);
        try std.testing.expectEqualSlices(i32, &.{ 25, -20, 10 }, values);
    }
    {
        const values = try chunk_reader.readAllocInterleavedu32(allocator, 3);
        defer allocator.free(values);
        try std.testing.expectEqualSlices(u32, &.{ 25, 20, 10 }, values);
    }
    {
        const values = try chunk_reader.readAllocInterleavedi64(allocator, 3);
        defer allocator.free(values);
        try std.testing.expectEqualSlices(i64, &.{ 2500, -2000, 1000 }, values);
    }
    {
        const values = try chunk_reader.readAllocInterleavedu64(allocator, 3);
        defer allocator.free(values);
        try std.testing.expectEqualSlices(u64, &.{ 2500, 2000, 1000 }, values);
    }
    {
        const values = try chunk_reader.readAllocInterleavedf32(allocator, 3);
        defer allocator.free(values);
        try std.testing.expectEqualSlices(f32, &.{ 1.23, -4.26, 55.55 }, values);
    }
    {
        const interleaved = ReadInterleaved(struct { u32, u32, i64 }, struct {
            fn inner(buf: *[16]u8) struct { u32, u32, i64 } {
                return .{
                    std.mem.readVarInt(u32, buf[8..12], .little),
                    std.mem.readVarInt(u32, buf[12..16], .little),
                    std.mem.readVarInt(i64, buf[0..8], .little),
                };
            }
        }.inner);
        const values = try interleaved(allocator, chunk_reader.readChunk(16), 1);
        defer allocator.free(values);
        try std.testing.expectEqual(1, values.len);
        for (values) |id| {
            const random, const time, const index = id;
            try std.testing.expectEqual(487392513, random);
            try std.testing.expectEqual(706351630, time);
            try std.testing.expectEqual(2165675077009410052, index);
        }
    }
}

test "Write interleaved" {
    const allocator = std.testing.allocator;

    try testWriteFn(
        i32,
        allocator,
        WriteInterleavedi32,
        &.{ 100, 50, -100 },
        &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 100, 199 },
    );

    // zig fmt: off
    try testWriteFn(
        i32,
        allocator,
        WriteInterleavedi32,
        &.{
            1,2,-1,2,2,-1,-5,7,1,1,1,1,1,1,1,1,
            1,1,1,2,1,-2,3,1,1,1,1,1,1,2,-1,2,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            2,1,1,1,1,-5,6,1,1,1,1,1,
        },
        &.{
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,2,4,1,4,4,1,9,14,2,2,2,2,
            2,2,2,2,2,2,2,4,2,3,6,2,2,2,2,2,
            2,4,1,4,2,2,2,2,2,2,2,2,2,2,2,2,
            2,2,2,2,4,2,2,2,2,9,12,2,2,2,2,2,
        },
    );
    // zig fmt: on

    try testWriteFn(
        u32,
        allocator,
        WriteInterleavedu32,
        &.{ 100, 50, 200 },
        &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 50, 200 },
    );

    try testWriteFn(
        i64,
        allocator,
        WriteInterleavedi64,
        &.{ 10000, 5000, -10000 },
        &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 78, 39, 78, 32, 16, 31 },
    );

    try testWriteFn(
        u64,
        allocator,
        WriteInterleavedu64,
        &.{ 10000, 5000, 20000 },
        &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 19, 78, 16, 136, 32 },
    );

    try testWriteFn(
        f32,
        allocator,
        WriteInterleavedf32,
        &.{ -1.0189406e9, 1.0189406e9, 5.410652e8 },
        &.{ 156, 156, 156, 229, 229, 2, 222, 222, 0, 103, 102, 0 },
    );
}

test "Read interleaved" {
    const allocator = std.testing.allocator;

    try testReadFn(
        i32,
        allocator,
        ReadInterleavedi32,
        &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 100, 199 },
        &.{ 100, 50, -100 },
    );

    // zig fmt: off
    try testReadFn(
        i32,
        allocator,
        ReadInterleavedi32,
        &.{
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,2,4,1,4,4,1,9,14,2,2,2,2,
            2,2,2,2,2,2,2,4,2,3,6,2,2,2,2,2,
            2,4,1,4,2,2,2,2,2,2,2,2,2,2,2,2,
            2,2,2,2,4,2,2,2,2,9,12,2,2,2,2,2,
        },
        &.{
            1,2,-1,2,2,-1,-5,7,1,1,1,1,1,1,1,1,
            1,1,1,2,1,-2,3,1,1,1,1,1,1,2,-1,2,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            2,1,1,1,1,-5,6,1,1,1,1,1,
        },
    );
    // zig fmt: on

    try testReadFn(
        u32,
        allocator,
        ReadInterleavedu32,
        &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 50, 200 },
        &.{ 100, 50, 200 },
    );

    try testReadFn(
        i64,
        allocator,
        ReadInterleavedi64,
        &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 78, 39, 78, 32, 16, 31 },
        &.{ 10000, 5000, -10000 },
    );

    try testReadFn(
        u64,
        allocator,
        ReadInterleavedu64,
        &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 19, 78, 16, 136, 32 },
        &.{ 10000, 5000, 20000 },
    );

    try testReadFn(
        f32,
        allocator,
        ReadInterleavedf32,
        &.{ 156, 156, 156, 229, 229, 2, 222, 222, 0, 103, 102, 0 },
        &.{ -1.0189406e9, 1.0189406e9, 5.410652e8 },
    );
}
