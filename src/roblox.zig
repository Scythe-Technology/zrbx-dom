const std = @import("std");

const Base64 = @import("utils/base64.zig");

pub const DeserializeError = error{
    UnknownEncoding,
    BadEncoding,
    BadData,
    BadChunkType,
};

pub const Class = struct {
    index: usize,
    name: []const u8,
    is_service: bool,
    instances_len: usize,
    instances: []i32,
};

pub const Instance = struct {
    ClassName: []const u8,
    Referent: i32,
    Parent: ?u32,
    Childs: std.ArrayList(*Instance),
    Properties: std.StringArrayHashMap(Property),
    IsService: bool,
};

pub const DocumentInfo = struct {
    version: u16,
    classes: u32,
    instances: u32,
    reserved: i64,
};

pub const Signature = struct {
    Type: Type,
    PublicKey: i64,
    Value: []const u8,

    pub const Type = enum { Ed25519 };
};

pub const Value = union(enum) {
    Unknown: void,
    String: []const u8,
    Bool: bool,
    Int: i32,
    Float: f32,
    Double: f64,
    UDim: struct { f32, i32 },
    UDim2: struct { f32, i32, f32, i32 },
    Ray: struct { f32, f32, f32, f32, f32, f32 },
    Faces: u8,
    Axes: u8,
    BrickColor: i32,
    Color3: struct { f32, f32, f32 },
    Vector2: struct { f32, f32 },
    Vector3: struct { f32, f32, f32 },
    CFrame: struct { f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32 },
    Enum: struct { u32, []const u8 }, // TODO
    Ref: ?u32,
    Vector3int16: struct { i16, i16, i16 },
    NumberSequence: []struct { f32, f32, f32 },
    ColorSequence: []struct { f32, f32, f32, f32, i32 },
    NumberRange: struct { f32, f32 },
    Rect: struct { f32, f32, f32, f32 },
    PhysicalProperties: ?struct { f32, f32, f32, f32, f32 },
    Color3uint8: struct { u8, u8, u8 },
    Int64: i64,
    SharedString: u32,
    ProtectedString: []const u8,
    OptionalCFrame: ?struct { f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32 },
    UniqueId: struct { u32, u32, i64 },
    FontFace: struct { []const u8, u16, u8, []const u8 },
    SecurityCapabilities: u64,
};

pub const Property = struct {
    Name: []const u8,
    Type: Type,
    Raw: ?[]const u8,
    Value: Value,

    pub const Type = enum {
        Unknown,
        String,
        Bool,
        Int,
        Float,
        Double,
        UDim,
        UDim2,
        Ray,
        Faces,
        Axes,
        BrickColor,
        Color3,
        Vector2,
        Vector3,
        CFrame,
        Quaternion,
        Enum,
        Ref,
        Vector3int16,
        NumberSequence,
        ColorSequence,
        NumberRange,
        Rect,
        PhysicalProperties,
        Color3uint8,
        Int64,
        SharedString,
        ProtectedString,
        OptionalCFrame,
        UniqueId,
        FontFace,
        SecurityCapabilities,
    };
};

pub const SharedString = struct {
    Lookup: std.StringHashMap([]const u8),
    Key: []const u8,

    pub fn init(allocator: std.mem.Allocator, buf: []const u8) !SharedString {
        var hash: [std.crypto.hash.blake2.Blake2b128.digest_length]u8 = undefined;

        std.crypto.hash.blake2.Blake2b128.hash(buf, &hash, .{});

        const hash_base64 = try Base64.toBase64(allocator, &hash);
        errdefer allocator.free(hash_base64);

        var lookup = std.StringHashMap([]const u8).init(allocator);
        errdefer lookup.deinit();

        const buf_copy = try allocator.dupe(u8, buf);
        errdefer allocator.free(buf_copy);
        try lookup.put(hash_base64, buf_copy);

        return .{
            .Lookup = lookup,
            .Key = hash_base64,
        };
    }

    pub fn register(self: *SharedString, key: []const u8, value: []const u8) !void {
        if (self.Lookup.contains(key))
            return;
        self.Lookup.put(key, value);
    }

    pub fn find(self: *SharedString, key: []const u8) ?[]const u8 {
        return self.Lookup.get(key);
    }

    pub fn deinit(self: *SharedString) void {
        var iter = self.Lookup.iterator();
        while (iter.next()) |entry| {
            self.Lookup.allocator.free(entry.value_ptr.*);
            self.Lookup.allocator.free(entry.key_ptr.*);
        }
        self.Lookup.deinit();
    }
};

pub const Document = struct {
    allocator: std.mem.Allocator,
    info: DocumentInfo,

    instances: std.ArrayList(Instance),
    classes: std.ArrayList(Class),
    metadata: std.StringHashMap([]const u8),
    sharedstrings: std.ArrayList(SharedString),
    signatures: ?[]Signature = null,

    pub fn init(allocator: std.mem.Allocator, info: DocumentInfo) !Document {
        var doc = Document{
            .info = info,
            .allocator = allocator,
            .sharedstrings = std.ArrayList(SharedString).init(allocator),
            .instances = std.ArrayList(Instance).init(allocator),
            .classes = std.ArrayList(Class).init(allocator),
            .metadata = std.StringHashMap([]const u8).init(allocator),
        };

        try doc.classes.ensureTotalCapacityPrecise(info.classes);
        try doc.instances.ensureTotalCapacityPrecise(info.instances);
        doc.classes.expandToCapacity();
        doc.instances.expandToCapacity();

        return doc;
    }

    pub fn deinit(self: *Document) void {
        for (self.instances.items) |*instance| {
            var iter = instance.Properties.iterator();
            while (iter.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
                switch (entry.value_ptr.Value) {
                    .ProtectedString, .String => |s| self.allocator.free(s),
                    .FontFace => |a| {
                        const family, _, _, const cached = a;
                        self.allocator.free(family);
                        self.allocator.free(cached);
                    },
                    .ColorSequence => |a| self.allocator.free(a),
                    .NumberSequence => |a| self.allocator.free(a),
                    else => {},
                }
            }
            self.allocator.free(instance.ClassName);
            instance.Properties.deinit();
            instance.Childs.deinit();
        }
        self.instances.deinit();
        for (self.classes.items) |class| {
            self.allocator.free(class.name);
            self.allocator.free(class.instances);
        }
        self.classes.deinit();

        var meta_iter = self.metadata.iterator();
        while (meta_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();

        for (self.sharedstrings.items) |*item| item.deinit();
        self.sharedstrings.deinit();
    }
};
