// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// Serves Gemma 4 per-layer embeddings (PLE) from outside the module graph.
///
/// The E-series `embed_tokens_per_layer` table is the largest tensor in those checkpoints
/// (about 1.3 GB of Gemma 4 E2B's 2.4 GB at 4 bits), but each token reads one row of it.
/// A model created while ``provider`` is bound builds no table and asks the provider for
/// those rows instead.
///
/// ```swift
/// let provider = try Gemma4PLEOffload.mappedProvider(modelDirectory: directory)
/// let container = try await Gemma4PLEOffload.$provider.withValue(provider) {
///     try await LLMModelFactory.shared.loadContainer(from: directory, using: tokenizerLoader)
/// }
/// ```
public enum Gemma4PLEOffload {
    /// Maps token ids `[B, L]` to rows of the table, `[B, L, layers * hidden_size_per_layer_input]`,
    /// in the dtype the table's own `Embedding` would return.
    public typealias Provider = @Sendable (MLXArray) -> MLXArray

    /// Read once, when a Gemma 4 text model with per-layer inputs is created.
    @TaskLocal public static var provider: Provider?

    /// A provider that copies the rows of the tokens it sees out of the memory-mapped
    /// checkpoint in `modelDirectory`, then dequantizes them on the GPU.
    ///
    /// Only the pages of rows that tokens touch become resident. They are clean file-backed
    /// pages, which the process footprint does not count and the kernel can evict. Wrapping the
    /// mapping as a no-copy Metal buffer instead wires the whole table and saves nothing.
    public static func mappedProvider(modelDirectory: URL) throws -> Provider {
        let table = try MappedPerLayerEmbeddingTable(modelDirectory: modelDirectory)
        return { table($0) }
    }
}

public enum Gemma4PLEOffloadError: LocalizedError, Equatable {
    case tableNotFound
    case unsupportedTable(String)

    public var errorDescription: String? {
        switch self {
        case .tableNotFound:
            "No safetensors file holds an embed_tokens_per_layer table."
        case .unsupportedTable(let reason):
            "The embed_tokens_per_layer table cannot be served from the file: \(reason)."
        }
    }
}

private struct MappedRows: @unchecked Sendable {
    let file: Data
    let offset: Int
    let rows: Int
    let columns: Int
    let dtype: DType

    var rowBytes: Int { columns * dtype.size }

    func gather(_ ids: [Int32]) -> MLXArray {
        let rowBytes = rowBytes
        var gathered = Data(count: ids.count * rowBytes)
        gathered.withUnsafeMutableBytes { destination in
            file.withUnsafeBytes { source in
                for (index, id) in ids.enumerated() {
                    precondition(id >= 0 && Int(id) < rows, "token id \(id) outside the table")
                    memcpy(
                        destination.baseAddress! + index * rowBytes,
                        source.baseAddress! + offset + Int(id) * rowBytes, rowBytes)
                }
            }
        }
        return MLXArray(gathered, [ids.count, columns], dtype: dtype)
    }
}

final class MappedPerLayerEmbeddingTable: Sendable {
    private let weight: MappedRows
    private let scales: MappedRows?
    private let biases: MappedRows?
    private let quantization: BaseConfiguration.Quantization?
    let dimensions: Int

    init(modelDirectory: URL) throws {
        var found = [String: MappedRows]()
        var tableName: String?
        for url in try safetensorWeightURLs(in: modelDirectory) {
            let file = try Data(contentsOf: url, options: .alwaysMapped)
            for (name, rows) in try Self.tensors(in: file)
            where name.contains("embed_tokens_per_layer.") {
                found[name] = rows
                if name.hasSuffix(".weight") { tableName = String(name.dropLast(".weight".count)) }
            }
        }
        guard let tableName, let weight = found[tableName + ".weight"] else {
            throw Gemma4PLEOffloadError.tableNotFound
        }
        if found[tableName + ".global_scale"] != nil {
            throw Gemma4PLEOffloadError.unsupportedTable("a global scale")
        }
        self.weight = weight
        self.scales = found[tableName + ".scales"]
        self.biases = found[tableName + ".biases"]

        guard scales != nil else {
            quantization = nil
            dimensions = weight.columns
            return
        }
        let configuration = try JSONDecoder.json5().decode(
            BaseConfiguration.self,
            from: Data(contentsOf: modelDirectory.appending(component: "config.json")))
        guard let quantization = configuration.perLayerQuantization?.quantization(layer: tableName)
        else {
            throw Gemma4PLEOffloadError.unsupportedTable("scales without a quantization config")
        }
        self.quantization = quantization
        dimensions = weight.columns * weight.dtype.size * 8 / quantization.bits
        guard weight.dtype == .uint32, let scales,
            scales.rows == weight.rows, scales.columns * quantization.groupSize == dimensions,
            biases.map({ $0.rows == weight.rows && $0.columns == scales.columns }) ?? true
        else {
            throw Gemma4PLEOffloadError.unsupportedTable("shapes that disagree with the config")
        }
    }

    func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        let ids = inputs.asType(.int32).asArray(Int32.self)
        let rows = weight.gather(ids)
        let embedded: MLXArray
        if let quantization, let scales {
            embedded = dequantized(
                rows, scales: scales.gather(ids), biases: biases?.gather(ids),
                groupSize: quantization.groupSize, bits: quantization.bits,
                mode: quantization.mode)
        } else {
            embedded = rows
        }
        return embedded.reshaped(inputs.shape + [dimensions])
    }

    /// Two-dimensional tensors of a safetensors file, by name, from its header.
    private static func tensors(in file: Data) throws -> [String: MappedRows] {
        let malformed = Gemma4PLEOffloadError.unsupportedTable("a malformed safetensors header")
        guard file.count >= 8 else { throw malformed }
        let declaredLength = file.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
        guard declaredLength > 0, declaredLength <= UInt64(file.count - 8) else { throw malformed }
        let dataStart = 8 + Int(declaredLength)
        guard
            let header = try JSONSerialization.jsonObject(with: file.subdata(in: 8 ..< dataStart))
                as? [String: Any]
        else { throw malformed }

        let dtypes: [String: DType] = [
            "U32": .uint32, "BF16": .bfloat16, "F16": .float16, "F32": .float32,
        ]
        var tensors = [String: MappedRows]()
        for (name, value) in header {
            guard let entry = value as? [String: Any],
                let shape = entry["shape"] as? [Int], shape.count == 2,
                let offsets = entry["data_offsets"] as? [Int], offsets.count == 2,
                let dtypeName = entry["dtype"] as? String
            else { continue }
            guard let dtype = dtypes[dtypeName] else {
                if name.contains("embed_tokens_per_layer.") {
                    throw Gemma4PLEOffloadError.unsupportedTable("dtype \(dtypeName)")
                }
                continue
            }
            let rowBytes = shape[1].multipliedReportingOverflow(by: dtype.size)
            let bytes = shape[0].multipliedReportingOverflow(by: rowBytes.partialValue)
            guard shape[0] >= 0, shape[1] >= 0, !rowBytes.overflow, !bytes.overflow,
                offsets[0] >= 0, offsets[1] >= offsets[0],
                offsets[1] - offsets[0] == bytes.partialValue,
                offsets[1] <= file.count - dataStart
            else { throw malformed }
            tensors[name] = MappedRows(
                file: file, offset: dataStart + offsets[0], rows: shape[0], columns: shape[1],
                dtype: dtype)
        }
        return tensors
    }
}
