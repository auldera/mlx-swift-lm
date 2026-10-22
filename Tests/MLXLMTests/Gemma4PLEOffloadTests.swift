// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@_spi(GemmaEncoder) @testable import MLXLLM

struct Gemma4PLEOffloadTests {
    private static let tableName = "language_model.model.embed_tokens_per_layer"
    private static var ids: MLXArray { MLXArray([3, 0, 31, 7, 7, 1] as [Int32]).reshaped(2, 3) }

    private struct UncheckedTable: @unchecked Sendable {
        let embedding: Embedding
    }

    @Test("The mapped provider returns the rows the quantized table's embedding returns")
    func mappedQuantizedRowsMatchTheEmbedding() throws {
        let embedding = Self.quantizedEmbedding(groupSize: 64, bits: 4)
        let directory = try Self.checkpoint(
            Self.arrays(of: embedding), quantization: #"{"group_size": 64, "bits": 4}"#)
        defer { try? FileManager.default.removeItem(at: directory) }

        let provider = try Gemma4PLEOffload.mappedProvider(modelDirectory: directory)
        let expected = embedding(Self.ids)

        let actual = provider(Self.ids)
        #expect(actual.shape == [2, 3, 128])
        #expect(actual.dtype == .bfloat16)
        #expect(Self.bits(actual) == Self.bits(expected))
    }

    @Test("A per-layer quantization entry for the table overrides the default")
    func perLayerQuantizationOverridesTheDefault() throws {
        let embedding = Self.quantizedEmbedding(groupSize: 32, bits: 8)
        let directory = try Self.checkpoint(
            Self.arrays(of: embedding),
            quantization:
                #"{"group_size": 64, "bits": 4, "\#(Self.tableName)": {"group_size": 32, "bits": 8}}"#
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        let expected = embedding(Self.ids)
        let actual = try Gemma4PLEOffload.mappedProvider(modelDirectory: directory)(Self.ids)
        #expect(actual.shape == [2, 3, 128])
        #expect(Self.bits(actual) == Self.bits(expected))
    }

    @Test("The mapped provider returns an unquantized table's rows unchanged")
    func mappedUnquantizedRowsMatchTheTable() throws {
        let table = MLXRandom.normal([32, 24]).asType(.float16)
        let directory = try Self.checkpoint(
            ["model.embed_tokens_per_layer.weight": table], quantization: nil)
        defer { try? FileManager.default.removeItem(at: directory) }

        let actual = try Gemma4PLEOffload.mappedProvider(modelDirectory: directory)(Self.ids)
        #expect(actual.shape == [2, 3, 24])
        #expect(Self.bits(actual) == Self.bits(table[Self.ids]))
    }

    @Test("A checkpoint without the table has no mapped provider")
    func missingTableThrows() throws {
        let directory = try Self.checkpoint(
            ["model.embed_tokens.weight": MLXArray.zeros([4, 4])], quantization: nil)
        defer { try? FileManager.default.removeItem(at: directory) }

        #expect(throws: Gemma4PLEOffloadError.tableNotFound) {
            try Gemma4PLEOffload.mappedProvider(modelDirectory: directory)
        }
    }

    @Test("A table whose header points outside its data is refused")
    func outOfRangeOffsetsThrow() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gemma4-ple-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let header = Data(
            #"{"model.embed_tokens_per_layer.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [-8, 0]}}"#
                .utf8)
        var file = Data()
        withUnsafeBytes(of: UInt64(header.count).littleEndian) { file.append(contentsOf: $0) }
        file.append(header)
        file.append(Data(count: 8))
        try file.write(to: directory.appendingPathComponent("model.safetensors"))
        try Data(#"{"model_type": "gemma4"}"#.utf8)
            .write(to: directory.appendingPathComponent("config.json"))

        #expect(throws: Gemma4PLEOffloadError.unsupportedTable("a malformed safetensors header")) {
            try Gemma4PLEOffload.mappedProvider(modelDirectory: directory)
        }
    }

    @Test("A model created with a provider owns no table and computes the same logits")
    func offloadedModelMatchesTheResidentModel() throws {
        let config = try Self.configuration()
        let resident = Gemma4TextModel(config)
        eval(resident)
        let table = UncheckedTable(embedding: try #require(resident.model.embedTokensPerLayer))

        let provider: Gemma4PLEOffload.Provider = { table.embedding($0) }
        let offloaded = Gemma4PLEOffload.$provider.withValue(provider) {
            Gemma4TextModel(config)
        }
        #expect(offloaded.model.embedTokensPerLayer == nil)
        let checkpoint = Dictionary(uniqueKeysWithValues: resident.parameters().flattened())
        let sanitized = offloaded.sanitize(weights: checkpoint)
        #expect(
            Set(sanitized.keys)
                == Set(checkpoint.keys).subtracting(["model.embed_tokens_per_layer.weight"]))
        try offloaded.update(
            parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
        eval(offloaded)

        let input = MLXArray([1, 2, 3, 4] as [Int32]).reshaped(1, 4)
        let expected = resident(input, cache: try resident.newCache(parameters: nil))
        let actual = offloaded(input, cache: try offloaded.newCache(parameters: nil))
        #expect(Self.bits(actual) == Self.bits(expected))
    }

    @Test("A model created outside a provider binding builds the table")
    func unboundProviderKeepsTheTable() throws {
        let model = Gemma4TextModel(try Self.configuration())
        #expect(model.model.embedTokensPerLayer != nil)
        #expect(model.model.perLayerEmbeddingProvider == nil)
    }

    private static func quantizedEmbedding(groupSize: Int, bits: Int) -> QuantizedEmbedding {
        let embedding = QuantizedEmbedding(
            weight: MLXRandom.normal([32, 128]).asType(.bfloat16), groupSize: groupSize,
            bits: bits)
        eval(embedding)
        return embedding
    }

    private static func arrays(of embedding: QuantizedEmbedding) -> [String: MLXArray] {
        var arrays = [
            "\(tableName).weight": embedding.weight, "\(tableName).scales": embedding.scales,
        ]
        arrays["\(tableName).biases"] = embedding.biases
        return arrays
    }

    private static func bits(_ array: MLXArray) -> [UInt32] {
        array.asType(.float32).asArray(Float.self).map(\.bitPattern)
    }

    private static func checkpoint(_ arrays: [String: MLXArray], quantization: String?) throws
        -> URL
    {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gemma4-ple-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try save(arrays: arrays, url: directory.appendingPathComponent("model.safetensors"))
        let config =
            #"{"model_type": "gemma4"\#(quantization.map { ", \"quantization\": \($0)" } ?? "")}"#
        try Data(config.utf8).write(to: directory.appendingPathComponent("config.json"))
        return directory
    }

    private static func configuration() throws -> Gemma4TextConfiguration {
        let json = """
            {
              "model_type": "gemma4_text",
              "hidden_size": 16,
              "num_hidden_layers": 2,
              "intermediate_size": 32,
              "num_attention_heads": 2,
              "head_dim": 32,
              "global_head_dim": 32,
              "global_partial_rotary_factor": 0.25,
              "rms_norm_eps": 0.000001,
              "vocab_size": 32,
              "vocab_size_per_layer_input": 32,
              "num_key_value_heads": 1,
              "num_global_key_value_heads": 1,
              "num_kv_shared_layers": 0,
              "hidden_size_per_layer_input": 8,
              "sliding_window": 8,
              "sliding_window_pattern": 1,
              "max_position_embeddings": 64,
              "attention_k_eq_v": false,
              "final_logit_softcapping": 30.0,
              "use_double_wide_mlp": false,
              "layer_types": ["full_attention", "full_attention"],
              "tie_word_embeddings": true
            }
            """
        return try JSONDecoder().decode(Gemma4TextConfiguration.self, from: Data(json.utf8))
    }
}
