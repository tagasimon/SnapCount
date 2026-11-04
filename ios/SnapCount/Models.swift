import Foundation

/// One counting model the server offers (or "auto", which runs several and combines them).
struct ModelInfo: Decodable, Equatable, Identifiable, Hashable {
    let id: String
    let name: String
    /// "free", "paid" or "test"
    let kind: String
    let description: String

    /// Used until the server's list arrives, or if it can't be fetched.
    static let fallback: [ModelInfo] = [
        ModelInfo(id: "auto", name: "Auto", kind: "free", description: "Runs the free models and combines their counts."),
        ModelInfo(id: "owlv2", name: "OWLv2", kind: "free", description: "Good on small, dense items."),
        ModelInfo(id: "grounding_dino", name: "Grounding DINO", kind: "free", description: "Good at matching the item name."),
        ModelInfo(id: "omdet_turbo", name: "OmDet-Turbo", kind: "free", description: "The fastest free model."),
    ]
}

struct ModelsResponse: Decodable {
    let `default`: String?
    let models: [ModelInfo]
}

struct CountResult: Decodable, Equatable {
    struct Section: Decodable, Equatable {
        let label: String
        let count: Int
    }

    enum Confidence: String, Decodable {
        case high, medium, low
    }

    /// One model's answer. In Auto mode there is one per model that ran.
    struct ModelResult: Decodable, Equatable, Identifiable {
        let modelId: String
        let modelName: String
        let count: Int?
        let confidence: Confidence
        let notes: String
        /// Centre of each counted item, as [x, y] fractions (0...1) of the photo.
        let points: [[Double]]
        let seconds: Double
        let error: String?
        var id: String { modelId }
    }

    let item: String
    /// Which model was asked for, e.g. "auto" or "owlv2".
    let model: String
    /// nil means the photo didn't contain enough information to count.
    let count: Int?
    let confidence: Confidence
    let notes: String
    let retakeSuggested: Bool
    let retakeTip: String
    let sections: [Section]
    /// Centre of each counted item, as [x, y] fractions of the photo. Empty for Claude.
    let points: [[Double]]
    /// In Auto mode, the model whose count and dots are shown.
    let bestModel: String?
    let results: [ModelResult]

    private enum CodingKeys: String, CodingKey {
        case item, model, count, confidence, notes, retakeSuggested, retakeTip, sections, points, bestModel, results
    }

    // Tolerates an older server that doesn't send the newer fields.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        item = try c.decode(String.self, forKey: .item)
        model = try c.decodeIfPresent(String.self, forKey: .model) ?? "auto"
        count = try c.decodeIfPresent(Int.self, forKey: .count)
        confidence = try c.decode(Confidence.self, forKey: .confidence)
        notes = try c.decodeIfPresent(String.self, forKey: .notes) ?? ""
        retakeSuggested = try c.decodeIfPresent(Bool.self, forKey: .retakeSuggested) ?? (confidence != .high)
        retakeTip = try c.decodeIfPresent(String.self, forKey: .retakeTip) ?? ""
        sections = try c.decodeIfPresent([Section].self, forKey: .sections) ?? []
        points = try c.decodeIfPresent([[Double]].self, forKey: .points) ?? []
        bestModel = try c.decodeIfPresent(String.self, forKey: .bestModel)
        results = try c.decodeIfPresent([ModelResult].self, forKey: .results) ?? []
    }

    /// Dots to draw: a chosen model's, or the main result's.
    func points(for modelID: String?) -> [[Double]] {
        if let modelID, let r = results.first(where: { $0.modelId == modelID }) {
            return r.points
        }
        return points
    }
}

/// The server always returns errors as {"error": "..."}.
struct ServerError: Decodable {
    let error: String
}
