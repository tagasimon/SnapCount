import Observation
import SwiftUI
import UIKit

@MainActor
@Observable
final class CountViewModel {
    enum Phase: Equatable {
        case idle
        case counting
        case done(CountResult)
        case failed(String)
    }

    /// ContentView clears the result when this changes (see .onChange there).
    var itemName = ""
    private(set) var photo: UIImage?
    private(set) var phase: Phase = .idle

    /// Models the server offers; replaced by the server's list once it loads.
    private(set) var models: [ModelInfo] = ModelInfo.fallback
    /// The model the next count uses. Remembered between launches.
    var selectedModelID: String = UserDefaults.standard.string(forKey: "selectedModel") ?? "auto" {
        didSet { UserDefaults.standard.set(selectedModelID, forKey: "selectedModel") }
    }
    /// In Auto mode, whose dots are drawn on the photo. nil = the combined result's.
    var dotsModelID: String? {
        didSet { if dotsModelID != oldValue { editedPoints = nil } }
    }
    /// The worker's corrections: the dots after removing wrong ones and adding missed ones.
    /// nil = no corrections, show the model's dots.
    private(set) var editedPoints: [[Double]]?

    var result: CountResult? {
        if case .done(let r) = phase { return r }
        return nil
    }

    /// The model's own dots for the current result.
    private var modelPoints: [[Double]] {
        result?.points(for: dotsModelID) ?? []
    }

    /// Dots to draw, numbered in reading order (top to bottom, left to right).
    var shownPoints: [[Double]] {
        let pts = editedPoints ?? modelPoints
        guard pts.count > 1 else { return pts }
        // Group into rows about one dot-gap tall, then left to right within a row.
        let ys = pts.map { $0.count > 1 ? $0[1] : 0 }.sorted()
        let gaps = zip(ys, ys.dropFirst()).map { $1 - $0 }.filter { $0 > 0 }.sorted()
        let rowHeight = max(0.02, gaps.isEmpty ? 0.02 : gaps[gaps.count / 2] * 0.6)
        return pts.sorted {
            let r0 = Int(($0.count > 1 ? $0[1] : 0) / rowHeight)
            let r1 = Int(($1.count > 1 ? $1[1] : 0) / rowHeight)
            return r0 == r1 ? $0[0] < $1[0] : r0 < r1
        }
    }

    /// Dots can be corrected only when the model gave positions (Claude doesn't).
    var canEditDots: Bool { !modelPoints.isEmpty || editedPoints != nil }
    var isEdited: Bool { editedPoints != nil }

    /// The number to show: the corrected count if the worker changed the dots.
    var displayCount: Int? {
        if let editedPoints { return editedPoints.count }
        return result?.count
    }

    func removeDot(_ point: [Double]) {
        var pts = editedPoints ?? modelPoints
        if let i = pts.firstIndex(of: point) {
            pts.remove(at: i)
            editedPoints = pts
        }
    }

    func addDot(x: Double, y: Double) {
        var pts = editedPoints ?? modelPoints
        pts.append([x, y])
        editedPoints = pts
    }

    func undoEdits() {
        editedPoints = nil
    }

    private var countTask: Task<Void, Never>?

    /// Waiting for the worker to drag a line along one side of a stack.
    var isDrawingLine = false
    /// The line the last count used, [x0, y0, x1, y1] as fractions of the photo.
    private(set) var countLine: [Double]?

    /// Offer "count one side" for stacks: results from the stack counter, or the stack model.
    var offersLineCount: Bool {
        selectedModelID == "stack" || result?.bestModel == "stack" || result?.model == "stack"
    }

    func startDrawingLine() {
        guard photo != nil, phase != .counting else { return }
        isDrawingLine = true
    }

    func cancelDrawingLine() {
        isDrawingLine = false
    }

    /// Count the edges crossed by the line the worker drew.
    func countAlongLine(_ line: [Double]) {
        isDrawingLine = false
        countLine = line
        count(line: line)
    }

    var canCount: Bool {
        photo != nil && phase != .counting
    }

    var selectedModel: ModelInfo {
        models.first(where: { $0.id == selectedModelID }) ?? models.first ?? ModelInfo.fallback[0]
    }

    func loadModels() async {
        guard let reply = await APIClient.shared.models(), !reply.models.isEmpty else { return }
        models = reply.models
        if !models.contains(where: { $0.id == selectedModelID }) {
            selectedModelID = reply.default ?? models[0].id
        }
    }

    func setPhoto(_ image: UIImage) {
        countTask?.cancel()
        photo = image
        phase = .idle
        dotsModelID = nil
        editedPoints = nil
        countLine = nil
        isDrawingLine = false
    }

    func clearResult() {
        guard phase != .counting else { return }
        phase = .idle
        dotsModelID = nil
        editedPoints = nil
    }

    func count(line: [Double]? = nil) {
        guard let photo, phase != .counting else { return }
        if line == nil {
            countLine = nil
            // The stack model counts along a line: ask for one first.
            if selectedModelID == "stack" {
                isDrawingLine = true
                return
            }
        }
        let item = itemName.trimmingCharacters(in: .whitespacesAndNewlines)
        let modelID = selectedModelID
        // The free models search for the named item; without a name they look for "object".
        if item.isEmpty && modelID != "claude" && line == nil {
            phase = .failed("Type what you're counting first, e.g. \"chairs\" or \"logs\".")
            return
        }
        phase = .counting
        dotsModelID = nil
        editedPoints = nil

        countTask = Task {
            let jpeg = await Task.detached(priority: .userInitiated) {
                ImageProcessing.uploadJPEG(from: photo)
            }.value

            guard !Task.isCancelled else { return }
            guard let jpeg else {
                phase = .failed("That photo couldn't be prepared. Please try another one.")
                return
            }

            do {
                let result = try await APIClient.shared.count(jpeg: jpeg, item: item,
                                                              model: line == nil ? modelID : "stack", line: line)
                guard !Task.isCancelled else { return }
                phase = .done(result)
            } catch is CancellationError {
                // A new photo was chosen; nothing to show.
            } catch {
                guard !Task.isCancelled else { return }
                phase = .failed(error.localizedDescription)
            }
        }
    }
}
