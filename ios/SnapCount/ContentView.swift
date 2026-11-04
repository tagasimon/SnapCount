import AVFoundation
import PhotosUI
import SwiftUI

/// One screen, no scrolling: item field on top, a photo area that takes whatever
/// height is left (with the result laid over its bottom edge), and the buttons pinned
/// at the bottom.
struct ContentView: View {
    @State private var model = CountViewModel()
    @State private var pickerItem: PhotosPickerItem?
    @State private var showCamera = false
    @State private var showDetails = false
    @State private var cardCollapsed = false
    @State private var cameraAlert: CameraAlert?
    @FocusState private var itemFieldFocused: Bool

    var body: some View {
        VStack(spacing: 12) {
            header
            itemField
            photoArea
            controls
        }
        .padding(.horizontal, 16)
        .padding(.top, 8)
        .padding(.bottom, 12)
        .background(Color(.systemGroupedBackground).ignoresSafeArea())
        .fullScreenCover(isPresented: $showCamera) {
            CameraPicker { model.setPhoto($0) }
                .ignoresSafeArea()
        }
        .sheet(isPresented: $showDetails) {
            if case .done(let result) = model.phase {
                ResultDetails(result: result, dotsModelID: $model.dotsModelID)
                    .presentationDetents([.medium, .large])
                    .presentationDragIndicator(.visible)
            }
        }
        .onChange(of: model.itemName) { _, _ in
            model.clearResult()
        }
        .onChange(of: model.selectedModelID) { _, _ in
            model.clearResult()
        }
        .onChange(of: model.phase) { _, newPhase in
            if case .done = newPhase { cardCollapsed = false }
        }
        .task { await model.loadModels() }
        .onChange(of: pickerItem) { _, newItem in
            guard let newItem else { return }
            Task { await loadPickedPhoto(newItem) }
        }
        .alert(item: $cameraAlert) { alert in
            switch alert {
            case .denied:
                return Alert(
                    title: Text("Camera access is off"),
                    message: Text("To take photos, allow camera access for SnapCount in Settings. You can still choose a photo from your library."),
                    primaryButton: .default(Text("Open Settings")) {
                        if let url = URL(string: UIApplication.openSettingsURLString) {
                            UIApplication.shared.open(url)
                        }
                    },
                    secondaryButton: .cancel(Text("Not now"))
                )
            case .unavailable:
                return Alert(
                    title: Text("No camera available"),
                    message: Text("This device has no camera. Choose a photo from your library instead."),
                    dismissButton: .default(Text("OK"))
                )
            case .photoLoadFailed:
                return Alert(
                    title: Text("Couldn't open that photo"),
                    message: Text("Please try another photo. Photos stored only in iCloud may need a connection to download."),
                    dismissButton: .default(Text("OK"))
                )
            }
        }
    }

    // MARK: - Sections

    private var header: some View {
        HStack {
            Text("SnapCount")
                .font(.title2.bold())
            Spacer()
            modelMenu
        }
    }

    /// Pick one model, or Auto to run the free models and compare them.
    private var modelMenu: some View {
        Menu {
            Picker("Counting model", selection: $model.selectedModelID) {
                ForEach(model.models) { m in
                    Text(m.kind == "paid" ? "\(m.name) (paid)" : m.name).tag(m.id)
                }
            }
            Section(model.selectedModel.description) {}
        } label: {
            HStack(spacing: 5) {
                Image(systemName: model.selectedModelID == "auto" ? "square.stack.3d.up" : "cpu")
                Text(model.selectedModel.name)
                    .lineLimit(1)
                Image(systemName: "chevron.down")
                    .font(.caption2.weight(.bold))
            }
            .font(.subheadline.weight(.semibold))
            .padding(.horizontal, 12)
            .frame(minHeight: 36)
            .background(Color(.secondarySystemGroupedBackground), in: Capsule())
        }
        .disabled(model.phase == .counting)
        .accessibilityLabel("Counting model: \(model.selectedModel.name)")
    }


    private var countingMessage: String {
        if model.selectedModelID == "auto" {
            let n = max(model.models.filter { $0.id != "auto" && $0.kind == "free" }.count, 2)
            return "Counting with \(n) models… can take a minute"
        }
        return "Counting with \(model.selectedModel.name)…"
    }

    private var itemField: some View {
        HStack(spacing: 10) {
            Image(systemName: "tag")
                .foregroundStyle(.secondary)
            TextField("What are you counting? e.g. plastic chairs", text: $model.itemName)
                .textInputAutocapitalization(.never)
                .submitLabel(.done)
                .focused($itemFieldFocused)
        }
        .padding(.horizontal, 14)
        .frame(minHeight: 48)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 12))
        .disabled(model.phase == .counting)
    }

    /// Takes all the height that's left, so the screen never needs to scroll.
    private var photoArea: some View {
        ZStack(alignment: .bottom) {
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.secondarySystemGroupedBackground))

            if let photo = model.photo {
                PhotoWithDots(
                    photo: photo,
                    points: model.isDrawingLine ? [] : model.shownPoints,
                    line: model.isDrawingLine ? nil : model.countLine,
                    editable: model.canEditDots && !model.isDrawingLine,
                    onRemove: { model.removeDot($0) },
                    onAdd: { model.addDot(x: $0, y: $1) }
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .overlay {
                    if model.isDrawingLine {
                        LineDrawer(photo: photo) { model.countAlongLine($0) }
                    }
                }
                .accessibilityLabel(model.shownPoints.isEmpty ? "Selected photo" : "Selected photo, \(model.shownPoints.count) items numbered")
            } else {
                VStack(spacing: 10) {
                    Image(systemName: "camera.viewfinder")
                        .font(.system(size: 44))
                        .foregroundStyle(.secondary)
                    Text("Get everything in the frame, in good light, as straight-on as you can.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 32)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            resultOverlay
                .padding(10)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .simultaneousGesture(TapGesture().onEnded { itemFieldFocused = false })
    }

    @ViewBuilder
    private var resultOverlay: some View {
        if model.isDrawingLine {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "line.diagonal")
                    .foregroundStyle(.blue)
                VStack(alignment: .leading, spacing: 4) {
                    Text("Drag along one side of the stack")
                        .font(.subheadline.weight(.semibold))
                    Text("From the top item to the bottom one, crossing every seat edge or rim once. Avoid the backrest.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
                Button("Cancel") { model.cancelDrawingLine() }
                    .font(.subheadline.weight(.semibold))
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
        } else {
            phaseOverlay
        }
    }

    @ViewBuilder
    private var phaseOverlay: some View {
        switch model.phase {
        case .idle:
            EmptyView()
        case .counting:
            HStack(spacing: 10) {
                ProgressView()
                Text(countingMessage)
                    .font(.subheadline)
            }
            .padding(14)
            .frame(maxWidth: .infinity)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
        case .done(let result):
            CompactResult(
                result: result,
                count: model.displayCount,
                edited: model.isEdited,
                canEdit: model.canEditDots,
                collapsed: $cardCollapsed,
                undo: { model.undoEdits() },
                showDetails: { showDetails = true },
                countOneSide: model.offersLineCount ? { model.startDrawingLine() } : nil
            )
        case .failed(let message):
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                Text(message)
                    .font(.subheadline)
                    .lineLimit(3)
                Spacer(minLength: 0)
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
        }
    }

    private var controls: some View {
        let hasPhoto = model.photo != nil
        let counting = model.phase == .counting
        let failed: Bool = { if case .failed = model.phase { return true } else { return false } }()

        return HStack(spacing: 10) {
            Button {
                itemFieldFocused = false
                openCamera()
            } label: {
                Image(systemName: "camera")
                    .font(.title3.weight(.semibold))
                    .frame(width: 56, height: 56)
            }
            .buttonStyle(.bordered)
            .accessibilityLabel(hasPhoto ? "Retake photo" : "Take photo")
            .disabled(counting)

            PhotosPicker(selection: $pickerItem, matching: .images, photoLibrary: .shared()) {
                Image(systemName: "photo.on.rectangle")
                    .font(.title3.weight(.semibold))
                    .frame(width: 56, height: 56)
            }
            .buttonStyle(.bordered)
            .accessibilityLabel(hasPhoto ? "Choose another photo" : "Choose photo")
            .disabled(counting)

            Button {
                itemFieldFocused = false
                model.count()
            } label: {
                HStack(spacing: 8) {
                    if counting {
                        ProgressView().tint(.white)
                    } else {
                        Image(systemName: failed ? "arrow.clockwise" : "number")
                    }
                    Text(counting ? "Counting…" : failed ? "Try Again" : "Count")
                }
                .font(.title3.weight(.semibold))
                .frame(maxWidth: .infinity, minHeight: 56)
            }
            .buttonStyle(.borderedProminent)
            .disabled(!model.canCount)
            .accessibilityHint(hasPhoto ? "" : "Take or choose a photo first")
        }
    }

    // MARK: - Actions

    private func openCamera() {
        guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
            cameraAlert = .unavailable
            return
        }
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            showCamera = true
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                Task { @MainActor in
                    if granted { showCamera = true } else { cameraAlert = .denied }
                }
            }
        default:
            cameraAlert = .denied
        }
    }

    private func loadPickedPhoto(_ item: PhotosPickerItem) async {
        defer { pickerItem = nil }
        if let data = try? await item.loadTransferable(type: Data.self),
           let image = UIImage(data: data) {
            model.setPhoto(image)
        } else {
            cameraAlert = .photoLoadFailed
        }
    }
}

private enum CameraAlert: Identifiable {
    case denied, unavailable, photoLoadFailed
    var id: Self { self }
}

// MARK: - Photo with a dot on every counted item

private struct PhotoWithDots: View {
    let photo: UIImage
    /// [x, y] fractions of the photo, already in numbering order.
    let points: [[Double]]
    /// The line the stack was counted along, [x0, y0, x1, y1] as fractions.
    var line: [Double]? = nil
    let editable: Bool
    let onRemove: ([Double]) -> Void
    let onAdd: (Double, Double) -> Void

    var body: some View {
        GeometryReader { geo in
            // Where the scaled-to-fit photo actually sits inside the available space.
            let rect = AVMakeRect(aspectRatio: photo.size, insideRect: CGRect(origin: .zero, size: geo.size))
            let style = DotStyle(count: points.count)
            ZStack {
                Image(uiImage: photo)
                    .resizable()
                    .scaledToFit()
                    .frame(width: geo.size.width, height: geo.size.height)
                Canvas { context, _ in
                    if let line, line.count == 4 {
                        var path = Path()
                        path.move(to: CGPoint(x: rect.minX + CGFloat(line[0]) * rect.width,
                                              y: rect.minY + CGFloat(line[1]) * rect.height))
                        path.addLine(to: CGPoint(x: rect.minX + CGFloat(line[2]) * rect.width,
                                                 y: rect.minY + CGFloat(line[3]) * rect.height))
                        context.stroke(path, with: .color(.blue.opacity(0.7)),
                                       style: StrokeStyle(lineWidth: 4, lineCap: .round, dash: [8, 6]))
                    }
                    for (i, p) in points.enumerated() where p.count >= 2 {
                        let c = CGPoint(x: rect.minX + CGFloat(p[0]) * rect.width,
                                        y: rect.minY + CGFloat(p[1]) * rect.height)
                        let d = style.diameter
                        let dot = Path(ellipseIn: CGRect(x: c.x - d / 2, y: c.y - d / 2, width: d, height: d))
                        context.fill(dot, with: .color(.red))
                        context.stroke(dot, with: .color(.white), lineWidth: 1.5)
                        if style.showNumbers {
                            context.draw(
                                Text("\(i + 1)")
                                    .font(.system(size: style.fontSize, weight: .bold, design: .rounded))
                                    .foregroundStyle(.white),
                                at: c
                            )
                        }
                    }
                }
                .allowsHitTesting(false)
            }
            .contentShape(Rectangle())
            .gesture(
                SpatialTapGesture().onEnded { tap in
                    guard editable, rect.contains(tap.location), rect.width > 0, rect.height > 0 else { return }
                    let w = Double(rect.width), h = Double(rect.height)
                    let x = Double(tap.location.x - rect.minX) / w
                    let y = Double(tap.location.y - rect.minY) / h
                    // On-screen distance, in points, from the tap to a dot.
                    func distance(_ p: [Double]) -> Double {
                        guard p.count >= 2 else { return .infinity }
                        return ((p[0] - x) * w).magnitude.squared + ((p[1] - y) * h).magnitude.squared
                    }
                    // Tapping on a dot removes it; tapping anywhere else adds one there.
                    let radius = max(22, Double(style.diameter))
                    if let hit = points.min(by: { distance($0) < distance($1) }),
                       distance(hit) <= radius * radius {
                        onRemove(hit)
                    } else {
                        onAdd(x, y)
                    }
                    UIImpactFeedbackGenerator(style: .light).impactOccurred()
                }
            )
        }
    }
}

/// Drag a straight line across the photo; reports it as [x0, y0, x1, y1] fractions.
private struct LineDrawer: View {
    let photo: UIImage
    let onFinish: ([Double]) -> Void
    @State private var start: CGPoint?
    @State private var end: CGPoint?

    var body: some View {
        GeometryReader { geo in
            let rect = AVMakeRect(aspectRatio: photo.size, insideRect: CGRect(origin: .zero, size: geo.size))
            ZStack {
                Color.black.opacity(0.15)
                if let start, let end {
                    Path { p in
                        p.move(to: start)
                        p.addLine(to: end)
                    }
                    .stroke(Color.blue, style: StrokeStyle(lineWidth: 5, lineCap: .round))
                    Circle().fill(Color.blue).frame(width: 14, height: 14).position(start)
                    Circle().fill(Color.blue).frame(width: 14, height: 14).position(end)
                }
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 8)
                    .onChanged { g in
                        start = clamp(g.startLocation, rect)
                        end = clamp(g.location, rect)
                    }
                    .onEnded { g in
                        let a = clamp(g.startLocation, rect), b = clamp(g.location, rect)
                        start = nil
                        end = nil
                        // Ignore taps and tiny drags.
                        guard hypot(b.x - a.x, b.y - a.y) > 40, rect.width > 0, rect.height > 0 else { return }
                        onFinish([
                            Double((a.x - rect.minX) / rect.width), Double((a.y - rect.minY) / rect.height),
                            Double((b.x - rect.minX) / rect.width), Double((b.y - rect.minY) / rect.height),
                        ])
                    }
            )
        }
        .accessibilityLabel("Drag a line along one side of the stack")
    }

    private func clamp(_ p: CGPoint, _ r: CGRect) -> CGPoint {
        CGPoint(x: min(max(p.x, r.minX), r.maxX), y: min(max(p.y, r.minY), r.maxY))
    }
}

/// Bigger numbered dots when there are few items; small plain dots for hundreds.
private struct DotStyle {
    let diameter: CGFloat
    let fontSize: CGFloat
    let showNumbers: Bool

    init(count: Int) {
        if count < 40 {
            diameter = 24; fontSize = 12; showNumbers = true
        } else if count < 100 {
            diameter = 18; fontSize = 9; showNumbers = true
        } else if count < 200 {
            diameter = 14; fontSize = 7; showNumbers = true
        } else {
            diameter = 8; fontSize = 0; showNumbers = false
        }
    }
}

private extension Double {
    var squared: Double { self * self }
}

// MARK: - Result (compact, laid over the photo)

private struct CompactResult: View {
    let result: CountResult
    /// The number to show; differs from result.count once the worker has corrected the dots.
    let count: Int?
    let edited: Bool
    let canEdit: Bool
    @Binding var collapsed: Bool
    let undo: () -> Void
    let showDetails: () -> Void
    /// For stacks: let the worker draw a line along one side and recount.
    let countOneSide: (() -> Void)?

    var body: some View {
        if collapsed {
            collapsedBar
        } else {
            fullCard
        }
    }

    /// A slim bar, so the whole photo and its dots can be seen and tapped.
    private var collapsedBar: some View {
        Button {
            collapsed = false
        } label: {
            HStack(spacing: 10) {
                Text(count.map { $0.formatted() } ?? "–")
                    .font(.title2.weight(.bold).monospacedDigit())
                Text(result.item).font(.subheadline).lineLimit(1)
                Spacer(minLength: 0)
                if edited {
                    Text("Corrected").font(.caption.weight(.semibold)).foregroundStyle(.blue)
                }
                Image(systemName: "chevron.up").font(.caption.weight(.bold))
            }
            .foregroundStyle(.primary)
            .padding(.horizontal, 14)
            .frame(minHeight: 44)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(count ?? 0) \(result.item). Show result")
    }

    private var fullCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                if let count {
                    Text(count.formatted())
                        .font(.system(size: 48, weight: .bold, design: .rounded))
                        .monospacedDigit()
                        .minimumScaleFactor(0.5)
                        .lineLimit(1)
                    Text(result.item)
                        .font(.headline)
                        .lineLimit(1)
                } else {
                    Label("Couldn't count from this photo", systemImage: "eye.slash")
                        .font(.headline)
                }
                Spacer(minLength: 0)
                Button {
                    collapsed = true
                } label: {
                    Image(systemName: "chevron.down")
                        .font(.subheadline.weight(.bold))
                        .frame(width: 36, height: 36)
                        .background(Color(.tertiarySystemFill), in: Circle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Hide result to see the whole photo")
            }
            .accessibilityElement(children: .contain)

            HStack(spacing: 8) {
                if edited {
                    Label("Corrected by you · AI said \(result.count.map { String($0) } ?? "–")",
                          systemImage: "hand.tap.fill")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.blue)
                        .lineLimit(1)
                        .minimumScaleFactor(0.8)
                    Spacer(minLength: 0)
                    Button("Undo", action: undo)
                        .font(.subheadline.weight(.semibold))
                } else {
                    ConfidenceBadge(confidence: result.confidence)
                    Spacer(minLength: 0)
                    Button("Details", action: showDetails)
                        .font(.subheadline.weight(.semibold))
                }
            }

            if !edited && result.results.count > 1 {
                Text(result.results.map { "\($0.modelName) \($0.count.map { String($0) } ?? "–")" }
                    .joined(separator: "  ·  "))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
            }

            if let countOneSide {
                Button(action: countOneSide) {
                    Label("Count one side of the stack", systemImage: "line.diagonal")
                        .font(.subheadline.weight(.semibold))
                        .frame(maxWidth: .infinity, minHeight: 40)
                }
                .buttonStyle(.bordered)
            }

            if canEdit {
                Label("Check the numbers. Tap a wrong dot to remove it, or tap a missed item to add one.",
                      systemImage: "hand.tap")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            } else if result.retakeSuggested && !result.retakeTip.isEmpty {
                Label(result.retakeTip, systemImage: "camera.rotate")
                    .font(.footnote)
                    .lineLimit(2)
            } else if !result.notes.isEmpty {
                Text(result.notes)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
    }
}

// MARK: - Full result (sheet)

private struct ResultDetails: View {
    let result: CountResult
    @Binding var dotsModelID: String?

    private func isShown(_ r: CountResult.ModelResult) -> Bool {
        (dotsModelID ?? result.bestModel ?? result.results.first?.modelId) == r.modelId
    }

    var body: some View {
        NavigationStack {
            List {
                Section {
                    HStack {
                        Text(result.count.map { $0.formatted() } ?? "—")
                            .font(.system(size: 44, weight: .bold, design: .rounded))
                            .monospacedDigit()
                        Text(result.item).font(.headline)
                        Spacer()
                    }
                    ConfidenceBadge(confidence: result.confidence)
                }
                if result.results.count > 1 {
                    Section {
                        ForEach(result.results) { r in
                            Button {
                                dotsModelID = r.modelId
                            } label: {
                                HStack(spacing: 12) {
                                    Image(systemName: isShown(r) ? "largecircle.fill.circle" : "circle")
                                        .foregroundStyle(.tint)
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(r.modelName)
                                            .foregroundStyle(.primary)
                                        Text(r.error ?? "\(Int(r.seconds.rounded())) s")
                                            .font(.caption)
                                            .foregroundStyle(r.error == nil ? Color.secondary : Color.red)
                                    }
                                    Spacer()
                                    Text(r.count.map { $0.formatted() } ?? "–")
                                        .font(.title3.weight(.semibold))
                                        .monospacedDigit()
                                        .foregroundStyle(.primary)
                                }
                            }
                            .disabled(r.points.isEmpty)
                        }
                    } header: {
                        Text("Each model")
                    } footer: {
                        Text("Tap a model to show its dots on the photo. Auto shows the count in the middle.")
                    }
                }
                if !result.notes.isEmpty {
                    Section(result.results.count > 1 ? "Notes" : "What limits accuracy") { Text(result.notes) }
                }
                if result.retakeSuggested && !result.retakeTip.isEmpty {
                    Section("For a better count") {
                        Label(result.retakeTip, systemImage: "camera.rotate")
                    }
                }
                if result.sections.count > 1 {
                    Section("How it was counted") {
                        ForEach(Array(result.sections.enumerated()), id: \.offset) { _, section in
                            HStack {
                                Text(section.label.capitalized)
                                Spacer()
                                Text(section.count.formatted()).monospacedDigit()
                            }
                        }
                    }
                }
                Section {
                    Text("This is an AI estimate. Check it before recording it as your stock count.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Count details")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

private struct ConfidenceBadge: View {
    let confidence: CountResult.Confidence

    var body: some View {
        Label(text, systemImage: icon)
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(color)
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(color.opacity(0.15), in: Capsule())
            .accessibilityLabel(text)
    }

    private var text: String {
        switch confidence {
        case .high: "High confidence"
        case .medium: "Medium confidence"
        case .low: "Low confidence"
        }
    }

    private var icon: String {
        switch confidence {
        case .high: "checkmark.seal.fill"
        case .medium: "exclamationmark.circle.fill"
        case .low: "exclamationmark.triangle.fill"
        }
    }

    private var color: Color {
        switch confidence {
        case .high: .green
        case .medium: .orange
        case .low: .red
        }
    }
}

#Preview {
    ContentView()
}
