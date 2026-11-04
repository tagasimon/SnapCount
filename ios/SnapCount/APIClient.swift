import Foundation

/// An error with a message that can be shown to the user as-is.
struct UserFacingError: LocalizedError, Equatable {
    let message: String
    var errorDescription: String? { message }
}

final class APIClient {
    static let shared = APIClient()

    private let session: URLSession
    private let decoder: JSONDecoder

    init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = Config.requestTimeout
        config.timeoutIntervalForResource = Config.requestTimeout + 30
        config.waitsForConnectivity = false
        session = URLSession(configuration: config)

        decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
    }

    /// The models the server offers. Falls back to the built-in list on any failure.
    func models() async -> ModelsResponse? {
        var request = URLRequest(url: Config.backendURL.appendingPathComponent("models"))
        request.timeoutInterval = 10
        if !Config.appToken.isEmpty {
            request.setValue(Config.appToken, forHTTPHeaderField: "X-App-Token")
        }
        guard let reply = try? await session.data(for: request),
              (reply.1 as? HTTPURLResponse)?.statusCode == 200 else { return nil }
        return try? decoder.decode(ModelsResponse.self, from: reply.0)
    }

    /// `line` is [x0, y0, x1, y1] as fractions of the photo: count edges along it (stacks).
    func count(jpeg: Data, item: String, model: String, line: [Double]? = nil) async throws -> CountResult {
        let boundary = "SnapCount-\(UUID().uuidString)"
        var request = URLRequest(url: Config.backendURL.appendingPathComponent("count"))
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if !Config.appToken.isEmpty {
            request.setValue(Config.appToken, forHTTPHeaderField: "X-App-Token")
        }
        let lineText = line.map { $0.map { String(format: "%.4f", $0) }.joined(separator: ",") } ?? ""
        request.httpBody = Self.multipartBody(boundary: boundary, jpeg: jpeg, item: item, model: model, line: lineText)

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch let error as URLError {
            throw Self.friendly(error)
        }

        guard let http = response as? HTTPURLResponse else {
            throw UserFacingError(message: "The server sent an unexpected response.")
        }

        guard (200..<300).contains(http.statusCode) else {
            if let serverError = try? decoder.decode(ServerError.self, from: data) {
                throw UserFacingError(message: serverError.error)
            }
            throw UserFacingError(message: "The server had a problem (\(http.statusCode)). Please try again.")
        }

        do {
            return try decoder.decode(CountResult.self, from: data)
        } catch {
            throw UserFacingError(message: "The server's answer couldn't be read. Please try again.")
        }
    }

    private static func multipartBody(boundary: String, jpeg: Data, item: String, model: String, line: String) -> Data {
        var body = Data()
        func add(_ string: String) { body.append(Data(string.utf8)) }

        if !line.isEmpty {
            add("--\(boundary)\r\n")
            add("Content-Disposition: form-data; name=\"line\"\r\n\r\n")
            add("\(line)\r\n")
        }

        add("--\(boundary)\r\n")
        add("Content-Disposition: form-data; name=\"model\"\r\n\r\n")
        add("\(model)\r\n")

        add("--\(boundary)\r\n")
        add("Content-Disposition: form-data; name=\"item\"\r\n\r\n")
        add("\(item)\r\n")

        add("--\(boundary)\r\n")
        add("Content-Disposition: form-data; name=\"image\"; filename=\"photo.jpg\"\r\n")
        add("Content-Type: image/jpeg\r\n\r\n")
        body.append(jpeg)
        add("\r\n")

        add("--\(boundary)--\r\n")
        return body
    }

    private static func friendly(_ error: URLError) -> Error {
        switch error.code {
        case .cancelled:
            return CancellationError()
        case .notConnectedToInternet, .networkConnectionLost, .dataNotAllowed:
            return UserFacingError(message: "No internet connection. Check your connection and try again.")
        case .timedOut:
            return UserFacingError(message: "Counting took too long. Please try again.")
        case .cannotFindHost, .cannotConnectToHost, .dnsLookupFailed:
            return UserFacingError(message: "Can't reach the SnapCount server. Check that it's running and that the address in Config.swift is right.")
        case .appTransportSecurityRequiresSecureConnection:
            return UserFacingError(message: "This server address needs HTTPS. Use an https:// address in Config.swift.")
        default:
            return UserFacingError(message: "Something went wrong sending the photo. Please try again.")
        }
    }
}
