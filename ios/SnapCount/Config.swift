import Foundation

/// The only two values you normally need to change.
enum Config {
    /// Where the SnapCount backend runs.
    /// - Local testing on a real iPhone: your computer's LAN IP, e.g. "http://192.168.1.20:8000"
    ///   (iPhone and computer on the same Wi-Fi).
    /// - Deployed: your HTTPS domain, e.g. "https://snapcount.example.com"
    static let backendURL = URL(string: "http://192.168.100.234:8000")!

    /// Must match APP_TOKEN on the server. Leave empty if the server has no APP_TOKEN set.
    static let appToken = ""

    /// How long to wait for a count before giving up. Auto runs three models one after
    /// another on a CPU server, which can take a minute or two.
    static let requestTimeout: TimeInterval = 180
}
