import Foundation

enum AppConfig {

    // MARK: - Frame Sender

    static let frameSenderHost = Bundle.main.object(forInfoDictionaryKey: "FrameSenderHost") as? String ?? "192.168.1.2"
    static let frameSenderPort = UInt16(Bundle.main.object(forInfoDictionaryKey: "FrameSenderPort") as? String ?? "1234") ?? 1234
    static let frameSendFPS: Double = 1.0
    static var frameSendInterval: TimeInterval { 1.0 / frameSendFPS }

    // MARK: - Auto Save

    static let autoSaveInterval: TimeInterval = 10.0

    // MARK: - Gemini

    static let geminiAPIKey = Bundle.main.object(forInfoDictionaryKey: "GeminiAPIKey") as? String ?? ""
    static let geminiModel = "gemini-2.5-flash"

    // MARK: - Data Logger

    static let dataLogFPS: Double = 30.0
    static let dataLogEnabled: Bool = true
}
