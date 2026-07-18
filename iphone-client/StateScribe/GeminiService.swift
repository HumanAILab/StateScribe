import Foundation
import UIKit

class GeminiService {

    static let shared = GeminiService()

    private let baseURL = "https://generativelanguage.googleapis.com/v1beta/models"
    private let apiKey: String
    private let model: String
    private let session: URLSession

    /// Initialise with values from AppConfig by default.
    init(apiKey: String = AppConfig.geminiAPIKey,
         model: String = AppConfig.geminiModel) {
        self.apiKey = apiKey
        self.model = model
        self.session = URLSession(configuration: .default)
    }

    // MARK: - Public

    /// Send a single text prompt to Gemini and receive the reply string.
    func sendMessage(_ prompt: String, completion: @escaping (Result<String, Error>) -> Void) {
        let endpoint = "\(baseURL)/\(model):generateContent"
        guard let url = URL(string: endpoint) else {
            completion(.failure(GeminiError.invalidURL))
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue(apiKey, forHTTPHeaderField: "x-goog-api-key")
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "contents": [
                [
                    "parts": [
                        ["text": prompt]
                    ]
                ]
            ]
        ]

        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        } catch {
            completion(.failure(error))
            return
        }

        session.dataTask(with: request) { data, response, error in
            if let error = error {
                DispatchQueue.main.async { completion(.failure(error)) }
                return
            }

            guard let data = data else {
                DispatchQueue.main.async { completion(.failure(GeminiError.noData)) }
                return
            }

            do {
                let reply = try self.parseResponse(data)
                DispatchQueue.main.async { completion(.success(reply)) }
            } catch {
                DispatchQueue.main.async { completion(.failure(error)) }
            }
        }.resume()
    }

    /// Send a multi-turn conversation (chat history) to Gemini.
    /// Each message is a tuple of (role: "user" | "model", text: String).
    func sendChat(messages: [(role: String, text: String)], completion: @escaping (Result<String, Error>) -> Void) {
        let endpoint = "\(baseURL)/\(model):generateContent"
        guard let url = URL(string: endpoint) else {
            completion(.failure(GeminiError.invalidURL))
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue(apiKey, forHTTPHeaderField: "x-goog-api-key")
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")

        let contents: [[String: Any]] = messages.map { msg in
            [
                "role": msg.role,
                "parts": [["text": msg.text]]
            ]
        }

        let body: [String: Any] = ["contents": contents]

        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        } catch {
            completion(.failure(error))
            return
        }

        session.dataTask(with: request) { data, _, error in
            if let error = error {
                DispatchQueue.main.async { completion(.failure(error)) }
                return
            }
            guard let data = data else {
                DispatchQueue.main.async { completion(.failure(GeminiError.noData)) }
                return
            }
            do {
                let reply = try self.parseResponse(data)
                DispatchQueue.main.async { completion(.success(reply)) }
            } catch {
                DispatchQueue.main.async { completion(.failure(error)) }
            }
        }.resume()
    }

    /// Send an image with a text prompt to Gemini (multimodal).
    func sendImageMessage(image: UIImage, prompt: String, completion: @escaping (Result<String, Error>) -> Void) {
        guard let jpegData = image.jpegData(compressionQuality: 0.6) else {
            completion(.failure(GeminiError.invalidResponse))
            return
        }
        let base64 = jpegData.base64EncodedString()

        let endpoint = "\(baseURL)/\(model):generateContent"
        guard let url = URL(string: endpoint) else {
            completion(.failure(GeminiError.invalidURL))
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue(apiKey, forHTTPHeaderField: "x-goog-api-key")
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "contents": [
                [
                    "parts": [
                        ["text": prompt],
                        [
                            "inline_data": [
                                "mime_type": "image/jpeg",
                                "data": base64
                            ]
                        ]
                    ]
                ]
            ]
        ]

        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        } catch {
            completion(.failure(error))
            return
        }

        session.dataTask(with: request) { data, _, error in
            if let error = error {
                DispatchQueue.main.async { completion(.failure(error)) }
                return
            }
            guard let data = data else {
                DispatchQueue.main.async { completion(.failure(GeminiError.noData)) }
                return
            }
            do {
                let reply = try self.parseResponse(data)
                DispatchQueue.main.async { completion(.success(reply)) }
            } catch {
                DispatchQueue.main.async { completion(.failure(error)) }
            }
        }.resume()
    }

    // MARK: - Private

    private func parseResponse(_ data: Data) throws -> String {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw GeminiError.invalidResponse
        }

        // Check for API error
        if let error = json["error"] as? [String: Any],
           let message = error["message"] as? String {
            throw GeminiError.apiError(message)
        }

        guard let candidates = json["candidates"] as? [[String: Any]],
              let first = candidates.first,
              let content = first["content"] as? [String: Any],
              let parts = content["parts"] as? [[String: Any]],
              let text = parts.first?["text"] as? String else {
            throw GeminiError.invalidResponse
        }

        return text
    }
}

// MARK: - Error Types

enum GeminiError: LocalizedError {
    case invalidURL
    case noData
    case invalidResponse
    case apiError(String)

    var errorDescription: String? {
        switch self {
        case .invalidURL:       return "Invalid Gemini API URL."
        case .noData:           return "No data received from Gemini."
        case .invalidResponse:  return "Could not parse Gemini response."
        case .apiError(let msg): return "Gemini API error: \(msg)"
        }
    }
}
