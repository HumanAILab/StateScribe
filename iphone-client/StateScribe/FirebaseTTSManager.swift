import Foundation
import FirebaseFirestore
import AVFoundation

class FirebaseTTSManager: NSObject, AVSpeechSynthesizerDelegate {

    private var db: Firestore
    private var listener: ListenerRegistration?
    private var ignoreSnapshotCount = 0
    private let speechSynthesizer = AVSpeechSynthesizer()
    private var isSpeaking = false
    private var latestMessage: String?
    private var suppressResumption = false
    private var suppressNewSpeech = false

    /// Called when TTS starts speaking a new message. Invoked on main queue.
    var onSpeechStarted: ((String) -> Void)?
    /// Called when TTS finishes or is cancelled. Invoked on main queue.
    var onSpeechEnded: (() -> Void)?

    override init() {
        self.db = Firestore.firestore()
        super.init()
        self.speechSynthesizer.delegate = self
        configureAudioSession()
    }

    private func configureAudioSession() {
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default, options: [.duckOthers, .interruptSpokenAudioAndMixWithOthers])
            try AVAudioSession.sharedInstance().setActive(true)
            print("Audio session configured successfully.")
        } catch {
            print("Failed to configure audio session: \(error.localizedDescription)")
        }
    }

    func startListening() {
        ignoreSnapshotCount = 1
        let docRef = db.collection("test").document("test_document")

        listener = docRef.addSnapshotListener { [weak self] documentSnapshot, error in
            guard let self = self else { return }

            if let error = error {
                print("Error fetching document: \(error.localizedDescription)")
                return
            }

            guard let document = documentSnapshot, document.exists else {
                print("Document does not exist or data was empty.")
                return
            }

            if self.ignoreSnapshotCount > 0 {
                self.ignoreSnapshotCount -= 1
                return
            }

            if let response = document.data()?["response"] as? String,
               !response.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                print("Current data: [\"response\": \"\(response)\"]")
                self.handleNewMessage(response)
                // Clear the response so we don't read it again on next launch
                docRef.updateData(["response": ""]) { err in
                    if let err = err {
                        print("Failed to clear 'response': \(err.localizedDescription)")
                    }
                }
            }
        }
    }

    func stopListening() {
        listener?.remove()
        speechSynthesizer.stopSpeaking(at: .immediate)
        latestMessage = nil
        isSpeaking = false
        print("FirebaseTTSManager stopped listening and cleared latest message.")
    }

    private func handleNewMessage(_ message: String) {
        latestMessage = message
        if suppressNewSpeech {
            return
        }
        if !isSpeaking {
            speakLatestMessage()
        }
    }

    private func speakLatestMessage() {
        guard let messageToSpeak = latestMessage, !isSpeaking else { return }
        isSpeaking = true
        latestMessage = nil
        let speechUtterance = AVSpeechUtterance(string: messageToSpeak)
        // speechUtterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        speechUtterance.voice = AVSpeechSynthesisVoice(identifier: "com.apple.voice.premium.en-US.Zoe")
        speechUtterance.rate = 0.55 // Increase speech rate (default is 0.5, range is 0.0-1.0)
        DispatchQueue.main.async {
            self.speechSynthesizer.speak(speechUtterance)
            self.onSpeechStarted?(messageToSpeak)
            print("Speaking: \(messageToSpeak)")
        }
    }

    // MARK: - AVSpeechSynthesizerDelegate

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        isSpeaking = false
        DispatchQueue.main.async { self.onSpeechEnded?() }
        print("Finished speaking: \(utterance.speechString)")
        speakLatestMessage()
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        isSpeaking = false
        DispatchQueue.main.async { self.onSpeechEnded?() }
        print("Cancelled speaking: \(utterance.speechString)")
        if suppressResumption {
            suppressResumption = false
            latestMessage = nil
            return
        }
        speakLatestMessage()
    }
    
    func startUserSpeech() {
        suppressNewSpeech = true
        suppressResumption = true
        latestMessage = nil
        if speechSynthesizer.isSpeaking {
            speechSynthesizer.stopSpeaking(at: .immediate)
        }
        isSpeaking = false
    }

    func stopForUserSpeech() {
        suppressResumption = true
        latestMessage = nil
        if speechSynthesizer.isSpeaking {
            speechSynthesizer.stopSpeaking(at: .immediate)
        }
        isSpeaking = false
        suppressNewSpeech = false
    }
    
    deinit {
        stopListening()
    }
}
