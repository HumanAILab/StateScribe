/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
Main view controller for the AR experience.
*/

import UIKit
import SceneKit
import ARKit
import AVFoundation
import CoreHaptics
import Speech
import FirebaseFirestore

extension Notification.Name {
    static let mapDidSave = Notification.Name("mapDidSave")
}

class ViewController: UIViewController, ARSCNViewDelegate, ARSessionDelegate {
    // MARK: - IBOutlets
    
    @IBOutlet weak var sessionInfoView: UIView!
    @IBOutlet weak var sessionInfoLabel: UILabel!
    @IBOutlet weak var sceneView: ARSCNView!
    @IBOutlet weak var saveExperienceButton: UIButton!
    @IBOutlet weak var statusLabel: UILabel!
    @IBOutlet weak var snapshotThumbnail: UIImageView!
    
    // MARK: - Properties
    var mapURLToLoad: URL? // Added to specify which map to load, if any
    var virtualObjectAnchor: ARAnchor?
    let virtualObjectAnchorName = "virtualObject"
    var virtualObject: SCNNode = {
        guard let sceneURL = Bundle.main.url(forResource: "cup", withExtension: "scn", subdirectory: "Assets.scnassets/cup"),
            let referenceNode = SCNReferenceNode(url: sceneURL) else {
                fatalError("can't load virtual object")
        }
        referenceNode.load()
        return referenceNode
    }()
    
    var isRelocalizingMap = false
    
    // Store generated map name for new scenes (display / file name)
    var generatedMapName: String?
    // Payload name sent to ARDataSender: sanitized_with_underscores + timestamp (same as saved file name when new)
    private var mapPayloadName: String?
    private var creationTimestamp: String?

    var defaultConfiguration: ARWorldTrackingConfiguration {
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = .horizontal
        configuration.frameSemantics.insert(.sceneDepth)
        configuration.environmentTexturing = .automatic
        return configuration
    }
    
    let mapsDirectoryURL: URL = { // Added maps directory URL
        do {
            return try FileManager.default
                .url(for: .documentDirectory,
                     in: .userDomainMask,
                     appropriateFor: nil,
                     create: true)
                .appendingPathComponent("ARWorldMaps")
        } catch {
            fatalError("Can't get maps directory URL: \(error.localizedDescription)")
        }
    }()
    
    var arDataSender: ARDataSender? // add ARDataSender property
    private var arDataLogger: ARDataLogger?
    private var firebaseTTSManager: FirebaseTTSManager? // Add FirebaseTTSManager property
    private var isShowingTTSMessage = false // When true, sessionInfoLabel shows TTS text; AR updates are skipped
    private var autoSaveTimer: Timer?
    private var hasRequestedGeminiName = false
    private var recordingTimerLabel: UILabel?
    private var recordingDisplayTimer: Timer?
    private var depthOverlayImageView: UIImageView?
    private var lastDepthUpdateTime: CFTimeInterval = 0
    private let depthOverlayInterval: CFTimeInterval = 0.1
    private var hapticEngine: CHHapticEngine?
    
    // MARK: - Speech Recognition
    private let audioEngine = AVAudioEngine()
    private var speechRecognizer: SFSpeechRecognizer? = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var currentTranscript: String = ""
    private var recordingGeneration: Int = 0
    private var isRecordingSpeech = false
    private let firestore = Firestore.firestore()

    // MARK: - Frame Sender Config (from Firestore)
    private var frameSenderHost: String?
    private var frameSenderPort: UInt16?
    private var isFrameSenderDisabled: Bool = true
    
    private func fetchFrameSenderConfig() {
        let docRef = firestore.collection("test").document("test_document")
        docRef.getDocument { [weak self] snapshot, error in
            guard let self = self else { return }
            if let error = error {
                print("Failed to fetch frame sender config: \(error.localizedDescription)")
                self.isFrameSenderDisabled = true
                return
            }
            guard let data = snapshot?.data() else {
                self.isFrameSenderDisabled = true
                return
            }
            // Accept ip as String or Number
            var ipString: String?
            if let ipVal = data["ip"] as? String { ipString = ipVal }
            else if let ipNum = data["ip"] as? NSNumber { ipString = ipNum.stringValue }
            else if let ipInt = data["ip"] as? Int { ipString = String(ipInt) }
            
            var portValue: UInt16?
            if let portNum = data["port"] as? NSNumber { portValue = portNum.uint16Value }
            else if let portInt = data["port"] as? Int { portValue = UInt16(portInt) }
            else if let portStr = data["port"] as? String, let p = UInt16(portStr) { portValue = p }

            if let ip = ipString, ip != "-1", let port = portValue {
                self.frameSenderHost = ip
                self.frameSenderPort = port
                self.isFrameSenderDisabled = false
                print("[FrameSender] config: using IP \(ip), port \(port)")
            } else {
                self.isFrameSenderDisabled = true
                self.frameSenderHost = nil
                self.frameSenderPort = nil
                print("[FrameSender] config: disabled (ip=\(ipString ?? "nil"), port=\(portValue?.description ?? "nil"))")
            }
        }
    }

    private func maybeStartFrameSenderIfNeeded(worldName: String, frame: ARFrame) {
        if isFrameSenderDisabled {
            print("[FrameSender] blocked: isFrameSenderDisabled=true")
            return
        }
        guard let host = frameSenderHost, let port = frameSenderPort else {
            print("[FrameSender] blocked: host=\(frameSenderHost ?? "nil"), port=\(String(describing: frameSenderPort))")
            return
        }
        if arDataSender == nil {
            print("[FrameSender] creating ARDataSender → \(host):\(port)")
            arDataSender = ARDataSender(host: host, port: port, sceneView: sceneView)
        }
        if arDataSender?.isSendingActive == false {
            print("[FrameSender] starting sender for world: \(worldName)")
            arDataSender?.start(worldName: worldName)
        }
    }

    // MARK: - View Life Cycle
    override var shouldAutorotate: Bool {
        return false
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Create maps directory if it doesn't exist
        let fileManager = FileManager.default
        if (!fileManager.fileExists(atPath: mapsDirectoryURL.path)) {
            do {
                try fileManager.createDirectory(at: mapsDirectoryURL, withIntermediateDirectories: true, attributes: nil)
            } catch {
                print("Error creating maps directory: \(error.localizedDescription)")
            }
        }
        
        // If a mapURLToLoad is provided, attempt to load it
        if mapURLToLoad != nil {
            // The actual loading will happen in viewDidAppear after ARSession is running
            // We can hide the save button initially if we are loading,
            // and only show it if the user modifies the scene or wants to save a new copy.
            // For simplicity, we'll keep it visible.
            // The session will be configured to load the map in viewDidAppear.
        }
        
        // Initialize FirebaseTTSManager
        let tts = FirebaseTTSManager()
        tts.onSpeechStarted = { [weak self] message in
            DispatchQueue.main.async {
                self?.sessionInfoLabel.text = message
                self?.sessionInfoView.isHidden = false
                self?.isShowingTTSMessage = true
            }
        }
        tts.onSpeechEnded = { [weak self] in
            DispatchQueue.main.async {
                self?.isShowingTTSMessage = false
            }
        }
        firebaseTTSManager = tts
        
        // Set up Core Haptics engine for strong vibration
        if CHHapticEngine.capabilitiesForHardware().supportsHaptics {
            do {
                hapticEngine = try CHHapticEngine()
                hapticEngine?.isAutoShutdownEnabled = false
                try hapticEngine?.start()
            } catch {
                print("Haptic engine failed to start: \(error)")
            }
        }

        // Add press-and-hold gesture for speech input
        let holdRecognizer = UILongPressGestureRecognizer(target: self, action: #selector(handleHoldToTalk(_:)))
        holdRecognizer.minimumPressDuration = 0
        holdRecognizer.cancelsTouchesInView = false
        sceneView.addGestureRecognizer(holdRecognizer)

        // Fetch frame sender config (ip, port) from Firestore
        fetchFrameSenderConfig()

        // Save and stop logger when app goes to background or terminates
        NotificationCenter.default.addObserver(self, selector: #selector(appWillResignActive), name: UIApplication.willResignActiveNotification, object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(appWillTerminate), name: UIApplication.willTerminateNotification, object: nil)

        // Recording time overlay
        setupRecordingTimerLabel()

        // Depth map overlay
        setupDepthOverlay()
    }

    private func setupDepthOverlay() {
        let iv = UIImageView()
        iv.contentMode = .scaleAspectFit
        iv.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        iv.layer.cornerRadius = 8
        iv.clipsToBounds = true
        iv.translatesAutoresizingMaskIntoConstraints = false
        iv.isUserInteractionEnabled = false
        sceneView.addSubview(iv)
        NSLayoutConstraint.activate([
            iv.leadingAnchor.constraint(equalTo: sceneView.safeAreaLayoutGuide.leadingAnchor, constant: 8),
            iv.bottomAnchor.constraint(equalTo: sceneView.safeAreaLayoutGuide.bottomAnchor, constant: -8),
            iv.widthAnchor.constraint(equalToConstant: 120),
            iv.heightAnchor.constraint(equalToConstant: 90),
        ])
        depthOverlayImageView = iv
    }

    private func updateDepthOverlay(with depthMap: CVPixelBuffer) {
        let now = CACurrentMediaTime()
        guard now - lastDepthUpdateTime >= depthOverlayInterval else { return }
        lastDepthUpdateTime = now

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let image = self?.depthBufferToImage(depthMap) else { return }
            DispatchQueue.main.async {
                self?.depthOverlayImageView?.image = image
            }
        }
    }

    private func depthBufferToImage(_ depthBuffer: CVPixelBuffer) -> UIImage? {
        let width = CVPixelBufferGetWidth(depthBuffer)
        let height = CVPixelBufferGetHeight(depthBuffer)
        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(depthBuffer) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(depthBuffer)
        let floatData = base.assumingMemoryBound(to: Float32.self)

        let minDepth: Float = 0.1
        let maxDepth: Float = 5.0

        var gray = [UInt8](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let f = floatData[y * (bytesPerRow / 4) + x]
                let clamped = (f < minDepth || f > maxDepth) ? maxDepth : f
                let t = (clamped - minDepth) / (maxDepth - minDepth)
                gray[y * width + x] = UInt8((1.0 - t) * 255)
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let cgData = CFDataCreate(nil, gray, width * height),
              let provider = CGDataProvider(data: cgData) else { return nil }
        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else { return nil }
        return UIImage(cgImage: cgImage)
    }

    private func setupRecordingTimerLabel() {
        let label = UILabel()
        label.font = .monospacedDigitSystemFont(ofSize: 14, weight: .semibold)
        label.textColor = .white
        label.backgroundColor = UIColor.systemRed.withAlphaComponent(0.75)
        label.textAlignment = .center
        label.layer.cornerRadius = 10
        label.clipsToBounds = true
        label.isHidden = true
        label.translatesAutoresizingMaskIntoConstraints = false
        sceneView.addSubview(label)
        NSLayoutConstraint.activate([
            label.topAnchor.constraint(equalTo: sceneView.safeAreaLayoutGuide.topAnchor, constant: 8),
            label.trailingAnchor.constraint(equalTo: sceneView.trailingAnchor, constant: -12),
            label.heightAnchor.constraint(equalToConstant: 28),
            label.widthAnchor.constraint(greaterThanOrEqualToConstant: 70),
        ])
        recordingTimerLabel = label
    }

    private func startRecordingDisplayTimer() {
        recordingDisplayTimer?.invalidate()
        recordingTimerLabel?.isHidden = false
        updateRecordingTimerLabel()
        recordingDisplayTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.updateRecordingTimerLabel()
        }
    }

    private func stopRecordingDisplayTimer() {
        recordingDisplayTimer?.invalidate()
        recordingDisplayTimer = nil
        recordingTimerLabel?.isHidden = true
    }

    private func updateRecordingTimerLabel() {
        guard let startDate = arDataLogger?.recordingStartDate else {
            recordingTimerLabel?.text = ""
            return
        }
        let elapsed = Int(Date().timeIntervalSince(startDate))
        let minutes = elapsed / 60
        let seconds = elapsed % 60
        recordingTimerLabel?.text = String(format: " ⏺ %d:%02d ", minutes, seconds)
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        
        guard ARWorldTrackingConfiguration.isSupported else {
            fatalError("""
                ARKit is not available on this device. For apps that require ARKit
                for core functionality, use the `arkit` key in the key in the
                `UIRequiredDeviceCapabilities` section of the Info.plist to prevent
                the app from installing. (If the app can't be installed, this error
                can't be triggered in a production scenario.)
                In apps where AR is an additive feature, use `isSupported` to
                determine whether to show UI for launching AR experiences.
            """) // For details, see https://developer.apple.com/documentation/arkit
        }
        
        // Start the view's AR session.
        sceneView.session.delegate = self
        
        if let mapURL = mapURLToLoad, let worldMap = loadWorldMap(from: mapURL) {
            print("Loading map from URL: \(mapURL)")
            let configuration = self.defaultConfiguration
            configuration.initialWorldMap = worldMap
            sceneView.session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
            isRelocalizingMap = true
            virtualObjectAnchor = worldMap.anchors.first(where: { $0.name == virtualObjectAnchorName })
            // Remove snapshot anchors from the world map since they are only used for thumbnails in the map list.
            worldMap.anchors.removeAll(where: { $0 is SnapshotAnchor })
        } else {
            print("Starting new session or map not found.")
            let df = DateFormatter()
            df.dateFormat = "yyyyMMdd_HHmmss"
            creationTimestamp = df.string(from: Date())
            sceneView.session.run(defaultConfiguration)
        }
        
        sceneView.debugOptions = [ .showFeaturePoints ]
        
        // Prevent the screen from being dimmed after a while as users will likely
        // have long periods of interaction without touching the screen or buttons.
        UIApplication.shared.isIdleTimerDisabled = true
        
        // Determine map name
        let mapName: String?
        if let url = mapURLToLoad {
            mapName = url.deletingPathExtension().lastPathComponent
            generatedMapName = nil
        } else {
            // If new scene, generate a unique map name (fallback until Gemini responds)
            if generatedMapName == nil {
                let dateFormatter = DateFormatter()
                dateFormatter.dateFormat = "yyyyMMdd HHmmss"
                generatedMapName = "Map \(dateFormatter.string(from: Date()))"
            }
            mapName = generatedMapName
        }
        
        // Frame sender will be started in session(didUpdate:) when tracking is normal and config is available
        
        // Local data logger will be started after Gemini returns the name (in requestMapNameFromGemini)
        // For existing maps, start it immediately
        if AppConfig.dataLogEnabled, mapURLToLoad != nil, let name = mapName {
            let df = DateFormatter()
            df.dateFormat = "yyyyMMdd_HHmmss"
            arDataLogger = ARDataLogger(sceneView: sceneView)
            arDataLogger?.start(sessionName: "\(name)_\(df.string(from: Date()))")
            startRecordingDisplayTimer()
        }

        // Start auto-save timer
        startAutoSaveTimer()

        // Start listening for TTS messages
        firebaseTTSManager?.startListening()

    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)

        // Save before pausing — session must still be active to get the world map
        stopAutoSaveTimer()
        saveWorldMap()

        // Stop logger and finalize video before pausing session (back button / swipe back path)
        stopRecordingDisplayTimer()
        arDataLogger?.stop()
        arDataLogger = nil

        // Pause the view's AR session.
        sceneView.session.pause()
        arDataSender?.stop() // stop sender when leaving scene
        arDataSender = nil   // Deinitialize ARDataSender to close TCP connection

        // Stop listening for TTS messages
        firebaseTTSManager?.stopListening()

        NotificationCenter.default.removeObserver(self, name: UIApplication.willResignActiveNotification, object: nil)
        NotificationCenter.default.removeObserver(self, name: UIApplication.willTerminateNotification, object: nil)
    }
    
    // MARK: - ARSCNViewDelegate
    
    func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
        guard anchor.name == virtualObjectAnchorName
            else { return }
        
        // save the reference to the virtual object anchor when the anchor is added from relocalizing
        if virtualObjectAnchor == nil {
            virtualObjectAnchor = anchor
        }
        node.addChildNode(virtualObject)
    }
    
    // MARK: - ARSessionDelegate
    
    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        print("[Tracking] state changed to: \(camera.trackingState), isRelocalizingMap=\(isRelocalizingMap)")
        updateSessionInfoLabel(for: session.currentFrame!, trackingState: camera.trackingState)

        // Relocalization is complete once tracking returns to normal
        if isRelocalizingMap && camera.trackingState == .normal {
            isRelocalizingMap = false
            print("[Tracking] Relocalization complete, isRelocalizingMap set to false")
        }
    }
    
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // Enable Save button only when the mapping status is good
        switch frame.worldMappingStatus {
        case .extending, .mapped:
            saveExperienceButton.isEnabled = true
        default:
            saveExperienceButton.isEnabled = false
        }
        statusLabel.text = """
        Mapping: \(frame.worldMappingStatus.description)
        Tracking: \(frame.camera.trackingState.description)
        """
        
        updateSessionInfoLabel(for: frame, trackingState: frame.camera.trackingState)

        // For new scenes, request a map name from Gemini using the first good frame
        if mapURLToLoad == nil && !hasRequestedGeminiName && frame.camera.trackingState == .normal {
            hasRequestedGeminiName = true
            requestMapNameFromGemini(frame: frame)
        }

        // Start/stop ARDataSender based on tracking state and remote config
        // For new scenes, wait until Gemini has provided the name before sending
        let isNewScene = (mapURLToLoad == nil)
        let senderName: String?
        if isNewScene {
            senderName = mapPayloadName
        } else {
            senderName = mapURLToLoad?.deletingPathExtension().lastPathComponent
        }

        if let name = senderName {
            if isNewScene {
                // New scenes: require normal tracking (Gemini name is already resolved at this point)
                if frame.camera.trackingState == .normal {
                    maybeStartFrameSenderIfNeeded(worldName: name, frame: frame)
                } else {
                    arDataSender?.stop()
                }
            } else {
                // Existing maps: start sending even during relocalization
                maybeStartFrameSenderIfNeeded(worldName: name, frame: frame)
            }
        }

        if let depthMap = frame.sceneDepth?.depthMap ?? frame.smoothedSceneDepth?.depthMap {
            updateDepthOverlay(with: depthMap)
        }
    }
    
    // MARK: - ARSessionObserver
    
    func sessionWasInterrupted(_ session: ARSession) {
        // Inform the user that the session has been interrupted, for example, by presenting an overlay.
        sessionInfoLabel.text = "Session was interrupted"
    }
    
    func sessionInterruptionEnded(_ session: ARSession) {
        // Reset tracking and/or remove existing anchors if consistent tracking is required.
        sessionInfoLabel.text = "Session interruption ended"
    }
    
    func session(_ session: ARSession, didFailWithError error: Error) {
        sessionInfoLabel.text = "Session failed: \(error.localizedDescription)"
        guard error is ARError else { return }
        
        let errorWithInfo = error as NSError
        let messages = [
            errorWithInfo.localizedDescription,
            errorWithInfo.localizedFailureReason,
            errorWithInfo.localizedRecoverySuggestion
        ]
        
        // Remove optional error messages.
        let errorMessage = messages.compactMap({ $0 }).joined(separator: "\n")
        
        DispatchQueue.main.async {
            // Present an alert informing about the error that has occurred.
            let alertController = UIAlertController(title: "The AR session failed.", message: errorMessage, preferredStyle: .alert)
            let restartAction = UIAlertAction(title: "Restart Session", style: .default) { _ in
                alertController.dismiss(animated: true, completion: nil)
                self.resetTracking(nil)
            }
            alertController.addAction(restartAction)
            self.present(alertController, animated: true, completion: nil)
        }
    }
    
    func sessionShouldAttemptRelocalization(_ session: ARSession) -> Bool {
        return true
    }
    
    // MARK: - Persistence: Saving and Loading
    
    @IBAction func saveExperience(_ button: UIButton) {
        print("[VideoDebug] Save Experience tapped; finalizing video first, then saving map")
        // Finalize video while AR session is still active, then save map and pop
        stopRecordingDisplayTimer()
        arDataLogger?.stop()
        arDataLogger = nil

        saveWorldMap { [weak self] success in
            print("[VideoDebug] saveWorldMap completed: success=\(success)")
            guard success, let self = self else { return }
            DispatchQueue.main.async {
                if self.navigationController?.topViewController == self {
                    self.navigationController?.popViewController(animated: true)
                }
            }
        }
    }

    /// Core save logic shared by manual save and auto-save.
    /// Calls completion on the main queue with `true` if save succeeded.
    private func saveWorldMap(completion: ((Bool) -> Void)? = nil) {
        sceneView.session.getCurrentWorldMap { [weak self] worldMap, error in
            guard let self = self, let map = worldMap else {
                if let error = error {
                    print("Auto-save: Can't get current world map – \(error.localizedDescription)")
                }
                DispatchQueue.main.async { completion?(false) }
                return
            }

            map.anchors.removeAll(where: { $0 is SnapshotAnchor })
            if let snapshotAnchor = SnapshotAnchor(capturing: self.sceneView) {
                map.anchors.append(snapshotAnchor)
            }

            let saveURL: URL
            if let existingURL = self.mapURLToLoad {
                saveURL = existingURL
            } else {
                let mapName = self.mapPayloadName ?? self.generatedMapName ?? "Map_Unknown"
                saveURL = self.mapsDirectoryURL.appendingPathComponent(mapName).appendingPathExtension("arexperience")
            }

            do {
                let data = try NSKeyedArchiver.archivedData(withRootObject: map, requiringSecureCoding: true)
                try data.write(to: saveURL, options: [.atomic])
                print("Map saved to: \(saveURL.lastPathComponent)")
                DispatchQueue.main.async {
                    NotificationCenter.default.post(name: .mapDidSave, object: nil)
                    completion?(true)
                }
            } catch {
                print("Failed to save ARWorldMap: \(error.localizedDescription)")
                DispatchQueue.main.async { completion?(false) }
            }
        }
    }

    // MARK: - Auto Save

    private func startAutoSaveTimer() {
        stopAutoSaveTimer()
        let interval = AppConfig.autoSaveInterval
        guard interval > 0 else { return }
        autoSaveTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            self?.autoSaveTick()
        }
        print("Auto-save enabled every \(Int(interval))s")
    }

    private func stopAutoSaveTimer() {
        autoSaveTimer?.invalidate()
        autoSaveTimer = nil
    }

    @objc private func autoSaveTick() {
        guard let frame = sceneView.session.currentFrame else { return }
        if mapURLToLoad == nil && mapPayloadName == nil { return }
        switch frame.worldMappingStatus {
        case .extending, .mapped:
            saveWorldMap()
        default:
            break
        }
    }

    // MARK: - Gemini Map Naming

    private func requestMapNameFromGemini(frame: ARFrame) {
        let ciImage = CIImage(cvPixelBuffer: frame.capturedImage).oriented(.right)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }
        let uiImage = UIImage(cgImage: cgImage)

        let prompt = """
        Describe where is the scene in 3 to 5 words that would make a good short title. \
        Reply with ONLY the title, no quotes, no punctuation, no explanation. \
        Example: Living Room with a couch and a TV, Bedroom with a bed and a dresser, Kitchen with a table and chairs
        """

        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyyMMdd_HHmmss"
        let timestamp = self.creationTimestamp ?? dateFormatter.string(from: Date())

        GeminiService.shared.sendImageMessage(image: uiImage, prompt: prompt) { [weak self] result in
            guard let self = self else { return }
            switch result {
            case .success(let rawName):
                let displayName = self.sanitizeMapName(rawName, keepSpaces: true)
                let payloadName = self.sanitizeMapName(rawName, keepSpaces: false)
                if !displayName.isEmpty {
                    self.generatedMapName = displayName
                    self.mapPayloadName = "\(payloadName)_\(timestamp)"
                    // Update the running sender's world name immediately
                    self.arDataSender?.currentWorldName = self.mapPayloadName
                    print("Map created at: \(timestamp)")
                    print("Display name: \(displayName)")
                    print("Payload name: \(self.mapPayloadName ?? "")")

                    // Start local data logger now that we have the name
                    if AppConfig.dataLogEnabled, self.arDataLogger == nil {
                        self.arDataLogger = ARDataLogger(sceneView: self.sceneView)
                        self.arDataLogger?.start(sessionName: "\(payloadName)_\(timestamp)")
                        self.startRecordingDisplayTimer()
                    }
                }
            case .failure(let error):
                print("Gemini map naming failed, keeping default name: \(error.localizedDescription)")
            }
        }
    }

    /// Sanitize a raw name. When `keepSpaces` is true, spaces are preserved (for display / file name).
    /// When false, spaces become underscores (for payload).
    private func sanitizeMapName(_ raw: String, keepSpaces: Bool) -> String {
        var name = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        name = name.replacingOccurrences(of: "\"", with: "")
        name = name.replacingOccurrences(of: "'", with: "")
        if !keepSpaces {
            name = name.replacingOccurrences(of: " ", with: "_")
        }
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: keepSpaces ? " _-" : "_-"))
        name = String(name.unicodeScalars.filter { allowed.contains($0) })
        if name.count > 50 { name = String(name.prefix(50)) }
        return name
    }

    @objc private func appWillResignActive() {
        saveWorldMap()
        // Finalize video so the file is intact even if the app is later killed
        stopRecordingDisplayTimer()
        arDataLogger?.stop()
        arDataLogger = nil
        // Stop sending to TCP server when app goes to background (resumes on next frame when app becomes active)
        arDataSender?.stop()
    }

    @objc private func appWillTerminate() {
        stopRecordingDisplayTimer()
        arDataLogger?.stop()
        arDataLogger = nil
        arDataSender?.stop()
    }

    // Helper function to load a world map from a given URL
    func loadWorldMap(from url: URL) -> ARWorldMap? {
        guard let data = try? Data(contentsOf: url) else {
            print("Could not load data from URL: \(url)")
            return nil
        }
        do {
            guard let worldMap = try NSKeyedUnarchiver.unarchivedObject(ofClass: ARWorldMap.self, from: data)
                else {
                    print("No ARWorldMap in archive at URL: \(url)")
                    return nil
            }
            return worldMap
        } catch {
            print("Can't unarchive ARWorldMap from file data at URL \(url): \(error)")
            return nil
        }
    }

    // MARK: - AR session management
    
    @IBAction func resetTracking(_ sender: UIButton?) {
        sceneView.session.run(defaultConfiguration, options: [.resetTracking, .removeExistingAnchors])
        isRelocalizingMap = false
        virtualObjectAnchor = nil
        arDataSender?.stop() // stop sender on reset
        startAutoSaveTimer() // restart auto-save timer
    }
    
    private func updateSessionInfoLabel(for frame: ARFrame, trackingState: ARCamera.TrackingState) {
        if isShowingTTSMessage { return }

        // Update the UI to provide feedback on the state of the AR experience.
        let message: String

        switch (trackingState, frame.worldMappingStatus) {
        case (.normal, .mapped),
             (.normal, .extending):
            if saveExperienceButton.isEnabled {
                message = "Tap 'Save Experience' to save the current map, or tap on the screen to place an object."
            } else {
                message = "Move around to map the environment. Tap on the screen to place an object once mapping is sufficient."
            }
            
        case (.normal, _) where mapURLToLoad != nil && !isRelocalizingMap:
            message = "Move around to map the environment or tap 'Load Experience' to load a saved experience."
            
        case (.normal, _) where mapURLToLoad == nil:
            message = "Move around to map the environment."
            
        case (.limited(.relocalizing), _) where isRelocalizingMap:
            message = "Move your device to the saved location to relocalize."
            
        default:
            message = trackingState.localizedFeedback
        }
        
        sessionInfoLabel.text = message
        sessionInfoView.isHidden = message.isEmpty
    }
    
    // MARK: - Placing AR Content
    
    @IBAction func handleSceneTap(_ sender: UITapGestureRecognizer) {
        if isRecordingSpeech { return }

        // Disable placing objects when the session is still relocalizing
        if isRelocalizingMap && virtualObjectAnchor == nil {
            return
        }
        // Hit test to find a place for a virtual object.
        guard let hitTestResult = sceneView
            .hitTest(sender.location(in: sceneView), types: [.existingPlaneUsingGeometry, .estimatedHorizontalPlane])
            .first
            else { return }
        
        // Remove exisitng anchor and add new anchor
        if let existingAnchor = virtualObjectAnchor {
            sceneView.session.remove(anchor: existingAnchor)
        }
        virtualObjectAnchor = ARAnchor(name: virtualObjectAnchorName, transform: hitTestResult.worldTransform)
        sceneView.session.add(anchor: virtualObjectAnchor!)
    }
}

// MARK: - Speech Handling
extension ViewController {
    private func requestSpeechAndMicPermissions(completion: @escaping (Bool) -> Void) {
        SFSpeechRecognizer.requestAuthorization { status in
            let speechGranted = (status == .authorized)
            AVAudioSession.sharedInstance().requestRecordPermission { micGranted in
                completion(speechGranted && micGranted)
            }
        }
    }

    @objc func handleHoldToTalk(_ gesture: UILongPressGestureRecognizer) {
        switch gesture.state {
        case .began:
            isRecordingSpeech = true
            playStrongHaptic()
            firebaseTTSManager?.startUserSpeech()
            requestSpeechAndMicPermissions { [weak self] granted in
                guard let self = self else { return }
                if granted {
                    DispatchQueue.main.async {
                        guard self.isRecordingSpeech else { return }
                        self.startRecording()
                    }
                } else {
                    print("Speech or Mic permission not granted")
                }
            }
        case .ended, .cancelled, .failed:
            isRecordingSpeech = false
            playReleaseHaptic()
            firebaseTTSManager?.stopForUserSpeech()
            stopRecordingAndSend()
        default:
            break
        }
    }

    private func configureRecordingSession() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.duckOthers, .defaultToSpeaker, .allowBluetooth])
        try session.setActive(true, options: .notifyOthersOnDeactivation)
    }

    private func startRecording() {
        do {
            try configureRecordingSession()
        } catch {
            print("Failed to configure recording session: \(error)")
            return
        }

        recognitionTask?.cancel()
        recognitionTask = nil
        currentTranscript = ""

        recordingGeneration += 1
        let thisGeneration = recordingGeneration

        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else { return }
        recognitionRequest.shouldReportPartialResults = true

        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        inputNode.removeTap(onBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] (buffer, _) in
            self?.recognitionRequest?.append(buffer)
        }

        audioEngine.prepare()
        do {
            try audioEngine.start()
        } catch {
            print("audioEngine couldn't start: \(error)")
            return
        }

        self.sessionInfoLabel.text = "Listening…"
        self.sessionInfoView.isHidden = false
        self.isShowingTTSMessage = true

        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            guard let self = self else { return }
            if let result = result {
                DispatchQueue.main.async {
                    guard self.recordingGeneration == thisGeneration else { return }
                    self.currentTranscript = result.bestTranscription.formattedString
                    self.sessionInfoLabel.text = self.currentTranscript
                }
            }
            if error != nil || (result?.isFinal ?? false) {
                DispatchQueue.main.async {
                    guard self.recordingGeneration == thisGeneration else { return }
                    self.audioEngine.stop()
                    self.audioEngine.inputNode.removeTap(onBus: 0)
                    self.recognitionRequest = nil
                    self.recognitionTask = nil
                }
            }
        }
    }

    private func stopRecordingAndSend() {
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
            recognitionRequest?.endAudio()
        }
        recognitionTask?.cancel()
        recognitionRequest = nil
        recognitionTask = nil

        let text = currentTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        currentTranscript = ""

        // Show final transcript briefly, then clear
        DispatchQueue.main.async {
            if text.isEmpty {
                self.sessionInfoLabel.text = "(nothing heard)"
            } else {
                self.sessionInfoLabel.text = "Sent: \(text)"
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                self.isShowingTTSMessage = false
                self.sessionInfoView.isHidden = true
            }
        }

        // Restore playback category for TTS
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .default, options: [.duckOthers, .interruptSpokenAudioAndMixWithOthers])
        try? AVAudioSession.sharedInstance().setActive(true)

        guard !text.isEmpty else { return }

        print("[Question] received: \(text)")
        // Send to Firestore: set 'question'
        let docRef = firestore.collection("test").document("test_document")
        docRef.updateData(["question": text]) { error in
            if let error = error {
                // Fallback to set if doc missing
                docRef.setData(["question": text], merge: true) { setErr in
                    if let setErr = setErr { print("Failed to write question: \(setErr.localizedDescription)") }
                }
            }
        }
    }

    private func playStrongHaptic() {
        guard CHHapticEngine.capabilitiesForHardware().supportsHaptics, let engine = hapticEngine else {
            UINotificationFeedbackGenerator().notificationOccurred(.error)
            return
        }
        do {
            let sharp = CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.3)
            let events: [CHHapticEvent] = [
                CHHapticEvent(eventType: .hapticContinuous,
                              parameters: [CHHapticEventParameter(parameterID: .hapticIntensity, value: 1.0), sharp],
                              relativeTime: 0, duration: 0.35),
                CHHapticEvent(eventType: .hapticTransient,
                              parameters: [CHHapticEventParameter(parameterID: .hapticIntensity, value: 1.0), sharp],
                              relativeTime: 0.35)
            ]
            let pattern = try CHHapticPattern(events: events, parameters: [])
            try engine.makePlayer(with: pattern).start(atTime: CHHapticTimeImmediate)
        } catch {
            print("Haptic playback failed: \(error)")
        }
    }

    private func playReleaseHaptic() {
        guard CHHapticEngine.capabilitiesForHardware().supportsHaptics, let engine = hapticEngine else {
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
            return
        }
        do {
            let sharp = CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.6)
            let events: [CHHapticEvent] = [
                CHHapticEvent(eventType: .hapticContinuous,
                              parameters: [CHHapticEventParameter(parameterID: .hapticIntensity, value: 1.0), sharp],
                              relativeTime: 0, duration: 0.1),
                CHHapticEvent(eventType: .hapticTransient,
                              parameters: [CHHapticEventParameter(parameterID: .hapticIntensity, value: 1.0), sharp],
                              relativeTime: 0.1)
            ]
            let pattern = try CHHapticPattern(events: events, parameters: [])
            try engine.makePlayer(with: pattern).start(atTime: CHHapticTimeImmediate)
        } catch {
            print("Haptic playback failed: \(error)")
        }
    }
}
