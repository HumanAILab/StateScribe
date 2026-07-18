import Foundation
import ARKit
import UIKit
import AVFoundation

class ARDataLogger {

    private weak var sceneView: ARSCNView?
    private var timer: Timer?
    private let ciContext = CIContext()
    private let ioQueue = DispatchQueue(label: "com.statescribe.arDataLogger", qos: .utility)
    private let videoQueue = DispatchQueue(label: "com.statescribe.arDataLogger.video", qos: .utility)

    private var sessionDirectory: URL?
    private var rgbDirectory: URL?
    private var depthDirectory: URL?
    private var confidenceDirectory: URL?
    private var metadataFileHandle: FileHandle?

    // Video recording
    private var assetWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var videoOutputURL: URL?
    private var videoStartTime: CMTime?
    private var isVideoSessionStarted = false
    private var videoFramesAppended: Int = 0
    private var hasLoggedAppendSkip = false

    private var frameIndex: Int = 0
    private(set) var isLogging: Bool = false
    private(set) var recordingStartDate: Date?

    // MARK: - Lifecycle

    init(sceneView: ARSCNView) {
        self.sceneView = sceneView
    }

    deinit {
        if isLogging {
            stop()
        }
    }

    // MARK: - Public

    func start(sessionName: String) {
        guard !isLogging else { return }

        do {
            try createSessionDirectories(sessionName: sessionName)
        } catch {
            print("[ARDataLogger] Failed to create directories: \(error.localizedDescription)")
            return
        }

        frameIndex = 0
        isLogging = true
        recordingStartDate = Date()

        // Setup video recording
        print("[ARDataLogger] start: sessionDirectory=\(sessionDirectory?.path ?? "nil")")
        setupVideoWriter()
        print("[ARDataLogger] start: after setupVideoWriter, assetWriter=\(assetWriter == nil ? "nil" : "ok")")

        let interval = 1.0 / AppConfig.dataLogFPS
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            self?.captureFrame()
        }

        print("[ARDataLogger] Started logging at \(Int(AppConfig.dataLogFPS)) FPS → \(sessionDirectory?.path ?? "")")
    }

    func stop() {
        print("[ARDataLogger] stop() called, isLogging=\(isLogging), frameIndex=\(frameIndex), assetWriter=\(assetWriter == nil ? "nil" : "set"), status=\(assetWriter?.status.rawValue ?? -1)")
        guard isLogging else { return }
        isLogging = false
        recordingStartDate = nil
        timer?.invalidate()
        timer = nil

        // Drain pending IO writes, then close metadata file
        ioQueue.sync {}
        metadataFileHandle?.closeFile()
        metadataFileHandle = nil

        // Finalize video synchronously so the file is complete before caller deallocates us
        finalizeVideoSync()

        if frameIndex > 0 {
            print("[ARDataLogger] Stopped. \(frameIndex) frames logged to \(sessionDirectory?.lastPathComponent ?? "")")
        }
    }

    // MARK: - Directory Setup

    private func createSessionDirectories(sessionName: String) throws {
        let docs = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let logsRoot = docs.appendingPathComponent("ARLogs")

        let safeName = sessionName.replacingOccurrences(of: "/", with: "_")
        let sessionDir = logsRoot.appendingPathComponent(safeName)

        let rgb = sessionDir.appendingPathComponent("rgb")
        let depth = sessionDir.appendingPathComponent("depth")
        let confidence = sessionDir.appendingPathComponent("confidence")

        let fm = FileManager.default
        try fm.createDirectory(at: rgb, withIntermediateDirectories: true)
        try fm.createDirectory(at: depth, withIntermediateDirectories: true)
        try fm.createDirectory(at: confidence, withIntermediateDirectories: true)

        let metadataURL = sessionDir.appendingPathComponent("metadata.jsonl")
        fm.createFile(atPath: metadataURL.path, contents: nil)
        metadataFileHandle = try FileHandle(forWritingTo: metadataURL)

        sessionDirectory = sessionDir
        rgbDirectory = rgb
        depthDirectory = depth
        confidenceDirectory = confidence
    }

    // MARK: - Frame Capture

    private func captureFrame() {
        guard isLogging,
              let view = sceneView,
              let frame = view.session.currentFrame else { return }

        let index = frameIndex
        frameIndex += 1
        print("[ARDataLogger] Logging frame \(index) at timestamp \(frame.timestamp)")

        let timestamp = frame.timestamp
        let cameraPose = extractCameraPose(from: frame)

        let capturedImage = frame.capturedImage

        // Append frame to video
        appendVideoFrame(pixelBuffer: capturedImage, timestamp: timestamp)

        let rgbResult = processRGBImage(from: capturedImage)

        let depthResult: (data: Data, width: Int, height: Int, bytesPerRow: Int, pixelFormat: UInt32)?
        if let depthMap = frame.sceneDepth?.depthMap ?? frame.smoothedSceneDepth?.depthMap {
            depthResult = extractRawPixelBuffer(depthMap)
        } else {
            depthResult = nil
        }

        let confidenceResult: (data: Data, width: Int, height: Int, bytesPerRow: Int, pixelFormat: UInt32)?
        if let confMap = frame.sceneDepth?.confidenceMap ?? frame.smoothedSceneDepth?.confidenceMap {
            if index < 3 {
                debugPixelBuffer(confMap, label: "confidence", frameIndex: index)
            }
            confidenceResult = extractRawPixelBuffer(confMap)
        } else {
            confidenceResult = nil
        }

        let rgbDir = self.rgbDirectory
        let depthDir = self.depthDirectory
        let confDir = self.confidenceDirectory

        ioQueue.async { [weak self] in
            guard let self = self else { return }
            let frameName = String(format: "%06d", index)

            // Write RGB JPEG
            if let rgb = rgbResult, let dir = rgbDir {
                let url = dir.appendingPathComponent("\(frameName).jpg")
                do {
                    try rgb.jpegData.write(to: url)
                } catch {
                    print("[ARDataLogger] RGB write failed: \(error.localizedDescription)")
                }
            }

            // Write depth binary
            if let depth = depthResult, let dir = depthDir {
                let url = dir.appendingPathComponent("\(frameName).bin")
                do {
                    try depth.data.write(to: url)
                } catch {
                    print("[ARDataLogger] Depth write failed: \(error.localizedDescription)")
                }
            }

            // Write confidence binary
            if let conf = confidenceResult, let dir = confDir {
                let url = dir.appendingPathComponent("\(frameName).bin")
                do {
                    try conf.data.write(to: url)
                } catch {
                    print("[ARDataLogger] Confidence write failed: \(error.localizedDescription)")
                }
            }

            // Write metadata line
            var meta: [String: Any] = [
                "frame": index,
                "timestamp": timestamp,
                "cameraPose": cameraPose
            ]
            if let rgb = rgbResult {
                meta["rgb"] = ["width": rgb.width, "height": rgb.height]
            }
            if let depth = depthResult {
                meta["depth"] = ["width": depth.width, "height": depth.height, "bytesPerRow": depth.bytesPerRow, "pixelFormat": depth.pixelFormat]
            }
            if let conf = confidenceResult {
                meta["confidence"] = ["width": conf.width, "height": conf.height, "bytesPerRow": conf.bytesPerRow, "pixelFormat": conf.pixelFormat]
            }

            if let jsonData = try? JSONSerialization.data(withJSONObject: meta, options: []) {
                var line = jsonData
                line.append(0x0A) // newline
                self.metadataFileHandle?.write(line)
            }
        }
    }

    // MARK: - Video Recording

    private func setupVideoWriter() {
        guard let sessionDir = sessionDirectory else {
            print("[ARDataLogger] setupVideoWriter: sessionDirectory is nil, skipping")
            return
        }

        let videoURL = sessionDir.appendingPathComponent("video.mp4")
        print("[ARDataLogger] setupVideoWriter: videoURL=\(videoURL.path)")

        // Remove if exists (e.g. from a previous incomplete session)
        try? FileManager.default.removeItem(at: videoURL)

        do {
            assetWriter = try AVAssetWriter(outputURL: videoURL, fileType: .mp4)
        } catch {
            print("[ARDataLogger] Failed to create AVAssetWriter: \(error.localizedDescription)")
            return
        }

        // AR camera captures in landscape; use raw dimensions for encoding
        let rawWidth: Int
        let rawHeight: Int
        if let frame = sceneView?.session.currentFrame {
            rawWidth = CVPixelBufferGetWidth(frame.capturedImage)
            rawHeight = CVPixelBufferGetHeight(frame.capturedImage)
        } else {
            print("[ARDataLogger] setupVideoWriter: no currentFrame, using default 1920x1440")
            rawWidth = 1920
            rawHeight = 1440
        }

        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: rawWidth,
            AVVideoHeightKey: rawHeight,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: 6_000_000,
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel
            ]
        ]

        let input = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        input.expectsMediaDataInRealTime = true
        // Rotate 90° so playback displays in portrait orientation
        input.transform = CGAffineTransform(rotationAngle: .pi / 2)

        let sourcePixelAttrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: rawWidth,
            kCVPixelBufferHeightKey as String: rawHeight
        ]

        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: sourcePixelAttrs)

        guard let writer = assetWriter, writer.canAdd(input) else {
            print("[ARDataLogger] setupVideoWriter: cannot add video input (writer=\(assetWriter != nil), canAdd=\(assetWriter?.canAdd(input) ?? false))")
            assetWriter = nil
            return
        }

        writer.add(input)
        writer.startWriting()
        videoOutputURL = videoURL
        print("[ARDataLogger] setupVideoWriter: startWriting() called, status=\(writer.status.rawValue)")

        videoInput = input
        pixelBufferAdaptor = adaptor
        videoStartTime = nil
        isVideoSessionStarted = false

        print("[ARDataLogger] Video writer ready → \(videoURL.lastPathComponent)")
    }

    private func appendVideoFrame(pixelBuffer: CVPixelBuffer, timestamp: TimeInterval) {
        // Convert YCbCr → BGRA on the calling thread while the pixel buffer is still valid
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard let bgraBuffer = createBGRAPixelBuffer(from: ciImage, width: width, height: height) else {
            if frameIndex % 30 == 0 { print("[ARDataLogger] appendVideoFrame: createBGRAPixelBuffer failed for frame \(frameIndex)") }
            return
        }

        let presentationTime = CMTime(seconds: timestamp, preferredTimescale: 600)

        videoQueue.async { [weak self] in
            guard let self = self,
                  let writer = self.assetWriter,
                  let input = self.videoInput,
                  let adaptor = self.pixelBufferAdaptor,
                  writer.status == .writing else {
                if let s = self, !s.hasLoggedAppendSkip {
                    s.hasLoggedAppendSkip = true
                    print("[ARDataLogger] appendVideoFrame: skip (writer/input/adaptor nil or status=\(s.assetWriter?.status.rawValue ?? -1))")
                }
                return
            }

            if !self.isVideoSessionStarted {
                writer.startSession(atSourceTime: presentationTime)
                self.videoStartTime = presentationTime
                self.isVideoSessionStarted = true
                print("[ARDataLogger] appendVideoFrame: first session started at \(presentationTime.seconds)")
            }

            if input.isReadyForMoreMediaData {
                adaptor.append(bgraBuffer, withPresentationTime: presentationTime)
                self.videoFramesAppended += 1
            }
        }
    }

    private func createBGRAPixelBuffer(from ciImage: CIImage, width: Int, height: Int) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pixelBuffer)
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }
        ciContext.render(ciImage, to: buffer)
        return buffer
    }

    private func finalizeVideoSync() {
        print("[ARDataLogger] finalizeVideoSync: draining video queue...")
        videoQueue.sync {}
        print("[ARDataLogger] finalizeVideoSync: drained. assetWriter=\(assetWriter == nil ? "nil" : "set"), status=\(assetWriter?.status.rawValue ?? -1), isVideoSessionStarted=\(isVideoSessionStarted), videoFramesAppended=\(videoFramesAppended)")

        guard let writer = assetWriter else {
            print("[ARDataLogger] finalizeVideoSync: no writer; cleanup")
            cleanupVideoState()
            return
        }

        // Always close the writer so the file has a valid header/footer (AVAssetWriter allows finishWriting in any state)
        videoInput?.markAsFinished()
        print("[ARDataLogger] finalizeVideoSync: marked input finished, calling finishWriting...")

        let outputURL = videoOutputURL
        let framesAppended = videoFramesAppended

        let semaphore = DispatchSemaphore(value: 0)
        writer.finishWriting {
            let status = writer.status
            let err = writer.error
            if status == .completed {
                print("[ARDataLogger] Video saved successfully (\(framesAppended) frames)")
            } else {
                print("[ARDataLogger] Video finalize: status=\(status.rawValue), error=\(err?.localizedDescription ?? "nil")")
                if let e = err { print("[ARDataLogger] Video error detail: \(e)") }
                // Remove incomplete/corrupt file so the UI doesn't show a broken video
                if let url = outputURL, FileManager.default.fileExists(atPath: url.path) {
                    try? FileManager.default.removeItem(at: url)
                    print("[ARDataLogger] Removed incomplete video file")
                }
            }
            semaphore.signal()
        }
        semaphore.wait()
        cleanupVideoState()
    }

    private func cleanupVideoState() {
        assetWriter = nil
        videoInput = nil
        pixelBufferAdaptor = nil
        videoOutputURL = nil
        videoStartTime = nil
        isVideoSessionStarted = false
        videoFramesAppended = 0
        hasLoggedAppendSkip = false
    }

    // MARK: - Image Processing

    private struct RGBResult {
        let jpegData: Data
        let width: Int
        let height: Int
    }

    private func processRGBImage(from pixelBuffer: CVPixelBuffer) -> RGBResult? {
        var ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        ciImage = ciImage.oriented(.right)

        let scaleFactor: CGFloat = 0.5
        ciImage = ciImage.transformed(by: CGAffineTransform(scaleX: scaleFactor, y: scaleFactor))

        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent),
              let jpegData = UIImage(cgImage: cgImage).jpegData(compressionQuality: 0.6) else {
            return nil
        }
        return RGBResult(jpegData: jpegData, width: cgImage.width, height: cgImage.height)
    }

    private func debugPixelBuffer(_ buffer: CVPixelBuffer, label: String, frameIndex: Int) {
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let format = CVPixelBufferGetPixelFormatType(buffer)
        let planeCount = CVPixelBufferGetPlaneCount(buffer)

        let formatBytes: [UInt8] = [
            UInt8((format >> 24) & 0xFF), UInt8((format >> 16) & 0xFF),
            UInt8((format >> 8) & 0xFF), UInt8(format & 0xFF)
        ]
        let formatStr = String(bytes: formatBytes, encoding: .ascii) ?? String(format: "0x%08X", format)

        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        let base = CVPixelBufferGetBaseAddress(buffer)
        let totalBytes = height * bytesPerRow
        var nonZero = 0
        if let ptr = base?.assumingMemoryBound(to: UInt8.self) {
            for i in 0..<totalBytes { if ptr[i] != 0 { nonZero += 1 } }
        }
        print("[ARDataLogger] \(label) debug frame=\(frameIndex): format=\(formatStr), \(width)x\(height), bpr=\(bytesPerRow), planes=\(planeCount), baseAddr=\(base != nil ? "ok" : "NIL"), nonZeroBytes=\(nonZero)/\(totalBytes)")
    }

    private func extractRawPixelBuffer(_ buffer: CVPixelBuffer) -> (data: Data, width: Int, height: Int, bytesPerRow: Int, pixelFormat: UInt32)? {
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let pixelFormat = CVPixelBufferGetPixelFormatType(buffer)

        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let data = Data(bytes: baseAddress, count: height * bytesPerRow)
        return (data, width, height, bytesPerRow, pixelFormat)
    }

    // MARK: - Camera Pose

    private func extractCameraPose(from frame: ARFrame) -> [String: Any] {
        let transform = frame.camera.transform
        let intrinsics = frame.camera.intrinsics
        let position = simd_float3(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)

        let rotationMatrix = simd_float3x3(
            simd_float3(transform.columns.0.x, transform.columns.0.y, transform.columns.0.z),
            simd_float3(transform.columns.1.x, transform.columns.1.y, transform.columns.1.z),
            simd_float3(transform.columns.2.x, transform.columns.2.y, transform.columns.2.z)
        )
        let eulerAngles = simd_float3(
            atan2(rotationMatrix[2][1], rotationMatrix[2][2]),
            atan2(-rotationMatrix[2][0], sqrt(rotationMatrix[2][1] * rotationMatrix[2][1] + rotationMatrix[2][2] * rotationMatrix[2][2])),
            atan2(rotationMatrix[1][0], rotationMatrix[0][0])
        )

        return [
            "position": ["x": position.x, "y": position.y, "z": position.z],
            "rotation": ["x": eulerAngles.x, "y": eulerAngles.y, "z": eulerAngles.z],
            "transform": [
                transform.columns.0.x, transform.columns.0.y, transform.columns.0.z, transform.columns.0.w,
                transform.columns.1.x, transform.columns.1.y, transform.columns.1.z, transform.columns.1.w,
                transform.columns.2.x, transform.columns.2.y, transform.columns.2.z, transform.columns.2.w,
                transform.columns.3.x, transform.columns.3.y, transform.columns.3.z, transform.columns.3.w
            ],
            "intrinsics": [
                intrinsics[0][0], intrinsics[0][1], intrinsics[0][2],
                intrinsics[1][0], intrinsics[1][1], intrinsics[1][2],
                intrinsics[2][0], intrinsics[2][1], intrinsics[2][2]
            ]
        ]
    }
}
