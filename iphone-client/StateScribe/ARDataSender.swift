import Foundation
import ARKit
import UIKit

class ARDataSender {
    private let tcpClient: TCPClient
    private var timer: Timer?
    private let ciContext = CIContext()
    var currentWorldName: String?
    private weak var sceneView: ARSCNView?

    var isSendingActive: Bool = false

    init(host: String, port: UInt16, sceneView: ARSCNView) {
        self.tcpClient = TCPClient(host: host, port: port)
        self.sceneView = sceneView
        print("ARDataSender initialized with host: \(host), port: \(port)")
        // Start TCP client and listening for messages on initialization
        self.tcpClient.start()
        self.tcpClient.startListeningForMessages()
    }

    // Add deinit to stop the TCP client when ARDataSender is deallocated
    deinit {
        self.tcpClient.stop()
        self.timer?.invalidate() // Ensure timer is also invalidated
        print("ARDataSender deinitialized, TCP client stopped.")
    }

    func start(worldName: String) {
        guard let sceneView = self.sceneView, let _ = sceneView.session.currentFrame else {
            // print("ARDataSender: Cannot start, ARFrame not available or sceneView is nil.")
            return
        }
        
        self.currentWorldName = worldName
        self.isSendingActive = true
        // self.tcpClient.start() // Removed: TCP client is started in init
        // self.tcpClient.startListeningForMessages() // Removed: TCP client listens from init

        self.timer?.invalidate()
        self.timer = Timer.scheduledTimer(timeInterval: AppConfig.frameSendInterval, target: self, selector: #selector(sendARDataTick), userInfo: nil, repeats: true)
        print("ARDataSender: Started sending data for world: \(worldName)")
    }

    func stop() {
        self.isSendingActive = false
        self.timer?.invalidate()
        self.timer = nil
        // self.tcpClient.stop() // Removed: TCP client is stopped in deinit
        self.currentWorldName = nil // It's good practice to clear this when stopping
//        print("ARDataSender: Stopped sending data.")
    }

    @objc private func sendARDataTick() {
        guard isSendingActive,
              let view = self.sceneView,
              let frame = view.session.currentFrame,
              let name = self.currentWorldName,
              tcpClient.server_is_processing == false else {
            return
        }

        tcpClient.server_is_processing = true

        var payload: [String: Any] = [
            "worldName": name,
            "timestamp": frame.timestamp,
            "cameraPose": extractCameraPose(from: frame)
        ]

        let capturedImage = frame.capturedImage
        if let rgbData = processRGBImage(from: capturedImage) {
            payload["rgbImage"] = rgbData
        }

        if let depthData = frame.sceneDepth?.depthMap ?? frame.smoothedSceneDepth?.depthMap {
            if let processedDepth = processDepthData(from: depthData) {
                payload["depthMap"] = processedDepth
            }
        }
        
        // Add confidence map to the payload
        if let confidenceMap = frame.sceneDepth?.confidenceMap ?? frame.smoothedSceneDepth?.confidenceMap {
            if let processedConfidence = processConfidenceData(from: confidenceMap) {
                payload["confidenceMap"] = processedConfidence
            }
        }
        
        sendPayload(payload)
        print("ARDataSender: Sent AR data for world: \(name) at timestamp: \(frame.timestamp)")
    }

    private func extractCameraPose(from frame: ARFrame) -> [String: Any] {
        let transform = frame.camera.transform
        let intrinsics = frame.camera.intrinsics
        
        let rotationMatrix = simd_float3x3(
            simd_float3(transform.columns.0.x, transform.columns.0.y, transform.columns.0.z),
            simd_float3(transform.columns.1.x, transform.columns.1.y, transform.columns.1.z),
            simd_float3(transform.columns.2.x, transform.columns.2.y, transform.columns.2.z)
        )
        
        let position = simd_float3(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
        
        let eulerAngles = simd_float3(
            atan2(rotationMatrix[2][1], rotationMatrix[2][2]),
            atan2(-rotationMatrix[2][0], sqrt(rotationMatrix[2][1] * rotationMatrix[2][1] + rotationMatrix[2][2] * rotationMatrix[2][2])),
            atan2(rotationMatrix[1][0], rotationMatrix[0][0])
        )
        
        return [
            "position": [
                "x": position.x,
                "y": position.y,
                "z": position.z
            ],
            "rotation": [
                "x": eulerAngles.x,
                "y": eulerAngles.y,
                "z": eulerAngles.z
            ],
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
    
    private func processRGBImage(from pixelBuffer: CVPixelBuffer) -> [String: Any]? {
        var ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        ciImage = ciImage.oriented(.right)
        
        let scaleFactor: CGFloat = 0.5
        let transformCGA = CGAffineTransform(scaleX: scaleFactor, y: scaleFactor)
        ciImage = ciImage.transformed(by: transformCGA)
        
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent),
              let jpegData = UIImage(cgImage: cgImage).jpegData(compressionQuality: 0.6) else {
            return nil
        }
        
        return [
            "width": cgImage.width,
            "height": cgImage.height,
            "data": jpegData.base64EncodedString()
        ]
    }
    
    private func processDepthData(from depthMap: CVPixelBuffer) -> [String: Any]? {
        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(depthMap)
        let pixelFormat = CVPixelBufferGetPixelFormatType(depthMap)
        
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(depthMap) else {
            return nil
        }
        
        let dataSize = height * bytesPerRow
        let rawDepthData = Data(bytes: baseAddress, count: dataSize)
        
        return [
            "width": width,
            "height": height,
            "bytesPerRow": bytesPerRow,
            "pixelFormat": pixelFormat,
            "data": rawDepthData.base64EncodedString()
        ]
    }
    
    private var confidenceSenderDebugCount: Int = 0

    private func processConfidenceData(from confidenceMap: CVPixelBuffer) -> [String: Any]? {
        let width = CVPixelBufferGetWidth(confidenceMap)
        let height = CVPixelBufferGetHeight(confidenceMap)
        let pixelFormat = CVPixelBufferGetPixelFormatType(confidenceMap)
        let planeCount = CVPixelBufferGetPlaneCount(confidenceMap)

        CVPixelBufferLockBaseAddress(confidenceMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(confidenceMap, .readOnly) }

        let effectiveBase: UnsafeMutableRawPointer?
        let bytesPerRow: Int
        if planeCount > 0 {
            effectiveBase = CVPixelBufferGetBaseAddressOfPlane(confidenceMap, 0)
            bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(confidenceMap, 0)
        } else {
            effectiveBase = CVPixelBufferGetBaseAddress(confidenceMap)
            bytesPerRow = CVPixelBufferGetBytesPerRow(confidenceMap)
        }

        guard let baseAddress = effectiveBase else {
            print("[ARDataSender] processConfidenceData: baseAddress is nil (planeCount=\(planeCount))")
            return nil
        }

        if confidenceSenderDebugCount < 3 {
            confidenceSenderDebugCount += 1
            let formatStr = fourCC(pixelFormat)
            let totalBytes = height * bytesPerRow
            let ptr = baseAddress.assumingMemoryBound(to: UInt8.self)
            var nonZero = 0
            for i in 0..<totalBytes { if ptr[i] != 0 { nonZero += 1 } }
            print("[ARDataSender] ConfidenceDebug: format=\(formatStr) (\(pixelFormat)), \(width)x\(height), bpr=\(bytesPerRow), planes=\(planeCount), nonZeroBytes=\(nonZero)/\(totalBytes)")
        }

        let dataSize = height * bytesPerRow
        let rawConfidenceData = Data(bytes: baseAddress, count: dataSize)

        return [
            "width": width,
            "height": height,
            "bytesPerRow": bytesPerRow,
            "pixelFormat": pixelFormat,
            "data": rawConfidenceData.base64EncodedString()
        ]
    }

    private func fourCC(_ code: OSType) -> String {
        let bytes: [UInt8] = [
            UInt8((code >> 24) & 0xFF),
            UInt8((code >> 16) & 0xFF),
            UInt8((code >> 8) & 0xFF),
            UInt8(code & 0xFF)
        ]
        if let s = String(bytes: bytes, encoding: .ascii) { return s }
        return String(format: "0x%08X", code)
    }
    
    private func sendPayload(_ payload: [String: Any]) {
        do {
            let jsonData = try JSONSerialization.data(withJSONObject: payload, options: [])
            let packetData = jsonData + tcpClient.tailData
            tcpClient.send(data: packetData)
        } catch {
            // print("ARDataSender: Failed to serialize payload: \(error)")
            tcpClient.server_is_processing = false
        }
    }
}
