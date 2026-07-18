//
//  NetworkManager.swift
//  RGBDepth
//
//  Created by jxr on 5/21/25.
//

import Foundation
import Network

class TCPClient {
    let host: NWEndpoint.Host
    let port: NWEndpoint.Port
    let connection: NWConnection
    
    let tailData: Data
    var server_is_processing : Bool!

    init(host: String, port: UInt16) {
        self.host = NWEndpoint.Host(host)
        self.port = NWEndpoint.Port(rawValue: port)!
        self.connection = NWConnection(host: self.host, port: self.port, using: .tcp)
        self.tailData = "_TAIL".data(using: .utf8)!
    }
    
    func startListeningForMessages() {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536, completion: { [weak self] (data, _, isComplete, error) in
            guard let strongSelf = self else { return }

            if let data = data, !data.isEmpty {
                // Handle the received data
                let message = String(data: data, encoding: .utf8) ?? ""
                self?.server_is_processing = false
                print("Received message: \(message)")
            }

            if isComplete {
                // Connection closed by the server or end of data
                strongSelf.connection.cancel()
            } else if let error = error {
                print("Received error: \(error)")
            } else {
                // Continue listening for further messages
                strongSelf.startListeningForMessages()
            }
        })
    }

    func start() {
        self.connection.stateUpdateHandler = { state in
            switch state {
            case .ready:
                print("Connected to the host.")
            case .failed(let error):
                print("Failed to connect: \(error)")
            default:
                break
            }
        }

        self.connection.start(queue: .global())
        self.server_is_processing = false
    }

    func send(data: Data) {
        self.connection.send(content: data, completion: .contentProcessed({ error in
            if let error = error {
                print("Error sending data: \(error)")
                return
            }
//            print("Data sent successfully.")
        }))
    }

    func stop() {
        self.connection.cancel()
    }
}
