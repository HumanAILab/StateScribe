# StateScribe iOS Client

The iOS client captures RGB-D frames with ARKit, sends them to the StateScribe server, saves and reloads AR world maps, and uses Firebase for server discovery and question answering.

## Requirements

- A Mac with Xcode 16 or later
- An iPhone with LiDAR running iOS 18 or later
- An Apple ID added to Xcode for device signing
- A Gemini API key
- A Firebase project with Cloud Firestore enabled
- A host machine capable of running the StateScribe Python server
- The host machine and iPhone connected to the same local network

The app must run on a physical iPhone. The simulator doesn't provide the required ARKit camera and depth data.

## 1. Install the server

Follow the installation instructions in the [main README](../README.md#installation). From the repository root, verify that the server starts:

```bash
python main.py
```

The default TCP port is `1234`. Leave this process running while using the iPhone app.

## 2. Find the host machine's local IP address

Find the local-network IP address of the computer running `python main.py`, for example `192.168.1.20`. The host machine can be the Mac used for Xcode or a different computer.

The address must be reachable from the iPhone. Don't use `127.0.0.1` or `localhost`.

## 3. Create the local iOS configuration

From the repository root, run:

```bash
cp iphone-client/Configuration/Local.xcconfig.example iphone-client/Configuration/Local.xcconfig
```

Open `iphone-client/Configuration/Local.xcconfig` and replace every placeholder:

```xcconfig
PRODUCT_BUNDLE_IDENTIFIER = com.your-organization.StateScribe
DEVELOPMENT_TEAM = YOUR_TEAM_ID
GEMINI_API_KEY = YOUR_GEMINI_API_KEY
FRAME_SENDER_HOST = 192.168.1.20
FRAME_SENDER_PORT = 1234
```

- `PRODUCT_BUNDLE_IDENTIFIER` must be unique and must match the bundle identifier registered in Firebase.
- `DEVELOPMENT_TEAM` is your Apple development Team ID. You can also select your team later in Xcode under **Signing & Capabilities**.
- `FRAME_SENDER_HOST` is the host machine IP address from the previous step.
- `FRAME_SENDER_PORT` must match `PORT` in the top-level `config.py`.

## 4. Configure Firebase

First complete the shared Firestore and server configuration in the [Firebase section of the main README](../README.md#firebase). The iOS app must use that same Firebase project.

Then add the iOS app configuration:

1. Open the [Firebase console](https://console.firebase.google.com/) and select the project.
2. Add an iOS app to the project.
3. Enter the same bundle identifier used for `PRODUCT_BUNDLE_IDENTIFIER`.
4. Download `GoogleService-Info.plist`.
5. Copy it to:

   ```text
   iphone-client/StateScribe/GoogleService-Info.plist
   ```

The Firestore `ip` field described in the main README must contain the host machine's local IP address. The iOS app reads it to locate the running StateScribe server.

## 5. Open the project in Xcode

Open:

```text
iphone-client/StateScribe.xcodeproj
```

Wait for Xcode to finish resolving the Firebase Swift packages. This may take several minutes the first time.

Then:

1. Connect the iPhone to the Mac.
2. Unlock the iPhone and trust the Mac if prompted.
3. Select the `StateScribe` project in Xcode.
4. Select the `StateScribe` target and open **Signing & Capabilities**.
5. Enable **Automatically manage signing**.
6. Select your development team.
7. Select the connected iPhone as the run destination.

## 6. Run StateScribe

1. Start the server from the repository root:

   ```bash
   python main.py
   ```

2. In Xcode, press **Run** or `Command-R`.
3. On the iPhone, allow camera, microphone, and speech-recognition access.
4. Move the phone slowly while ARKit maps the environment.

The web interface is available at [http://127.0.0.1:8765](http://127.0.0.1:8765) on the host machine when the server's web visualizer is enabled.

## Acknowledgment

This client is based on Apple's [Saving and loading world data](https://developer.apple.com/documentation/arkit/saving-and-loading-world-data) sample. The original license is included in [LICENSE.txt](LICENSE.txt).
