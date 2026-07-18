# StateScribe: Towards Accessible Change Awareness Across Real-World Revisits

StateScribe is a scene-change awareness system for blind and low-vision users. It supports both offline benchmarking on revisit datasets and real-time RGB-D streaming from a mobile capture client.

## Installation

```bash
conda create -n statescribe python=3.12 -y
conda activate statescribe
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Benchmark on a Dataset

Download the dataset from [StateScribe Dataset](https://placeholder.example.com/statescribe-dataset), then extract it so each scene directory looks like this:

```text
/path/to/statescribe-dataset/
  Grocery.zip
  Office.zip
  Outdoor.zip
  Grocery/
    annotations.json
    Grocery_20260308_155134/
      metadata.jsonl
      rgb/
      depth/
      confidence/
    Grocery_...
  Office/
    annotations.json
    Office_...
  Outdoor/
    annotations.json
    Outdoor_...
```

Each scene directory such as `Grocery/` or `Office/` is one benchmark dataset root. Each capture directory should contain at least `metadata.jsonl`, `rgb/`, `depth/`, and `confidence/`.

Run StateScribe on a scene dataset:

- `--mode realtime`: replay the dataset using the original capture timing
- `--mode fast`: feed frames as quickly as possible for faster experiments

```bash
python benchmark/run.py --dataset /path/to/statescribe-dataset/Grocery --mode realtime
python benchmark/run.py --dataset /path/to/statescribe-dataset/Grocery --mode fast
```

Outputs are written to:

```text
output/benchmark/Grocery/statescribe/<run_timestamp>/
```

Run the baselines:

```bash
python benchmark/run_baseline.py --baseline online_gemini --dataset /path/to/statescribe-dataset/Grocery
python benchmark/run_baseline.py --baseline offline_gemini --dataset /path/to/statescribe-dataset/Grocery
```

Their outputs are written to:

```text
output/benchmark/Grocery/baseline_online_gemini_flash/<run_timestamp>/
output/benchmark/Grocery/baseline_offline_gemini_flash/<run_timestamp>/
```

Evaluate a benchmark run:

```bash
python -m benchmark.eval.cli \
  --annotations /path/to/statescribe-dataset/Grocery/annotations.json \
  --benchmark-output output/benchmark/Grocery/statescribe/<run_timestamp>
```

Evaluate a baseline run the same way, but point `--benchmark-output` to the corresponding baseline output directory.

## Run with an iPhone

The iPhone capture backend is available at [StateScribe iPhone Backend](https://placeholder.example.com/statescribe-iphone-backend).
You need an iPhone model with a LiDAR sensor.

To run the live system:

1. Put the iPhone and the host machine on the same Wi-Fi network.
2. Configure the iPhone backend to send frames to the host machine's LAN IP and port `1234`.
3. If your mobile setup uses Firebase-based discovery, also publish that LAN IP there so the phone can discover the active server.
4. Start StateScribe:

```bash
python main.py
```

By default, StateScribe listens on all interfaces with `HOST=0.0.0.0` and `PORT=1234` in `config.py`.

## View the Web UI

The web visualizer is enabled by default. The relevant settings in `config.py` are:

```python
VISUALIZATION_ENABLED = True
VISUALIZATION_BACKEND = "web"
VISUALIZATION_ENABLE_3D_RECONSTRUCTION = False
WEB_VIS_HOST = "127.0.0.1"
WEB_VIS_PORT = 8765
```

Keep `VISUALIZATION_ENABLE_3D_RECONSTRUCTION = False` unless you specifically need 3D reconstruction. Turning it on significantly slows the system down.

Then start StateScribe normally:

```bash
python main.py
```

Open the Web UI at:

```text
http://127.0.0.1:8765
```

## Firebase

Firebase is optional unless you want remote question answering and answer publishing.

To set it up:

1. Open the Firebase Console and select your project.
2. Enable Firestore Database for that project.
3. In Firestore, create a collection and a document. With the current default config, they are:

```text
collection: test
document: test_document
```

4. Inside that document, create fields like this:

```text
ip: "YOUR_HOST_LAN_IP"
port: "1234"
question: null
response: ""
```

5. Set `ip` to the LAN IP address of the machine running StateScribe, and keep `port` consistent with the port used by the live server.
6. Start the iPhone backend with the same Firebase project so it can find the server and exchange questions and answers through the same Firestore document.

StateScribe watches the `question` field, clears it after reading, and writes replies into `response`.

To download the Firestore credentials JSON:

1. Open the Firebase Console and select your project.
2. Click the gear icon next to `Project Overview`, then open `Project settings`.
3. Open the `Service accounts` tab.
4. In the `Firebase Admin SDK` section, click `Generate new private key`.
5. Confirm and download the `.json` file.
6. Put that file on the machine running StateScribe and point `FIREBASE_CREDENTIALS_PATH` in `config.py` to it.

If you want different collection, document, or field names, you can change `FIREBASE_COLLECTION`, `FIREBASE_DOCUMENT`, `FIREBASE_QUESTION_FIELD`, and `FIREBASE_ANSWER_FIELD` in `config.py`.

## API Keys and Configuration

The main runtime settings live in `config.py`. The most important ones are:

- `GEMINI_API_KEY`
- `FIREBASE_CREDENTIALS_PATH`
- `FIREBASE_COLLECTION`
- `FIREBASE_DOCUMENT`
- `FIREBASE_QUESTION_FIELD`
- `FIREBASE_ANSWER_FIELD`
- `HOST`
- `PORT`

`GEMINI_API_KEY` can also be supplied through the environment.
