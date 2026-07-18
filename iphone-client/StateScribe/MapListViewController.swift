import UIKit
import ARKit
import AVFoundation

// MARK: - MapCardCell

class MapCardCell: UICollectionViewCell {

    static let reuseIdentifier = "MapCardCell"

    let snapshotImageView: UIImageView = {
        let iv = UIImageView()
        iv.contentMode = .scaleAspectFill
        iv.clipsToBounds = true
        iv.backgroundColor = .secondarySystemBackground
        iv.translatesAutoresizingMaskIntoConstraints = false
        return iv
    }()

    private let placeholderImageView: UIImageView = {
        let iv = UIImageView()
        iv.contentMode = .scaleAspectFit
        iv.tintColor = .tertiaryLabel
        iv.image = UIImage(systemName: "camera.fill")
        iv.translatesAutoresizingMaskIntoConstraints = false
        return iv
    }()

    let nameLabel: UILabel = {
        let label = UILabel()
        label.font = .systemFont(ofSize: 11, weight: .semibold)
        label.textColor = .label
        label.numberOfLines = 0
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()

    let dateLabel: UILabel = {
        let label = UILabel()
        label.font = .systemFont(ofSize: 11, weight: .regular)
        label.textColor = .secondaryLabel
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()

    let sessionCountLabel: UILabel = {
        let label = UILabel()
        label.font = .systemFont(ofSize: 11, weight: .regular)
        label.textColor = .tertiaryLabel
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()

    let videoLengthLabel: UILabel = {
        let label = UILabel()
        label.font = .systemFont(ofSize: 11, weight: .regular)
        label.textColor = .tertiaryLabel
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()

    let sizeLabel: UILabel = {
        let label = UILabel()
        label.font = .systemFont(ofSize: 11, weight: .regular)
        label.textColor = .tertiaryLabel
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()

    let deleteButton: UIButton = {
        let button = UIButton(type: .system)
        button.setTitle("Delete", for: .normal)
        button.setTitleColor(.systemRed, for: .normal)
        button.titleLabel?.font = .systemFont(ofSize: 13, weight: .medium)
        button.backgroundColor = .systemRed.withAlphaComponent(0.12)
        button.layer.cornerRadius = 8
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }()

    let renameButton: UIButton = {
        let button = UIButton(type: .system)
        button.setTitle("Rename", for: .normal)
        button.setTitleColor(.systemBlue, for: .normal)
        button.titleLabel?.font = .systemFont(ofSize: 13, weight: .medium)
        button.backgroundColor = .systemBlue.withAlphaComponent(0.12)
        button.layer.cornerRadius = 8
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }()

    var onDelete: (() -> Void)?
    var onRename: (() -> Void)?

    override init(frame: CGRect) {
        super.init(frame: frame)
        setupUI()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupUI()
    }

    private func setupUI() {
        contentView.backgroundColor = .systemBackground
        contentView.layer.cornerRadius = 14
        contentView.layer.masksToBounds = true

        layer.cornerRadius = 14
        layer.shadowColor = UIColor.black.cgColor
        layer.shadowOpacity = 0.08
        layer.shadowOffset = CGSize(width: 0, height: 2)
        layer.shadowRadius = 8
        layer.masksToBounds = false

        // Placeholder sits behind the snapshot image
        contentView.addSubview(placeholderImageView)
        contentView.addSubview(snapshotImageView)
        contentView.addSubview(nameLabel)
        contentView.addSubview(dateLabel)

        let infoStack = UIStackView(arrangedSubviews: [sessionCountLabel, videoLengthLabel, sizeLabel])
        infoStack.axis = .horizontal
        infoStack.spacing = 6
        infoStack.alignment = .center
        infoStack.translatesAutoresizingMaskIntoConstraints = false
        contentView.addSubview(infoStack)

        let buttonStack = UIStackView(arrangedSubviews: [deleteButton, renameButton])
        buttonStack.axis = .horizontal
        buttonStack.spacing = 8
        buttonStack.distribution = .fillEqually
        buttonStack.translatesAutoresizingMaskIntoConstraints = false
        contentView.addSubview(buttonStack)

        NSLayoutConstraint.activate([
            snapshotImageView.topAnchor.constraint(equalTo: contentView.topAnchor),
            snapshotImageView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
            snapshotImageView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
            snapshotImageView.heightAnchor.constraint(equalTo: contentView.widthAnchor, multiplier: 1.2),

            placeholderImageView.centerXAnchor.constraint(equalTo: snapshotImageView.centerXAnchor),
            placeholderImageView.centerYAnchor.constraint(equalTo: snapshotImageView.centerYAnchor),
            placeholderImageView.widthAnchor.constraint(equalToConstant: 36),
            placeholderImageView.heightAnchor.constraint(equalToConstant: 36),

            nameLabel.topAnchor.constraint(equalTo: snapshotImageView.bottomAnchor, constant: 10),
            nameLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 13),
            nameLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -12),

            dateLabel.topAnchor.constraint(equalTo: nameLabel.bottomAnchor, constant: 3),
            dateLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 13),
            dateLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -12),

            infoStack.topAnchor.constraint(equalTo: dateLabel.bottomAnchor, constant: 2),
            infoStack.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 13),
            infoStack.trailingAnchor.constraint(lessThanOrEqualTo: contentView.trailingAnchor, constant: -12),

            buttonStack.topAnchor.constraint(equalTo: infoStack.bottomAnchor, constant: 10),
            buttonStack.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 12),
            buttonStack.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -12),
            buttonStack.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -12),
            buttonStack.heightAnchor.constraint(equalToConstant: 30),
        ])

        deleteButton.addTarget(self, action: #selector(deleteTapped), for: .touchUpInside)
        renameButton.addTarget(self, action: #selector(renameTapped), for: .touchUpInside)
    }

    @objc private func deleteTapped() { onDelete?() }
    @objc private func renameTapped() { onRename?() }

    override func prepareForReuse() {
        super.prepareForReuse()
        snapshotImageView.image = nil
        nameLabel.text = nil
        dateLabel.text = nil
        sessionCountLabel.text = nil
        videoLengthLabel.text = nil
        sizeLabel.text = nil
        onDelete = nil
        onRename = nil
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        layer.shadowPath = UIBezierPath(roundedRect: bounds, cornerRadius: 14).cgPath
    }
}

// MARK: - MapListViewController

class MapListViewController: UIViewController {

    var mapURLs: [URL] = []
    private var snapshotCache: [URL: UIImage] = [:]

    let mapsDirectoryURL: URL = {
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

    private lazy var collectionView: UICollectionView = {
        let layout = UICollectionViewFlowLayout()
        layout.minimumLineSpacing = 16
        layout.minimumInteritemSpacing = 12
        layout.sectionInset = UIEdgeInsets(top: 16, left: 16, bottom: 16, right: 16)

        let cv = UICollectionView(frame: .zero, collectionViewLayout: layout)
        cv.backgroundColor = .systemGroupedBackground
        cv.dataSource = self
        cv.delegate = self
        cv.register(MapCardCell.self, forCellWithReuseIdentifier: MapCardCell.reuseIdentifier)
        cv.alwaysBounceVertical = true
        cv.translatesAutoresizingMaskIntoConstraints = false
        return cv
    }()

    private let emptyLabel: UILabel = {
        let label = UILabel()
        label.text = "No saved scenes yet.\nTap + to create one."
        label.textColor = .secondaryLabel
        label.font = .systemFont(ofSize: 16, weight: .regular)
        label.textAlignment = .center
        label.numberOfLines = 0
        label.translatesAutoresizingMaskIntoConstraints = false
        label.isHidden = true
        return label
    }()

    private let dateFormatter: DateFormatter = {
        let df = DateFormatter()
        df.dateStyle = .medium
        df.timeStyle = .short
        return df
    }()

    // MARK: - Lifecycle

    override func viewDidLoad() {
        super.viewDidLoad()
        title = "My Scenes"
        view.backgroundColor = .systemGroupedBackground
        navigationItem.rightBarButtonItem = UIBarButtonItem(barButtonSystemItem: .add, target: self, action: #selector(createNewScene))

        let fileManager = FileManager.default
        if !fileManager.fileExists(atPath: mapsDirectoryURL.path) {
            do {
                try fileManager.createDirectory(at: mapsDirectoryURL, withIntermediateDirectories: true, attributes: nil)
            } catch {
                print("Error creating maps directory: \(error.localizedDescription)")
            }
        }

        NotificationCenter.default.addObserver(self, selector: #selector(handleMapDidSave), name: .mapDidSave, object: nil)

        view.addSubview(collectionView)
        view.addSubview(emptyLabel)
        NSLayoutConstraint.activate([
            collectionView.topAnchor.constraint(equalTo: view.topAnchor),
            collectionView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            collectionView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            collectionView.bottomAnchor.constraint(equalTo: view.bottomAnchor),

            emptyLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            emptyLabel.centerYAnchor.constraint(equalTo: view.centerYAnchor),
        ])
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        loadMapURLs()
        collectionView.reloadData()
        updateEmptyState()
    }

    // MARK: - Data

    func loadMapURLs() {
        let fileManager = FileManager.default
        do {
            let contents = try fileManager.contentsOfDirectory(at: mapsDirectoryURL, includingPropertiesForKeys: [.contentModificationDateKey], options: .skipsHiddenFiles)
            mapURLs = contents.filter { $0.pathExtension == "arexperience" }
                .sorted { url1, url2 in
                    let date1 = (try? url1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                    let date2 = (try? url2.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                    return date1 > date2
                }
        } catch {
            print("Error loading map URLs: \(error.localizedDescription)")
            mapURLs = []
        }
        snapshotCache.removeAll()
    }

    private func updateEmptyState() {
        emptyLabel.isHidden = !mapURLs.isEmpty
    }

    // MARK: - Snapshot Loading

    private func loadSnapshotImage(for url: URL, completion: @escaping (UIImage?) -> Void) {
        if let cached = snapshotCache[url] {
            completion(cached)
            return
        }
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let data = try? Data(contentsOf: url),
                  let worldMap = try? NSKeyedUnarchiver.unarchivedObject(ofClass: ARWorldMap.self, from: data),
                  let snapshotAnchor = worldMap.anchors.first(where: { $0 is SnapshotAnchor }) as? SnapshotAnchor,
                  let image = UIImage(data: snapshotAnchor.imageData) else {
                DispatchQueue.main.async { completion(nil) }
                return
            }
            DispatchQueue.main.async {
                self?.snapshotCache[url] = image
                completion(image)
            }
        }
    }

    // MARK: - Actions

    @objc private func handleMapDidSave() {
        loadMapURLs()
        collectionView.reloadData()
        updateEmptyState()
    }

    @objc func createNewScene() {
        performSegue(withIdentifier: "showARScene", sender: nil)
    }

    private func deleteMap(at indexPath: IndexPath) {
        let mapURL = mapURLs[indexPath.item]
        let name = mapURL.deletingPathExtension().lastPathComponent

        let alert = UIAlertController(title: "Delete Scene", message: "Are you sure you want to delete \"\(name)\"?", preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
        alert.addAction(UIAlertAction(title: "Delete", style: .destructive) { [weak self] _ in
            guard let self = self else { return }
            do {
                try FileManager.default.removeItem(at: mapURL)
                self.snapshotCache.removeValue(forKey: mapURL)
                self.mapURLs.remove(at: indexPath.item)
                self.collectionView.deleteItems(at: [indexPath])
                self.updateEmptyState()
                self.deleteMatchingARLogs(for: name)
            } catch {
                let errorAlert = UIAlertController(title: "Error", message: "Could not delete the scene. \(error.localizedDescription)", preferredStyle: .alert)
                errorAlert.addAction(UIAlertAction(title: "OK", style: .default))
                self.present(errorAlert, animated: true)
            }
        })
        present(alert, animated: true)
    }

    private func renameMap(at indexPath: IndexPath) {
        let oldURL = mapURLs[indexPath.item]
        let oldName = oldURL.deletingPathExtension().lastPathComponent

        let alert = UIAlertController(title: "Rename Scene", message: "Enter a new name for this scene.", preferredStyle: .alert)
        alert.addTextField { textField in
            textField.text = oldName
            textField.clearButtonMode = .whileEditing
        }
        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
        alert.addAction(UIAlertAction(title: "Rename", style: .default) { [weak self] _ in
            guard let self = self,
                  let newName = alert.textFields?.first?.text?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !newName.isEmpty, newName != oldName else { return }

            let newURL = self.mapsDirectoryURL.appendingPathComponent(newName).appendingPathExtension("arexperience")

            if FileManager.default.fileExists(atPath: newURL.path) {
                let errorAlert = UIAlertController(title: "Error", message: "A scene named \"\(newName)\" already exists.", preferredStyle: .alert)
                errorAlert.addAction(UIAlertAction(title: "OK", style: .default))
                self.present(errorAlert, animated: true)
                return
            }

            do {
                try FileManager.default.moveItem(at: oldURL, to: newURL)
                if let cached = self.snapshotCache.removeValue(forKey: oldURL) {
                    self.snapshotCache[newURL] = cached
                }
                self.mapURLs[indexPath.item] = newURL
                self.collectionView.reloadItems(at: [indexPath])
            } catch {
                let errorAlert = UIAlertController(title: "Error", message: "Could not rename the scene. \(error.localizedDescription)", preferredStyle: .alert)
                errorAlert.addAction(UIAlertAction(title: "OK", style: .default))
                self.present(errorAlert, animated: true)
            }
        })
        present(alert, animated: true)
    }

    /// Find matching ARLog directories and return session count + latest video duration.
    private func loadARLogInfo(for mapName: String, completion: @escaping (_ sessionCount: Int, _ videoDuration: String?) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            guard let docs = try? FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false) else {
                DispatchQueue.main.async { completion(0, nil) }
                return
            }

            let logsRoot = docs.appendingPathComponent("ARLogs")
            guard FileManager.default.fileExists(atPath: logsRoot.path),
                  let contents = try? FileManager.default.contentsOfDirectory(at: logsRoot, includingPropertiesForKeys: nil) else {
                DispatchQueue.main.async { completion(0, nil) }
                return
            }

            let nameWithSpaces = mapName
            let nameWithUnderscores = mapName.replacingOccurrences(of: " ", with: "_")

            let matchingDirs = contents.filter {
                let n = $0.lastPathComponent
                return n.hasPrefix(nameWithSpaces) || n.hasPrefix(nameWithUnderscores)
            }

            let count = matchingDirs.count

            // Pick the latest session by video.mp4 modification date (most recently recorded)
            let logDir: URL? = matchingDirs
                .compactMap { dir -> (URL, Date)? in
                    let videoURL = dir.appendingPathComponent("video.mp4")
                    guard FileManager.default.fileExists(atPath: videoURL.path),
                          let date = try? videoURL.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate else { return nil }
                    return (dir, date)
                }
                .max(by: { $0.1 < $1.1 })
                .map { $0.0 }

            guard let logDir = logDir else {
                DispatchQueue.main.async { completion(count, nil) }
                return
            }

            let videoURL = logDir.appendingPathComponent("video.mp4")

            let asset = AVURLAsset(url: videoURL)
            let durationSeconds = CMTimeGetSeconds(asset.duration)
            guard durationSeconds.isFinite, durationSeconds > 0 else {
                DispatchQueue.main.async { completion(count, nil) }
                return
            }

            let minutes = Int(durationSeconds) / 60
            let seconds = Int(durationSeconds) % 60
            let durationText = String(format: "🎬 %d:%02d", minutes, seconds)

            DispatchQueue.main.async { completion(count, durationText) }
        }
    }

    /// Delete all ARLog directories whose name starts with the map name.
    /// Handles both space-based display names and underscore-based payload names.
    private func deleteMatchingARLogs(for mapName: String) {
        let nameWithSpaces = mapName
        let nameWithUnderscores = mapName.replacingOccurrences(of: " ", with: "_")

        do {
            let docs = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
            let logsRoot = docs.appendingPathComponent("ARLogs")
            guard FileManager.default.fileExists(atPath: logsRoot.path) else { return }

            let contents = try FileManager.default.contentsOfDirectory(at: logsRoot, includingPropertiesForKeys: nil)
            for dir in contents {
                let dirName = dir.lastPathComponent
                if dirName.hasPrefix(nameWithSpaces) || dirName.hasPrefix(nameWithUnderscores) {
                    try FileManager.default.removeItem(at: dir)
                    print("[MapList] Deleted ARLog: \(dirName)")
                }
            }
        } catch {
            print("[MapList] Failed to clean ARLogs: \(error.localizedDescription)")
        }
    }

    // MARK: - Navigation

    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if segue.identifier == "showARScene" {
            guard let viewController = segue.destination as? ViewController else {
                fatalError("Unexpected destination: \(segue.destination)")
            }
            if let mapURL = sender as? URL {
                viewController.mapURLToLoad = mapURL
            } else {
                viewController.mapURLToLoad = nil
            }
        }
    }
}

// MARK: - UICollectionViewDataSource

extension MapListViewController: UICollectionViewDataSource {

    func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
        return mapURLs.count
    }

    func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
        guard let cell = collectionView.dequeueReusableCell(withReuseIdentifier: MapCardCell.reuseIdentifier, for: indexPath) as? MapCardCell else {
            fatalError("Could not dequeue MapCardCell")
        }

        let mapURL = mapURLs[indexPath.item]
        cell.nameLabel.text = mapURL.deletingPathExtension().lastPathComponent

        if let date = try? mapURL.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate {
            cell.dateLabel.text = dateFormatter.string(from: date)
        } else {
            cell.dateLabel.text = "Unknown date"
        }

        // Session count + video length
        let mapName = mapURL.deletingPathExtension().lastPathComponent
        loadARLogInfo(for: mapName) { [weak cell] sessionCount, durationText in
            cell?.sessionCountLabel.text = sessionCount > 0 ? "📁 \(sessionCount)" : nil
            cell?.videoLengthLabel.text = durationText
        }

        if let fileSize = try? mapURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
            cell.sizeLabel.text = ByteCountFormatter.string(fromByteCount: Int64(fileSize), countStyle: .file)
        } else {
            cell.sizeLabel.text = nil
        }

        loadSnapshotImage(for: mapURL) { [weak cell] image in
            cell?.snapshotImageView.image = image
        }

        cell.onDelete = { [weak self] in
            guard let self = self,
                  let currentIndex = self.mapURLs.firstIndex(of: mapURL) else { return }
            self.deleteMap(at: IndexPath(item: currentIndex, section: 0))
        }
        cell.onRename = { [weak self] in
            guard let self = self,
                  let currentIndex = self.mapURLs.firstIndex(of: mapURL) else { return }
            self.renameMap(at: IndexPath(item: currentIndex, section: 0))
        }

        return cell
    }
}

// MARK: - UICollectionViewDelegate & FlowLayout

extension MapListViewController: UICollectionViewDelegate, UICollectionViewDelegateFlowLayout {

    func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
        performSegue(withIdentifier: "showARScene", sender: mapURLs[indexPath.item])
    }

    func collectionView(_ collectionView: UICollectionView, layout collectionViewLayout: UICollectionViewLayout, sizeForItemAt indexPath: IndexPath) -> CGSize {
        let sectionInset: CGFloat = 16
        let interitemSpacing: CGFloat = 12
        let columns: CGFloat = 2
        let totalHorizontalPadding = (sectionInset * 2) + (interitemSpacing * (columns - 1))
        let cardWidth = floor((collectionView.bounds.width - totalHorizontalPadding) / columns)
        let imageHeight = cardWidth * 1.2

        // Assume name is always 2 lines for uniform card height
        let nameFont = UIFont.systemFont(ofSize: 11, weight: .semibold)
        let twoLineNameHeight = ceil(nameFont.lineHeight * 2)

        // 10 (top) + name + 3 (gap) + 13 (date) + 2 (gap) + 13 (info) + 10 (gap) + 30 (buttons) + 12 (bottom)
        let textAndButtonsHeight: CGFloat = 10 + twoLineNameHeight + 3 + 13 + 2 + 13 + 10 + 30 + 12

        return CGSize(width: cardWidth, height: imageHeight + textAndButtonsHeight)
    }
}
