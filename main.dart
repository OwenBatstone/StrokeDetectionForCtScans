import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:archive/archive.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;

// IMPORTANT: Windows package you said works
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Stroke ZIP Classifier + Locator',
      theme: ThemeData(useMaterial3: true),
      home: const StrokeZipHome(),
    );
  }
}

class StrokeZipHome extends StatefulWidget {
  const StrokeZipHome({super.key});
  @override
  State<StrokeZipHome> createState() => _StrokeZipHomeState();
}

class _StrokeZipHomeState extends State<StrokeZipHome> {
  final _svc = StrokeInferenceService();

  bool _busy = false;
  String _status = 'Upload a .zip with images (png/jpg/jpeg).';
  String _modelInfo = '';

  PatientSummary? _summary;
  List<SliceResult> _rows = [];

  @override
  void dispose() {
    _svc.dispose();
    super.dispose();
  }

  Future<void> _pickZipAndRun() async {
    setState(() {
      _busy = true;
      _status = 'Picking zip...';
      _rows = [];
      _summary = null;
    });

    try {
      final picked = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: const ['zip'],
        withData: true,
      );

      if (picked == null || picked.files.isEmpty) {
        setState(() {
          _busy = false;
          _status = 'Cancelled.';
        });
        return;
      }

      final file = picked.files.single;
      final zipBytes = file.bytes ?? await File(file.path!).readAsBytes();

      setState(() => _status = 'Extracting images...');
      final images = _extractImagesFromZip(zipBytes);

      if (images.isEmpty) {
        setState(() {
          _busy = false;
          _status = 'No images found inside the zip.';
        });
        return;
      }

      setState(() => _status = 'Loading model...');
      await _svc.ensureLoaded();
      setState(() => _modelInfo = _svc.modelInfo);

      setState(() => _status = 'Running inference on ${images.length} slices...');

      final out = <SliceResult>[];

      // Patient-level aggregation: average logits across slices
      final aggLogits = List<double>.filled(StrokeInferenceService.labels.length, 0.0);
      int usedForAgg = 0;

      for (final n in images) {
        // OPTION A: Decode using dart:ui (handles more PNG formats robustly, incl. 16-bit cases)
        final decoded = await decodeWithUi(n.bytes);

        if (decoded == null) {
          out.add(SliceResult(
            fileName: n.name,
            typeLabel: 'Decode failed',
            confidence: 0,
            logits: const [],
            originalPng: n.bytes, // keep original bytes just in case
          ));
          continue;
        }

        final pred = await _svc.predictType(decoded);

        // Aggregate logits
        if (pred.logits.isNotEmpty && pred.logits.length == aggLogits.length) {
          for (int i = 0; i < aggLogits.length; i++) {
            aggLogits[i] += pred.logits[i];
          }
          usedForAgg++;
        }

        // Segmentation (mask + dot)
        final seg = await _svc.predictMask(decoded);

        out.add(SliceResult(
          fileName: n.name,
          typeLabel: pred.label,
          confidence: pred.confidence,
          logits: pred.logits,
          originalPng: _ensurePngBytes(decoded), // consistent display
          maskOverlayPng: seg.overlayPng,
          centroid: seg.centroid,
          maskScore: seg.maskScore,
        ));
      }

      // Patient-level overall prediction
      PatientSummary? summary;
      if (usedForAgg > 0) {
        for (int i = 0; i < aggLogits.length; i++) {
          aggLogits[i] /= usedForAgg.toDouble();
        }
        final probs = _softmax(aggLogits);
        final idx = _argmax(probs);
        summary = PatientSummary(
          label: StrokeInferenceService.labels[idx],
          confidence: probs[idx],
          perClassProb: probs,
          slicesUsed: usedForAgg,
          totalSlices: images.length,
        );
      }

      setState(() {
        _busy = false;
        _rows = out;
        _summary = summary;
        _status = 'Done.';
      });
    } catch (e) {
      setState(() {
        _busy = false;
        _status = 'Error: $e';
      });
    }
  }

  List<_NamedBytes> _extractImagesFromZip(Uint8List zipBytes) {
    final archive = ZipDecoder().decodeBytes(zipBytes, verify: true);
    final out = <_NamedBytes>[];

    for (final f in archive.files) {
      if (!f.isFile) continue;
      final name = f.name.toLowerCase();
      final isImg = name.endsWith('.png') || name.endsWith('.jpg') || name.endsWith('.jpeg');
      if (!isImg) continue;

      final content = f.content;
      if (content is List<int>) {
        out.add(_NamedBytes(name: f.name, bytes: Uint8List.fromList(content)));
      }
    }

    out.sort((a, b) => a.name.compareTo(b.name));
    return out;
  }

  @override
  Widget build(BuildContext context) {
    final summary = _summary;

    return Scaffold(
      appBar: AppBar(title: const Text('Stroke ZIP Classifier + Locator')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            FilledButton.icon(
              onPressed: _busy ? null : _pickZipAndRun,
              icon: const Icon(Icons.upload_file),
              label: const Text('Upload ZIP + Run'),
            ),
            const SizedBox(height: 10),
            Text(_status),
            if (_modelInfo.isNotEmpty) ...[
              const SizedBox(height: 10),
              Text(
                _modelInfo,
                style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
              ),
            ],
            if (summary != null) ...[
              const SizedBox(height: 12),
              _PatientSummaryCard(summary: summary),
            ],
            const SizedBox(height: 10),
            Expanded(
              child: _rows.isEmpty
                  ? const Center(child: Text('No results yet.'))
                  : ListView.separated(
                      itemCount: _rows.length,
                      separatorBuilder: (_, __) => const Divider(height: 1),
                      itemBuilder: (context, i) {
                        final r = _rows[i];
                        final hasMask = r.maskOverlayPng != null && r.centroid != null;

                        return ListTile(
                          title: Text(r.fileName),
                          subtitle: Text(
                            'Type: ${r.typeLabel}  |  ${(r.confidence * 100).toStringAsFixed(1)}%'
                            '${hasMask ? '  |  Mask: ${(r.maskScore * 100).toStringAsFixed(1)}%' : ''}',
                          ),
                          trailing: hasMask ? const Icon(Icons.image_search) : null,
                          onTap: (r.originalPng == null)
                              ? null
                              : () {
                                  Navigator.of(context).push(
                                    MaterialPageRoute(
                                      builder: (_) => SliceViewerScreen(result: r),
                                    ),
                                  );
                                },
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }
}

/// -------------------- OPTION A: UI DECODER --------------------
/// Robust decoding using dart:ui -> RGBA8, then convert to image package Image.
/// Helps with tricky PNG formats (including many 16-bit cases).
Future<img.Image?> decodeWithUi(Uint8List bytes) async {
  try {
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    final uiImage = frame.image;

    final byteData = await uiImage.toByteData(format: ui.ImageByteFormat.rawRgba);
    if (byteData == null) return null;

    final rgba = byteData.buffer.asUint8List();

    final out = img.Image.fromBytes(
      width: uiImage.width,
      height: uiImage.height,
      bytes: rgba.buffer,
      order: img.ChannelOrder.rgba,
    );

    return out;
  } catch (_) {
    return null;
  }
}

/// -------------------- VIEWER SCREEN (IMAGE + MASK + DOT) --------------------

class SliceViewerScreen extends StatelessWidget {
  final SliceResult result;
  const SliceViewerScreen({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    final baseBytes = result.originalPng!;
    final overlayBytes = result.maskOverlayPng;
    final c = result.centroid;

    return Scaffold(
      appBar: AppBar(
        title: Text(result.fileName),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Type: ${result.typeLabel} • ${(result.confidence * 100).toStringAsFixed(1)}%',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 6),
            Text(
              overlayBytes == null || c == null
                  ? 'No location mask available for this slice.'
                  : 'Location mask confidence: ${(result.maskScore * 100).toStringAsFixed(1)}%',
            ),
            const SizedBox(height: 12),
            Expanded(
              child: Center(
                child: AspectRatio(
                  aspectRatio: 1,
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      Image.memory(baseBytes, fit: BoxFit.contain),
                      if (overlayBytes != null) Image.memory(overlayBytes, fit: BoxFit.contain),
                      if (c != null)
                        CustomPaint(
                          painter: _DotPainter(nx: c.dx, ny: c.dy),
                        ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _DotPainter extends CustomPainter {
  final double nx; // normalized 0..1
  final double ny;
  _DotPainter({required this.nx, required this.ny});

  @override
  void paint(Canvas canvas, Size size) {
    final p = Offset(nx.clamp(0, 1) * size.width, ny.clamp(0, 1) * size.height);

    final paintOuter = Paint()..color = Colors.white.withOpacity(0.95);
    final paintInner = Paint()..color = Colors.redAccent.withOpacity(0.95);

    canvas.drawCircle(p, math.max(6, size.shortestSide * 0.02), paintOuter);
    canvas.drawCircle(p, math.max(3.5, size.shortestSide * 0.012), paintInner);
  }

  @override
  bool shouldRepaint(covariant _DotPainter oldDelegate) {
    return oldDelegate.nx != nx || oldDelegate.ny != ny;
  }
}

/// -------------------- SUMMARY CARD --------------------

class _PatientSummaryCard extends StatelessWidget {
  final PatientSummary summary;
  const _PatientSummaryCard({required this.summary});

  @override
  Widget build(BuildContext context) {
    final probs = summary.perClassProb;
    final labels = StrokeInferenceService.labels;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Overall prediction (patient-level)',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 6),
            Text(
              '${summary.label} • ${(summary.confidence * 100).toStringAsFixed(1)}%',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 6),
            Text('Slices used: ${summary.slicesUsed}/${summary.totalSlices}'),
            const SizedBox(height: 10),
            for (int i = 0; i < labels.length; i++)
              Text('${labels[i]}: ${(probs[i] * 100).toStringAsFixed(1)}%'),
          ],
        ),
      ),
    );
  }
}

/// -------------------- ONNX INFERENCE SERVICE --------------------

class StrokeInferenceService {
  // Asset paths MUST match pubspec.yaml
  static const String clsAsset = 'assets/models/stroke_type_classifier_single.onnx';
  static const String segAsset = 'assets/models/stroke_location_segmenter_single.onnx';

  // Keep this label order identical to training
  static const List<String> labels = ['Normal', 'Ischemic', 'Hemorrhagic'];

  static const int clsW = 224;
  static const int clsH = 224;

  static const int segW = 256;
  static const int segH = 256;

  final OnnxRuntime _ort = OnnxRuntime();

  OrtSession? _clsSession;
  OrtSession? _segSession;

  String modelInfo = '';

  Future<void> ensureLoaded() async {
    if (_clsSession != null && _segSession != null) return;

    _clsSession = await _ort.createSessionFromAsset(clsAsset);
    _segSession = await _ort.createSessionFromAsset(segAsset);

    final ins1 = _clsSession!.inputNames;
    final outs1 = _clsSession!.outputNames;
    final ins2 = _segSession!.inputNames;
    final outs2 = _segSession!.outputNames;

    modelInfo = 'CLS inputs: $ins1\nCLS outputs: $outs1\nSEG inputs: $ins2\nSEG outputs: $outs2';
  }

  Future<TypePred> predictType(img.Image src) async {
    final session = _clsSession!;
    final inputName = session.inputNames.isNotEmpty ? session.inputNames.first : 'input';
    final outputName = session.outputNames.isNotEmpty ? session.outputNames.first : 'output';

    // Build [1,3,224,224] float tensor in CHW order (0..1 scaling)
    final chw = _preprocessRgbCHW(src, clsW, clsH);

    final inputs = <String, OrtValue>{
      inputName: await OrtValue.fromList(chw, [1, 3, clsH, clsW]),
    };

    final outputs = await session.run(inputs);
    final outVal = outputs[outputName] ?? outputs.values.first;

    final raw = await outVal!.asList();
    final flat = _flattenToDoubles(raw);

    final probs = _softmax(flat);
    final idx = _argmax(probs);

    return TypePred(
      label: (idx >= 0 && idx < labels.length) ? labels[idx] : 'Class#$idx',
      confidence: probs[idx].clamp(0.0, 1.0),
      logits: flat,
      probs: probs,
    );
  }

Future<MaskPred> predictMask(img.Image src) async {
  final session = _segSession!;
  final inputName = session.inputNames.isNotEmpty ? session.inputNames.first : 'input';
  final outputName = session.outputNames.isNotEmpty ? session.outputNames.first : 'output';

  // Preprocess grayscale for model
  final chw = _preprocessGrayCHW(src, segW, segH);

  final inputs = <String, OrtValue>{
    inputName: await OrtValue.fromList(chw, [1, 1, segH, segW]),
  };

  final outputs = await session.run(inputs);
  final outVal = outputs[outputName] ?? outputs.values.first;

  final raw = await outVal!.asList();
  final flat = _flattenToDoubles(raw);

  final hw = segH * segW;
  if (flat.length < hw) {
    return const MaskPred(null, null, 0.0);
  }

  final start = flat.length - hw;
  final logits = flat.sublist(start);

  // sigmoid
  final probs = logits.map((v) => 1.0 / (1.0 + math.exp(-v))).toList();

  // 🔑 BASE IMAGE (same size as mask)
  final base = img.copyResize(
    src,
    width: segW,
    height: segH,
    interpolation: img.Interpolation.linear,
  );

  final overlay = img.Image.from(base);

  const thr = 0.5;
  double sumX = 0, sumY = 0, sumW = 0;
  int idx = 0;
  int onCount = 0;

  for (int y = 0; y < segH; y++) {
    for (int x = 0; x < segW; x++) {
      final p = probs[idx++];
      if (p >= thr) {
        onCount++;
        sumX += x * p;
        sumY += y * p;
        sumW += p;

        // Paint translucent red ON TOP of the image
        overlay.setPixelRgba(x, y, 255, 0, 0, 120);
      }
    }
  }

  if (onCount < 25 || sumW <= 0) {
    return const MaskPred(null, null, 0.0);
  }

  final cx = sumX / sumW;
  final cy = sumY / sumW;

  final nx = cx / (segW - 1);
  final ny = cy / (segH - 1);

  final maskScore = (onCount / (segW * segH)).clamp(0.0, 1.0);

  final overlayPng = Uint8List.fromList(img.encodePng(overlay));

  return MaskPred(overlayPng, Offset(nx, ny), maskScore);
}


  void dispose() {
    _clsSession?.close();
    _segSession?.close();
    _clsSession = null;
    _segSession = null;
  }

  // ---------- preprocessing helpers ----------

  // RGB -> CHW, 0..1
  List<double> _preprocessRgbCHW(img.Image src, int w, int h) {
    final resized = img.copyResize(src, width: w, height: h, interpolation: img.Interpolation.linear);
    final plane = w * h;
    final out = List<double>.filled(3 * plane, 0);

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final p = resized.getPixel(x, y);
        final i = y * w + x;
        out[i] = p.r / 255.0;
        out[plane + i] = p.g / 255.0;
        out[2 * plane + i] = p.b / 255.0;
      }
    }
    return out;
  }

  // grayscale -> CHW, 0..1
  List<double> _preprocessGrayCHW(img.Image src, int w, int h) {
    final resized = img.copyResize(src, width: w, height: h, interpolation: img.Interpolation.linear);

    final plane = w * h;
    final out = List<double>.filled(plane, 0);

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final p = resized.getPixel(x, y);
        final i = y * w + x;
        final g = (0.299 * p.r + 0.587 * p.g + 0.114 * p.b) / 255.0;
        out[i] = g;
      }
    }
    return out;
  }
}

/// -------------------- DATA MODELS --------------------

class _NamedBytes {
  final String name;
  final Uint8List bytes;
  _NamedBytes({required this.name, required this.bytes});
}

class SliceResult {
  final String fileName;
  final String typeLabel;
  final double confidence;

  final List<double> logits;

  final Uint8List? originalPng;
  final Uint8List? maskOverlayPng;
  final Offset? centroid; // normalized 0..1
  final double maskScore;

  SliceResult({
    required this.fileName,
    required this.typeLabel,
    required this.confidence,
    required this.logits,
    required this.originalPng,
    this.maskOverlayPng,
    this.centroid,
    this.maskScore = 0.0,
  });
}

class PatientSummary {
  final String label;
  final double confidence;
  final List<double> perClassProb;
  final int slicesUsed;
  final int totalSlices;

  const PatientSummary({
    required this.label,
    required this.confidence,
    required this.perClassProb,
    required this.slicesUsed,
    required this.totalSlices,
  });
}

class TypePred {
  final String label;
  final double confidence;
  final List<double> logits;
  final List<double> probs;
  const TypePred({
    required this.label,
    required this.confidence,
    required this.logits,
    required this.probs,
  });
}

class MaskPred {
  final Uint8List? overlayPng;
  final Offset? centroid; // normalized 0..1
  final double maskScore;
  const MaskPred(this.overlayPng, this.centroid, this.maskScore);
}

/// -------------------- MATH / UTILS --------------------

int _argmax(List<double> a) {
  var bestI = 0;
  var bestV = -double.infinity;
  for (var i = 0; i < a.length; i++) {
    if (a[i] > bestV) {
      bestV = a[i];
      bestI = i;
    }
  }
  return bestI;
}

List<double> _softmax(List<double> logits) {
  if (logits.isEmpty) return const [];
  final m = logits.reduce(math.max);
  double denom = 0;
  for (final v in logits) {
    denom += math.exp(v - m);
  }
  final out = <double>[];
  for (final v in logits) {
    out.add(math.exp(v - m) / denom);
  }
  return out;
}

List<double> _flattenToDoubles(dynamic x) {
  final out = <double>[];

  void rec(dynamic v) {
    if (v is List) {
      for (final e in v) rec(e);
    } else if (v is num) {
      out.add(v.toDouble());
    }
  }

  rec(x);
  return out;
}

Uint8List _ensurePngBytes(img.Image image) {
  final png = img.encodePng(image);
  return Uint8List.fromList(png);
}
