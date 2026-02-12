
//Imports
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:archive/archive.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
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
      _busy = true; //disables button
      _status = 'Picking zip...'; //message indicating user is picking a zip
      _rows = []; //clears any results if they exist
      _summary = null; //same as above
    });

    try {
      final picked = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: const ['zip'], //makes the user have to pick a zip file **MAY WANT TO CHANGE**
        withData: true,
      );

      //if user cancels out of picking a files
      if (picked == null || picked.files.isEmpty) {
        setState(() {
          _busy = false; 
          _status = 'Cancelled.';
        });
        return;
      }

      final file = picked.files.single; //grabs the file
      final zipBytes = file.bytes ?? await File(file.path!).readAsBytes(); //chcks for in memory bytes, if not reads from path

      //extracting file
      setState(() => _status = 'Extracting'); 
      final images = _extractImagesFromZip(zipBytes);

      //error handeling, if theres no image inside the zip
      if (images.isEmpty) {
        setState(() {
          _busy = false;
          _status = 'No images found';
        });
        return;
      }

      setState(() => _status = 'Loading ONNX');  //loading onnx file for classification and segmentation
      await _svc.ensureLoaded();
      setState(() => _modelInfo = _svc.modelInfo);

      setState(() => _status = 'Running inference on ${images.length} slices...'); //shows how many slices are analyzed (takes about 1 mins per 100-200)

      //slice by slice results are here, give state at the end
      final out = <SliceResult>[];

      //sum of the slice averages logits
      final aggLogits = List<double>.filled(StrokeInferenceService.labels.length, 0.0);
      int usedForAgg = 0;

      //goes through each image in the zip
      for (final n in images) {
        final decoded = await decodeWithUi(n.bytes); //uses dartUI so we dont have to worry about weird formats

        if (decoded == null) { //if it failes record a new row and continue
          out.add(SliceResult(
            fileName: n.name, //orginial file name
            typeLabel: 'Decode failed', //failed, no guess
            confidence: 0, //no confidence (failed)
            logits: const [], //no Logits
            originalPng: n.bytes, //keep original bytes just in case
          ));
          continue; //just skip this one
        }

        final pred = await _svc.predictType(decoded); //clasification on slice

        //if size matches, add logits to the patient-level accumulator
        if (pred.logits.isNotEmpty && pred.logits.length == aggLogits.length) {
          for (int i = 0; i < aggLogits.length; i++) {
            aggLogits[i] += pred.logits[i];
          }
          usedForAgg++; //counter for the amount of slices contributing
        }

        //Segmentation (to find Leissions)
        final seg = await _svc.predictMask(decoded);

        out.add(SliceResult(
          fileName: n.name,
          typeLabel: pred.label,
          confidence: pred.confidence,
          logits: pred.logits, ///raw logits
          originalPng: _ensurePngBytes(decoded), //cre-encoded to keep the display consistent
          maskOverlayPng: seg.overlayPng, //bytes of the overlay
          centroid: seg.centroid, //dot location
          maskScore: seg.maskScore, //about how much of the image is filled by the leision
        ));
      }

      //Overall prediciton
      PatientSummary? summary; //checks if we have usable slices
      if (usedForAgg > 0) { //only compute if we have atleast one usable slice
        for (int i = 0; i < aggLogits.length; i++) { 
          aggLogits[i] /= usedForAgg.toDouble(); //converts sumed logits to average logits
        }
        final probs = _softmax(aggLogits); //turns average into probabilities
        final idx = _argmax(probs); //find the label with the highest probability
        //Overall summary object *****COME HERE FOR DATABASE STUFF!!*****
        summary = PatientSummary(
          label: StrokeInferenceService.labels[idx], //final choice
          confidence: probs[idx], //probability
          perClassProb: probs, //all classes probabilities
          slicesUsed: usedForAgg, //the slices that contributed
          totalSlices: images.length, //total slices in the zip
        );
      }
      //set the UI once at the end (way faster then doing it after each slice is ready, could change later if we value showing them as they come)
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
  //extract images from zip files
  List<_NamedBytes> _extractImagesFromZip(Uint8List zipBytes) {
    final archive = ZipDecoder().decodeBytes(zipBytes, verify: true); //checks integrity of the zip
    final out = <_NamedBytes>[];

    //goes through each file
    for (final f in archive.files) {
      if (!f.isFile) continue; //skips non images
      final name = f.name.toLowerCase();
      final isImg = name.endsWith('.png') || name.endsWith('.jpg') || name.endsWith('.jpeg'); //only take png, jpg and jpeg **NEED TO ADD DICOM EVENTUALLY!
      if (!isImg) continue;

      final content = f.content; //we only handle raw byte lists
      if (content is List<int>) { //if the content is a llist of bye, wrap it in a uint8list and store it
        out.add(_NamedBytes(name: f.name, bytes: Uint8List.fromList(content)));
      }
    }

    out.sort((a, b) => a.name.compareTo(b.name)); //sort results alphabetically (then numerically)
    return out;
  }

  @override
  Widget build(BuildContext context) {
    final summary = _summary;

    return Scaffold( //basic screen for now
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

//decoding using dart:ui, helps to decodes weird png formats
Future<img.Image?> decodeWithUi(Uint8List bytes) async {
  try {
    final codec = await ui.instantiateImageCodec(bytes); //creates a image codec to encode bytes
    final frame = await codec.getNextFrame(); //decodes the first frame, because most are single framed
    final uiImage = frame.image; //puts it in the engine format

    final byteData = await uiImage.toByteData(format: ui.ImageByteFormat.rawRgba); //convert to raw RGB Byte images
    if (byteData == null) return null;

    final rgba = byteData.buffer.asUint8List();

    final out = img.Image.fromBytes(
      width: uiImage.width,
      height: uiImage.height,
      bytes: rgba.buffer, //buffer containing rgb bytes
      order: img.ChannelOrder.rgba, //chanel ordering
    );

    //returns decoded image object
    return out;
  } catch (_) {
    return null; //if for some reason it failes, it returns null
  }
}

//screen with all the slice images ##COME HERE FOR UI CHANGES
class SliceViewerScreen extends StatelessWidget {
  final SliceResult result;
  const SliceViewerScreen({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    final baseBytes = result.originalPng!; //base image
    final overlayBytes = result.maskOverlayPng; //mask bytes or null if none
    final c = result.centroid; //centroid coordinates, or null if none exists

    return Scaffold(
      appBar: AppBar(
        title: Text(result.fileName),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column( //vertical layoutm with text and hte images
          crossAxisAlignment: CrossAxisAlignment.start, //moves all text to the left
          children: [
            Text(
              'Type: ${result.typeLabel} • ${(result.confidence * 100).toStringAsFixed(1)}%',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 6),
            Text(
              overlayBytes == null || c == null
                  ? 'No location mask available for this slice.' //if it cant show a location mask
                  : 'Location mask confidence: ${(result.maskScore * 100).toStringAsFixed(1)}%',
            ),
            const SizedBox(height: 12),
            Expanded(
              child: Center(
                child: AspectRatio(
                  aspectRatio: 1, //makes it as close to a square as we can
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      Image.memory(baseBytes, fit: BoxFit.contain),
                      if (overlayBytes != null) Image.memory(overlayBytes, fit: BoxFit.contain),
                      if (c != null) //if centroid exists, put it in place
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

class _DotPainter extends CustomPainter { //for making the dot based off the normalized coordinates
  //normalized coordinates
  final double nx; 
  final double ny;

  //painter constructor
  _DotPainter({required this.nx, required this.ny});

  @override
  void paint(Canvas canvas, Size size) {
    //converts the normalized coords into the pixel coords
    final p = Offset(nx.clamp(0, 1) * size.width, ny.clamp(0, 1) * size.height);

    final paintOuter = Paint()..color = Colors.white.withOpacity(0.95); //white ouutline of dot
    final paintInner = Paint()..color = Colors.redAccent.withOpacity(0.95); //red inside of dot

    canvas.drawCircle(p, math.max(6, size.shortestSide * 0.02), paintOuter);
    canvas.drawCircle(p, math.max(3.5, size.shortestSide * 0.012), paintInner);
  }

  @override
  bool shouldRepaint(covariant _DotPainter oldDelegate) { //if dot moves repaint
    return oldDelegate.nx != nx || oldDelegate.ny != ny;
  }
}


class _PatientSummaryCard extends StatelessWidget { //for showing the final prediction
  final PatientSummary summary;
  const _PatientSummaryCard({required this.summary}); //summary data about patient

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
              'Overall prediction',
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
  //Asset paths
  static const String clsAsset = 'assets/models/stroke_type_classifier_single.onnx';
  static const String segAsset = 'assets/models/stroke_location_segmenter_single.onnx';

  //same as training label order
  static const List<String> labels = ['Normal', 'Ischemic', 'Hemorrhagic'];

  static const int clsW = 224; //base width of classifier
  static const int clsH = 224; //base height of classifier

  static const int segW = 256; //base width of segmenter
  static const int segH = 256; //base height of segmenter

  final OnnxRuntime _ort = OnnxRuntime();

  OrtSession? _clsSession;
  OrtSession? _segSession;

  String modelInfo = '';

  Future<void> ensureLoaded() async { //makes sure the onnx is loaded
    if (_clsSession != null && _segSession != null) return;
    //loads the classifier and segmenter from the onnx
    _clsSession = await _ort.createSessionFromAsset(clsAsset);
    _segSession = await _ort.createSessionFromAsset(segAsset);

    //reads the input/output names for both the classifier and segmenter
    final ins1 = _clsSession!.inputNames;
    final outs1 = _clsSession!.outputNames;
    final ins2 = _segSession!.inputNames;
    final outs2 = _segSession!.outputNames;

    modelInfo = 'CLS inputs: $ins1\nCLS outputs: $outs1\nSEG inputs: $ins2\nSEG outputs: $outs2'; //saves a readable string for debugging
  }

  //rune classification on one image
  Future<TypePred> predictType(img.Image src) async {
    final session = _clsSession!;
    final inputName = session.inputNames.isNotEmpty ? session.inputNames.first : 'input'; //choses first input name
    final outputName = session.outputNames.isNotEmpty ? session.outputNames.first : 'output'; //choses first output name

    //Build [1,3,224,224] float tensor in CHW order (0..1 scaling)
    final chw = _preprocessRgbCHW(src, clsW, clsH);

    final inputs = <String, OrtValue>{
      inputName: await OrtValue.fromList(chw, [1, 3, clsH, clsW]),
    };

    final outputs = await session.run(inputs); //runs the inference
    final outVal = outputs[outputName] ?? outputs.values.first;

    final raw = await outVal.asList(); //converets the output to a dart list
    final flat = _flattenToDoubles(raw); //flattens the list

    final probs = _softmax(flat); //converst hte logits to a probability
    final idx = _argmax(probs); //picks whichever class wins

    return TypePred( //returns the predicition result with structure (for a individfual slice)
      label: (idx >= 0 && idx < labels.length) ? labels[idx] : 'Class#$idx',
      confidence: probs[idx].clamp(0.0, 1.0),
      logits: flat,
      probs: probs,
    );
  }

Future<MaskPred> predictMask(img.Image src) async { //runs segmentation to make the mask and dot
  final session = _segSession!;
  final inputName = session.inputNames.isNotEmpty ? session.inputNames.first : 'input';
  final outputName = session.outputNames.isNotEmpty ? session.outputNames.first : 'output';

  //Preprocess grayscale for model
  final chw = _preprocessGrayCHW(src, segW, segH);

  final inputs = <String, OrtValue>{ //gets the grayscale tensor from the onnx
    inputName: await OrtValue.fromList(chw, [1, 1, segH, segW]),
  };

  final outputs = await session.run(inputs); //runs the inference
  final outVal = outputs[outputName] ?? outputs.values.first;

  final raw = await outVal.asList();
  final flat = _flattenToDoubles(raw);

  final hw = segH * segW; //number of pixels in mask
  if (flat.length < hw) { //if its a weird shape or too smalre return no mask
    return const MaskPred(null, null, 0.0);
  }

  final start = flat.length - hw;
  final logits = flat.sublist(start);

  //gets probabiltiies for each logit using sigmoid
  final probs = logits.map((v) => 1.0 / (1.0 + math.exp(-v))).toList();

  //Base image size (may need to resize here dependent on input)
  final base = img.copyResize(
    src,
    width: segW,
    height: segH,
    interpolation: img.Interpolation.linear,
  );

  final overlay = img.Image.from(base); //copies base image so we can paint over it

  const thr = 0.5; //pixel mask threshold
  double sumX = 0, sumY = 0, sumW = 0; //centroid sum determinants
  int idx = 0;
  int onCount = 0;

  for (int y = 0; y < segH; y++) { //goes through each pixel location in mask
    for (int x = 0; x < segW; x++) {
      final p = probs[idx++];
      if (p >= thr) {
        onCount++;
        sumX += x * p;
        sumY += y * p;
        sumW += p;

        //Paint translucent red ON TOP of the image
        overlay.setPixelRgba(x, y, 255, 0, 0, 120);
      }
    }
  }

  if (onCount < 25 || sumW <= 0) { //if the mask is tiny or weird return no mask
    return const MaskPred(null, null, 0.0);
  }

  //computing centroid
  final cx = sumX / sumW;
  final cy = sumY / sumW;

  //normalize the centroids location or weird images and ui drawing
  final nx = cx / (segW - 1);
  final ny = cy / (segH - 1);

  final maskScore = (onCount / (segW * segH)).clamp(0.0, 1.0); //amount of pixels that are in the mask

  final overlayPng = Uint8List.fromList(img.encodePng(overlay)); //encodes the overlay image

  return MaskPred(overlayPng, Offset(nx, ny), maskScore); //returns overlay, centroid and score
}


  void dispose() {
    _clsSession?.close();
    _segSession?.close();
    _clsSession = null;
    _segSession = null;
  }

  //preprocessing helpers

  // RGB (the classifier needs 3 channels)
  List<double> _preprocessRgbCHW(img.Image src, int w, int h) {
    final resized = img.copyResize(src, width: w, height: h, interpolation: img.Interpolation.linear); //resize to expected model input size
    final plane = w * h; //number of pixels per channel
    final out = List<double>.filled(3 * plane, 0); //output tensor (in RGB)

    for (int y = 0; y < h; y++) { //goes pixel by pixels
      for (int x = 0; x < w; x++) {
        final p = resized.getPixel(x, y);
        final i = y * w + x;
        //normalize each pixel
        out[i] = p.r / 255.0; //R
        out[plane + i] = p.g / 255.0; //G
        out[2 * plane + i] = p.b / 255.0; //B
      }
    }
    return out;
  }

  // grayscale (Segmenter only needs 1 channel)
  List<double> _preprocessGrayCHW(img.Image src, int w, int h) {
    final resized = img.copyResize(src, width: w, height: h, interpolation: img.Interpolation.linear); //resize segmenter input size

    final plane = w * h; //number of pixels
    final out = List<double>.filled(plane, 0); //input is just the 1 plane

    for (int y = 0; y < h; y++) { //goes through each pixel
      for (int x = 0; x < w; x++) {
        final p = resized.getPixel(x, y);
        final i = y * w + x;
        final g = (0.299 * p.r + 0.587 * p.g + 0.114 * p.b) / 255.0; //converts rgb to grayscale using luminiosity, then normalizes
        out[i] = g; //store grayscale value
      }
    }
    return out;
  }
}

//Data Models

class _NamedBytes { //file structure
  final String name;
  final Uint8List bytes;
  _NamedBytes({required this.name, required this.bytes});
}

class SliceResult { ///stores all outputs for a single slice
  final String fileName;
  final String typeLabel;
  final double confidence;

  final List<double> logits;

  final Uint8List? originalPng;
  final Uint8List? maskOverlayPng;
  final Offset? centroid; // normalized 0..1
  final double maskScore;

  SliceResult({ //for slice results
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

class PatientSummary {//full patient summary storage
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

class TypePred { //classification prediction
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

class MaskPred { //segmentation predition
  final Uint8List? overlayPng;
  final Offset? centroid; 
  final double maskScore;
  const MaskPred(this.overlayPng, this.centroid, this.maskScore);
}

//Utils

//returns the largest value in teh list
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

//turn logits into probabilities
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

List<double> _flattenToDoubles(dynamic x) { //flattens the nested list into a List<double>
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

//makes sure the image is turned into apng for display
Uint8List _ensurePngBytes(img.Image image) {
  final png = img.encodePng(image);
  return Uint8List.fromList(png);
}
