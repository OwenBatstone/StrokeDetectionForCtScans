import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui';


//data classes 
import '../data_classes/type_pred.dart';
import '../data_classes/mask_pred.dart';

//utils
import '../utils/argmax.dart';
import '../utils/softmax.dart';
import '../utils/flatten_to_doubles.dart';


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
    final flat = flattenToDoubles(raw); //flattens the list

    final probs = softmax(flat); //converst hte logits to a probability
    final idx = argmax(probs); //picks whichever class wins

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
  final flat = flattenToDoubles(raw);

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