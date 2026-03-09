import  'dart:typed_data';
import 'dart:ui';

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