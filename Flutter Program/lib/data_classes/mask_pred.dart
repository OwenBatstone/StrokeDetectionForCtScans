import  'dart:typed_data';
import 'dart:ui';


class MaskPred { //segmentation predition
  final Uint8List? overlayPng;
  final Offset? centroid; 
  final double maskScore;
  const MaskPred(this.overlayPng, this.centroid, this.maskScore);
}