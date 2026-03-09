import 'dart:typed_data';
import 'package:image/image.dart' as img;

//makes sure the image is turned into apng for display
Uint8List ensurePngBytes(img.Image image) {
  final png = img.encodePng(image);
  return Uint8List.fromList(png);
}
