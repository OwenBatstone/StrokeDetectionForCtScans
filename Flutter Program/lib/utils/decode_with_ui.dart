import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'dart:ui' as ui;

Future<img.Image?> decodeWithUi(Uint8List bytes) async {
  try {
    final codec = await ui.instantiateImageCodec(
      bytes,
    ); //creates a image codec to encode bytes
    final frame = await codec
        .getNextFrame(); //decodes the first frame, because most are single framed
    final uiImage = frame.image; //puts it in the engine format

    final byteData = await uiImage.toByteData(
      format: ui.ImageByteFormat.rawRgba,
    ); //convert to raw RGB Byte images
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
