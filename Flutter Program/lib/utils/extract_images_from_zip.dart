import 'dart:typed_data';
import 'package:archive/archive.dart';

import '../data_classes/named_bytes.dart';

List<NamedBytes> extractImagesFromZip(Uint8List zipBytes) {
  final archive = ZipDecoder().decodeBytes(
    zipBytes,
    verify: true,
  ); //checks integrity of the zip
  final out = <NamedBytes>[];

  //goes through each file
  for (final f in archive.files) {
    if (!f.isFile) continue; //skips non images
    final name = f.name.toLowerCase();
    final isImg =
        name.endsWith('.png') ||
        name.endsWith('.jpg') ||
        name.endsWith(
          '.jpeg',
        ); //only take png, jpg and jpeg **NEED TO ADD DICOM EVENTUALLY!
    if (!isImg) continue;

    final content = f.content; //we only handle raw byte lists
    if (content is List<int>) {
      //if the content is a llist of bye, wrap it in a uint8list and store it
      out.add(NamedBytes(name: f.name, bytes: Uint8List.fromList(content)));
    }
  }

  out.sort(
    (a, b) => a.name.compareTo(b.name),
  ); //sort results alphabetically (then numerically)
  return out;
}
