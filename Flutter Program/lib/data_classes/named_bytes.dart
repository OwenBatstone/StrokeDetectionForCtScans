import 'dart:typed_data';

class NamedBytes { //file structure
  final String name;
  final Uint8List bytes;
  NamedBytes({required this.name, required this.bytes});
}