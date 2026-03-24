//flattens the nested list into a List<double>
List<double> flattenToDoubles(dynamic x) {
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
