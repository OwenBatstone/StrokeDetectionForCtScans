import 'dart:math' as math;

//turn logits into probabilities
List<double> softmax(List<double> logits) {
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