
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