
class PatientSummary {//full patient summary storage
  final String label;
  final double confidence;
  final List<double> perClassProb;
  final int slicesUsed;
  final int totalSlices;

  const PatientSummary({
    required this.label,
    required this.confidence,
    required this.perClassProb,
    required this.slicesUsed,
    required this.totalSlices,
  });
}