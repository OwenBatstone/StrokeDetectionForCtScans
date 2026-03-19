
class PatientSummary {//full patient summary storage
  final String? run_by;
  final String label;
  final double confidence;
  final List<double> perClassProb;
  final int slicesUsed;
  final int totalSlices;
  final List<String> slicesIds;
  final List<String> imageUrl;
  final String scanId; 

  const PatientSummary({
    this.scanId = "0",
    this.run_by,
    required this.label,
    required this.confidence,
    required this.perClassProb,
    required this.slicesUsed,
    required this.totalSlices,
    this.slicesIds = const[],
    this.imageUrl = const [],   //this is const so that it can be a const class use copy with
    });
}