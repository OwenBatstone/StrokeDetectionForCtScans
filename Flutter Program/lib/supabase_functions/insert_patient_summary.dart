import 'package:supabase_flutter/supabase_flutter.dart';
import '../data_classes/paitient_summary.dart';


Future<String?> insertPatientSummary(PatientSummary summary) async {
  final supabase = Supabase.instance.client; 
  
  try {

    final response = await supabase
        .from('Scan_Overview')
        .insert({
          'scan_id' : summary.scanId,
          'run_by' : summary.run_by,
          'prediction': summary.label,
          'confidence': summary.confidence,
          'total_slices': summary.totalSlices,
          'file_url': summary.imageUrl,
        })
        .select()
        .single();

    final scanId = response['scan_id']?.toString();

    for (int i=0; i<summary.slicesIds.length; i++){
      await supabase.from('Scan_Slices').insert({
        'slices_Id':summary.slicesIds[i],
        'scan_id': scanId,
        'file_url': summary.imageUrl.length > i ? summary.imageUrl[i] : null,
        'run_by':summary.run_by,

      }
        
      );
    }

    return scanId;

  } on PostgrestException catch (e) {
    print('Insert summary error: ${e.message}');
    return null;
  }
}