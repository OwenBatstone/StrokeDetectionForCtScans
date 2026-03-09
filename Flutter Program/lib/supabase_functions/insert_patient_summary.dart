import '../supabase_functions/initilize_supabase.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../data_classes/paitient_summary.dart';


Future<String?> insertPatientSummary(PatientSummary summary) async {
  initSupabase();
  final supabase = Supabase.instance.client; 
  
  try {
    final response = await supabase
        .from('Scan_Overview')
        .insert({
          'prediction': summary.label,
          'confidence': summary.confidence,
          'slices_used': summary.slicesUsed,
          'total_slices': summary.totalSlices,
        })
        .select()
        .single();
    return response['scan_id']?.toString();
  } on PostgrestException catch (e) {
    print('Insert summary error: ${e.message}');
    return null;
  }
}