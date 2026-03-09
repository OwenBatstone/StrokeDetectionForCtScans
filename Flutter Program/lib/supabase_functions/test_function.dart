import '../supabase_functions/initilize_supabase.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

//testFunction


Future<List<Map<String, dynamic>>?>fetchInstruments() async{ 
  initSupabase();
  final supabase = Supabase.instance.client; 

  try{ 
    final List<Map<String,dynamic>> data = await supabase
    .from('Employee')
    .select("*");
    return data;
  } on PostgrestException catch (error) {
    print('Error fetching data: ${error.message}');
    return null;
  }catch (error){ 
    print("Error with supabase: ${error}");
    return null; 
  }
}