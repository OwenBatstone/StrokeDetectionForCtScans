import 'package:supabase_flutter/supabase_flutter.dart';
  // ScanRecord, SliceRecord

Future<bool> isAdmin() async {
  print('Users Called');
  final User? user = Supabase.instance.client.auth.currentUser;
  if (user ==null) return false; //if no user return

  final supabase = Supabase.instance.client; //get current instance for context

  final result = await supabase
  .from('Users')
  .select('admin')
  .eq('id', user.id)
  .maybeSingle();
  
  print('user id: ${user.id}'); 
  print('raw result: $result');

  if (result == null) return false;

  
  print('$result');

  return result['admin'] == true;
  

}

