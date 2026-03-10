import 'package:supabase_flutter/supabase_flutter.dart';

Future<void> signInWithEmail(String email, String password) async {
  final supabase = Supabase.instance.client; 
  // ignore: unused_local_variable
  final AuthResponse response = await supabase.auth.signInWithPassword(
    email: email,
    password: password
  );
} 