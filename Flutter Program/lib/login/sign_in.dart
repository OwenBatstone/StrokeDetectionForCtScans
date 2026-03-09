import 'package:supabase_flutter/supabase_flutter.dart';

Future<void> signInWithEmail(String email, String password) async {
  final supabase = Supabase.instance.client; 
  final AuthResponse response = await supabase.auth.signInWithPassword(
    email: email,
    password: password
  );
} 