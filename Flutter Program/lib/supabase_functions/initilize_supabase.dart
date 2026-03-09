import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

Future<void> initSupabase() async { 
  WidgetsFlutterBinding.ensureInitialized(); 
  await Supabase.initialize(
    url:'https://pxanrviwtzuexbcztbnk.supabase.co' ,
    anonKey:'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB4YW5ydml3dHp1ZXhiY3p0Ym5rIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE3MTE2MzMsImV4cCI6MjA4NzI4NzYzM30.G3Px4yuOhE-1Dc3lwzx8IQnLN__1esRM1VpIJwEe-Bg' ,
  ); 
}
