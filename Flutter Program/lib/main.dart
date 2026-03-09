
import 'package:flutter/material.dart';
import 'package:stroketry3/pages/login.dart';
import '../supabase_functions/initilize_supabase.dart';


Future<void> main() async {
  //initilize connection to supabasedatabase
  await initSupabase();
  runApp(const MyApp());
}
 
class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Stroke ZIP Classifier + Locator',
      theme: ThemeData(useMaterial3: true),
      home: const LoginPage(),
    );
  }
}



