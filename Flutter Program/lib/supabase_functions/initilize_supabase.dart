//Function to initilize supabase Used in Main. 
//Should be called once

import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

Future<void> initSupabase() async { 
  WidgetsFlutterBinding.ensureInitialized(); 
  await Supabase.initialize(
    url:'https://pxanrviwtzuexbcztbnk.supabase.co' ,
    anonKey:'sb_publishable_SY7RXWMI54IYXOEM8HuhdA_Bp9iXGPs' ,
  ); 
}
