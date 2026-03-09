import 'package:flutter/material.dart';
import '../pages/stroke_zip_home.dart';
import '../login/sign_in.dart';

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

  class _LoginPageState extends State<LoginPage> {
  
  final GlobalKey<FormState> formKey = GlobalKey<FormState>();
  final TextEditingController emailController = TextEditingController(); 
  final TextEditingController passwordController = TextEditingController();
  bool _isLoading = false;

  Future<void> _loginStates() async { 
    if (!formKey.currentState!.validate()) return; 
    
    setState(() => _isLoading = true); 
    try {
      final email = emailController.text.trim(); 
      final password = passwordController.text; 

      await signInWithEmail(email, password);

      if (mounted) { 
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (context) => const StrokeZipHome()),
        );
      }
    } catch (e) { 
      if (mounted) { 
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Login failed: ${e.toString()}')), 
        );
      }
    } finally { 
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  void dispose() {
    emailController.dispose();
    passwordController.dispose();
    super.dispose();
  }                         

  


  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;
    return Scaffold(
    appBar: AppBar(
      title: const Text("Stroke Zip Classifier + Locator"),
      ),body: 
      Padding(
        padding: EdgeInsets.only( //control height from top 
          top: size.height*.1,
          ),
      child :Align(
        alignment: Alignment.topCenter,
        child: 
        SizedBox(
          
          width: size.width *.8,
          child:
          Card(
        elevation: 4,
        child: 
        Padding(
        padding: EdgeInsets.all(24) ,
          
            child: Form(
          key: formKey,
          child: Column(
            mainAxisSize: 
            MainAxisSize.min,
            mainAxisAlignment: 
            MainAxisAlignment.start,
          children: [
            TextFormField(
              controller: emailController,
              decoration: const InputDecoration(
                labelText: "Email",
                border: OutlineInputBorder()
              ),
              //Validator to check for something
              validator: (value) {
                if(value == null || value.isEmpty)
                {
                  return "Enter Email";
                }
                return null;  
              },
              
            ),
            const SizedBox( 
              height : 20
              ),
              TextFormField(
                controller: passwordController,
                obscureText: true,
                decoration: const InputDecoration(
                  labelText: "Password",
                  border: OutlineInputBorder()
                ),
                validator: (value) { 
                  if( value == null || value.isEmpty){ 
                    return "Enter Password";
                  }
                  return null;
                }
                ),
                const SizedBox(height: 20), 
                //elevated button for submit information
                ElevatedButton(onPressed: _isLoading ? null : _loginStates, 
                 child: _isLoading
                 ?const SizedBox(
                  height: 20, 
                  width: 20, 
                  child: CircularProgressIndicator(strokeWidth: 2),
                 )
                 : const Text('Login'), 
                )
          ]
        ),
        )
      )
      ),
      ),
      ) 
      )
      );
    
  
  
  }
}