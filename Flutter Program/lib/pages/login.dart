import 'package:flutter/material.dart';
import '../pages/stroke_zip_home.dart';


class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  
  final GlobalKey<FormState> formKey = GlobalKey<FormState>();
  final TextEditingController emailController = TextEditingController(); 
  final TextEditingController passwordController = TextEditingController();
  
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
                ElevatedButton(onPressed: () {
                  if( formKey.currentState!.validate()){ 
                    //add   new class with login logic
                    // Steps planned to implement
                    // hash then encrypt then compare to database 
                    // could aso hash email if we really want?
                    
                  }
                  //navigator push must be moved when we implement login logic
                  //maybe admin flag here so no other extra buttons are needed 
                  Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(builder: (context) => const StrokeZipHome()
                  ));
                }, 
                child: const Text('Login'),
                ),
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